"""Interactive Streamlit term harvesting: TF-IDF, MiniLM semantic, hybrid RRF.

No LLM generation. All returned candidates occur in retrieved PubMed records.
The lightweight ONNX backend is loaded only when a semantic mode is selected.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime, timezone

import streamlit as st

from pubmed_proximity import PubMedError, fetch_pubmed_candidates
from ris_proximity import parse_ris, ris_bytes
from term_harvesting import title_tfidf, ranked_mesh, rows_to_csv
from semantic_ranking import (
    MODEL_ID, RRF_K, MAX_EMBED_CANDIDATES, EmbeddingError,
    load_fastembed_model, rerank_title_terms, default_semantic_focus,
    compare_top_k, annotate_seed_presence,
)


@st.cache_resource(show_spinner=False)
def _get_embedder():
    """Reuse a single CPU ONNX MiniLM model between app reruns."""
    return load_fastembed_model()


@st.cache_data(ttl=3600, max_entries=30, show_spinner=False)
def _cached_semantic(
    rows: list[dict], focus: str, mode: str, semantic_weight: float,
    top_n: int, diversify: bool, diversity_weight: float,
    term_key: str, score_key: str, score_label: str,
) -> list[dict]:
    return rerank_title_terms(
        rows, focus=focus, encoder=_get_embedder(), mode=mode,
        semantic_weight=semantic_weight, top_n=top_n,
        diversify=diversify, diversity_weight=diversity_weight,
        term_key=term_key, score_key=score_key, base_rank_label=score_label,
    )


def _rank_columns(mode: str, is_title: bool, diversify: bool) -> list[str]:
    if is_title:
        base = ["Rank", "Title term", "Mean TF-IDF",
                "Document frequency", "% titles", "Occurrences"]
    else:
        base = ["Rank", "MeSH descriptor", "Indexed records",
                "% indexed records", "Major topic records", "% all retrieved"]
    if mode == "TF-IDF / Frequency":
        return base
    prefix = base[:2] + ["Semantic similarity", "Semantic rank"]
    prefix += ["TF-IDF rank" if is_title else "Frequency rank"]
    if mode == "Hybrid":
        prefix.append("Hybrid RRF")
    if diversify:
        prefix.append("MMR score")
    return prefix + base[2:]


def render() -> None:
    st.subheader("Term Harvesting from PubMed")
    st.caption(
        "MeSH search → title words/phrases, or TIAB search → assigned MeSH descriptors. "
        "Compare statistical ranking with semantic similarity "
        "and a hybrid ranking. No generative AI."
    )
    direction = st.radio(
        "Analysis direction",
        ["MeSH → Title terms", "TIAB → MeSH descriptors"],
        horizontal=True, key="harvest_direction",
    )
    is_title = direction.startswith("MeSH")
    default_query = (
        '"Students"[mh] AND "Social Media"[mh]' if is_title
        else '("social media"[tiab] AND (academic[tiab] OR university[tiab]))'
    )
    with st.form("harvest_pubmed_form"):
        st.caption(
            "PubMed syntax: e.g. 'Students'[mh] with explosion or "
            "'Students'[mh:noexp] without explosion; free text uses [tiab]."
        )
        query = st.text_area(
            "PubMed query", value=default_query,
            key=("harvest_query_mesh" if is_title else "harvest_query_tiab"),
            height=95,
        )
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            sample_limit = st.number_input(
                "Maximum publications to retrieve", min_value=1,
                max_value=10000, value=500, step=100, key="harvest_limit")
        with c2:
            email = st.text_input("NCBI email (optional)", key="harvest_email")
        with c3:
            api_key = st.text_input(
                "NCBI API key (optional)", type="password", key="harvest_key")
        fetch = st.form_submit_button(
            "Retrieve from PubMed", type="primary")

    if fetch:
        if not query.strip():
            st.error("Enter a PubMed query.")
        elif is_title and not re.search(
            r"\[(?:mh|mesh|majr|mesh terms)(?::noexp)?\]", query, re.I
        ):
            st.error("For MeSH → Title, include at least one MeSH [mh] tag.")
        elif not is_title and not re.search(
            r"\[(?:tiab|title/abstract)\]", query, re.I
        ):
            st.error("For TIAB → MeSH, include at least one [tiab] tag.")
        else:
            st.session_state.pop("harvest_data", None)
            bar = st.progress(0.0)
            try:
                with st.spinner("PubMed ESearch / EFetch"):
                    retrieved = fetch_pubmed_candidates(
                        query.strip(), int(sample_limit), email=email,
                        api_key=api_key,
                        progress=lambda done, total: bar.progress(
                            done / total if total else 1.0
                        ),
                    )
                st.session_state.harvest_data = (
                    retrieved, direction, query.strip()
                )
                st.success(
                    f"{retrieved.fetched:,} of "
                    f"{retrieved.candidate_count:,} hits downloaded."
                )
            except (PubMedError, ValueError) as exc:
                st.error(f"PubMed retrieval failed: {exc}")
            finally:
                bar.empty()

    if "harvest_data" not in st.session_state:
        return
    retrieved, saved_direction, saved_query = st.session_state.harvest_data
    if direction != saved_direction or query.strip() != saved_query:
        st.warning("Suchmodus oder Abfrage geändert – bitte erneut Retrieve from PubMed.")
        return

    records = retrieved.records
    indexed = sum(bool(x.fields.get("MH")) for x in records)
    titled = sum(bool(x.title) for x in records)
    metrics = st.columns(4)
    metrics[0].metric("PubMed hits", f"{retrieved.candidate_count:,}")
    metrics[1].metric("Retrieved", f"{len(records):,}")
    metrics[2].metric("With title", f"{titled:,}")
    metrics[3].metric("MeSH-indexed", f"{indexed:,}")

    if retrieved.candidate_count != len(records):
        st.warning(
            f"Only {len(records):,} of {retrieved.candidate_count:,} hits were "
            "analyzed (sorted by PubMed relevance). Rankings apply "
            "only to the retrieved sample, not to all hits."
        )
    if not records:
        st.info("No records retrieved.")
        return

    st.markdown("#### Ranking and filters")
    mode = st.radio(
        "Ranking method",
        ["TF-IDF / Frequency", "Semantic", "Hybrid"],
        horizontal=True, key="harvest_ranking_mode",
    )
    options = st.columns(4)
    with options[0]:
        top_n = st.number_input(
            "Show top N", min_value=10, max_value=1000,
            value=100, step=10, key="harvest_top_n")
    if is_title:
        if not titled:
            st.error("No titles available.")
            return
        with options[1]:
            min_df = st.number_input(
                "Minimum title document frequency", min_value=1,
                max_value=titled, value=min(2, titled),
                step=1, key="harvest_min_df")
        with options[2]:
            ngram = st.selectbox(
                "N-grams", [1, 2, 3], index=1, key="harvest_ngram",
                format_func=lambda n: (
                    "Single words" if n == 1
                    else f"Single words up to {n}-word phrases"
                ))
        with options[3]:
            remove_query = st.checkbox(
                "Exclude search terms", value=True, key="harvest_exclude_query")
        extra_stopwords = st.text_input(
            "Additional English stop words (optional)",
            key="harvest_extra_stop")
        title_args = {
            "min_df": int(min_df), "max_ngram": int(ngram),
            "query": saved_query, "remove_query_terms": bool(remove_query),
            "extra_stopwords": extra_stopwords,
        }
        term_key, score_key, rank_label = (
            "Title term", "Mean TF-IDF", "TF-IDF rank"
        )
        base = title_tfidf(
            records, top_n=MAX_EMBED_CANDIDATES, **title_args)
        st.caption(
            "TF-IDF: smoothed IDF + L2-normalized document weights. "
            "Candidates come exclusively from titles; "
            "stop words do not create artificial phrases."
        )
        filters = dict(title_args)
    else:
        with options[1]:
            exclude_checks = st.checkbox(
                "Exclude general MeSH check tags",
                value=True, key="harvest_exclude_checktags")
        term_key, score_key, rank_label = (
            "MeSH descriptor", "Indexed records", "Frequency rank"
        )
        base = ranked_mesh(
            records, top_n=MAX_EMBED_CANDIDATES,
            exclude_check_tags=bool(exclude_checks))
        filters = {"exclude_check_tags": bool(exclude_checks)}
        st.caption(
            "Base ranking: number of actually assigned MeSH descriptors per "
            "indexed publication. No automatic thesaurus expansion."
        )
        if not indexed:
            st.error("Keine MeSH-indexeden Datensätze im Abruf.")
        elif indexed < len(records):
            st.warning(
                f"{len(records) - indexed:,} publications without MeSH indexing "
                "are not counted as negatives. Percentages refer "
                "to the {indexed:,} indexed publications."
            )

    score_weight = 0.6
    pool_size = min(600, MAX_EMBED_CANDIDATES)
    diversify = False
    diversity_weight = 0.15
    focus = ""
    if mode != "TF-IDF / Frequency":
        st.markdown("#### Semantic model")
        st.caption(
            "Original English all-MiniLM-L6-v2 via the lightweight "
            "FastEmbed ONNX implementation. On first use, the model is "
            "downloaded to the Streamlit server and run locally. "
            "No search terms are generated."
        )
        focus = st.text_input(
            "Semantic target concept (in English)",
            value=default_semantic_focus(saved_query),
            help="Beispiel: academic self-concept among university students. "
                 "Use a specific target concept for each construct.",
            key="harvest_semantic_focus",
        )
        tuning = st.columns([2, 2, 2])
        with tuning[0]:
            pool_size = st.slider(
                "Embedding candidates",
                min_value=100, max_value=MAX_EMBED_CANDIDATES,
                value=600, step=100, key="harvest_semantic_pool",
                help="Candidates are preselected by TF-IDF / frequency. "
                     "Terms outside the candidate pool cannot improve through semantic ranking.",
            )
        with tuning[1]:
            if mode == "Hybrid":
                score_weight = st.slider(
                    "Semantic weight in hybrid RRF",
                    min_value=0.0, max_value=1.0,
                    value=0.6, step=0.05,
                    key="harvest_semantic_weight",
                    help="0 = purely statistical ranks, 1 = purely semantic "
                         "ranks; k=60 for reciprocal rank fusion.",
                )
        with tuning[2]:
            diversify = st.checkbox(
                "Reduce redundancy (MMR)", value=False,
                key="harvest_diversify",
                help="Optional MMR step to downrank very similar "
                     "suggestions in the output.",
            )
        if diversify:
            diversity_weight = st.slider(
                "Redundancy penalty", 0.0, 0.45, 0.15, step=0.05,
                key="harvest_diversity_weight")
        st.warning(
            "Semantic similarity does not imply construct equivalence: "
            "Related constructs such as academic self-efficacy and academic "
            "self-concept must be evaluated separately."
        )

    rows = base.rows[:int(top_n)]
    if mode != "TF-IDF / Frequency" and base.rows:
        try:
            if len(focus.strip()) < 3:
                st.error("Enter an English target concept.")
                return
            with st.spinner(
                "Computing MiniLM embeddings (initial model download required)"
            ):
                rows = _cached_semantic(
                    base.rows[:int(pool_size)], focus.strip(),
                    "semantic" if mode == "Semantic" else "hybrid",
                    float(score_weight), int(top_n), bool(diversify),
                    float(diversity_weight), term_key, score_key, rank_label,
                )
            overlap = compare_top_k(
                base.rows, rows, k=min(20, len(rows)),
                term_key=term_key,
            )
            st.caption(
                f"Of the top {overlap['Top K']} suggestions, "
                f"{overlap['Shared terms']} also appeared in the statistical top "
                f"{overlap['Top K']}. {len(overlap['New terms'])} "
                "are new relative to that ranking."
            )
            if len(base.rows) > int(pool_size):
                st.caption(
                    f"Semantically scored: {min(len(base.rows), int(pool_size)):,} "
                    f"statistically preselected candidates. Other "
                    "terms cannot rise in the semantic/hybrid ranking."
                )
        except (EmbeddingError, ValueError, RuntimeError) as exc:
            st.error(f"Semantic ranking unavailable: {exc}")
            st.info("TF-IDF / Frequency still works without the model.")
            return

    seed_records = []
    if is_title:
        st.markdown("#### Optional seed-paper check")
        uploaded = st.file_uploader(
            "Upload seed papers as RIS (optional)",
            type=["ris"], key="harvest_seed_ris",
            help="Only counts whether a suggested term occurs in the title/abstract "
                 "of known relevant papers. This is not a "
                 "PubMed retrieval validation or sensitivity estimate.",
        )
        if uploaded:
            if uploaded.size > 10 * 1024 * 1024:
                st.warning("Seed RIS file too large (maximum 10 MB).")
            else:
                seed_records, seed_encoding = parse_ris(uploaded.getvalue())
                if seed_records:
                    rows = annotate_seed_presence(rows, seed_records)
                    st.caption(
                        f"{len(seed_records)} seed papers from RIS ({seed_encoding}) "
                        "analyzed: exact phrase in title or abstract, without "
                        "inferring PubMed query retrieval performance."
                    )
    if not rows:
        st.info("No term meets the selection criteria.")
    columns = _rank_columns(mode, is_title, diversify)
    if seed_records:
        columns += ["Seed records containing term", "% seed records"]
    st.dataframe(
        [{col: row.get(col, "") for col in columns} for row in rows],
        use_container_width=True, hide_index=True,
    )

    meta = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "direction": saved_direction,
        "ranking_mode": mode,
        "pubmed_query": saved_query,
        "pubmed_translation": retrieved.translation,
        "pubmed_sort": retrieved.sort,
        "total_pubmed_hits": retrieved.candidate_count,
        "retrieved": len(records),
        "complete_retrieval": len(records) == retrieved.candidate_count,
        "title_count": titled,
        "mesh_indexed": indexed,
        "analysis_docs": base.included_docs,
        "missing_field_docs": base.missing_docs,
        "statistical_method": base.note,
        "filters": filters,
        "candidate_pool": min(int(pool_size), len(base.rows))
                          if mode != "TF-IDF / Frequency" else len(base.rows),
        "model": MODEL_ID if mode != "TF-IDF / Frequency" else None,
        "embedding_runtime": "FastEmbed (CPU ONNX)"
                             if mode != "TF-IDF / Frequency" else None,
        "focus": focus if mode != "TF-IDF / Frequency" else None,
        "hybrid_method": (
            f"weighted RRF with k={RRF_K}, semantic_weight={score_weight}"
            if mode == "Hybrid" else None
        ),
        "diversification": (
            f"greedy MMR, penalty={diversity_weight}"
            if diversify else None
        ),
        "seed_records": len(seed_records),
        "seed_validation_note": (
            "Title/abstract term presence ONLY, not additional PubMed "
            "retrieval or evidence of improved search sensitivity."
            if seed_records else None
        ),
    }
    table_csv = rows_to_csv(rows, columns)
    buttons = st.columns(2)
    with buttons[0]:
        st.download_button(
            "Ranking as CSV", table_csv,
            "searchbridge_harvested_terms.csv", "text/csv",
            use_container_width=True,
        )
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("ranked_terms.csv", table_csv)
        z.writestr("search_method.json",
                   json.dumps(meta, ensure_ascii=False, indent=2))
        z.writestr("retrieved_pubmed_records.ris", ris_bytes(records))
        if seed_records:
            z.writestr("seed_papers.ris", ris_bytes(seed_records))
    with buttons[1]:
        st.download_button(
            "CSV + search log + RIS (ZIP)", archive.getvalue(),
            "searchbridge_harvesting.zip", "application/zip",
            type="primary", use_container_width=True,
        )
    with st.expander("Methods and search log"):
        st.json(meta)
    st.info(
        "The results are candidates for manual review. "
        "A higher rank does not establish synonymy or "
        "additional retrieval recall."
    )
