"""Streamlit Review Pipeline, an auditable research workflow.

Direct data source APIs: PubMed, OpenAlex OQL, Scopus (key).
Other database exports: multi-file RIS (e.g. PsycINFO, WoS, Embase).
Merge + DOI/PMID/title-year conservative dedup + TF-IDF ranking.
Optional OpenAI screening requires explicit consent and user API key.
No autonomous inclusion/exclusion; manual decisions are stored separately.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import io
import json
import zipfile

import streamlit as st

from ris_proximity import parse_ris
from review_core import (
    csv_bytes, ris_bytes, as_json, from_ris,
    lexical_rank, merge_records,
)
from review_retrieval import (
    RetrievalError, pubmed_search, openalex_search, scopus_search,
)
from openai_review_screening import screen_batch, DEFAULT_MODEL, SCREENING_MODELS


def _record_fields(record) -> dict:
    """Consistent bibliographic metadata in every screening result view."""
    return {
        "Title": record.title,
        "Abstract": record.abstract,
        "Authors": record.authors,
        "Year": record.year,
        "Journal": record.journal,
        "DOI": record.doi,
        "Database": "; ".join(record.sources) or record.source,
    }


def _screening_row(record, ai, overrides) -> dict:
    human = overrides.get(record.uid, {})
    if isinstance(human, dict):
        human_decision = human.get("decision", "not_reviewed")
        human_reason = human.get("reason", "")
        reviewer = human.get("reviewer", "")
        reviewed_utc = human.get("timestamp_utc", "")
    else:
        human_decision = human or "not_reviewed"
        human_reason = reviewer = reviewed_utc = ""
    suggestion = ai.get(record.uid, {})
    return {
        **_record_fields(record),
        "UID": record.uid,
        "Rank": record.rank_score,
        "AI suggestion": suggestion.get("decision", ""),
        "AI reason": suggestion.get("reason", ""),
        "AI evidence": suggestion.get("evidence", ""),
        "AI uncertainty": suggestion.get("uncertainty", ""),
        "Human decision": human_decision,
        "Human reason": human_reason,
        "Reviewer": reviewer,
        "Reviewed UTC": reviewed_utc,
    }


def _bundle(merge, ordered, search_log, ai, overrides, screening_runs) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("deduplicated.csv", csv_bytes(ordered, ai))
        z.writestr("deduplicated.ris", ris_bytes(ordered))
        z.writestr("deduplicated.json", as_json(ordered))
        audit = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "raw_input": merge.input_count,
            "deduplication_engine": search_log[-1].get("dedup_engine", "Python DOI/PMID/title-year") if search_log else "unknown",
            "deduplicated": len(ordered),
            "duplicate_records_removed": merge.duplicates_removed,
            "per_source_counts": merge.counts,
            "fuzzy_review_count": len(merge.fuzzy_review),
            "searches": search_log,
            "ranking": "within-sample title-weighted lexical TF-IDF cosine",
            "warning": "Counts are NOT a completed PRISMA diagram; "
                       "sample limits, source overlap, database coverage "
                       "and screening require human review.",
        }
        z.writestr("retrieval_audit.json",
                   json.dumps(audit, ensure_ascii=False, indent=2))
        z.writestr("deduplication_log.json",
                   json.dumps(merge.audit, ensure_ascii=False, indent=2))
        z.writestr("fuzzy_candidates_manual_review.csv",
                   _dict_csv(merge.fuzzy_review))
        # Screening exports are self-contained: metadata accompany every decision.
        z.writestr("screening_results.csv", _dict_csv([
            _screening_row(r, ai, overrides) for r in ordered
        ]))
        z.writestr("human_decisions.csv", _dict_csv([
            _screening_row(r, ai, overrides)
            for r in ordered if r.uid in overrides
        ]))
        if ai:
            z.writestr("ai_screening_suggestions.json",
                       json.dumps(ai, ensure_ascii=False, indent=2))
            z.writestr("ai_screening_suggestions.csv", _dict_csv([
                _screening_row(r, ai, overrides)
                for r in ordered if r.uid in ai
            ]))
        z.writestr("screening_criteria_and_run_audit.json",
                   json.dumps(screening_runs, ensure_ascii=False, indent=2))
    return b.getvalue()


def _dict_csv(rows: list[dict]) -> bytes:
    import csv
    cols = list(dict.fromkeys(k for row in rows for k in row))
    f = io.StringIO(newline="")
    if cols:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    return f.getvalue().encode("utf-8-sig")


def render() -> None:
    st.subheader("Review Pipeline · Search → Merge → Deduplication → Ranking → Screening")
    st.caption(
        "Reproducible multi-database workflow with provenance tracking. "
        "Not a substitute for comprehensive database searching, "
        "PRISMA documentation, or independent screening."
    )

    st.markdown("#### 1 · Select and search data sources")
    translations = st.session_state.get("translations") or {}
    initial_pubmed = (translations.get("PubMed") or {}).get(
        "query", '("social media"[tiab] AND student*[tiab])')
    initial_oa = (translations.get("OpenAlex") or {}).get(
        "query", 'get works about "academic self concept"')
    initial_scopus = (translations.get("Scopus") or {}).get("query", "")

    with st.form("review_source_form"):
        selection = st.columns(3)
        with selection[0]:
            use_pubmed = st.checkbox("PubMed E-utilities", value=True)
        with selection[1]:
            use_openalex = st.checkbox("OpenAlex OQL", value=False)
        with selection[2]:
            use_scopus = st.checkbox("Scopus Search API", value=False)
        pubmed_q = st.text_area(
            "PubMed query (PubMed syntax)",
            initial_pubmed,
            height=78, help=("Use PubMed syntax, not Ovid: [tiab], [mh], etc. "
                              "A PubMed query generated in the translator "
                              "is prefilled automatically."))
        openalex_q = st.text_area(
            "OpenAlex OQL (can be imported from Search String Translator)",
            initial_oa, height=110)
        scopus_q = st.text_area(
            "Scopus Advanced Search (e.g. TITLE-ABS-KEY(...))",
            initial_scopus, height=75)
        opts = st.columns(4)
        with opts[0]:
            per_source_limit = st.number_input(
                "Maximum publications per API", min_value=1,
                max_value=2000, value=100, step=25)
        with opts[1]:
            pubmed_email = st.text_input("NCBI email (optional)")
            pubmed_key = st.text_input("NCBI API key", type="password")
        with opts[2]:
            openalex_key = st.text_input("OpenAlex API key", type="password")
        with opts[3]:
            scopus_key = st.text_input("Scopus API key", type="password")
            scopus_token = st.text_input(
                "Scopus institutional token (optional)", type="password")
        submitted = st.form_submit_button(
            "Search databases and merge records", type="primary",
            key="review_fetch_and_merge")
    ris_files = st.file_uploader(
        "Other sources: RIS files (e.g. Web of Science, Ovid, Embase)",
        type=["ris"], accept_multiple_files=True, key="review_ris_files",
        help="One RIS file per source. The filename is preserved "
             "as provenance. Uploaded files are not processed automatically; "
             "click the search and merge button to include them."
    )

    st.caption(
        "Deduplication uses conservative Python rules for DOI, PMID, and "
        "normalized title/year. Similar titles are flagged for manual "
        "review, not removed automatically. "
        "ASySD is not offered until its R integration "
        "has been fully validated."
    )
    if submitted:
        if not (use_pubmed or use_openalex or use_scopus or ris_files):
            st.warning("Select at least one API or upload an RIS file.")
        else:
            raw = []
            audit = []
            failures = []
            source_jobs = []
            if use_pubmed:
                source_jobs.append((
                    "PubMed", pubmed_q,
                    lambda callback: pubmed_search(
                        pubmed_q.strip(), int(per_source_limit),
                        email=pubmed_email, api_key=pubmed_key,
                        progress=callback
                    ),
                ))
            if use_openalex:
                source_jobs.append((
                    "OpenAlex", openalex_q,
                    lambda callback: openalex_search(
                        openalex_q.strip(), int(per_source_limit),
                        api_key=openalex_key, progress=callback
                    ),
                ))
            if use_scopus:
                source_jobs.append((
                    "Scopus", scopus_q,
                    lambda callback: scopus_search(
                        scopus_q.strip(), int(per_source_limit),
                        api_key=scopus_key, inst_token=scopus_token,
                        progress=callback
                    ),
                ))
            total_jobs = len(source_jobs) + len(ris_files or [])
            overall = st.progress(
                0.0, text=f"Starting: 0 of {total_jobs} sources processed")
            current = st.progress(
                0.0, text="Waiting for first data source")
            steps_text = st.empty()
            steps_text.caption(
                "The bar reflects records actually retrieved and "
                "completed processing stages, not an estimated time remaining."
            )

            def source_progress(index, source, done, total):
                completed = min(1.0, done / total) if total else 1.0
                overall.progress(
                    min(0.8, 0.8 * (index + completed) / max(1, total_jobs)),
                    text=f"Source {index + 1}/{total_jobs}: {source}"
                )
                current.progress(
                    completed,
                    text=f"{source}: {done:,} of "
                         f"{total:,} planned records"
                )

            job_index = 0
            for enabled_source, original_query, load in source_jobs:
                source_progress(job_index, enabled_source, 0,
                                int(per_source_limit))
                if not original_query.strip():
                    failures.append(f"{enabled_source}: Missing search query.")
                    audit.append({
                        "source": enabled_source, "query": original_query,
                        "status": "failed", "error": "Missing search query",
                        "complete": False, "retrieved": 0,
                    })
                else:
                    try:
                        result = load(
                            lambda done, total, ix=job_index, name=enabled_source:
                            source_progress(ix, name, done, total))
                        raw.extend(result.records)
                        audit.append({
                            "source": enabled_source, "query": original_query.strip(),
                            "total_hits": result.total,
                            "retrieved": len(result.records),
                            "complete": result.complete,
                            "warning": result.warning,
                            "status": "success",
                        })
                    except (RetrievalError, ValueError) as exc:
                        failures.append(f"{enabled_source}: {exc}")
                        audit.append({
                            "source": enabled_source,
                            "query": original_query.strip(),
                            "status": "failed", "error": str(exc),
                            "complete": False, "retrieved": 0,
                        })
                source_progress(job_index, enabled_source, 1, 1)
                job_index += 1

            for upload in ris_files or []:
                name = upload.name.rsplit(".", 1)[0][:90]
                label = "RIS: " + name
                source_progress(job_index, label, 0, 1)
                try:
                    if upload.size > 25 * 1024 * 1024:
                        raise ValueError("File exceeds 25 MB.")
                    recs, encoding = parse_ris(upload.getvalue())
                    converted = from_ris(recs, label)
                    raw.extend(converted)
                    audit.append({
                        "source": label, "query": "RIS import",
                        "total_hits": len(converted),
                        "retrieved": len(converted),
                        "complete": True, "status": "imported",
                        "encoding": encoding,
                    })
                except (ValueError, UnicodeError) as exc:
                    failures.append(f"{upload.name}: {exc}")
                    audit.append({
                        "source": label, "query": "RIS import",
                        "status": "failed", "error": str(exc),
                        "complete": False, "retrieved": 0,
                    })
                source_progress(job_index, label, 1, 1)
                job_index += 1

            for error in failures:
                st.error(error)

            if raw:
                # Progress of the three *actual* dedup phases.
                phases = {
                    "Matching identifiers": (0.80, 0.91),
                    "Merging metadata": (0.91, 0.97),
                    "Checking similar titles": (0.97, 1.00),
                }

                def dedup_progress(phase, done, total):
                    start, finish = phases.get(phase, (0.80, 1.0))
                    frac = min(1.0, done / total) if total else 1.0
                    overall.progress(
                        start + (finish - start) * frac,
                        text="Deduplication: " + phase
                    )
                    current.progress(
                        frac, text=f"{phase}: {done:,} / {total:,}"
                    )

                merged = merge_records(raw, progress=dedup_progress)
                engine = "Python DOI/PMID/title+year + manual similarity candidates"
                audit.append({
                    "source": "DEDUPLICATION", "query": "",
                    "status": "completed", "dedup_engine": engine,
                    "retrieved": len(merged.records),
                    "complete": True,
                })
                st.session_state.review_data = (merged, audit)
                st.session_state.review_ai = {}
                st.session_state.review_human = {}
                st.session_state.review_screening_runs = []
                st.session_state.review_dedup_engine = engine
                overall.progress(1.0, text="Retrieval and deduplication complete")
                current.empty()
                steps_text.caption(
                    f"Completed: {len(raw):,} raw records, "
                    f"{len(merged.records):,} unique records, "
                    f"{merged.duplicates_removed:,} duplicates removed."
                )
                st.success(
                    f"{len(raw):,} records loaded → "
                    f"{len(merged.records):,} after deduplication."
                )
            else:
                st.session_state.pop("review_data", None)
                overall.progress(1.0, text="Retrieval finished – no records available")
                current.empty()
                st.error("No records available; screening is not possible.")

    if "review_data" not in st.session_state:
        return

    merged, search_log = st.session_state.review_data
    records = merged.records
    st.divider()
    st.markdown("#### 2 · Merge and deduplication")
    m = st.columns(4)
    m[0].metric("Raw records", merged.input_count)
    m[1].metric("Unique after deduplication", len(records))
    m[2].metric("Automatically removed duplicates", merged.duplicates_removed)
    m[3].metric("Potential duplicates (manual review)", len(merged.fuzzy_review))
    if merged.fuzzy_skipped:
        st.warning(
            "For some very large year groups, the "
            "quadratic similarity check was skipped for performance reasons: "
            "" + ", ".join(
                f"{year}: {n} records"
                for year, n in merged.fuzzy_skipped.items()
            ) + ". These records have not been confirmed as duplicate-free."
        )
    st.caption(
        "Method: " + st.session_state.get(
            "review_dedup_engine", "Python DOI/PMID/title+year"
        ) + ". Unconfirmed duplicates are never removed automatically. "
        "ASySD candidates come from the official R pipeline; "
        "Python candidates use a similarity threshold of 0.94."
    )
    st.dataframe([
        {"Source": a.get("source"),
         "Total hits": (
             str(a["total_hits"]) if a.get("total_hits") is not None else "—"
         ),
         "Retrieved": a.get("retrieved", 0),
         "Complete": a.get("complete", False),
         "Status": a.get("status")} for a in search_log
    ], hide_index=True, use_container_width=True)
    if any(a.get("complete") is False for a in search_log):
        st.warning(
            "At least one search source failed or was limited. "
            "This result set is NOT complete. "
            "For systematic reviews, document and complete "
            "all searches separately."
        )
    if merged.audit:
        with st.expander("Automatically merged duplicates (audit)"):
            st.dataframe(merged.audit, hide_index=True, use_container_width=True)
    if merged.fuzzy_review:
        st.warning(
            f"{len(merged.fuzzy_review)} similar titles require manual review "
            "and have NOT been merged automatically."
        )
        with st.expander("Potential duplicates requiring manual decisions"):
            st.dataframe(merged.fuzzy_review, hide_index=True,
                         use_container_width=True)

    st.divider()
    st.markdown("#### 3 · Ranking for screening prioritization")
    focus = st.text_area(
        "Research topic / target concept for ranking",
        "social media use and academic self concept among university students",
        key="review_focus", height=75,
        help="This ranking is lexical, not an "
             "inclusion/exclusion decision."
    )
    ordered = lexical_rank(records, focus)
    st.caption(
        "Title-weighted lexical TF-IDF cosine ranking. "
        "A low score does NOT justify excluding a study "
        "and no records are removed by ranking."
    )
    ai = st.session_state.setdefault("review_ai", {})
    human = st.session_state.setdefault("review_human", {})
    items = [_screening_row(r, ai, human) for r in ordered[:300]]
    st.dataframe(items, hide_index=True, use_container_width=True)
    if len(ordered) > 300:
        st.caption("Table preview limited to 300 records; exports contain all records.")

    st.divider()
    st.markdown("#### 4 · Optional OpenAI title/abstract screening")
    st.warning(
        "OpenAI screening is **prioritization**, not an automatic "
        "review decision. Titles and abstracts are sent to the OpenAI API "
        "and API charges may apply. "
        "Only records with sufficient context should be used. "
        "Records without abstracts should receive MANUAL REVIEW. "
        "For licensed records (e.g. Scopus), sharing permissions "
        "must also be checked."
    )
    criteria = st.text_area(
        "Eligibility criteria / PICO / PCC (explicit and operationalized)",
        value="",
        placeholder=("Describe every mandatory inclusion criterion "
                     "and explicit exclusion criteria (PICO/PCC). "
                     "Only clearly unmet criteria justify "
                     "a suggestion to exclude."),
        height=125, key="review_eligibility",
    )
    q1, q2, q3 = st.columns([1, 1, 1])
    with q1:
        openai_model = st.selectbox(
            "OpenAI model",
            options=list(SCREENING_MODELS.values()),
            index=list(SCREENING_MODELS.values()).index(DEFAULT_MODEL),
            format_func=lambda m: next(
                (name + " · " + m for name, value in SCREENING_MODELS.items()
                 if value == m), m),
            key="review_openai_model",
            help="Terra is the default. Sol is suited to more complex "
                 "cases; Luna to cost-sensitive larger samples. "
                 "Availability depends on API access."
        )
    with q2:
        openai_key = st.text_input(
            "OpenAI API-Key", type="password", key="review_openai_key")
    with q3:
        screen_limit = st.number_input(
            "Maximum records to pre-screen", min_value=1, max_value=60,
            value=20, step=5, key="review_screen_limit")
    share = st.checkbox(
        "I confirm that I have reviewed sharing these titles and abstracts "
        "with the OpenAI API and any applicable costs.",
        value=False, key="review_screen_consent")
    st.caption(
        "Only the first N ranked records are transmitted. "
        "No API keys are included in exports or committed to GitHub. "
        "API requests set store=false."
    )
    if st.button(
        "Run OpenAI pre-screening",
        disabled=not share or not openai_key.strip() or len(criteria.strip()) < 12,
        key="review_screen_start",
    ):
        bar = st.progress(0.0)
        with st.spinner("Processing OpenAI structured outputs"):
            result = screen_batch(
                ordered, criteria=criteria, key=openai_key,
                model=openai_model.strip(), max_items=int(screen_limit),
                progress=lambda n, total: bar.progress(n / total),
            )
        bar.empty()
        st.session_state.review_ai.update(result.decisions)
        st.session_state.setdefault("review_screening_runs", []).append({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "criteria": criteria,
            "model": result.model,
            "prompt_protocol": "SearchBridge conservative title-abstract v1",
            "attempted": result.attempted,
            "count_by_suggestion": dict(Counter(
                x.get("decision") for x in result.decisions.values()
            )),
            "errors": result.errors,
            "note": "Only AI suggestions. Human decisions are separate.",
        })
        for err in result.errors:
            st.warning(err)
        st.success(
            f"{result.attempted} records processed for prioritization. "
            "All suggestions require human review."
        )
        st.rerun()

    if ai:
        count = Counter(v.get("decision", "") for v in ai.values())
        st.caption(
            "OpenAI suggestions: "
            f"{count.get('include',0)} Include, "
            f"{count.get('exclude',0)} Exclude, "
            f"{count.get('manual_review',0)} Manual review."
        )
        with st.expander("OpenAI suggestions with evidence and uncertainty"):
            st.dataframe([
                _screening_row(r, ai, human)
                for r in ordered if r.uid in ai
            ], hide_index=True, use_container_width=True)

    st.markdown("#### 5 · Human title/abstract screening")
    st.caption(
        "Every final decision must be based on the displayed title "
        "and full abstract. The OpenAI assessment is "
        "an independent suggestion, never a preselected decision."
    )
    def _human_label(uid):
        state = human.get(uid, "not_reviewed")
        return state.get("decision", "not_reviewed") if isinstance(state, dict) else state

    total_done = sum(_human_label(r.uid) != "not_reviewed" for r in ordered)
    st.progress(total_done / len(ordered), text=(
        f"{total_done:,} / {len(ordered):,} screened manually"
    ))
    only_pending = st.checkbox(
        "Show only unscreened studies",
        value=True, key="review_only_pending")
    pool = [r for r in ordered if not only_pending or
            _human_label(r.uid) == "not_reviewed"]
    if not pool:
        st.success("All available studies have been screened manually.")
        st.caption("Uncheck 'Show only unscreened studies' "
                   "to revisit completed decisions.")
    else:
        options_by_uid = {r.uid: r for r in pool}
        selected_uid = st.selectbox(
            "Select a study for human review",
            options=list(options_by_uid),
            format_func=lambda uid: (
                f"{next((i + 1 for i, r in enumerate(ordered) if r.uid == uid), '?')}"
                f" · {options_by_uid[uid].title[:110]}"
                + (" · Abstract missing" if not options_by_uid[uid].abstract else "")
            ),
            key="review_selected_uid",
        )
        selected = options_by_uid[selected_uid]
        st.markdown("##### " + (selected.title or "Untitled record"))
        st.markdown("**Authors:** " + (selected.authors or "Not available"))
        st.caption(
            f"**Year:** {selected.year or 'Not available'} · "
            f"**Journal:** {selected.journal or 'Not available'} · "
            f"**DOI:** {selected.doi or 'Not available'}"
        )
        st.caption(
            "**Database:** " + ("; ".join(selected.sources) or selected.source or "Not available")
            + f" · PMID: {selected.pmid or 'Not available'}"
            + f" · Rank: {selected.rank_score}"
        )
        st.markdown("**Abstract**")
        if selected.abstract:
            st.markdown(selected.abstract)
        else:
            st.warning(
                "No abstract available. Without further information, "
                "eligibility often cannot be determined. "
                "Check the original publication and consider "
                "selecting 'Manual Review'."
            )
        suggestion = ai.get(selected_uid)
        if suggestion:
            with st.expander("OpenAI suggestion – not binding", expanded=True):
                st.markdown(
                    "**Suggestion:** " + suggestion.get("decision", "unknown")
                    + " · **Uncertainty:** "
                    + suggestion.get("uncertainty", "unknown")
                )
                st.markdown("**Model rationale:** "
                            + suggestion.get("reason", ""))
                if suggestion.get("evidence"):
                    st.markdown("**Evidence in article text:** "
                                + suggestion["evidence"])
                st.caption("Model: " + suggestion.get("model", ""))
        state = human.get(selected_uid, {})
        previous = (state.get("decision", "not_reviewed")
                    if isinstance(state, dict) else state)
        previous_reason = state.get("reason", "") if isinstance(state, dict) else ""
        previous_reviewer = state.get("reviewer", "") if isinstance(state, dict) else ""
        decisions = ["not_reviewed", "include", "exclude", "manual_review"]
        with st.form("review_human_full_abstract_form", clear_on_submit=False):
            decision = st.radio(
                "Your independent final title/abstract decision",
                options=decisions,
                index=decisions.index(previous) if previous in decisions else 0,
                horizontal=True,
                key=f"review_human_choice_{selected_uid}",
            )
            reason = st.text_area(
                "Reason for the decision",
                value=previous_reason, height=85,
                key=f"review_human_reason_{selected_uid}",
                placeholder="Which inclusion or exclusion criterion "
                            "is met or clearly not met?"
            )
            reviewer = st.text_input(
                "Reviewer initials (optional)",
                value=previous_reviewer,
                key=f"review_human_reviewer_{selected_uid}",
            )
            save_human = st.form_submit_button(
                "Save human decision and continue",
                type="primary"
            )
        if save_human:
            if decision == "exclude" and not reason.strip():
                st.error(
                    "For exclusion, an unmet "
                    "eligibility criterion must be documented."
                )
            elif decision == "include" and not selected.abstract:
                st.warning(
                    "No abstract available. Check the original and "
                    "full text; use 'Manual Review' for now."
                )
            else:
                if decision == "not_reviewed":
                    human.pop(selected_uid, None)
                else:
                    human[selected_uid] = {
                        "decision": decision,
                        "reason": reason.strip(),
                        "reviewer": reviewer.strip(),
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    }
                st.success("Human decision saved.")
                st.rerun()

    with st.expander("Saved human decisions"):
        st.dataframe([
            _screening_row(r, ai, human)
            for r in ordered if _human_label(r.uid) != "not_reviewed"
        ], hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("#### 6 · Reproducible export")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Deduplicated: RIS", ris_bytes(ordered),
                           "searchbridge_merged.ris",
                           "application/x-research-info-systems",
                           use_container_width=True)
    with c2:
        st.download_button("Records + AI: CSV", csv_bytes(ordered, ai),
                           "searchbridge_merged.csv", "text/csv",
                           use_container_width=True)
    with c3:
        st.download_button(
            "Audit + records + screening (ZIP)",
            _bundle(merged, ordered, search_log, ai, human,
                    st.session_state.get("review_screening_runs", [])),
            "searchbridge_review_pipeline.zip",
            "application/zip", use_container_width=True, type="primary")
    st.info(
        "The ZIP contains the complete deduplicated record list, "
        "source log, deduplication audit, potential duplicate list, "
        "separate AI suggestions and human decisions. "
        "This does not automatically create a complete PRISMA flowchart."
    )
