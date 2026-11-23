"""Streamlit UI for deterministic PubMed title/MeSH term harvesting."""
from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime, timezone

import streamlit as st

from pubmed_proximity import PubMedError, fetch_pubmed_candidates
from ris_proximity import ris_bytes
from term_harvesting import title_tfidf, ranked_mesh, rows_to_csv


def render() -> None:
    st.subheader("Term Harvesting aus PubMed")
    st.caption("Zwei komplementäre Wege zur Ergänzung einer Suchstrategie: "
               "MeSH-Suche → Titelbegriffe mit TF-IDF; "
               "TIAB-Suche → tatsächlich vergebene MeSH-Descriptors. "
               "Rein regelbasiert, keine LLMs.")

    direction = st.radio(
        "Auswertungsrichtung",
        ["MeSH → Title terms (TF-IDF)", "TIAB → MeSH descriptors (frequency)"],
        horizontal=True, key="harvest_direction",
    )
    mesh_mode = direction.startswith("MeSH")
    default_query = ("Students[mh] AND Social Media[mh]" if mesh_mode
                     else '("social media"[tiab] AND (academic[tiab] OR university[tiab]))')

    with st.form("harvest_pubmed_form"):
        st.caption("PubMed-Syntax verwenden: MeSH beispielsweise Students[mh] "
                   "(mit Explosion) oder Students[mh:noexp]. "
                   "Freitext mit [tiab] kennzeichnen. AND/OR-Kombinationen sind erlaubt.")
        query = st.text_area(
            "PubMed-Suchanfrage",
            value=default_query,
            key=("harvest_query_mesh" if mesh_mode else "harvest_query_tiab"),
            height=100,
        )
        h1, h2, h3 = st.columns([2, 1, 1])
        with h1:
            sample_limit = st.number_input(
                "Maximal abzurufende Publikationen", min_value=1,
                max_value=10000, value=500, step=100, key="harvest_limit")
        with h2:
            contact = st.text_input("NCBI-E-Mail (optional)", key="harvest_email")
        with h3:
            ncbi_key = st.text_input("NCBI API-Key (optional)",
                                     type="password", key="harvest_key")
        requested = st.form_submit_button(
            "PubMed abrufen und analysieren", type="primary")

    if requested:
        if not query.strip():
            st.error("Bitte eine PubMed-Suchanfrage eingeben.")
        elif mesh_mode and not re.search(
            r"\[(?:mh|mesh|majr|mesh terms)(?::noexp)?\]",
            query, flags=re.I,
        ):
            st.error("Für MeSH → Titel bitte mindestens einen expliziten "
                     "MeSH-Feldtag wie [mh], [mh:noexp] oder [majr] verwenden.")
        elif not mesh_mode and not re.search(
            r"\[(?:tiab|title/abstract)\]", query, flags=re.I,
        ):
            st.error("Für TIAB → MeSH bitte mindestens einen [tiab]-Feldtag "
                     "in der Suche verwenden.")
        else:
            st.session_state.pop("harvest_data", None)
            bar = st.progress(0.0)
            try:
                with st.spinner("PubMed ESearch / EFetch: Publikationen abrufen"):
                    retrieval = fetch_pubmed_candidates(
                        query.strip(), int(sample_limit), email=contact,
                        api_key=ncbi_key,
                        progress=lambda done, total: bar.progress(
                            done / total if total else 1.0),
                    )
                st.session_state.harvest_data = (
                    retrieval, direction, query.strip()
                )
                st.success(
                    f"{retrieval.fetched:,} von "
                    f"{retrieval.candidate_count:,} PubMed-Treffern abgerufen.")
            except (PubMedError, ValueError) as exc:
                st.error(f"PubMed-Abruf fehlgeschlagen: {exc}")
            finally:
                bar.empty()

    if "harvest_data" not in st.session_state:
        return
    retrieval, old_direction, old_query = st.session_state.harvest_data
    if direction != old_direction or query.strip() != old_query:
        st.warning("Suchmodus oder Suchanfrage geändert: bitte erneut abrufen.")
        return

    records = retrieval.records
    indexed = sum(bool(r.fields.get("MH")) for r in records)
    with_titles = sum(bool(r.title) for r in records)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PubMed Treffer", f"{retrieval.candidate_count:,}")
    m2.metric("Heruntergeladen", f"{len(records):,}")
    m3.metric("Mit Titel", f"{with_titles:,}")
    m4.metric("Mit MeSH", f"{indexed:,}")

    if retrieval.candidate_count != len(records):
        st.warning(
            f"Nur {len(records):,} von {retrieval.candidate_count:,} "
            "Publikationen wurden ausgewertet. Die Auswahl entspricht der "
            "PubMed-Relevanzsortierung, nicht einer Zufallsstichprobe. "
            "Dieses Ranking ist nicht für die gesamte Treffermenge repräsentativ.")

    if not records:
        st.info("Keine Datensätze abgerufen.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        top_n = st.number_input("Top-N Begriffe", min_value=10, max_value=1000,
                                value=100, step=10, key="harvest_top_n")

    if mesh_mode:
        if not with_titles:
            st.error("Keiner der Datensätze enthält einen Titel.")
            return
        with c2:
            min_df = st.number_input(
                "Minimale Dokumenthäufigkeit",
                min_value=1, max_value=with_titles,
                value=min(2, with_titles), step=1, key="harvest_min_df")
        with c3:
            ngram = st.selectbox(
                "N-Gramme", [1, 2], key="harvest_ngram",
                format_func=lambda n: "Nur Einzelwörter" if n == 1
                else "Einzelwörter + Zweierphrasen")
        with c4:
            remove_query = st.checkbox(
                "Wörter der Suchanfrage ausblenden",
                value=True, key="harvest_exclude_query")
        extra_stop = st.text_input(
            "Zusätzliche Stoppwörter (optional, mit Leerzeichen trennen)",
            key="harvest_extra_stop")
        result = title_tfidf(
            records, min_df=int(min_df), max_ngram=int(ngram),
            query=old_query, remove_query_terms=bool(remove_query),
            extra_stopwords=extra_stop, top_n=int(top_n),
        )
        st.caption(
            "TF-IDF ist keine Häufigkeit: Die Tabelle zeigt zusätzlich die "
            "Dokumenthäufigkeit und die Gesamtzahl der Vorkommen. "
            "Die IDF-Gewichte werden nur aus den heruntergeladenen Titeln berechnet.")
        columns = ["Rank", "Title term", "Mean TF-IDF",
                   "Document frequency", "% titles", "Occurrences"]
        filters = {
            "min_df": int(min_df), "ngram_max": int(ngram),
            "exclude_query_terms": bool(remove_query),
            "extra_stopwords": extra_stop,
        }
    else:
        with c2:
            exclude_check = st.checkbox(
                "Allgemeine MeSH-Check-Tags ausblenden", value=True,
                key="harvest_exclude_checktags")
        result = ranked_mesh(
            records, top_n=int(top_n),
            exclude_check_tags=bool(exclude_check),
        )
        columns = ["Rank", "MeSH descriptor", "Indexed records",
                   "% indexed records", "Major topic records", "% all retrieved"]
        filters = {"exclude_check_tags": bool(exclude_check)}
        st.caption(
            "Hier werden ausschliesslich tatsächlich vergebene MeSH-Descriptors "
            "gezählt, nicht alle Begriffe, die theoretisch unter einem EXP liegen.")
        if not indexed:
            st.error("Keine der abgerufenen Publikationen besitzt eine MeSH-Indexierung.")
        elif indexed < len(records):
            st.warning(
                f"{len(records) - indexed:,} Publikationen ohne MeSH-Indexierung "
                "werden für die MeSH-Rangliste nicht als negative Fälle gezählt. "
                "Prozentangaben beziehen sich primär auf die indexierten Datensätze.")

    st.dataframe(result.rows, use_container_width=True, hide_index=True)
    if not result.rows:
        st.info("Keine Begriffe erfüllen die Filtereinstellungen.")

    meta = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "mode": old_direction,
        "pubmed_query": old_query,
        "pubmed_translation": retrieval.translation,
        "pubmed_sort": retrieval.sort,
        "total_pubmed_hits": retrieval.candidate_count,
        "downloaded": len(records),
        "complete_retrieval": len(records) == retrieval.candidate_count,
        "titles_available": with_titles,
        "mesh_indexed": indexed,
        "analysis_docs": result.included_docs,
        "missing_data_docs": result.missing_docs,
        "method": result.method,
        "method_note": result.note,
        "filters": filters,
    }
    table_csv = rows_to_csv(result.rows, columns)
    left, right = st.columns(2)
    with left:
        st.download_button(
            "Begriffs-Ranking (CSV)", table_csv,
            "searchbridge_term_harvesting.csv", "text/csv",
            use_container_width=True)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("ranked_terms.csv", table_csv)
        z.writestr("search_method.json",
                   json.dumps(meta, ensure_ascii=False, indent=2))
        z.writestr("retrieved_pubmed_records.ris", ris_bytes(records))
    with right:
        st.download_button(
            "CSV + Suchprotokoll + RIS (ZIP)", archive.getvalue(),
            "searchbridge_term_harvesting.zip",
            "application/zip", type="primary", use_container_width=True)
    with st.expander("Methodik und Suchprotokoll"):
        st.json(meta)
    st.info(
        "Diese Begriffe sind datenbasierte Kandidaten für die manuelle "
        "Suchstrategieentwicklung, keine validierten Synonyme oder neuen MeSH-Mappings.")
