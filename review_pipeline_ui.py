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
from openai_review_screening import screen_batch


def _bundle(merge, ordered, search_log, ai, overrides, screening_runs) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("deduplicated.csv", csv_bytes(ordered, ai))
        z.writestr("deduplicated.ris", ris_bytes(ordered))
        z.writestr("deduplicated.json", as_json(ordered))
        audit = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "raw_input": merge.input_count,
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
        z.writestr("human_decisions.csv", _dict_csv([
            {"uid": k, "human_decision": v}
            for k, v in overrides.items()
        ]))
        if ai:
            z.writestr("ai_screening_suggestions.json",
                       json.dumps(ai, ensure_ascii=False, indent=2))
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
    st.subheader("Review Pipeline · Suche → Merge → Deduplikation → Ranking → Screening")
    st.caption(
        "Reproduzierbarer Mehrdatenbank-Workflow mit Herkunftsnachweisen. "
        "Nicht als automatischer Ersatz für vollständige Datenbankrecherchen, "
        "PRISMA-Dokumentation oder unabhängiges Screening."
    )

    st.markdown("#### 1 · Datenquellen auswählen und abfragen")
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
            "PubMed-Suchanfrage (PubMed-Syntax)",
            initial_pubmed,
            height=78, help=("Nicht Ovid Syntax: PubMed [tiab], [mh] usw. "
                              "Eine zuvor im Translator erzeugte PubMed-Abfrage "
                              "wird hier automatisch vorausgefüllt."))
        openalex_q = st.text_area(
            "OpenAlex OQL (kann aus Search String Translator übernommen werden)",
            initial_oa, height=110)
        scopus_q = st.text_area(
            "Scopus Advanced Search (z. B. TITLE-ABS-KEY(...))",
            initial_scopus, height=75)
        opts = st.columns(4)
        with opts[0]:
            per_source_limit = st.number_input(
                "Max. Publikationen je API", min_value=1,
                max_value=2000, value=100, step=25)
        with opts[1]:
            pubmed_email = st.text_input("NCBI E-Mail (optional)")
            pubmed_key = st.text_input("NCBI API-Key", type="password")
        with opts[2]:
            openalex_key = st.text_input("OpenAlex API-Key", type="password")
        with opts[3]:
            scopus_key = st.text_input("Scopus API-Key", type="password")
            scopus_token = st.text_input(
                "Scopus Institutions-Token (optional)", type="password")
        submitted = st.form_submit_button(
            "Datenbanken abfragen und zusammenführen", type="primary")
    ris_files = st.file_uploader(
        "Weitere Quellen: RIS-Exportdateien (z. B. Web of Science, Ovid, Embase)",
        type=["ris"], accept_multiple_files=True, key="review_ris_files",
        help="Eine RIS-Datei pro Quelle. Der Dateiname wird als Provenienz "
             "gespeichert. Uploads sind nicht automatisch ausgewertet; "
             "klicke anschliessend auf den Abruf-Button."
    )

    if submitted:
        if not (use_pubmed or use_openalex or use_scopus or ris_files):
            st.warning("Wähle mindestens eine API oder lade eine RIS-Datei hoch.")
        else:
            raw = []
            audit = []
            failures = []
            with st.spinner("Datenquellen abfragen und RIS-Dateien lesen"):
                for enabled, title, query, handler in [
                    (use_pubmed, "PubMed", pubmed_q,
                     lambda: pubmed_search(
                         pubmed_q.strip(), int(per_source_limit),
                         email=pubmed_email, api_key=pubmed_key)),
                    (use_openalex, "OpenAlex", openalex_q,
                     lambda: openalex_search(
                         openalex_q.strip(), int(per_source_limit),
                         api_key=openalex_key)),
                    (use_scopus, "Scopus", scopus_q,
                     lambda: scopus_search(
                         scopus_q.strip(), int(per_source_limit),
                         api_key=scopus_key, inst_token=scopus_token)),
                ]:
                    if not enabled:
                        continue
                    if not query.strip():
                        failures.append(f"{title}: Suchanfrage fehlt.")
                        continue
                    try:
                        result = handler()
                        raw.extend(result.records)
                        audit.append({
                            "source": title, "query": query.strip(),
                            "total_hits": result.total,
                            "retrieved": len(result.records),
                            "complete": result.complete,
                            "warning": result.warning,
                            "status": "success",
                        })
                    except (RetrievalError, ValueError) as exc:
                        failures.append(f"{title}: {exc}")
                        audit.append({
                            "source": title, "query": query.strip(),
                            "status": "failed", "error": str(exc),
                            "complete": False, "retrieved": 0,
                        })
                for upload in ris_files or []:
                    name = upload.name.rsplit(".", 1)[0][:90]
                    label = "RIS: " + name
                    try:
                        if upload.size > 25 * 1024 * 1024:
                            raise ValueError("Datei überschreitet 25 MB.")
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
            if failures:
                for err in failures:
                    st.error(err)
            if raw:
                merged = merge_records(raw)
                st.session_state.review_data = (merged, audit)
                st.session_state.review_ai = {}
                st.session_state.review_human = {}
                st.session_state.review_screening_runs = []
                st.success(
                    f"{len(raw):,} Datensätze eingelesen → "
                    f"{len(merged.records):,} nach Deduplikation."
                )
            else:
                st.session_state.pop("review_data", None)
                st.error("Keine Datensätze verfügbar; kein Screening möglich.")

    if "review_data" not in st.session_state:
        return

    merged, search_log = st.session_state.review_data
    records = merged.records
    st.divider()
    st.markdown("#### 2 · Merge und Deduplikation")
    m = st.columns(4)
    m[0].metric("Rohdatensätze", merged.input_count)
    m[1].metric("Eindeutig nach Deduplikation", len(records))
    m[2].metric("Automatisch entfernte Dubletten", merged.duplicates_removed)
    m[3].metric("Fuzzy-Kandidaten (manuell)", len(merged.fuzzy_review))
    st.dataframe([
        {"Quelle": a.get("source"), "Treffer insgesamt": a.get("total_hits", ""),
         "Abgerufen": a.get("retrieved", 0),
         "Vollständig": a.get("complete", False),
         "Status": a.get("status")} for a in search_log
    ], hide_index=True, use_container_width=True)
    if any(a.get("complete") is False for a in search_log):
        st.warning(
            "Mindestens eine Suchquelle ist fehlgeschlagen oder begrenzt. "
            "Diese Ergebnismenge ist NICHT vollständig. "
            "Für systematische Reviews bitte alle Suchläufe "
            "separat dokumentieren und vervollständigen."
        )
    if merged.audit:
        with st.expander("Automatisch zusammengeführte Dubletten (Audit)"):
            st.dataframe(merged.audit, hide_index=True, use_container_width=True)
    if merged.fuzzy_review:
        st.warning(
            f"{len(merged.fuzzy_review)} ähnliche Titel müssen manuell geprüft "
            "werden. Sie wurden bewusst NICHT automatisch zusammengeführt."
        )
        with st.expander("Fuzzy-Dubletten, manuelle Entscheidung erforderlich"):
            st.dataframe(merged.fuzzy_review, hide_index=True,
                         use_container_width=True)

    st.divider()
    st.markdown("#### 3 · Ranking zur Screening-Priorisierung")
    focus = st.text_area(
        "Forschungsthema / Zielkonzept zur Ranking-Berechnung",
        "social media use and academic self concept among university students",
        key="review_focus", height=75,
        help="Dieses Ranking ist lexikalisch und keine "
             "Ein-/Ausschlussentscheidung."
    )
    ordered = lexical_rank(records, focus)
    st.caption(
        "Titelgewichtiges lexikalisches TF-IDF-Cosinus-Ranking. "
        "Ein niedriger Score bedeutet NICHT, dass eine Studie ausgeschlossen "
        "werden darf. Kein Datensatz wird durch das Ranking entfernt."
    )
    ai = st.session_state.setdefault("review_ai", {})
    human = st.session_state.setdefault("review_human", {})
    items = []
    for r in ordered[:300]:
        choice = ai.get(r.uid, {})
        items.append({
            "uid": r.uid, "Title": r.title, "Year": r.year,
            "DOI": r.doi, "Sources": ", ".join(r.sources),
            "Rank": r.rank_score, "Abstract present": bool(r.abstract),
            "AI suggestion": choice.get("decision", ""),
            "AI reason": choice.get("reason", ""),
            "Human decision": human.get(r.uid, "not_reviewed"),
        })
    st.dataframe(items, hide_index=True, use_container_width=True)
    if len(ordered) > 300:
        st.caption("Tabellenvorschau auf 300 Datensätze begrenzt; Exporte enthalten alle.")

    st.divider()
    st.markdown("#### 4 · Optionales OpenAI-Titel-/Abstract-Screening")
    st.warning(
        "OpenAI-Screening ist eine **Vorsortierung**, kein automatischer "
        "Review-Entscheid. Titel und Abstracts werden dabei an die OpenAI API "
        "übertragen und es können API-Kosten anfallen. "
        "Nur Records mit vollständigem Kontext sollten verwendet werden. "
        "Publikationen ohne Abstract erhalten grundsätzlich MANUAL REVIEW. "
        "Bei lizenzierten Daten (z. B. Scopus) müssen Weitergaberechte "
        "zusätzlich geprüft werden."
    )
    criteria = st.text_area(
        "Einschlusskriterien / PICO / PCC (konkret und operationalisiert)",
        value="",
        placeholder=("Beschreibe alle obligatorischen Einschlusskriterien "
                     "und explizite Ausschlusskriterien (PICO/PCC). "
                     "Nur eindeutig nicht erfüllte Kriterien rechtfertigen "
                     "einen Exclude-Vorschlag."),
        height=125, key="review_eligibility",
    )
    q1, q2, q3 = st.columns([1, 1, 1])
    with q1:
        openai_model = st.text_input(
            "OpenAI-Modell", value="gpt-4o-mini", key="review_openai_model",
            help="Ein Modell verwenden, das Responses Structured Outputs unterstützt.")
    with q2:
        openai_key = st.text_input(
            "OpenAI API-Key", type="password", key="review_openai_key")
    with q3:
        screen_limit = st.number_input(
            "Max. Datensätze vorscreenen", min_value=1, max_value=60,
            value=20, step=5, key="review_screen_limit")
    share = st.checkbox(
        "Ich bestätige, dass ich die Weitergabe dieser Titel und Abstracts "
        "an die OpenAI API sowie mögliche Kosten geprüft habe.",
        value=False, key="review_screen_consent")
    st.caption(
        "Es werden nur die ersten N Datensätze des Rankings übertragen. "
        "Keine API-Keys werden in Exporte oder GitHub geschrieben. "
        "Die API-Anfragen setzen store=false."
    )
    if st.button(
        "OpenAI-Vorscreening ausführen",
        disabled=not share or not openai_key.strip() or len(criteria.strip()) < 12,
        key="review_screen_start",
    ):
        bar = st.progress(0.0)
        with st.spinner("OpenAI-Structured-Outputs auswerten"):
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
            f"{result.attempted} Studien zur Vorsortierung verarbeitet. "
            "Alle Vorschläge müssen von Menschen überprüft werden."
        )
        st.rerun()

    if ai:
        count = Counter(v.get("decision", "") for v in ai.values())
        st.caption(
            "OpenAI-Vorschläge: "
            f"{count.get('include',0)} Include, "
            f"{count.get('exclude',0)} Exclude, "
            f"{count.get('manual_review',0)} Manual review."
        )
        with st.expander("OpenAI-Vorschläge mit Evidenz und Unsicherheit"):
            st.dataframe([
                {"uid": r.uid, "Title": r.title,
                 "AI": ai.get(r.uid, {}).get("decision", ""),
                 "Reason": ai.get(r.uid, {}).get("reason", ""),
                 "Evidence": ai.get(r.uid, {}).get("evidence", ""),
                 "Uncertainty": ai.get(r.uid, {}).get("uncertainty", "")}
                for r in ordered if r.uid in ai
            ], hide_index=True, use_container_width=True)

    st.markdown("#### 5 · Menschliche Entscheidungen")
    st.caption(
        "Finale Einschluss-/Ausschlussentscheidungen werden separat "
        "gespeichert. KI-Vorschläge werden niemals automatisch übernommen. "
        "Diese erste Version unterstützt bis zu 100 manuelle "
        "Entscheidungen in der interaktiven Vorschau."
    )
    edited = st.data_editor(
        [
            {"uid": r.uid, "Title": r.title[:170],
             "Human decision": human.get(r.uid, "not_reviewed")}
            for r in ordered[:100]
        ],
        hide_index=True, use_container_width=True,
        key="review_human_editor",
        disabled=["uid", "Title"],
        column_config={
            "Human decision": st.column_config.SelectboxColumn(
                "Human decision",
                options=["not_reviewed", "include", "exclude", "manual_review"],
                required=True
            ),
        },
    )
    if st.button("Menschliche Entscheidungen speichern",
                 key="review_save_human"):
        for row in edited:
            if row["Human decision"] != "not_reviewed":
                st.session_state.review_human[row["uid"]] = row["Human decision"]
            else:
                st.session_state.review_human.pop(row["uid"], None)
        st.success(
            f"{len(st.session_state.review_human)} menschliche "
            "Entscheidungen gespeichert."
        )

    st.divider()
    st.markdown("#### 6 · Reproduzierbarer Export")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Dedupliziert: RIS", ris_bytes(ordered),
                           "searchbridge_merged.ris",
                           "application/x-research-info-systems",
                           use_container_width=True)
    with c2:
        st.download_button("Datensätze + AI: CSV", csv_bytes(ordered, ai),
                           "searchbridge_merged.csv", "text/csv",
                           use_container_width=True)
    with c3:
        st.download_button(
            "Audit + Datensätze + Screening (ZIP)",
            _bundle(merged, ordered, search_log, ai, human,
                    st.session_state.get("review_screening_runs", [])),
            "searchbridge_review_pipeline.zip",
            "application/zip", use_container_width=True, type="primary")
    st.info(
        "Das ZIP enthält die vollständige deduplizierte Trefferliste, "
        "Quellenprotokoll, Deduplikations-Audit, Fuzzy-Prüfliste, "
        "separate KI-Vorschläge und menschliche Entscheidungen. "
        "Ein vollständiger PRISMA-Flowchart ist damit noch nicht automatisch erstellt."
    )
