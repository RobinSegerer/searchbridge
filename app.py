"""SearchBridge – rule-based search translator and PubMed/RIS proximity checker."""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone

import streamlit as st

from translator_core import translate, ParseError
from ris_proximity import (
    ProximityClause, parse_ris, extract_clauses, check_ris_records,
    html_report, csv_bytes, ris_bytes,
)
from pubmed_proximity import (
    PubMedError, parsed_terms, build_pubmed_and_query, fetch_pubmed_candidates,
)

st.set_page_config(page_title="SearchBridge", page_icon="🔎", layout="wide")
st.markdown(
    """<style>
    .block-container{max-width:1260px;padding-top:1.7rem}
    .hero{background:linear-gradient(105deg,#173046,#136c79);padding:22px 28px;
      border-radius:15px;color:white;margin-bottom:22px}
    .hero h1{color:white;margin:0;font-size:2.25rem}
    .hero p{color:#d9ecf0;margin:5px 0 0}
    </style>""", unsafe_allow_html=True
)
st.markdown(
    '<div class="hero"><h1>SearchBridge</h1>'
    '<p>Ovid search translation · PubMed and RIS proximity verification</p>'
    '<p>Rule-based · No LLMs · No mandatory API keys</p></div>',
    unsafe_allow_html=True,
)
SAMPLE = """(
  ((College OR university) adj2 (student* OR graduate* OR undergraduate*)).ti,ab,kf.
  OR exp Students/
)
AND
(
  ((academic OR university) adj2 (performance OR self-concept*)).ti,ab,kf.
  OR exp Self Concept/
)
AND
(
  (social media OR Instagram OR TikTok).ti,ab,kf.
  OR exp Social Media/
)"""
PROX_SAMPLE = """((juvenil* OR young* OR adolescent* OR youth*) adj2
(perpetr* OR offend* OR crimin* OR delinqu* OR justice* OR convict* OR prison*)).ti,ab,kf."""


def parse_mappings(raw):
    result = {}
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Ungültige Mapping-Zeile: {line}")
        source, target = line.split("=", 1)
        if not source.strip() or not target.strip():
            raise ValueError(f"Leerer Mapping-Eintrag: {line}")
        result[source.strip()] = target.strip()
    return result


def download_results(records, rows, clause, filename, source_kind="RIS", query_log=None):
    matched = [rec for rec, row in zip(records, rows) if row["matched"]]
    unmatched = [rec for rec, row in zip(records, rows) if not row["matched"]]
    report = html_report(records, rows, clause, source_file=filename, source_kind=source_kind)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button("HTML report", report, "searchbridge_report.html",
                           "text/html", use_container_width=True)
    with c2:
        st.download_button("All results CSV", csv_bytes(rows), "searchbridge_results.csv",
                           "text/csv", use_container_width=True)
    with c3:
        st.download_button("Matched RIS", ris_bytes(matched), "searchbridge_matches.ris",
                           "application/x-research-info-systems", use_container_width=True)
    with c4:
        st.download_button("Non-matched RIS", ris_bytes(unmatched), "searchbridge_nonmatches.ris",
                           "application/x-research-info-systems", use_container_width=True)
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.html", report)
        z.writestr("results.csv", csv_bytes(rows))
        z.writestr("all_records.ris", ris_bytes(records))
        z.writestr("matched.ris", ris_bytes(matched))
        z.writestr("not_matched.ris", ris_bytes(unmatched))
        z.writestr("query_log.json", json.dumps(query_log or {
            "ovid_clause": clause.label,
            "count": len(rows),
            "matches": len(matched),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2, ensure_ascii=False))
    st.download_button("Complete report ZIP", bundle.getvalue(),
                       "searchbridge_results.zip", "application/zip", type="primary")


def display_results(records, rows, clause):
    count = sum(r["matched"] for r in rows)
    a, b, c, d = st.columns(4)
    a.metric("Records checked", len(rows))
    b.metric(f"adj{clause.n} matched", count)
    c.metric("Not matched", len(rows) - count)
    d.metric("Missing abstracts", sum(not r.abstract for r in records))
    choice = st.radio("Show", ["All", "Matches", "Non-matches"], horizontal=True)
    subset = [r for r in rows if choice == "All"
              or r["matched"] == (choice == "Matches")]
    st.dataframe([
        {key: row.get(key) for key in
         ("record_no", "id", "title", "matched", "field", "term_a",
          "term_b", "gap", "evidence")}
        for row in subset[:500]
    ], use_container_width=True, hide_index=True)
    if len(subset) > 500:
        st.caption(f"Preview limited to 500; exports contain all {len(subset)} records.")


translate_tab, pubmed_tab, ris_tab, help_tab = st.tabs([
    "Search String Translator", "PubMed Proximity Search",
    "RIS Proximity Checker", "Help",
])

with translate_tab:
    st.subheader("Translate Ovid MEDLINE")
    st.caption("Ovid PsycINFO maps .kf. to .id.; controlled vocabulary must be checked.")
    source = st.text_area("Ovid search strategy", SAMPLE, height=220)
    with st.expander("Optional manually verified mappings"):
        psy_map = st.text_area("MeSH = APA PsycINFO term (one per line)")
        em_map = st.text_area("MeSH = Emtree term (one per line)")
    if st.button("Translate", type="primary", key="translate"):
        try:
            mappings = {
                "Ovid PsycINFO": parse_mappings(psy_map),
                "Elsevier Embase.com": parse_mappings(em_map),
            }
            st.session_state.translations = translate(source, mappings)
            st.session_state.translation_error = None
        except (ParseError, ValueError) as ex:
            st.session_state.translations = None
            st.session_state.translation_error = str(ex)
    if st.session_state.get("translation_error"):
        st.error(st.session_state.translation_error)
    results = st.session_state.get("translations")
    if results:
        tabs = st.tabs(list(results))
        for tab, (name, item) in zip(tabs, results.items()):
            with tab:
                st.code(item["query"], language=None)
                for note in item["notes"]:
                    st.caption(note)
                if item["removed"]:
                    with st.expander("Removed controlled descriptors"):
                        st.write(item["removed"])
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for name, entry in results.items():
                z.writestr(name.replace(" ", "_") + ".txt", entry["query"])
            z.writestr("quality_report.json", json.dumps(results, indent=2,
                                                        ensure_ascii=False))
        st.download_button("Download translated queries + quality report",
                           data.getvalue(), "searchbridge_translations.zip",
                           "application/zip")
    st.info("Translations are syntax proposals, not validated retrieval equivalents.")

with pubmed_tab:
    st.subheader("Direct PubMed AND search followed by Ovid-style adjN filtering")
    st.caption("OR synonyms within each group; AND between groups. Titles, abstracts and "
               "individual author keywords are tested independently after retrieval.")
    with st.form("pubmed_form"):
        left_col, right_col = st.columns(2)
        with left_col:
            left_text = st.text_area(
                "Group A", "juvenil* OR young* OR adolescent* OR youth*", height=105
            )
        with right_col:
            right_text = st.text_area(
                "Group B",
                "perpetr* OR offend* OR crimin* OR delinqu* OR justice* OR convict* OR prison*",
                height=105
            )
        x, y = st.columns(2)
        with x:
            max_records = st.number_input("Maximum candidate records", min_value=1,
                                          max_value=10000, value=1000, step=100)
            email = st.text_input("NCBI email (recommended)")
        with y:
            key = st.text_input("NCBI API key (optional)", type="password")
            st.caption("API rate limits apply. Credentials are not written to disk.")
        submitted = st.form_submit_button("Retrieve PubMed records", type="primary")
    try:
        left, right = parsed_terms(left_text), parsed_terms(right_text)
        query = build_pubmed_and_query(left, right)
        st.code(query, language="text")
        error = None
    except ValueError as exc:
        error = str(exc)
        st.error(error)
    if submitted and not error:
        st.session_state.pop("pubmed_data", None)
        bar = st.progress(0.0)
        def progress(done, total):
            bar.progress(done / total if total else 1.0)
        try:
            with st.spinner("PubMed ESearch / EFetch"):
                retrieval = fetch_pubmed_candidates(
                    query, int(max_records), email=email, api_key=key,
                    progress=progress
                )
            st.session_state.pubmed_data = (retrieval, left, right)
            st.success(f"Downloaded {retrieval.fetched:,} of "
                       f"{retrieval.candidate_count:,} PubMed candidates.")
        except (PubMedError, ValueError) as exc:
            st.error(f"PubMed API request failed: {exc}")
        finally:
            bar.empty()
    if "pubmed_data" in st.session_state:
        retrieval, last_left, last_right = st.session_state.pubmed_data
        if (left, right) != (last_left, last_right):
            st.warning("Groups changed: retrieve new candidates before interpreting results.")
        else:
            distance = st.slider("Ovid adjacency operator", 1, 10, 2,
                                 key="pubmed_adj")
            check_keywords = st.checkbox("Include author keywords", True,
                                         key="pubmed_kw")
            clause = ProximityClause(
                distance, last_left, last_right,
                ("ti", "ab", "kf") if check_keywords else ("ti", "ab")
            )
            records = retrieval.records
            rows = check_ris_records(records, clause)
            st.caption(f"Original ESearch candidates: {retrieval.candidate_count:,}. "
                       f"Downloaded: {len(rows):,}.")
            if retrieval.candidate_count != len(rows):
                st.warning("INCOMPLETE candidate retrieval. This is not a complete "
                           "systematic-review result set.")
            display_results(records, rows, clause)
            download_results(records, rows, clause, retrieval.query, "PubMed", {
                "query": retrieval.query,
                "pubmed_query_translation": retrieval.translation,
                "total_candidates": retrieval.candidate_count,
                "downloaded": len(rows),
                "proximity_clause": clause.label,
                "matches": sum(x["matched"] for x in rows),
                "complete": retrieval.candidate_count == len(rows),
            })
            st.info("Changing adjN rechecks already downloaded records without a new PubMed call.")

with ris_tab:
    st.subheader("Proximity checking in uploaded RIS files")
    upload = st.file_uploader("Upload a broad AND-query RIS export", type=["ris"])
    strategy = st.text_area("Ovid adjacency clause", PROX_SAMPLE, height=100)
    try:
        clauses = extract_clauses(strategy)
        clause = st.selectbox("Clause to verify", clauses,
                              format_func=lambda x: x.label)
    except (ParseError, ValueError) as exc:
        clause = None
        st.error(str(exc))
    if st.button("Check RIS", type="primary",
                 disabled=upload is None or clause is None):
        try:
            if upload.size > 25 * 1024 * 1024:
                raise ValueError("RIS file exceeds the 25 MB interactive limit.")
            records, encoding = parse_ris(upload.getvalue())
            if not records:
                raise ValueError("No RIS records found.")
            rows = check_ris_records(records, clause)
            st.session_state.ris_data = (records, rows, clause, upload.name, encoding)
            st.session_state.pop("ris_error", None)
        except (ValueError, ParseError, UnicodeError) as exc:
            st.session_state.ris_error = str(exc)
            st.session_state.pop("ris_data", None)
    if st.session_state.get("ris_error"):
        st.error(st.session_state.ris_error)
    if "ris_data" in st.session_state:
        records, rows, matched_clause, name, encoding = st.session_state.ris_data
        st.caption(f"Imported {name} ({encoding}). One selected adjN clause is evaluated.")
        if clause != matched_clause:
            st.warning("The selected clause has changed. Click Check RIS again.")
        display_results(records, rows, matched_clause)
        download_results(records, rows, matched_clause, name, "RIS")

with help_tab:
    st.markdown("""
    ### How it works
    - The translator rewrites supported Ovid operators and field codes.
      Unverified MeSH-to-Emtree mappings are not invented.
    - PubMed mode runs a broad Boolean AND query through E-utilities and
      post-filters retrieved title, abstract and author keyword fields.
    - RIS mode evaluates selected adjN clauses on exported bibliographic records.
    - adj2 allows 0–1 intervening words; adj3 allows 0–2, in either order.
    - Missing abstracts or keywords can produce false negatives.
    - For a complete review, *all* candidates must be retrieved, not just the first 1,000.
    - This is an approximation of Ovid proximity tokenization, not a validated
      identical implementation.
    ### Deploy
    From [Streamlit Community Cloud](https://share.streamlit.io/), choose this
    GitHub repository, main branch and app.py. No paid API key is required.
    Uploaded RIS files are processed in the hosted Streamlit environment.
    """)
