"""SearchBridge – rule-based search translator and PubMed/RIS proximity checker."""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone

import streamlit as st
import requests
from urllib.parse import urlencode

from translator_core import translate, ParseError
from external_translation import translate_external
from ris_proximity import (
    ProximityClause, parse_ris, extract_clauses, check_ris_records,
    html_report, csv_bytes, ris_bytes,
)
from pubmed_proximity import (
    PubMedError, parsed_terms, build_pubmed_and_query, fetch_pubmed_candidates,
)
from harvesting_ui import render as render_harvesting
from review_pipeline_ui import render as render_review_pipeline
from mesh_expansion import (
    MeSHError, find_exp_headings, expand_mesh_heading, ExplodedHeading,
    display_rows as mesh_display_rows, csv_export as mesh_csv_export,
    labels_as_text as mesh_labels_as_text,
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


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def cached_mesh_expansion(name: str, qualifier: str):
    """Fetch official NLM MeSH RDF hierarchy; cache results for 24h."""
    return expand_mesh_heading(ExplodedHeading(name, qualifier))


translate_tab, pubmed_tab, ris_tab, harvest_tab, pipeline_tab, help_tab = st.tabs([
    "Search String Translator", "PubMed Proximity Search",
    "RIS Proximity Checker", "Term Harvesting", "Review Pipeline", "Help",
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
            results = translate(source, mappings)
            results.update(translate_external(source))
            st.session_state.translations = results
            st.session_state.translated_source = source
            st.session_state.exp_headings = find_exp_headings(source)
            st.session_state.mesh_expansions = {}
            st.session_state.mesh_errors = {}
            st.session_state.translation_error = None
        except (ParseError, ValueError) as ex:
            st.session_state.translations = None
            st.session_state.exp_headings = []
            st.session_state.mesh_expansions = {}
            st.session_state.mesh_errors = {}
            st.session_state.translation_error = str(ex)
    if st.session_state.get("translation_error"):
        st.error(st.session_state.translation_error)
    results = st.session_state.get("translations")
    if results:
        tabs = st.tabs(list(results))
        for tab, (name, item) in zip(tabs, results.items()):
            with tab:
                if item.get("mode"):
                    st.warning("Keine validierte 1:1-Übersetzung. Die API-Abfrage verändert die Suchsemantik; bitte alle Hinweise prüfen.")
                    st.caption(f"API-Endpunkt: {item['endpoint']} · Modus: {item['mode']}")
                st.code(item["query"], language=None)
                if name == "OpenAlex" and item.get("request_json"):
                    if not item.get("proximity_preserved", True):
                        skipped = [x for x in item.get("removed", [])
                                   if "nur für dieses Paar breites AND" in x]
                        st.warning(
                            f"OpenAlex kann bei {len(skipped)} Nähepaar(en) "
                            "die Kombination der Ovid-Wildcards nicht als echte "
                            "Proximity-Abfrage ausführen. Nur diese Paarzweige "
                            "wurden als breiteres AND übertragen, "
                            "ohne die Wildcards abzuschneiden. "
                            "Die konkreten Stellen stehen unten im Verlustbericht."
                        )
                    if item.get("uses_post"):
                        st.info(
                            "Die vollständige OQL-Strategie ist für einen GET-Link zu lang. "
                            "OpenAlex verwendet hierfür POST / mit dem JSON-Body unten. "
                            "Es wurde kein Suchterm abgeschnitten."
                        )
                    else:
                        st.caption("Die Abfrage kann direkt mit GET ausgeführt werden; "
                                   "der identische OQL-Ausdruck ist auch als POST-JSON verfügbar.")
                    with st.expander("OpenAlex POST-Request (JSON) und Syntaxvalidierung",
                                     expanded=bool(item.get("uses_post"))):
                        st.code(item["request_json"], language="json")
                        st.download_button(
                            "Vollständige OpenAlex-POST-Abfrage (JSON) herunterladen",
                            item["request_json"].encode("utf-8"),
                            "searchbridge_openalex_oql.json",
                            "application/json", key="download_openalex_oql")
                        if st.button(
                            "OQL-Syntax mit OpenAlex prüfen (kostenlos)",
                            key="validate_openalex_oql"):
                            with st.spinner("OpenAlex OQL-Parser wird aufgerufen"):
                                try:
                                    response = requests.post(
                                        "https://api.openalex.org/query",
                                        json={"oql": item["query"]},
                                        timeout=60)
                                    response.raise_for_status()
                                    validation = response.json().get("validation", {})
                                    if validation.get("valid") is True:
                                        st.success("OpenAlex bestätigt: OQL ist syntaktisch gültig.")
                                        st.caption(
                                            "Syntaxvalidität prüft nicht die Äquivalenz "
                                            "zu MEDLINE oder den tatsächlichen Retrievalgewinn.")
                                    else:
                                        st.error("OpenAlex meldet Syntax- oder Validierungsprobleme.")
                                        st.json(validation)
                                except (requests.RequestException, ValueError) as exc:
                                    st.error(f"OpenAlex-Syntaxvalidierung nicht verfügbar: {exc}")
                if item.get("url"):
                    if name == "OpenAlex":
                        st.link_button("OpenAlex API-Abfrage öffnen", item["url"])
                    elif name == "Crossref":
                        st.link_button("Crossref API-Abfrage öffnen", item["url"])
                    elif name == "Semantic Scholar":
                        st.link_button("Semantic Scholar Bulk-API-Abfrage öffnen", item["url"])
                if name == "Crossref" and item.get("anchors"):
                    st.warning(
                        "Crossref hat keine Boolesche Abfragesprache. Diese "
                        "Suchanker sind explorative Metadaten-Suchen und keine "
                        "äquivalente Übersetzung deiner Strategie."
                    )
                    with st.expander(
                        "Crossref-Suchanker je Themenblock und vollständige "
                        "Begriffsliste", expanded=True):
                        for anchor in item["anchors"]:
                            st.markdown(
                                f"**Themenblock {anchor['block']}:** "
                                f"`{anchor['query']}`"
                            )
                            st.link_button(
                                f"Crossref: Block {anchor['block']} durchsuchen",
                                anchor["url"])
                        st.caption(
                            f"Quellstrategie: {item.get('source_term_count', 0)} "
                            "unterschiedliche positive Begriffskandidaten; "
                            "die Reihenfolge und die erste Nähepaarung bestimmen "
                            "die automatisch gewählten Anker. Bitte fachlich prüfen."
                        )
                        st.write(item.get("source_terms", []))
                        if item.get("excluded_not_terms"):
                            st.warning(
                                "Ausgeschlossene NOT-Begriffe können nicht "
                                "durch Crossref ausgeschlossen werden."
                            )
                            st.write(item["excluded_not_terms"])
                    custom_crossref = st.text_input(
                        "Crossref-Bibliografie-Suche manuell anpassen",
                        value=item["query"],
                        key="crossref_custom_probe")
                    if custom_crossref.strip():
                        custom_url = ("https://api.crossref.org/works?"
                                      + urlencode({
                                          "query.bibliographic": custom_crossref.strip(),
                                          "rows": 20,
                                          "select": "DOI,title,score,type,published",
                                      }))
                        st.link_button("Angepasste Crossref-Abfrage öffnen",
                                       custom_url)
                if name == "Semantic Scholar":
                    st.caption(
                        "Diese Suche gilt nur für die Academic Graph Bulk API. "
                        "Die Websuche und /paper/search unterstützen sie nicht."
                    )
                    if item.get("url"):
                        with st.expander("Semantic-Scholar-API-Key und Live-Prüfung"):
                            s2_key = st.text_input(
                                "Semantic Scholar API-Key (optional)",
                                type="password", key="s2_search_key")
                            if st.button("Bulk-Suchabfrage live überprüfen",
                                         key="test_s2_search"):
                                try:
                                    response = requests.get(
                                        item["url"], timeout=35,
                                        headers=(
                                            {"x-api-key": s2_key.strip()}
                                            if s2_key.strip() else {}
                                        ))
                                    if response.ok:
                                        payload = response.json()
                                        st.success(
                                            "Semantic Scholar Bulk hat die "
                                            "Abfrage angenommen (HTTP 200)."
                                        )
                                        st.caption(
                                            f"Geschätzte Gesamttreffer: "
                                            f"{payload.get('total', 'nicht verfügbar')} · "
                                            f"Datensätze auf dieser Seite: "
                                            f"{len(payload.get('data') or [])}"
                                        )
                                        st.caption(
                                            "HTTP 200 bestätigt keine Äquivalenz "
                                            "zur Ovid-MEDLINE-Strategie."
                                        )
                                    elif response.status_code == 429:
                                        st.warning(
                                            "HTTP 429 – S2-Anfragelimit. "
                                            "Die Syntax konnte nicht live "
                                            "bestätigt werden; bitte mit "
                                            "eigenem API-Key erneut prüfen."
                                        )
                                    else:
                                        st.error(
                                            f"Semantic Scholar meldet HTTP "
                                            f"{response.status_code}: "
                                            f"{response.text[:500]}"
                                        )
                                except (requests.RequestException, ValueError) as exc:
                                    st.error(f"S2 Live-Test fehlgeschlagen: {exc}")
                    else:
                        st.error(
                            "Die API-URL ist zu lang. Diese vollständige Suche "
                            "kann nicht als einzelner S2-Bulk-GET-Aufruf "
                            "ausgeführt werden. Suchblöcke aufteilen und "
                            "die Resultate anhand der Identifier zusammenführen."
                        )
                for note in item["notes"]:
                    if ("PROVISORISCHE" in note or "KEINE Emtree" in note or
                            "kontrollierte Heading-Klausel(n) entfernt" in note):
                        st.warning(note)
                    else:
                        st.caption(note)
                if item.get("removed") and item.get("mode"):
                    with st.expander(f"Nicht exakt übertragbare Suchoperationen ({len(item['removed'])})"):
                        st.write(item["removed"])
                if item.get("unverified"):
                    with st.expander(f"Unverified subject headings ({len(item['unverified'])})", expanded=not bool(item.get("mode"))):
                        st.write(item["unverified"])
                        st.caption("Verify descriptors against the target database thesaurus and provide confirmed mappings above.")
                if item["removed"] and not item.get("mode"):
                    with st.expander("Removed controlled descriptors"):
                        st.write(item["removed"])
        st.divider()
        st.subheader("MeSH EXP: enthaltene untergeordnete Descriptors")
        st.caption(
            "Quelle: offizielle NLM-MeSH-RDF-Hierarchie (aktuelle Version). "
            "Angezeigt werden der Hauptdescriptor und alle engeren Descriptors aus der "
            "MeSH-Baumstruktur, mit Hierarchiestufe und Tree Number. "
            "Nicht mit MeSH Entry Terms oder kontrollierten Vokabularen von PsycINFO/Embase gleichsetzen."
        )
        exp_headings = st.session_state.get("exp_headings", [])
        source_current = source == st.session_state.get("translated_source")
        if not source_current:
            st.warning("Der eingegebene Suchstring wurde seit der letzten Übersetzung geändert. "
                       "Bitte erneut Translate ausführen, bevor du die EXP-Hierarchie lädst.")
        elif not exp_headings:
            st.info("Keine exp-Descriptors in dieser Ovid-MEDLINE-Strategie gefunden.")
        else:
            st.caption(f"{len(exp_headings)} unterschiedliche exp-Descriptors erkannt.")
            if st.button("Alle MeSH-EXP-Hierarchien laden", key="load_all_mesh", type="secondary"):
                for h in exp_headings:
                    identity = (h.name.casefold(), h.qualifier)
                    if identity not in st.session_state.mesh_expansions:
                        try:
                            with st.spinner(f"NLM MeSH: {h.label}"):
                                st.session_state.mesh_expansions[identity] = cached_mesh_expansion(
                                    h.name, h.qualifier)
                        except MeSHError as exc:
                            st.session_state.mesh_errors[identity] = str(exc)
            if st.session_state.get("mesh_expansions"):
                st.caption("Optional: zuvor geladene NLM-MeSH-Unterbegriffe können als zusätzliche "
                           "Freitext-Phrasen in OpenAlex und Semantic Scholar aufgenommen werden. "
                           "Das ersetzt KEINE MeSH-Indexierung.")
                if st.button("API-Abfragen mit geladenen MeSH-EXP-Unterbegriffen aktualisieren",
                             key="apply_exp_to_api"):
                    variants = {
                        exp.heading.name.casefold():
                        [exp.resolved_name] +
                        list(dict.fromkeys(p.descriptor for p in exp.positions if not p.is_root))
                        for exp in st.session_state.mesh_expansions.values()
                    }
                    st.session_state.translations.update(
                        translate_external(source, explosions=variants))
                    st.rerun()
            for index, h in enumerate(exp_headings):
                identity = (h.name.casefold(), h.qualifier)
                expansion = st.session_state.mesh_expansions.get(identity)
                label = (f"{h.label} — {expansion.children_count} untergeordnete Descriptors"
                         if expansion else h.label)
                with st.expander(label, expanded=expansion is not None):
                    if expansion is None:
                        if st.button("MeSH-Unterbegriffe laden", key=f"load_mesh_{index}"):
                            try:
                                with st.spinner(f"NLM-MeSH-Abfrage: {h.name}"):
                                    expansion = cached_mesh_expansion(h.name, h.qualifier)
                                st.session_state.mesh_expansions[identity] = expansion
                                st.session_state.mesh_errors.pop(identity, None)
                            except MeSHError as exc:
                                st.session_state.mesh_errors[identity] = str(exc)
                        if identity in st.session_state.mesh_errors:
                            st.error(st.session_state.mesh_errors[identity])
                    if expansion is not None:
                        c_a, c_b, c_c = st.columns(3)
                        c_a.metric("Untergeordnete Descriptors", expansion.children_count)
                        c_b.metric("MeSH-Descriptors inkl. Hauptbegriff", expansion.descriptor_count)
                        c_c.metric("Baumpositionen", len(expansion.positions))
                        st.caption(
                            f"MeSH-ID: {expansion.mesh_id} · "
                            f"MeSH-Root-Tree(s): {', '.join(expansion.root_trees) or 'none'} · "
                            f"Datenstand abgerufen: {expansion.fetched_utc[:10]}"
                        )
                        if h.qualifier:
                            st.warning("Der Subheading-/Qualifier-Suffix wurde bei der "
                                       "Hierarchieauflösung nicht angewendet. Qualifier-Vererbung "
                                       "muss separat überprüft werden.")
                        if not expansion.root_trees:
                            st.warning("Dieser Descriptor besitzt keine MeSH-Tree-Number; "
                                       "es konnten keine engeren Hierarchiepositionen abgeleitet werden.")
                        st.dataframe(mesh_display_rows(expansion), use_container_width=True,
                                     hide_index=True, column_config={
                                         "NLM": st.column_config.LinkColumn(
                                             "NLM descriptor", display_text="Open"),
                                         "Level": st.column_config.NumberColumn(
                                             "Level", help="0 = Ausgangsdescriptor, 1 = direkte Unterkategorie"),
                                     })
                        st.caption("Ein Descriptor kann mehrfach in verschiedenen MeSH-Baumzweigen vorkommen. "
                                   "Der Descriptor-Zähler ist nach MeSH-ID dedupliziert; "
                                   "die Tabelle enthält sämtliche Hierarchiepositionen.")
                        c_csv, c_txt = st.columns(2)
                        with c_csv:
                            st.download_button(
                                "Alle Tree-Positionen als CSV",
                                mesh_csv_export(expansion),
                                f"mesh_exp_{expansion.mesh_id}.csv",
                                "text/csv", key=f"mesh_csv_{index}",
                                use_container_width=True)
                        with c_txt:
                            st.download_button(
                                "Eindeutige Descriptors als TXT",
                                mesh_labels_as_text(expansion).encode("utf-8"),
                                f"mesh_exp_{expansion.mesh_id}.txt",
                                "text/plain", key=f"mesh_txt_{index}",
                                use_container_width=True)
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for name, entry in results.items():
                z.writestr(name.replace(" ", "_") + ".txt", entry["query"])
                if entry.get("url"):
                    z.writestr(name.replace(" ", "_") + "_api_url.txt", entry["url"])
            z.writestr("quality_report.json", json.dumps(results, indent=2,
                                                        ensure_ascii=False))
            if source_current:
                for expansion in st.session_state.get("mesh_expansions", {}).values():
                    z.writestr(f"MeSH_EXP_{expansion.mesh_id}.csv", mesh_csv_export(expansion))
                    z.writestr(f"MeSH_EXP_{expansion.mesh_id}_unique_terms.txt",
                               mesh_labels_as_text(expansion))
        st.download_button("Download translated queries + quality report",
                           data.getvalue(), "searchbridge_translations.zip",
                           "application/zip")
    st.info("These are syntax proposals, not validated retrieval equivalents. For PsycINFO, MeSH headings without verified mappings are provisional; for Embase, such headings are converted into labelled free-text fallbacks. Web of Science and Scopus omit indexed headings entirely.")

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

with harvest_tab:
    render_harvesting()


with pipeline_tab:
    render_review_pipeline()


with help_tab:
    st.markdown("""
    ### How it works
    - The Review Pipeline tab searches PubMed, OpenAlex (OQL API) and
      Scopus (Elsevier API key/entitlements) and accepts additional RIS exports
      from other databases. Results are merged preserving source provenance.
      Exact DOI, PMID and non-conflicting normalized title+year duplicates
      are merged; fuzzy matches are reported only for manual review.
    - Lexical TF-IDF ranks all deduplicated records without excluding any.
      OpenAI Responses API screening is optional, user-key-funded and opt-in:
      suggestions are structured Include/Exclude/Manual Review with evidence
      and uncertainty; only human decisions count as final.
      The full run exports RIS, CSV, JSON audits, deduplication logs,
      source counts, AI suggestions and human decisions.
    - The translator rewrites supported Ovid operators and field codes.
    - The OpenAlex tab now generates native **OQL proximity** rather than
      replacing Ovid adjN with AND. adj2 maps to within 1 intervening word,
      adj3 to within 2. OR word lists are distributed to full proximity
      pair combinations. Long strategies use a downloadable POST JSON request;
      use the syntax-validation button to check them with OpenAlex's parser.
      Textual searches and MeSH indexing are not equivalent.
    - The Semantic Scholar tab uses the Graph /paper/search/bulk endpoint,
      compact +(...)/-()/OR grouping, and explicit loss warnings for adjN,
      Ovid fields, wildcard approximations. With a Semantic Scholar API key,
      the request can be tested directly; anonymous requests often get 429.
      This grammar does not work on the website or /paper/search.
    - Crossref does NOT support Boolean queries. Instead of concatenating all
      synonyms, the converter now produces short discovery probes: one anchor
      per top-level AND topic block plus a short combined query. The full
      original term inventory is preserved, with a manual-edit field. These
      searches are explorative bibliographic matches, NOT systematic search
      equivalents; Crossref requires roughly 20% matching query words.
    - Previously loaded MeSH EXP descendants can optionally be added as
      labelled free-text candidate phrases for OpenAlex/Semantic Scholar.
      Unverified MeSH-to-Emtree mappings are not invented.
    - In the translator's **MeSH EXP** section, click *Alle MeSH-EXP-Hierarchien laden*
      or resolve one exp descriptor at a time. The NLM RDF API provides
      the source MeSH root and all narrower descriptors, levels and tree positions.
      This is the **MeSH** hierarchy only, not a PsycINFO or Emtree explosion.
      If NLM is unavailable or an exact descriptor cannot be resolved, the
      app displays an error instead of guessing.
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
    The optional Review Pipeline uses external APIs. PubMed is free subject
    to NCBI rate limits; OpenAlex has budget and paging rules (API key is
    recommended); Scopus requires subscription/permissions. OpenAI screening
    requires an API key, express consent and expert verification.
    """)
