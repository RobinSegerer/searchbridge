"""Direct PubMed ESearch + EFetch retrieval for local Ovid-style proximity verification.

The ESearch query is intentionally a BROAD candidate search: OR synonyms within
word groups, AND the groups, fielded to PubMed [tiab]. Proximity is evaluated
against title/abstract/author keywords independently after EFetch.

No LLM or semantic matching. NCBI limits and field differences mean this is an
approximation of an Ovid .ti,ab,kf. query, not an exact replication.
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Callable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ris_proximity import RISRecord, ProximityClause, _compile_alternative

ROOT = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OR_SEPARATOR = re.compile(r"\s+OR\s+", re.I)
WORD_PART = re.compile(r"[^\W_]+[\w?*]*", re.U)
MAX_INTERACTIVE = 10000


class PubMedError(RuntimeError):
    pass


def parsed_terms(expression: str) -> tuple[str, ...]:
    """Parse a simple OR-list. Deliberately reject nested Boolean logic."""
    text = expression.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    if not text:
        raise ValueError("Bitte mindestens einen Begriff pro Gruppe angeben.")
    if re.search(r"\b(?:AND|NOT|ADJ\d+)\b", text, re.I):
        raise ValueError("Je Wortgruppe bitte nur OR-verknüpfte Begriffe eingeben, kein AND/NOT/adjN.")
    terms = [t.strip().strip('"\'').strip() for t in OR_SEPARATOR.split(text)]
    if not terms or any(not x for x in terms):
        raise ValueError("Leerer Begriff in der OR-Liste.")
    if any(re.search(r"[()\[\]{}|;]", t) for t in terms):
        raise ValueError("Keine verschachtelten Klammern oder Feld-Tags in den Wortgruppen verwenden.")
    if len(terms) > 100:
        raise ValueError("Maximal 100 Synonyme pro Wortgruppe erlaubt.")
    for term in terms:
        _compile_alternative(term)
    return tuple(dict.fromkeys(terms))


def _broaden_term_for_pubmed(term: str) -> str:
    """Construct a conservative AND of per-token [tiab] queries.

    Quoted phrases are split to individual term AND clauses to avoid losing
    candidates on PubMed phrase-index differences. Internal wildcards (Ovid ?)
    are broadened to trailing '*' at the stable prefix; local matching restores
    the intended wildcard. This aims at high recall, not identical semantics.
    """
    components = WORD_PART.findall(term)
    if not components:
        raise ValueError(f"Kein verwertbarer Begriff: {term!r}")
    clauses = []
    for part in components:
        pos = min([i for i in (part.find("?"), part.find("*")) if i != -1], default=-1)
        if pos >= 0:
            stem = part[:pos]
            if len(stem) < 3:
                raise ValueError(f"Wildcard zu früh im Begriff {term!r}; breitere PubMed-Suche nicht zuverlässig erstellbar.")
            part = stem + "*"
        clauses.append(f"{part}[tiab]")
    return "(" + " AND ".join(clauses) + ")" if len(clauses) > 1 else clauses[0]


def build_pubmed_and_query(left: tuple[str, ...], right: tuple[str, ...]) -> str:
    if not left or not right:
        raise ValueError("Beide Wortgruppen müssen Begriffe enthalten.")
    a = " OR ".join(_broaden_term_for_pubmed(t) for t in left)
    b = " OR ".join(_broaden_term_for_pubmed(t) for t in right)
    return f"({a}) AND ({b})"


def _text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _ris_value(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _first(elem: ET.Element, *paths: str) -> str:
    for path in paths:
        value = _text(elem.find(path))
        if value:
            return value
    return ""


def pubmed_xml_to_ris(xml: bytes, start_number: int = 1) -> list[RISRecord]:
    try:
        doc = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise PubMedError(f"Ungültige EFetch-XML-Antwort: {exc}") from exc
    result = []
    for el in doc.findall(".//PubmedArticle"):
        pmid = _first(el, "./MedlineCitation/PMID")
        if not pmid:
            continue
        title = _first(el, "./MedlineCitation/Article/ArticleTitle")
        pieces = []
        for node in el.findall("./MedlineCitation/Article/Abstract/AbstractText"):
            label = node.attrib.get("Label", "")
            val = _text(node)
            if val:
                pieces.append(f"{label}: {val}" if label and not val.lower().startswith(label.lower() + ":") else val)
        abstract = " ".join(pieces)
        kws = list(dict.fromkeys(_text(k) for k in el.findall("./MedlineCitation/KeywordList/Keyword") if _text(k)))
        # Assigned NLM MeSH descriptors (not MeSH entry terms and not author keywords).
        # Preserve MajorTopicYN separately for optional major-topic counts.
        mesh_descriptors = []
        mesh_major = []
        for mesh_heading in el.findall("./MedlineCitation/MeshHeadingList/MeshHeading"):
            descriptor = mesh_heading.find("DescriptorName")
            label = _text(descriptor)
            if not label:
                continue
            if label not in mesh_descriptors:
                mesh_descriptors.append(label)
            if descriptor is not None and descriptor.attrib.get("MajorTopicYN") == "Y":
                if label not in mesh_major:
                    mesh_major.append(label)
        doi = ""
        for aid in el.findall("./PubmedData/ArticleIdList/ArticleId"):
            if aid.attrib.get("IdType", "").lower() == "doi":
                doi = _text(aid)
                break
        if not doi:
            for aid in el.findall("./MedlineCitation/Article/ELocationID"):
                if aid.attrib.get("EIdType", "").lower() == "doi":
                    doi = _text(aid)
                    break
        year = _first(el,
                      "./MedlineCitation/Article/Journal/JournalIssue/PubDate/Year",
                      "./MedlineCitation/DateCompleted/Year")
        if not year:
            datetxt = _first(el, "./MedlineCitation/Article/Journal/JournalIssue/PubDate/MedlineDate")
            match = re.search(r"\b(?:18|19|20)\d{2}\b", datetxt)
            year = match.group() if match else ""
        journal = _first(el, "./MedlineCitation/Article/Journal/Title")
        authors = []
        for author in el.findall("./MedlineCitation/Article/AuthorList/Author"):
            family = _first(author, "LastName")
            first = _first(author, "ForeName", "Initials")
            corporate = _first(author, "CollectiveName")
            who = " ".join(x for x in (family, first) if x) or corporate
            if who:
                authors.append(who)
        data: dict[str, list[str]] = {"TY": ["JOUR"], "ID": [pmid], "AN": [pmid]}
        for tag, value in (("TI", title), ("AB", abstract), ("DO", doi), ("PY", year), ("JO", journal), ("UR", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")):
            if value:
                data[tag] = [_ris_value(value)]
        if kws:
            data["KW"] = [_ris_value(k) for k in kws]
        if mesh_descriptors:
            data["MH"] = [_ris_value(k) for k in mesh_descriptors]
        if mesh_major:
            data["M1"] = [_ris_value(k) for k in mesh_major]
        if authors:
            data["AU"] = [_ris_value(a) for a in authors]
        lines = [f"{tag}  - {v}" for tag, values in data.items() for v in values]
        lines.append("ER  -")
        result.append(RISRecord(number=start_number + len(result), fields=data, raw_lines=lines))
    return result


@dataclass
class Retrieval:
    records: list[RISRecord]
    candidate_count: int
    query: str
    translation: str
    fetched: int
    sort: str


class NCBIClient:
    def __init__(self, email: str = "", api_key: str = ""):
        self.email = email.strip()
        self.api_key = api_key.strip()
        self.session = requests.Session()
        retries = Retry(total=4, connect=3, read=3, backoff_factor=0.7,
                        status_forcelist=[429, 500, 502, 503, 504],
                        allowed_methods=["GET"], respect_retry_after_header=True)
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.last_request = 0.0
        self.min_interval = 0.12 if self.api_key else 0.36

    def get(self, endpoint: str, params: dict) -> requests.Response:
        elapsed = time.monotonic() - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        payload = dict(params)
        payload["tool"] = "SearchBridge"
        if self.email:
            payload["email"] = self.email
        if self.api_key:
            payload["api_key"] = self.api_key
        try:
            resp = self.session.get(f"{ROOT}/{endpoint}.fcgi", params=payload, timeout=(10, 80))
            self.last_request = time.monotonic()
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", "?")
            raise PubMedError(f"NCBI E-utilities-Anfrage fehlgeschlagen (HTTP {status}; {type(exc).__name__}). Bitte Netzwerk, API-Key und NCBI-Limits prüfen.") from exc


def fetch_pubmed_candidates(query: str, limit: int = 1000,
                            email: str = "", api_key: str = "",
                            progress: Callable[[int, int], None] | None = None) -> Retrieval:
    if not (1 <= limit <= MAX_INTERACTIVE):
        raise ValueError(f"Abruflimit muss zwischen 1 und {MAX_INTERACTIVE:,} liegen.")
    client = NCBIClient(email=email, api_key=api_key)
    response = client.get("esearch", {"db": "pubmed", "term": query, "retmode": "json",
                                      "retmax": 0, "usehistory": "y", "sort": "relevance"})
    try:
        data = response.json()["esearchresult"]
        total = int(data["count"])
    except (ValueError, KeyError, TypeError) as exc:
        raise PubMedError(f"Unerwartetes PubMed ESearch-Ergebnis: {response.text[:500]}") from exc
    if data.get("ERROR") or data.get("ErrorList", {}).get("PhraseNotFound"):
        raise PubMedError(f"PubMed kann die Suchanfrage nicht verarbeiten: {data.get('ERROR') or data.get('ErrorList')}")
    target = min(total, limit)
    records = []
    if target:
        webenv, querykey = data.get("webenv"), data.get("querykey")
        if not webenv or not querykey:
            raise PubMedError("PubMed lieferte keinen WebEnv/query_key für den EFetch-Abruf.")
        for offset in range(0, target, 200):
            count = min(200, target - offset)
            response = client.get("efetch", {"db": "pubmed", "query_key": querykey,
                                             "WebEnv": webenv, "retmode": "xml",
                                             "retstart": offset, "retmax": count})
            batch = pubmed_xml_to_ris(response.content, start_number=len(records) + 1)
            if not batch and count:
                raise PubMedError(f"Keine PubMedArticle-Datensätze im EFetch-Batch ab Position {offset}.")
            records.extend(batch)
            if progress:
                progress(min(target, offset + count), target)
    return Retrieval(records=records, candidate_count=total, query=query,
                     translation=data.get("querytranslation", ""), fetched=len(records), sort="relevance")
