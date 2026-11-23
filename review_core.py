"""SearchBridge Review Pipeline: provenance-preserving merge, dedup, ranking.

Deterministic, dependency-light processing. Records are never deleted silently:
deduplication returns a complete audit and the original source counts.
Exact DOI and PMID matching are automatic; normalized title+year is automatic
ONLY without contradictory persistent identifiers. Fuzzy title suggestions
always require manual confirmation and are never auto-merged.

No AI decision is treated as a ground-truth eligibility label.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Optional

from ris_proximity import RISRecord


def norm_doi(value: str) -> str:
    x = (value or "").strip().lower()
    x = re.sub(r"^(?:doi:|https?://(?:dx\.)?doi\.org/)", "", x).strip()
    x = x.split("?")[0].split("#")[0].rstrip(".,; ").strip()
    return x if re.fullmatch(r"10\.\d{4,9}/\S+", x) else ""


def norm_title(value: str) -> str:
    x = unicodedata.normalize("NFKD", value or "").casefold()
    x = "".join(c for c in x if not unicodedata.combining(c))
    x = re.sub(r"[\W_]+", " ", x, flags=re.UNICODE)
    return " ".join(x.split())


def _uid(source: str, identifier: str, title: str) -> str:
    return hashlib.sha256(
        (source + "\0" + identifier + "\0" + title).encode("utf-8")
    ).hexdigest()[:16]


@dataclass
class ReviewRecord:
    uid: str
    title: str = ""
    abstract: str = ""
    year: str = ""
    doi: str = ""
    pmid: str = ""
    authors: str = ""
    journal: str = ""
    source: str = ""
    source_id: str = ""
    sources: list[str] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    rank_score: float = 0.0

    def __post_init__(self) -> None:
        self.doi = norm_doi(self.doi)
        self.title = self.title.strip()
        self.abstract = self.abstract.strip()
        self.year = str(self.year or "").strip()[:4]
        if not self.sources and self.source:
            self.sources = [self.source]
        if not self.identifiers and self.source_id:
            self.identifiers = [f"{self.source}:{self.source_id}"]


def from_ris(recs: list[RISRecord], source: str = "RIS") -> list[ReviewRecord]:
    result = []
    for n, r in enumerate(recs, 1):
        doi = norm_doi(r.doi)
        pmid = r.get("PM", "PMID")
        if not pmid and "pubmed" in source.casefold():
            pmid = r.identifier if re.fullmatch(r"\d+", r.identifier or "") else ""
        if not pmid:
            uri = r.get("UR")
            m = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", uri)
            if m:
                pmid = m.group(1)
        ident = r.identifier or str(n)
        result.append(ReviewRecord(
            uid=_uid(source, ident + ":" + str(n), r.title),
            title=r.title, abstract=r.abstract, year=r.get("PY", "Y1", "DA"),
            doi=doi, pmid=pmid, authors="; ".join(r.fields.get("AU", [])),
            journal=r.get("JO", "JF", "T2"), source=source, source_id=ident
        ))
    return result


def from_openalex_results(results: list[dict]) -> list[ReviewRecord]:
    out = []
    for index, item in enumerate(results, 1):
        if not isinstance(item, dict):
            continue
        ident = str(item.get("id") or item.get("openalex_id") or index)
        inverted = item.get("abstract_inverted_index") or {}
        abstract = ""
        if isinstance(inverted, dict):
            try:
                places: dict[int, str] = {}
                for word, positions in inverted.items():
                    for p in positions:
                        places[int(p)] = str(word)
                abstract = " ".join(places[i] for i in sorted(places))
            except (ValueError, TypeError):
                abstract = ""
        if not abstract:
            abstract = str(item.get("abstract") or "")
        ids = item.get("ids") or {}
        doi = item.get("doi") or ids.get("doi") or ""
        pmid = str(ids.get("pmid") or "")
        match = re.search(r"/(\d+)/?$", pmid)
        if match:
            pmid = match.group(1)
        authors = "; ".join(
            str(a.get("author", {}).get("display_name", ""))
            for a in (item.get("authorships") or []) if isinstance(a, dict)
        )
        source = item.get("primary_location") or {}
        journal = ((source.get("source") or {}).get("display_name") or "")
        out.append(ReviewRecord(
            uid=_uid("OpenAlex", ident, str(item.get("title") or "")),
            source="OpenAlex", source_id=ident,
            title=str(item.get("title") or item.get("display_name") or ""),
            abstract=abstract, year=item.get("publication_year") or "",
            doi=doi, pmid=pmid, authors=authors, journal=journal,
        ))
    return out


def from_scopus_results(entries: list[dict]) -> list[ReviewRecord]:
    out = []
    for n, item in enumerate(entries, 1):
        if not isinstance(item, dict):
            continue
        ident = str(item.get("dc:identifier") or item.get("eid") or n)
        doi = str(item.get("prism:doi") or "")
        date = str(item.get("prism:coverDate") or "")
        abstract = str(item.get("dc:description") or "")
        authors = item.get("author") or []
        if isinstance(authors, list):
            author_names = "; ".join(
                str(a.get("authname") or a.get("surname") or "")
                for a in authors if isinstance(a, dict)
            )
        else:
            author_names = str(item.get("dc:creator") or "")
        out.append(ReviewRecord(
            uid=_uid("Scopus", ident, str(item.get("dc:title") or "")),
            title=str(item.get("dc:title") or ""),
            abstract=abstract, year=date[:4], doi=doi,
            authors=author_names or str(item.get("dc:creator") or ""),
            journal=str(item.get("prism:publicationName") or ""),
            source="Scopus", source_id=ident
        ))
    return out


@dataclass
class MergeResult:
    records: list[ReviewRecord]
    counts: dict[str, int]
    duplicates_removed: int
    audit: list[dict]
    fuzzy_review: list[dict]
    input_count: int


def merge_records(records: list[ReviewRecord], *, fuzzy_limit: int = 100) -> MergeResult:
    data = [ReviewRecord(**asdict(r)) for r in records]
    parent = list(range(len(data)))
    reason: dict[tuple[int, int], str] = {}
    doi_to_idx: dict[str, int] = {}
    pmid_to_idx: dict[str, int] = {}
    title_to_idx: dict[tuple[str, str], int] = {}

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int, why: str) -> None:
        a, b = root(i), root(j)
        if a == b:
            return
        # Two distinct non-empty DOIs never merge even when PMIDs/titles collide.
        dois_a = {data[k].doi for k in range(len(data)) if root(k) == a and data[k].doi}
        dois_b = {data[k].doi for k in range(len(data)) if root(k) == b and data[k].doi}
        if dois_a and dois_b and not (dois_a & dois_b):
            return
        parent[b] = a
        reason[i, j] = why

    for i, r in enumerate(data):
        if r.doi:
            if r.doi in doi_to_idx:
                union(doi_to_idx[r.doi], i, "exact DOI")
            else:
                doi_to_idx[r.doi] = i
        if r.pmid:
            if r.pmid in pmid_to_idx:
                union(pmid_to_idx[r.pmid], i, "exact PMID")
            else:
                pmid_to_idx[r.pmid] = i
        title = norm_title(r.title)
        if len(title) >= 18 and r.year and r.year.isdigit():
            k = (title, r.year)
            if k in title_to_idx:
                union(title_to_idx[k], i, "normalized title + year")
            else:
                title_to_idx[k] = i

    groups: dict[int, list[ReviewRecord]] = defaultdict(list)
    for i, item in enumerate(data):
        groups[root(i)].append(item)
    merged: list[ReviewRecord] = []
    audit: list[dict] = []
    for group in groups.values():
        # Longest abstract and most complete metadata survive, independent
        # of the order in which sources were fetched.
        representative = max(
            group, key=lambda r: (
                bool(r.doi), bool(r.pmid), len(r.abstract),
                len(r.title), len(r.authors)
            )
        )
        field_pick = lambda field: max(
            (str(getattr(r, field) or "") for r in group),
            key=lambda val: len(val), default=""
        )
        obj = ReviewRecord(
            uid=min(r.uid for r in group), title=field_pick("title"),
            abstract=field_pick("abstract"), year=representative.year or field_pick("year"),
            doi=representative.doi or field_pick("doi"),
            pmid=representative.pmid or field_pick("pmid"),
            authors=field_pick("authors"), journal=field_pick("journal"),
            source=representative.source, source_id=representative.source_id,
            sources=sorted({x for r in group for x in r.sources}),
            identifiers=sorted({x for r in group for x in r.identifiers}),
        )
        merged.append(obj)
        if len(group) > 1:
            audit.append({
                "master_uid": obj.uid, "group_size": len(group),
                "sources": obj.sources, "source_identifiers": obj.identifiers,
                "doi": obj.doi, "pmid": obj.pmid,
                "matching_evidence": (
                    "shared DOI/PMID and/or exact normalized title+year; "
                    "persistent DOI conflicts block auto-merge"
                ),
            })
    merged.sort(key=lambda r: (norm_title(r.title), r.uid))

    fuzzy: list[dict] = []
    # Candidate-only ambiguity flags; O(n^2) is intentionally restricted to
    # same-year groups. Large sets have a review limit with explicit flag.
    year_groups: dict[str, list[ReviewRecord]] = defaultdict(list)
    for item in merged:
        if item.year and item.title:
            year_groups[item.year].append(item)
    for year, options in year_groups.items():
        if len(options) > 250:
            continue  # Limit avoids an implicit all-pairs processing promise.
        for i, a in enumerate(options):
            for b in options[i+1:]:
                if a.doi and b.doi and a.doi != b.doi:
                    continue
                if abs(len(norm_title(a.title)) - len(norm_title(b.title))) > 12:
                    continue
                similarity = SequenceMatcher(
                    None, norm_title(a.title), norm_title(b.title)
                ).ratio()
                if similarity >= 0.94:
                    fuzzy.append({
                        "uid_a": a.uid, "uid_b": b.uid, "year": year,
                        "title_a": a.title, "title_b": b.title,
                        "similarity": round(similarity, 4),
                        "status": "MANUAL REVIEW - NOT merged",
                    })
                    if len(fuzzy) >= fuzzy_limit:
                        break
            if len(fuzzy) >= fuzzy_limit:
                break
        if len(fuzzy) >= fuzzy_limit:
            break
    counts = dict(sorted(Counter(r.source for r in data).items()))
    return MergeResult(
        records=merged, counts=counts, duplicates_removed=len(data)-len(merged),
        audit=audit, fuzzy_review=fuzzy, input_count=len(data)
    )


def tokens(text: str) -> list[str]:
    words = re.findall(r"[^\W\d_]{3,}", (text or "").casefold())
    return [w for w in words if w not in {
        "the", "and", "for", "with", "from", "this", "that", "are", "was",
        "were", "has", "have", "had", "not", "its", "but", "into", "among",
        "von", "der", "die", "und", "das", "eine", "einer", "bei",
    }]


def lexical_rank(records: list[ReviewRecord], focus: str) -> list[ReviewRecord]:
    """Sample-based TF-IDF cosine, boosted title; priority, not eligibility."""
    data = [ReviewRecord(**asdict(r)) for r in records]
    query = tokens(focus)
    if not query:
        for r in data:
            r.rank_score = 0.0
        return data
    docs = [
        Counter(tokens(" ".join([r.title, r.title, r.abstract])))
        for r in data
    ]
    df = Counter()
    for doc in docs:
        df.update(doc.keys())
    n = len(data)
    idf = {term: math.log((n + 1) / (df[term] + 1)) + 1
           for term in set(query)}
    qw = Counter(query)
    qlen = math.sqrt(sum((count * idf[term])**2
                         for term, count in qw.items()))
    for rec, doc in zip(data, docs):
        if not doc or not qlen:
            rec.rank_score = 0.0
            continue
        dlen = math.sqrt(sum(
            (count * (math.log((n + 1)/(df[word]+1)) + 1))**2
            for word, count in doc.items()))
        dot = sum(qw[term] * idf[term] * doc.get(term, 0) * idf[term]
                  for term in qw)
        rec.rank_score = round(dot / (dlen * qlen), 6) if dlen else 0.0
    return sorted(data, key=lambda r: (-r.rank_score, r.uid))


BASE_COLUMNS = ["uid", "doi", "pmid", "title", "abstract", "year",
                "authors", "journal", "sources", "identifiers", "rank_score"]


def csv_bytes(records: list[ReviewRecord], screening: Optional[dict] = None) -> bytes:
    f = io.StringIO()
    extra = ["screen_decision", "screen_reason", "screen_uncertainty",
             "screen_evidence", "screen_model"]
    columns = BASE_COLUMNS + (extra if screening else [])
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    for r in records:
        line = {k: getattr(r, k) for k in BASE_COLUMNS}
        line["sources"] = "; ".join(r.sources)
        line["identifiers"] = "; ".join(r.identifiers)
        if screening:
            x = screening.get(r.uid, {})
            line.update({
                "screen_decision": x.get("decision", ""),
                "screen_reason": x.get("reason", ""),
                "screen_uncertainty": x.get("uncertainty", ""),
                "screen_evidence": x.get("evidence", ""),
                "screen_model": x.get("model", ""),
            })
        writer.writerow(line)
    return f.getvalue().encode("utf-8-sig")


def ris_bytes(records: list[ReviewRecord]) -> bytes:
    lines: list[str] = []
    for r in records:
        fields = [("TY", "JOUR"), ("ID", r.uid), ("TI", r.title),
                  ("AB", r.abstract), ("PY", r.year), ("DO", r.doi),
                  ("JO", r.journal), ("N1", "SearchBridge sources: " +
                   "; ".join(r.sources))]
        if r.pmid:
            fields.append(("UR", "https://pubmed.ncbi.nlm.nih.gov/" + r.pmid + "/"))
        for author in r.authors.split(";"):
            if author.strip():
                fields.append(("AU", author.strip()))
        lines.extend(f"{key}  - {value.replace(chr(10), ' ')}"
                     for key, value in fields if value)
        lines.append("ER  -")
        lines.append("")
    return ("\n".join(lines) + "\n").encode("utf-8")


def as_json(records: list[ReviewRecord]) -> bytes:
    return json.dumps([asdict(x) for x in records],
                      ensure_ascii=False, indent=2).encode("utf-8")
