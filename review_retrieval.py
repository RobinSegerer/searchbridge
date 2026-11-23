"""SearchBridge federated retrieval using public/credentialled official APIs.

PubMed uses existing NCBI ESearch/EFetch with history.
OpenAlex uses native OQL POST /, with explicit pagination/credit limits.
Scopus uses Elsevier's search endpoint with X-ELS-APIKey and optional
institutional token; STANDARD metadata may lack abstracts.

Each retrieval exposes the source count, retrieved count and errors
separately. No data is silently represented as complete.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pubmed_proximity import fetch_pubmed_candidates
from review_core import (
    ReviewRecord, from_openalex_results, from_scopus_results, from_ris
)


class RetrievalError(RuntimeError):
    pass


@dataclass
class SourceResult:
    source: str
    records: list[ReviewRecord]
    total: int
    query: str
    complete: bool
    warning: str = ""


def _http() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=2, backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def pubmed_search(query: str, limit: int, *, email: str = "",
                  api_key: str = "",
                  progress: Callable[[int, int], None] | None = None) -> SourceResult:
    r = fetch_pubmed_candidates(
        query, limit=limit, email=email, api_key=api_key, progress=progress
    )
    records = from_ris(r.records, "PubMed")
    return SourceResult(
        "PubMed", records, r.candidate_count, query,
        complete=r.candidate_count == len(records),
        warning="" if len(records) == r.candidate_count else
        "Nur ein Teil der PubMed-Treffer wurde abgerufen."
    )


def openalex_search(oql: str, limit: int = 100, *,
                    api_key: str = "",
                    progress: Callable[[int, int], None] | None = None) -> SourceResult:
    if not oql.strip().casefold().startswith("get works"):
        raise ValueError(
            "OpenAlex verlangt OQL beginnend mit 'get works'. "
            "Im Search String Translator kann die korrekte OQL-Suche erzeugt werden."
        )
    if not 1 <= limit <= 2000:
        raise ValueError("OpenAlex Abfrage-Limit: 1–2'000 Publikationen.")
    if not api_key and limit > 100:
        raise ValueError(
            "OpenAlex verlangt für die Pagination über die erste Seite "
            "einen eigenen API-Key. Bitte Limit ≤ 100 oder Key verwenden."
        )
    session = _http()
    headers = {"Content-Type": "application/json",
               "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    records: list[ReviewRecord] = []
    total = 0
    page = 1
    while len(records) < limit:
        count = min(100, limit - len(records))
        try:
            response = session.post(
                "https://api.openalex.org/",
                json={"oql": oql.strip(), "per_page": count, "page": page},
                headers=headers, timeout=(12, 90),
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", "?")
            diagnostic = ""
            if response is not None:
                try:
                    detail = response.json()
                    problems = (detail.get("validation") or {}).get("errors") or []
                    if problems:
                        diagnostic = " / ".join(
                            str(p.get("message", "")) for p in problems[:2])
                    elif isinstance(detail, dict):
                        diagnostic = str(
                            detail.get("message") or detail.get("error") or "")[:220]
                except (ValueError, TypeError, AttributeError):
                    diagnostic = ""
            if records:
                raise RetrievalError(
                    f"OpenAlex wurde nach {len(records)} Datensätzen unterbrochen "
                    f"(HTTP {status}). Dieser Teilabruf wurde NICHT als "
                    "vollständige Suche übernommen. " + diagnostic[:400]
                ) from exc
            raise RetrievalError(
                f"OpenAlex HTTP {status}: {diagnostic[:400] or type(exc).__name__}. "
                "OQL, API-Key und Rate-/Budget-Limits prüfen."
            ) from exc
        if page == 1:
            metadata = data.get("meta") or {}
            total = int(metadata.get("count") or metadata.get("total") or 0)
        items = data.get("results") or []
        if not isinstance(items, list):
            raise RetrievalError("OpenAlex: unerwartetes results-Format.")
        records.extend(from_openalex_results(items))
        if progress:
            progress(len(records), min(limit, total) if total else limit)
        if not items or len(items) < count:
            break
        page += 1
        if len(records) < limit:
            time.sleep(0.6 if api_key else 1.2)
    if total == 0 and records:
        total = len(records)  # Explicit fallback when no count is supplied.
    return SourceResult(
        "OpenAlex", records[:limit], total, oql,
        complete=total == len(records),
        warning="" if total == len(records) else
        f"OpenAlex: {len(records):,} von {total:,} Treffern abgerufen "
        "(API-Limit/Pagination)."
    )


def scopus_search(query: str, limit: int = 200, *, api_key: str,
                  inst_token: str = "",
                  progress: Callable[[int, int], None] | None = None) -> SourceResult:
    if not api_key.strip():
        raise ValueError("Scopus Search API erfordert einen Elsevier API-Key.")
    if not 1 <= limit <= 2000:
        raise ValueError("Scopus-Limit: 1–2'000 Datensätze.")
    headers = {
        "X-ELS-APIKey": api_key.strip(),
        "Accept": "application/json",
    }
    if inst_token.strip():
        headers["X-ELS-Insttoken"] = inst_token.strip()
    session = _http()
    entries: list[dict] = []
    total = 0
    offset = 0
    while len(entries) < limit:
        count = min(25, limit - len(entries))
        try:
            res = session.get(
                "https://api.elsevier.com/content/search/scopus",
                headers=headers,
                params={
                    "query": query, "start": offset, "count": count,
                    "view": "STANDARD",
                    "httpAccept": "application/json",
                },
                timeout=(12, 80),
            )
            res.raise_for_status()
            payload = res.json()
        except (requests.RequestException, ValueError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", "?")
            raise RetrievalError(
                f"Scopus API HTTP {status}: Suche/Entitlements, "
                "Institutionszugriff oder API-Limits prüfen. "
                "Kein Teilabruf als vollständig ausgewiesen."
            ) from exc
        response = payload.get("search-results") or {}
        if offset == 0:
            try:
                total = int(response.get("opensearch:totalResults") or 0)
            except (ValueError, TypeError):
                total = 0
        block = response.get("entry") or []
        if not isinstance(block, list):
            raise RetrievalError("Scopus lieferte kein standardisiertes Entry-Array.")
        entries.extend(block)
        if progress:
            progress(len(entries), min(limit, total) if total else limit)
        if not block or len(block) < count:
            break
        offset += len(block)
        if len(entries) < limit:
            time.sleep(0.2)
    records = from_scopus_results(entries[:limit])
    return SourceResult(
        "Scopus", records, total, query, complete=total == len(records),
        warning=(
            "Scopus STANDARD enthält möglicherweise keine Abstracts; "
            "Title/Abstract-Screening kann daher nur eingeschränkt erfolgen. "
            "Es gelten die institutionellen API-Lizenzbedingungen."
        )
    )
