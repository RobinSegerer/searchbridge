"""Ovid MEDLINE strategy conversion to three external scholarly APIs.

OpenAlex: native OQL preserving proximity via OpenAlex's own query syntax.
Crossref: explicitly NON-Boolean short exploratory bibliographic search probes.
Semantic Scholar: Academic Graph Bulk Boolean query, explicit lost Ovid features.

Every target produces a transparent loss report. None claims equivalence to
MeSH-indexed Ovid MEDLINE.
"""
from __future__ import annotations

from openalex_oql import translate_openalex_oql
from crossref_translation import translate_crossref
from semantic_scholar_translation import translate_semantic_scholar


def translate_external(
    source: str, *,
    explosions: dict[str, list[str]] | None = None,
) -> dict[str, dict]:
    """Return query, URL, mode, original semantic losses and audit metadata."""
    return {
        "OpenAlex": translate_openalex_oql(source, explosions=explosions),
        "Semantic Scholar": translate_semantic_scholar(
            source, explosions=explosions
        ),
        "Crossref": translate_crossref(source),
    }
