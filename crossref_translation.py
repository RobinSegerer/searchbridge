"""Crossref discovery probes from an Ovid strategy (NEVER an equivalent search).

Crossref REST /works?query.bibliographic does not support Boolean/proximity,
requires about 20% of query words, and indexes bibliographic metadata only.
Consequently a long concatenation of every synonym is deceptive.

Instead suggest short, transparent exploration queries: one automatically
selected Ovid lexical anchor per top-level AND concept, plus each anchor by
itself. All omitted positive alternatives and lost operators are reported.
No negative NOT term is included in the query; it cannot be enforced.

The short anchors are only convenience defaults, not automatically optimized
strategies. Researchers should edit them before serious use.
"""
from __future__ import annotations

import re
from urllib.parse import urlencode

from translator_core import Node, heading_details, parse_ovid

ROOT = "https://api.crossref.org/works"
MAX_ANCHOR_WORDS = 10


def _walk(node: Node):
    yield node
    if node.left:
        yield from _walk(node.left)
    if node.right:
        yield from _walk(node.right)


def _and_blocks(node: Node) -> list[Node]:
    if node.kind == "bool" and node.value == "AND":
        return _and_blocks(node.left) + _and_blocks(node.right)
    return [node]


def _first_leaf(node: Node) -> str:
    if node.kind == "heading":
        name, _, _ = heading_details(node.value)
        return name
    if node.kind == "text":
        return node.value
    if node.kind == "field":
        return _first_leaf(node.left)
    if node.kind in ("near", "bool"):
        return _first_leaf(node.left)
    return ""


def _first_proximity(node: Node) -> Node | None:
    for part in _walk(node):
        if part.kind == "near":
            return part
    return None


def _clean(text: str) -> str:
    text = text.strip().strip('"\'')
    # Crossref has no wildcards; this does NOT expand truncation to variants.
    text = re.sub(r"\$\d+|[*?$#]", "", text)
    text = re.sub(r"[\(\)\[\]{}]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _source_terms(node: Node) -> tuple[list[str], list[str]]:
    """Return every positive text/heading and every negative term separately."""
    positive: list[str] = []
    negative: list[str] = []

    def visit(part: Node, exclude: bool = False) -> None:
        if part.kind == "heading":
            val = _clean(heading_details(part.value)[0])
            if val:
                (negative if exclude else positive).append(val)
        elif part.kind == "text":
            val = _clean(part.value)
            if val:
                (negative if exclude else positive).append(val)
        elif part.kind == "field":
            visit(part.left, exclude)
        elif part.kind == "bool" and part.value == "NOT":
            visit(part.left, exclude)
            visit(part.right, not exclude)
        else:
            if part.left:
                visit(part.left, exclude)
            if part.right:
                visit(part.right, exclude)

    visit(node)
    return list(dict.fromkeys(positive)), list(dict.fromkeys(negative))


def _anchor(block: Node) -> str:
    """Pick an illustrative two-part phrase based on the first Ovid ADJn.

    This is a mechanical default, not semantic ranking, and is labelled so.
    """
    prox = _first_proximity(block)
    if prox:
        pieces = (_first_leaf(prox.left), _first_leaf(prox.right))
    else:
        pieces = (_first_leaf(block),)
    combined = " ".join(_clean(x) for x in pieces if x).strip()
    words = combined.split()
    return " ".join(words[:4])


def _api_url(q: str) -> str:
    return ROOT + "?" + urlencode({
        "query.bibliographic": q, "rows": 20,
        "select": "DOI,title,score,type,published",
    })


def translate_crossref(source: str) -> dict:
    tree = parse_ovid(source)
    blocks = _and_blocks(tree)
    source_terms, excluded = _source_terms(tree)
    anchors = []
    for number, block in enumerate(blocks, 1):
        value = _anchor(block)
        if value:
            anchors.append({
                "block": number,
                "query": value,
                "url": _api_url(value),
                "candidate_terms": _source_terms(block)[0],
                "selection": "first lexical pair from first adjN clause, else first term",
            })

    if not anchors:
        raise ValueError("No positive bibliographic text available for Crossref.")
    # One anchor per concept, with a reproducible length limit. Any truncation
    # of anchors is explicit and included in the query metadata.
    snippets = []
    used = 0
    for obj in anchors:
        part = obj["query"].split()
        if used + len(part) > MAX_ANCHOR_WORDS:
            break
        snippets.append(obj["query"])
        used += len(part)
    combined = " ".join(snippets)
    if not combined:
        combined = anchors[0]["query"]

    lost = []
    for node in _walk(tree):
        if node.kind == "near":
            lost.append(f"{node.value}: no proximity operator.")
        if node.kind == "bool":
            lost.append(f"{node.value}: Boolean operation not available.")
        if node.kind == "heading":
            lost.append(f"{node.value}: MeSH indexing/explosion is unavailable.")
        if node.kind == "field":
            lost.append(f"{node.value}: Ovid field restriction is unavailable.")
    notes = [
        "Exploratory bibliographic search only: NOT a Boolean or "
        "retrieval-equivalent translation.",
        "Crossref query.bibliographic requires around 20% of the "
        "query terms to match; a long OR synonym string can systematically "
        "miss relevant publications.",
        "The combined search automatically uses the first "
        "lexical anchor from each top-level AND topic block (up to "
        f"{MAX_ANCHOR_WORDS} words). This is an illustrative selection, "
        "not a validated expert selection.",
        f"Originally {len(source_terms)} distinct positive "
        f"term candidates, including {len(anchors)} topic-block anchors. "
        "All original term candidates are included in the export.",
        "Crossref search results must not be treated as complete "
        "systematic search result sets.",
    ]
    if len(snippets) < len(anchors):
        notes.append("Some topic-block anchors were omitted due to the "
                     "10-word limit of the combined search.")
    if excluded:
        notes.append(
            f"{len(excluded)} excluded terms (NOT branch) cannot be "
            "excluded and are NOT part "
            "of the Crossref query."
        )
    return {
        "query": combined,
        "endpoint": "GET /works?query.bibliographic=",
        "url": _api_url(combined),
        "mode": "exploratory Crossref metadata discovery (not Boolean)",
        "notes": list(dict.fromkeys(notes)),
        "removed": list(dict.fromkeys(lost)),
        "unverified": [],
        "validated_equivalence": False,
        "source_terms": source_terms,
        "excluded_not_terms": excluded,
        "source_term_count": len(source_terms),
        "anchors": anchors,
        "anchor_strategy": "first Ovid adjN lexical pair per top-level AND block",
    }
