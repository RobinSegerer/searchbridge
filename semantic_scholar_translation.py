"""Readable Semantic Scholar Academic Graph BULK Boolean query translation.

Only /graph/v1/paper/search/bulk supports its documented +, |, -, (),
quotes and prefix * syntax. /paper/search and the website DO NOT.

Ovid ADJn is *not* translated to S2's phrase slop: Ovid is unordered
and the phrase distance semantics differ. Instead use broad AND, explicitly
documenting proximity loss. Follow with local title/abstract verification.
"""
from __future__ import annotations

import re
from urllib.parse import urlencode

from translator_core import Node, heading_details, parse_ovid

ROOT = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
MAX_S2_URL = 7900


def _walk(n: Node):
    yield n
    if n.left is not None:
        yield from _walk(n.left)
    if n.right is not None:
        yield from _walk(n.right)


def _flatten(n: Node, op: str) -> list[Node]:
    if n.kind == "bool" and n.value == op:
        return _flatten(n.left, op) + _flatten(n.right, op)
    return [n]


def _term(raw: str, notes: list[str]) -> str:
    term = raw.strip().strip('"\'')
    if not term:
        raise ValueError("Leerer Ovid-Term im Semantic-Scholar-Translator.")
    if re.search(r"\$\d+|[#^]", term):
        notes.append(f"Ovid-Spezialtrunkierung {raw!r} ist nicht gleichwertig "
                     "in Semantic Scholar vorhanden.")
    term = re.sub(r"\$\d+", "*", term).replace("$", "*")
    if "?" in term:
        before = term
        def broaden(m: re.Match) -> str:
            prefix = m.group(1)
            if len(prefix) < 2:
                raise ValueError(f"Zu kurze Wildcard-Basis für {raw!r}.")
            return prefix + "*"
        term = re.sub(r"([^\W_]+)\?[^\W_]*", broaden, term)
        notes.append(
            f"Ein-Zeichen-Wildcard {before!r} als breitere Präfix-Wildcard "
            f"{term!r} übertragen. Ovid-Äquivalenz geht verloren."
        )
    # Hyphens can behave as query operators in the Lucene-style parser.
    # Replace with a broadly matched multiword group rather than risking
    # accidental NOT/exclusion or zero matches.
    if "-" in term and not term.startswith("-"):
        before = term
        term = term.replace("-", " ")
        notes.append(f"Bindestrich in {before!r} als Leerzeichen abgebildet "
                     "(andere Tokenisierung und Phrasenpräzision).")
        if "*" in term or "?" in term:
            tokens = [x for x in term.split() if x]
            return "(" + " ".join("+"+x for x in tokens) + ")"
    if " " in term:
        if "*" in term:
            notes.append(f"Mehrwort-Term {raw!r} mit Wildcard wird als "
                         "breite AND-Wortgruppe ausgegeben; keine exakte Phrase.")
            return "(" + " ".join("+" + x for x in term.split()) + ")"
        return '"' + term + '"'
    if "*" in term and not term.endswith("*"):
        notes.append(f"Nur Präfix-Wildcards sind garantiert; {term!r} "
                     "bitte manuell prüfen.")
    return term


def _or(parts: list[str]) -> str:
    parts = list(dict.fromkeys(parts))
    if len(parts) == 1:
        return parts[0]
    return "(" + " | ".join(parts) + ")"


def _prefix_group(symbol: str, part: str) -> str:
    # A parenthesized Boolean group is already enclosed; avoid adding
    # duplicate nesting while retaining required/excluded group semantics.
    if part.startswith("(") and part.endswith(")"):
        return symbol + part
    return symbol + "(" + part + ")"


def _and(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    return "(" + " ".join(_prefix_group("+", p) for p in parts) + ")"


def _render(node: Node, notes: list[str], lost: list[str],
            unverified: list[str],
            explosions: dict[str, list[str]] | None) -> str:
    if node.kind == "text":
        return _term(node.value, notes)
    if node.kind == "heading":
        name, exploded, qualifier = heading_details(node.value)
        unverified.append(node.value)
        notes.append(f"{node.value}: MeSH-Descriptor wird nur als Freitext "
                     "gesucht; keine MeSH-Indexierung.")
        if qualifier:
            lost.append(f"MeSH-Qualifier /{qualifier} entfällt.")
        labels = [name]
        if exploded and explosions and explosions.get(name.casefold()):
            labels.extend(explosions[name.casefold()])
            notes.append(f"exp {name}/: Unterbegriffe lediglich als "
                         "Freitextalternativen aufgenommen.")
        return _or([_term(x, notes) for x in dict.fromkeys(labels)])
    if node.kind == "field":
        lost.append(
            f"Ovid-Feldbeschränkung {node.value} nicht in S2 Bulk "
            "(Suchtext: Titel und Abstract)."
        )
        return _render(node.left, notes, lost, unverified, explosions)
    if node.kind == "near":
        left = _render(node.left, notes, lost, unverified, explosions)
        right = _render(node.right, notes, lost, unverified, explosions)
        lost.append(
            f"{node.value}: durch breites AND ersetzt; die gerichtete "
            "S2-Phrasensuche ist NICHT dasselbe wie bidirektionales Ovid ADJ. "
            "Proximity nach dem Abruf lokal prüfen."
        )
        return _and([left, right])
    if node.kind == "bool":
        if node.value in ("AND", "OR"):
            items = _flatten(node, node.value)
            rendered = [_render(n, notes, lost, unverified, explosions) for n in items]
            return _and(rendered) if node.value == "AND" else _or(rendered)
        if node.value == "NOT":
            left = _render(node.left, notes, lost, unverified, explosions)
            right = _render(node.right, notes, lost, unverified, explosions)
            return "(" + _prefix_group("+", left) + " " + _prefix_group("-", right) + ")"
    raise ValueError(f"Unbekannter Suchoperator: {node.kind}")


def translate_semantic_scholar(
    source: str, *, explosions: dict[str, list[str]] | None = None,
) -> dict:
    tree = parse_ovid(source)
    notes: list[str] = [
        "Nur der Academic Graph Bulk-Endpunkt /paper/search/bulk "
        "unterstützt diese Suchsyntax; NICHT die normale Websuche oder /paper/search.",
        "S2 Bulk durchsucht Titel und Abstract. Er hat keine gleichwertigen "
        "Ovid-Feldbegrenzungen und keine kontrollierte MeSH-Indexierung.",
        "Es sind höchstens 1'000 Treffer pro Seite verfügbar; "
        "für umfangreiche Suchläufe ist Token-Pagination nötig.",
        "Ohne eigenen S2-API-Key kann HTTP 429 auftreten; dann ist die "
        "Abfrage NICHT gegen den Live-Parser bestätigt.",
    ]
    lost: list[str] = []
    unverified: list[str] = []
    query = _render(tree, notes, lost, unverified, explosions)
    uri = ROOT + "?" + urlencode({
        "query": query, "fields": "paperId,title,abstract,year,externalIds",
    })
    if len(uri.encode("utf-8")) > MAX_S2_URL:
        notes.append(
            "Die GET-URL ist zu lang für einen verlässlichen API-Aufruf. "
            "Eine einzelne S2-Bulk-GET-Abfrage nicht ausführen; die "
            "Suche in mehrere Konzeptabfragen aufteilen und die "
            "identifizierten Publikationen lokal abgleichen."
        )
        uri = ""
    return {
        "query": query,
        "endpoint": "GET /graph/v1/paper/search/bulk?query=",
        "url": uri,
        "mode": "Boolean candidate search (unverified Ovid equivalence)",
        "notes": list(dict.fromkeys(notes)),
        "removed": list(dict.fromkeys(lost)),
        "unverified": list(dict.fromkeys(unverified)),
        "validated_equivalence": False,
        "backend": "S2 Academic Graph Bulk",
        "field_support": "title + abstract only",
        "proximity_preserved": not any(n.kind == "near" for n in _walk(tree)),
    }
