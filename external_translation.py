"""Adapt an Ovid MEDLINE search strategy to OpenAlex, Crossref, Semantic Scholar.

Honest translation semantics:
  * OpenAlex provides Boolean logic, wildcard and scoped text search. Ovid
    adjacency falls back to AND for broad candidate retrieval, not exact matching.
  * Semantic Scholar Graph paper/search/bulk supports Boolean groups (+, |, -)
    but NOT the relevance-search endpoint /paper/search. No per-field Ovid limit.
  * Crossref is a bibliographic RELEVANCE query, not a Boolean-equivalent search;
    the flattening is explicitly labelled and carries all query terms.

MeSH heading terms are converted to unverified text tokens (not subject headings).
Optional, previously verified NLM MeSH explosion labels can be appended as
free-text candidates, with explicit warnings for semantic and recall differences.

None of the generated results is represented as validated Ovid equivalence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from urllib.parse import urlencode

from translator_core import Node, heading_details, parse_ovid

BASES = {
    "OpenAlex": "https://api.openalex.org/works",
    "Crossref": "https://api.crossref.org/works",
    "Semantic Scholar": "https://api.semanticscholar.org/graph/v1/paper/search/bulk",
}


@dataclass
class Conversion:
    target: str
    query: str
    endpoint: str
    url: str
    mode: str
    notes: list[str] = field(default_factory=list)
    lost: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "query": self.query, "endpoint": self.endpoint, "url": self.url,
            "mode": self.mode, "notes": list(dict.fromkeys(self.notes)),
            "removed": list(dict.fromkeys(self.lost)),
            "unverified": list(dict.fromkeys(self.unverified)),
            "validated_equivalence": False,
        }


def _quoted_literal(text: str, target: str, notes: list[str]) -> str:
    text = text.strip().strip('"\'')
    if not text:
        raise ValueError("Empty Ovid search term.")
    if re.search(r"\$\d+|[#^]", text):
        notes.append(f"Ovid Spezialtrunkierung '{text}' nicht direkt übertragbar; manuelle Korrektur erforderlich.")
    # Standard Ovid $ truncation, if present.
    text = re.sub(r"\$(?!\d)", "*", text)
    if target == "Semantic Scholar" and "?" in text:
        before = text
        # S2 bulk supports prefix wildcards, not Ovid's one-char wildcard.
        text = re.sub(r"\?[\w]*", "*", text)
        notes.append(f"Ein-Zeichen-Wildcard {before!r} verbreitert zu {text!r}; Präzision prüfen.")
    # A hyphenated expression is still a normal full-text term but will be
    # tokenized by each target's backend differently.
    if " " in text:
        if "*" in text or "?" in text:
            notes.append(f"Wildcard innerhalb mehrteiliger Phrase '{text}': Ziel-API-Semantik prüfen.")
        return f'"{text}"'
    if target == "OpenAlex" and re.search(r"[*?]", text):
        stem = text.split("*")[0].split("?")[0]
        if len(stem) < 3:
            notes.append(f"OpenAlex verlangt mindestens 3 Zeichen vor einem Wildcard: '{text}'. Abfrage könnte fehlschlagen.")
    return text


def _walk(node: Node):
    yield node
    if node.left is not None:
        yield from _walk(node.left)
    if node.right is not None:
        yield from _walk(node.right)


def _fields(node: Node) -> set[str]:
    fields = set()
    for n in _walk(node):
        if n.kind == "field":
            fields.update(x.strip().lower() for x in n.value.strip(".").split(","))
    return fields


def _scoped_openalex_param(tree: Node, notes: list[str], lost: list[str]) -> str:
    field_nodes = [n for n in _walk(tree) if n.kind == "field"]
    scopes: set[str] = set()
    for n in field_nodes:
        fs = {x.strip().lower() for x in n.value.strip(".").split(",")}
        if fs <= {"ti"}:
            scopes.add("title")
        elif fs <= {"ti", "ab"}:
            scopes.add("title_and_abstract")
        elif fs <= {"ti", "ab", "kf", "kw", "id"}:
            scopes.add("title_abstract_keywords")
        else:
            scopes.add("title_abstract_keywords")
            lost.append(f"Feldcodes {n.value} auf Title/Abstract/Keywords verbreitert.")
    # Full Boolean scope at the request level: per-branch field limits cannot
    # be reproduced by a single scoped search parameter.
    if len(scopes) > 1:
        notes.append("Gemischte Ovid-Feldscopes können in einer einzelnen OpenAlex search-Abfrage nicht zweigweise erhalten bleiben. Auf Titel/Abstract/Keywords verbreitert.")
        return "search.title_abstract_keywords"
    if not scopes:
        return "search.title_abstract_keywords"
    mapping = {
        "title": "search.title",
        "title_and_abstract": "search.title_and_abstract",
        "title_abstract_keywords": "search.title_abstract_keywords",
    }
    return mapping[next(iter(scopes))]


def _expanded_heading(
    name: str, exploded: bool, target: str, notes: list[str],
    unverified: list[str], explosions: dict[str, list[str]] | None
) -> str:
    unverified.append(("exp " if exploded else "") + name + "/")
    labels = [name]
    if exploded and explosions:
        expanded = explosions.get(name.casefold(), [])
        if expanded:
            labels += [x for x in expanded if x.casefold() != name.casefold()]
            notes.append(
                f"exp {name}/: {len(labels)-1} MeSH-Descendants als zusätzliche "
                "FREITEXT-Phrasen aufgenommen (kein echtes Index-/Heading-Mapping)."
            )
    bits = [_quoted_literal(x, target, notes) for x in list(dict.fromkeys(labels))]
    if len(bits) > 1:
        return "(" + (" OR " if target == "OpenAlex" else " | ").join(bits) + ")"
    return bits[0]


def _render(tree: Node, target: str, notes: list[str], lost: list[str],
            unverified: list[str], explosions: dict[str, list[str]] | None) -> str:
    if tree.kind == "text":
        return _quoted_literal(tree.value, target, notes)
    if tree.kind == "heading":
        name, exploded, qualifier = heading_details(tree.value)
        if qualifier:
            lost.append(f"Heading-Qualifier {tree.value!r} entfällt.")
        notes.append(f"{tree.value}: kontrollierter MeSH-Descriptor durch Freitext ersetzt; kein gleichwertiger Thesaurus-Suchzweig.")
        return _expanded_heading(name, exploded, target, notes, unverified, explosions)
    if tree.kind == "field":
        # A scoped OpenAlex search can apply one combined scope. Semantic Scholar
        # bulk always matches title/abstract without per-branch Ovid field codes.
        if target == "Semantic Scholar":
            lost.append(f"Ovid-Feldbeschränkung {tree.value} entfällt (S2: Titel/Abstract).")
        return _render(tree.left, target, notes, lost, unverified, explosions)
    if tree.kind == "near":
        n = int(tree.value[3:])
        # Intentionally broader candidate query: broad AND is not a claimed
        # replacement for adjacency, which can be checked on downloaded texts.
        left = _render(tree.left, target, notes, lost, unverified, explosions)
        right = _render(tree.right, target, notes, lost, unverified, explosions)
        lost.append(f"{tree.value} durch breites AND ersetzt; anschliessende lokale Proximity-Prüfung empfohlen.")
        if target == "OpenAlex":
            return f"({left} AND {right})"
        return f"(+({left}) +({right}))"
    if tree.kind == "bool":
        left = _render(tree.left, target, notes, lost, unverified, explosions)
        right = _render(tree.right, target, notes, lost, unverified, explosions)
        if target == "OpenAlex":
            if tree.value == "NOT":
                return f"({left} NOT {right})"
            return f"({left} {tree.value} {right})"
        if tree.value == "NOT":
            return f"(+({left}) -({right}))"
        if tree.value == "OR":
            return f"({left} | {right})"
        if tree.value == "AND":
            return f"(+({left}) +({right}))"
    raise ValueError(f"Unsupported Ovid node: {tree.kind}")


def _crossref_terms(tree: Node, notes: list[str], lost: list[str]) -> list[str]:
    terms: list[str] = []
    for n in _walk(tree):
        if n.kind == "heading":
            label, exploded, qualifier = heading_details(n.value)
            terms.append(label)
            notes.append(f"{n.value}: nur Descriptor-Text für Crossref verwendet, keine MeSH-Indexierung.")
            if exploded:
                lost.append(f"exp {label}/: MeSH-Explosion entfällt.")
            if qualifier:
                lost.append(f"MeSH-Qualifier /{qualifier} entfällt.")
        elif n.kind == "text":
            source = n.value.strip().strip('"\'')
            source = re.sub(r"\$\d+|[$*?]", "", source)
            source = source.replace('"', "")
            if source:
                terms.append(source)
        elif n.kind == "near":
            lost.append(f"{n.value}: keine Proximity-Beschränkung bei Crossref.")
        elif n.kind == "bool":
            if n.value in ("AND", "OR", "NOT"):
                lost.append(f"Boolesche Operation {n.value} kann Crossref nicht ausführen.")
        elif n.kind == "field":
            lost.append(f"Feld {n.value} bei Crossref nicht als Ovid-Feld abbildbar.")
    return list(dict.fromkeys(terms))


def _crossref_target(tree: Node) -> Conversion:
    notes = [
        "EXPLORATIVE bibliografische Relevanzsuche – KEINE äquivalente Übersetzung.",
        "Crossref unterstützt keine Booleschen Suchoperatoren. Seine query.bibliographic-Suche verlangt ungefähr 20% der Query-Wörter und gewichtet Treffer nach Relevanz.",
        "Die Originalstruktur ist im Query-String nicht mehr vorhanden. Für systematische Recherchen ist diese Fassung nicht als reproduktionsgleichwertig einsetzbar.",
    ]
    lost: list[str] = []
    terms = _crossref_terms(tree, notes, lost)
    # The API's 20% minimum makes large OR groups especially risky; do not
    # truncate unannounced. Crossref is deliberately a discovery query.
    q = " ".join(terms)
    if not q:
        raise ValueError("No text left for Crossref search.")
    uri = BASES["Crossref"] + "?" + urlencode({"query.bibliographic": q, "rows": 20})
    if len(uri.encode("utf-8")) > 7000:
        notes.append("Die Crossref-GET-URL ist sehr lang; bitte manuell auf thematisch zentrale Begriffe kürzen oder Suchblöcke einzeln explorieren.")
    return Conversion("Crossref", q, "GET /works?query.bibliographic", uri,
                      "exploratory (NOT boolean equivalent)", notes, lost)


def translate_external(
    source: str, *,
    explosions: dict[str, list[str]] | None = None,
) -> dict[str, dict]:
    """Return API endpoint, actual parameter value, URL and loss notes."""
    tree = parse_ovid(source)
    result = {}
    for target in ("OpenAlex", "Semantic Scholar"):
        notes: list[str] = []
        lost: list[str] = []
        unverified: list[str] = []
        rendered = _render(tree, target, notes, lost, unverified, explosions)
        if target == "OpenAlex":
            param = _scoped_openalex_param(tree, notes, lost)
            # OpenAlex requires search.exact for wildcard queries.
            if re.search(r"(?<!\w)[\w-]+[*?]", rendered):
                param += ".exact"
                notes.append("Wildcards erfordern die ungestemmte OpenAlex-Variante '.exact'; Stemming ist für die gesamte kombinierte Suche deaktiviert.")
            uri = BASES[target] + "?" + urlencode({param: rendered, "per-page": 25})
            if len(uri.encode("utf-8")) > 8190:
                notes.append("GET-URL überschreitet OpenAlex-Limit von 8'190 Bytes. Ausführung erfordert OQL-POST oder Query-Splitting; diese GET-URL NICHT verwenden.")
                uri = ""
            notes.append("OpenAlex-Keywords werden algorithmisch vergeben; sie sind keine Ovid .kf.-Heading-Äquivalente.")
            notes.append("Für grösseren Zugriff empfiehlt OpenAlex einen kostenlosen API-Key; für Cursor-Pagination ist er erforderlich.")
            endpoint = f"GET /works?{param}="
            mode = "Boolean text candidate search (not Ovid-equivalent)"
        else:
            uri = BASES[target] + "?" + urlencode({
                "query": rendered, "fields": "paperId,title,abstract,year,externalIds",
            })
            notes.append("Nur Graph /paper/search/bulk versteht diese Boolesche Syntax; NICHT /paper/search oder die normale Website-Suche verwenden.")
            notes.append("S2-Bulk durchsucht Titel und Abstract, ohne Ovid-Feldbegrenzung; strukturierte MeSH-Indexierung ist nicht verfügbar.")
            endpoint = "GET /graph/v1/paper/search/bulk?query="
            mode = "Boolean text candidate search (not Ovid-equivalent)"
        obj = Conversion(target, rendered, endpoint, uri, mode,
                         notes, lost, unverified)
        result[target] = obj.as_dict()
    result["Crossref"] = _crossref_target(tree).as_dict()
    return result
