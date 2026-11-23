"""OpenAlex OQL conversion preserving Ovid ADJn proximity where representable.

Official native OQL syntax:
get works where title-abstract-keywords has (<boolean text expression>)
("word1" and "word2" within N words of each other)

Ovid (a OR b) adj3 (c OR d) becomes OR of 4 *real proximity*
clauses; Ovid adjN has N-1 intervening words.

The conversion is deliberately labelled non-equivalent because OpenAlex
keyword indexing and MeSH indexing differ from Ovid MEDLINE. The API may
require POST / with an OQL JSON body for long review strategies.
"""
from __future__ import annotations

import json
import re
from urllib.parse import urlencode

from translator_core import Node, heading_details, parse_ovid

ROOT = "https://api.openalex.org/"
QUERY_VALIDATE_URL = "https://api.openalex.org/query"
MAX_OQL_GET = 7900
MAX_PROX_PAIRS = 1000


class OQLConversionError(ValueError):
    pass


def _walk(tree: Node):
    yield tree
    if tree.left:
        yield from _walk(tree.left)
    if tree.right:
        yield from _walk(tree.right)


def _flatten(node: Node, operator: str) -> list[Node]:
    if node.kind == "bool" and node.value == operator:
        return _flatten(node.left, operator) + _flatten(node.right, operator)
    return [node]


def _clean_phrase(text: str, notes: list[str]) -> str:
    term = text.strip().strip('"\'')
    if not term:
        raise OQLConversionError("Leerer Begriff für OpenAlex OQL.")
    if "\\" in term:
        raise OQLConversionError(f"Backslash in Ovid-Begriff {term!r}: OQL kann diesen nicht übernehmen.")
    if '"' in term:
        raise OQLConversionError("Anführungszeichen innerhalb eines Terms müssen manuell geprüft werden.")
    if re.search(r"\$\d+|#", term):
        notes.append(f"Ovid Spezialtrunkierung {term!r} nicht identisch verfügbar; bitte manuell anpassen.")
        term = re.sub(r"\$\d+", "*", term).replace("#", "?")
    term = term.replace("$", "*")
    if re.search(r"(^|\s)[*?]", term):
        notes.append(f"OpenAlex unterstützt keine führenden Wildcards: {term!r}")
    for m in re.finditer(r"[\w-]+[*?]", term):
        prefix = re.split(r"[*?]", m.group())[0]
        if len(prefix) < 3:
            notes.append(f"OpenAlex erlaubt Wildcards erst nach mindestens 3 Zeichen: {term!r}")
    return term


def _literal(text: str, notes: list[str], exact: bool = False) -> str:
    term = _clean_phrase(text, notes)
    if exact or " " in term:
        return '"' + term + '"'
    return term


def _proximity_pair_supported(
    left: str, right: str, notes: list[str],
) -> bool:
    """Check the official OpenAlex wildcard limits without changing words.

    In proximity, a 2-wildcard pair requires 4 leading characters
    before each wildcard. Invalid pairs use a documented broader Boolean
    AND as a candidate search, not a narrower broken literal.
    """
    a = _clean_phrase(left, notes)
    b = _clean_phrase(right, notes)
    wildcards = []
    for term in (a, b):
        for match in re.finditer(r"[*?]", term):
            leading = re.search(r"([A-Za-z0-9]+)$", term[:match.start()])
            prefix_length = len(leading.group(1)) if leading else 0
            wildcards.append(prefix_length)
    if len(wildcards) > 2:
        return False
    if any(n < 3 for n in wildcards):
        return False
    if len(wildcards) == 2 and any(n < 4 for n in wildcards):
        return False
    return True


def _or_alternatives(node: Node) -> list[str] | None:
    """Only flatten ORs of lexical operands, never AND/NOT or headings."""
    if node.kind == "text":
        return [node.value]
    if node.kind == "bool" and node.value == "OR":
        a = _or_alternatives(node.left)
        b = _or_alternatives(node.right)
        if a is not None and b is not None:
            return a + b
    return None


def _render(
    node: Node, notes: list[str], lost: list[str], unverified: list[str],
    explosions: dict[str, list[str]] | None,
    depth: int = 0,
) -> str:
    if node.kind == "text":
        return _literal(node.value, notes)
    if node.kind == "heading":
        name, exploded, qualifier = heading_details(node.value)
        unverified.append(node.value)
        notes.append(
            f"{node.value}: MeSH-Descriptor nur als Titel-/Abstract-/Keyword-Freitext "
            "übertragen; keine indexierte MeSH-Suche."
        )
        if qualifier:
            lost.append(f"Qualifier {qualifier!r} an {name!r} nicht übertragbar.")
        labels = [name]
        if exploded and explosions:
            more = explosions.get(name.casefold(), [])
            if more:
                labels.extend(x for x in more if x.casefold() != name.casefold())
                notes.append(
                    f"exp {name}/: {len(labels)-1} untergeordnete MeSH-Labels "
                    "zusätzlich als FREITEXT, nicht als echte MeSH-Explosion."
                )
        labels = list(dict.fromkeys(labels))
        if len(labels) == 1:
            return _literal(labels[0], notes)
        return _join([_literal(x, notes) for x in labels], "OR", depth)
    if node.kind == "field":
        # The outer OQL search scope is determined conservatively once for
        # the complete query. Field codes cannot be copied literally.
        return _render(node.left, notes, lost, unverified, explosions, depth)
    if node.kind == "near":
        n = int(node.value[3:])
        if n < 1:
            raise OQLConversionError("Ovid ADJ must be >= 1.")
        a, b = _or_alternatives(node.left), _or_alternatives(node.right)
        if a is None or b is None:
            raise OQLConversionError(
                f"{node.value}: komplexe verschachtelte Proximity-Operanden. "
                "Bitte die Adjazenzklausel zuerst in reine OR-Wortgruppen umformen; "
                "SearchBridge entfernt die Nähebedingung hier nicht stillschweigend."
            )
        if len(a) * len(b) > MAX_PROX_PAIRS:
            raise OQLConversionError(
                f"{node.value}: {len(a)*len(b)} Proximity-Paare überschreiten "
                f"das Sicherheitslimit von {MAX_PROX_PAIRS}. Suche aufteilen."
            )
        # OQL insists all proximity operands share quoted/unquoted mode. Use
        # quoted (= exact/no stemming) consistently, retaining internal * and ?.
        pairs = []
        for left in a:
            for right in b:
                x = _literal(left, notes, exact=True)
                y = _literal(right, notes, exact=True)
                if _proximity_pair_supported(left, right, notes):
                    pairs.append(f"({x} and {y} within {n-1} words of each other)")
                else:
                    pairs.append(f"({x} AND {y})")
                    lost.append(
                        f"{node.value} für Paar {left!r} / {right!r}: "
                        "OpenAlex-Wildcard-Regeln verhindern native Proximity; "
                        "nur für dieses Paar breites AND (OHNE Wortnähe). "
                        "Die ursprüngliche Wildcard bleibt erhalten und "
                        "die Recherche wird für diesen Paarzweig weiter."
                    )
        notes.append(
            f"{node.value} → native OpenAlex-Nähe mit höchstens {n-1} "
            "dazwischenliegenden Wörtern; OR-Wortgruppen vollständig "
            f"auf {len(pairs)} Paarabfragen verteilt."
        )
        notes.append(
            "OpenAlex verlangt in Proximity-Ausdrücken einheitliche Quotierung "
            "aller Operanden. Hier sind die Begriffe exakt (kein zusätzliches Stemming), "
            "Wildcards bleiben erhalten."
        )
        return _join(pairs, "OR", depth)
    if node.kind == "bool":
        if node.value in ("OR", "AND"):
            parts = _flatten(node, node.value)
            return _join(
                [_render(p, notes, lost, unverified, explosions, depth + 1)
                 for p in parts],
                node.value, depth,
            )
        if node.value == "NOT":
            left = _render(node.left, notes, lost, unverified, explosions, depth+1)
            right = _render(node.right, notes, lost, unverified, explosions, depth+1)
            return f"({left} NOT {right})"
    raise OQLConversionError("Nicht unterstützter Ovid-AST-Knoten: " + node.kind)


def _join(parts: list[str], operator: str, depth: int) -> str:
    parts = list(dict.fromkeys(parts))
    if len(parts) == 1:
        return parts[0]
    # The only extra parentheses are the meaningful OR / AND group boundaries;
    # deep left-associated parser trees are flattened.
    if len(parts) > 8:
        indent = "  " * min(depth + 1, 5)
        sep = "\n" + indent + operator + " "
        return "(\n" + indent + sep.join(parts) + "\n" + "  " * min(depth, 5) + ")"
    return "(" + f" {operator} ".join(parts) + ")"


def _scope(tree: Node, notes: list[str], lost: list[str]) -> str:
    fields = [
        {x.strip().lower() for x in n.value.strip(".").split(",")}
        for n in _walk(tree) if n.kind == "field"
    ]
    mapped = set()
    for f in fields:
        if f <= {"ti"}:
            mapped.add("title")
        elif f <= {"ab"}:
            mapped.add("abstract")
        elif f <= {"ti", "ab"}:
            mapped.add("title-abstract")
        else:
            mapped.add("title-abstract-keywords")
            lost.append(f"Unvollständig abbildbare Ovid-Feldcodes {sorted(f)}.")
    if len(mapped) != 1:
        notes.append(
            "Unterschiedliche Suchfeldgrenzen werden auf einen gemeinsamen "
            "OpenAlex-Textbereich verbreitert."
        )
        return "title-abstract-keywords"
    return next(iter(mapped)) if mapped else "title-abstract-keywords"


def translate_openalex_oql(
    source: str, *, explosions: dict[str, list[str]] | None = None,
) -> dict:
    root = parse_ovid(source)
    notes: list[str] = []
    lost: list[str] = []
    unverified: list[str] = []
    scope = _scope(root, notes, lost)
    expr = _render(root, notes, lost, unverified, explosions)
    # Use where rather than 'get works about' to make the precise OpenAlex
    # search field explicit, including search.title_abstract_keywords.
    q = f"get works where {scope} has {expr}"
    uri = ROOT + "?" + urlencode({"oql": q, "per-page": 25})
    can_get = len(uri.encode("utf-8")) <= MAX_OQL_GET
    if not can_get:
        uri = ""
        notes.append(
            "OQL-Abfrage überschreitet die GET-Grenze. Die vollständige "
            "Suche über POST https://api.openalex.org/ mit JSON-Body ausführen. "
            "Keine Suchterme wurden abgeschnitten."
        )
    if scope == "title-abstract-keywords":
        notes.append(
            "OpenAlex Keywords und Ovid kf sind nicht identisch; OpenAlex verwendet "
            "algorithmisch erschlossene Keywords. MeSH-/Indexing-Semantik geht verloren."
        )
    notes.append(
        "Native OpenAlex OQL – syntaktische Validierung über OpenAlex "
        "/query erforderlich; nicht als validierte Ovid-MEDLINE-Äquivalenz behandeln."
    )
    body = {"oql": q}
    return {
        "query": q,
        "endpoint": ("GET /?oql= (OQL)" if can_get else "POST / (OQL JSON)"),
        "url": uri,
        "request_json": json.dumps(body, ensure_ascii=False, indent=2),
        "validation_url": QUERY_VALIDATE_URL,
        "mode": "Native OpenAlex OQL with proximity (not identical Ovid indexing)",
        "notes": list(dict.fromkeys(notes)),
        "removed": list(dict.fromkeys(lost)),
        "unverified": list(dict.fromkeys(unverified)),
        "validated_equivalence": False,
        "uses_post": not can_get,
        "proximity_preserved": not any(
            "nur für dieses Paar breites AND" in entry for entry in lost
        ),
    }
