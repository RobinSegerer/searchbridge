"""Conservative Ovid MEDLINE -> PubMed search syntax conversion.

Preserves MeSH (including Ovid EXP/no-EXP), Boolean grouping and fields
where possible. PubMed [tiab] includes author keywords ([ot]). PubMed
proximity "a b"[tiab:~N] is unordered with N *intervening* words, but
does NOT support wildcard terms. In such cases do NOT invent a proximity
query: broaden to AND of the lexical operand groups and flag the loss.

No silent deletion of MeSH branches. No external API required to convert.
"""
from __future__ import annotations

import re
from urllib.parse import urlencode

from translator_core import Node, heading_details, parse_ovid


PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/"
MAX_BROWSER_URL = 1900
MAX_EXPLICIT_PAIRS = 800
_SPELLINGS = {"wom?n": ("woman", "women")}
# An Ovid prefix shorter than four letters cannot be used as a PubMed wildcard.
# This *limited* curated candidate set allows the user's Web* strategy to run,
# but explicitly cannot claim exhaustiveness of every web* word.
_SHORT_PREFIXES = {
    "web*": ("web", "website*", "webpage*", "webinar*", "webcam*", "webapp*"),
}
_ALLOWED_FIELDS = {"ti", "ab", "kf", "kw", "id", "tw", "mp", "pt"}


class PubMedConversionError(ValueError):
    pass


def _flatten(node: Node, op: str) -> list[Node]:
    if node.kind == "bool" and node.value == op:
        return _flatten(node.left, op) + _flatten(node.right, op)
    return [node]


def _or_words(node: Node) -> list[str] | None:
    if node.kind == "text":
        return [node.value]
    if node.kind == "bool" and node.value == "OR":
        a, b = _or_words(node.left), _or_words(node.right)
        return a + b if a is not None and b is not None else None
    return None


def _join(parts: list[str], op: str) -> str:
    parts = list(dict.fromkeys(parts))
    if not parts:
        raise PubMedConversionError("Leere Boolesche Suchgruppe.")
    return parts[0] if len(parts) == 1 else "(" + f" {op} ".join(parts) + ")"


def _fields(value: str, notes: list[str]) -> tuple[str, ...]:
    raw = [x.strip().lower() for x in value.strip(".").split(",")]
    flags = set(raw)
    if not flags or any(x not in _ALLOWED_FIELDS for x in flags):
        raise PubMedConversionError(f"Nicht unterstützter Ovid-Feldcode: {value}")
    if flags == {"ti", "ab", "kf"} or flags == {"ti", "ab", "kw"}:
        return ("tiab",)
    if flags == {"ti", "ab"}:
        return ("ti", "ab")
    mapping = {"ti": "ti", "ab": "ab", "kf": "ot", "kw": "ot",
               "id": "ot", "tw": "tw", "mp": "tw", "pt": "pt"}
    result = tuple(dict.fromkeys(mapping[x] for x in raw))
    if "mp" in flags or "tw" in flags:
        notes.append(
            f"Ovid {value} ist NICHT identisch mit PubMed [tw]; "
            "Umfang der durchsuchten Indexfelder prüfen."
        )
    if "id" in flags:
        notes.append(
            f"Ovid {value}: .id. ist nicht exakt PubMed [ot]; "
            "hier nur als Keyword-Fallback übertragen."
        )
    if flags & {"kf", "kw"} and "tiab" not in result:
        notes.append("Ovid Keyword-Feld als PubMed [ot] übertragen; "
                     "author-keyword-Verfügbarkeit variiert.")
    return result


def _term_variants(raw: str, notes: list[str]) -> list[str]:
    term = raw.strip().strip('"\'')
    if not term:
        raise PubMedConversionError("Leerer Freitextbegriff.")
    if '"' in term:
        raise PubMedConversionError("Anführungszeichen innerhalb eines Terms.")
    if re.search(r"\$\d+|[#^]", term):
        raise PubMedConversionError(
            f"Ovid-Spezialtrunkierung {raw!r} kann nicht automatisch sicher "
            "in PubMed übertragen werden."
        )
    term = re.sub(r"\$(?!\d)", "*", term)
    if term.casefold() in _SHORT_PREFIXES:
        values = list(_SHORT_PREFIXES[term.casefold()])
        notes.append(
            f"PubMed kann {term!r} nicht trunkieren (weniger als 4 Zeichen "
            "vor *). Als begrenzte Liste konkreter Web-Begriffe übertragen: "
            + ", ".join(values) + ". NICHT vollständig äquivalent; "
            "weitere Varianten müssen bei Bedarf ergänzt werden."
        )
        return values
    if term.casefold() in _SPELLINGS:
        values = list(_SPELLINGS[term.casefold()])
        notes.append(
            f"Ovid-Einzelzeichenwildcard {term} explizit als "
            f"{', '.join(values)} übertragen."
        )
        return values
    if "?" in term:
        def repl(m: re.Match) -> str:
            prefix = m.group(1)
            if len(prefix) < 4:
                raise PubMedConversionError(
                    f"PubMed unterstützt die Ovid-Einzelzeichenwildcard "
                    f"{term!r} nicht: mindestens 4 Zeichen vor * notwendig. "
                    "Bitte Schreibvarianten explizit ergänzen."
                )
            notes.append(
                f"Ovid ? in {term!r} zu * verbreitert "
                "(nicht exakt äquivalent)."
            )
            return prefix + "*"
        term = re.sub(r"([A-Za-z]+)\?[A-Za-z]*", repl, term)
        if "?" in term:
            raise PubMedConversionError(
                f"Nicht übertragbares Ovid-Wildcardmuster {raw!r}."
            )
    if "*" in term:
        # PubMed accepts internal * but requires >= 4 characters before first *.
        prefix = term.split("*")[0].split()[-1]
        if len(prefix) < 4:
            raise PubMedConversionError(
                f"PubMed braucht mindestens vier Zeichen vor dem ersten "
                f"*: {term!r}. Suchterm manuell präzisieren."
            )
    return [term]


def _lexical(raw: str, tags: tuple[str, ...],
             notes: list[str]) -> str:
    forms = []
    for word in _term_variants(raw, notes):
        quoted = f'"{word}"' if " " in word else word
        forms.append(_join([f"{quoted}[{tag}]" for tag in tags], "OR"))
    return _join(forms, "OR")


def _heading(raw: str, notes: list[str]) -> str:
    name, exp, qualifier = heading_details(raw)
    # The full preferred descriptor should be checked against current MeSH.
    if qualifier:
        notes.append(
            f"{raw}: MeSH-Subheading /{qualifier} muss gegen die PubMed "
            "MeSH-Qualifier-Namen geprüft werden; in der generierten Suche "
            "nicht stillschweigend entfernt."
        )
        raise PubMedConversionError(
            f"MeSH-Qualifier /{qualifier} aus {raw!r} benötigt ein "
            "verifiziertes Ovid-zu-PubMed-Subheading-Mapping."
        )
    tag = "mh" if exp else "mh:noexp"
    return f'"{name}"[{tag}]'


def _render_near(
    node: Node, tags: tuple[str, ...], notes: list[str], losses: list[str],
) -> str:
    n = int(node.value[3:])
    if n < 1:
        raise PubMedConversionError("Ovid ADJ muss >=1 sein.")
    left, right = _or_words(node.left), _or_words(node.right)
    if left is None or right is None:
        # Do not flatten complex nested proximity into an unsupported phrase.
        losses.append(
            f"{node.value}: verschachtelte AND/NOT/Heading-Nähegruppe "
            "kann PubMed nicht exakt abbilden; breitere AND-Kandidatenabfrage."
        )
        return _join([
            _render(node.left, tags, notes, losses),
            _render(node.right, tags, notes, losses),
        ], "AND")

    lexis_l = [v for term in left for v in _term_variants(term, notes)]
    lexis_r = [v for term in right for v in _term_variants(term, notes)]
    lexical_pairs = len(lexis_l) * len(lexis_r)
    # If any original pair requires truncation, PubMed ignores its proximity
    # operator. Preserve the *entire* lexical concept as a broad AND instead,
    # and report how much specificity was lost.
    if (any("*" in x for x in lexis_l + lexis_r)
            or any(" " in x or "-" in x for x in lexis_l + lexis_r)
            or "ot" in tags or "tw" in tags or "pt" in tags):
        losses.append(
            f"{node.value} ({lexical_pairs} Wortpaare): PubMed-Nähe unterstützt "
            "hier nicht alle Ovid-Operanden (Trunkierung, Mehrwort-Phrase "
            "oder nicht unterstütztes Feld). Durch breites AND ersetzt; "
            "Proximity nach dem Trefferabruf lokal prüfen."
        )
        return _join([
            _join([_lexical(t, tags, notes) for t in left], "OR"),
            _join([_lexical(t, tags, notes) for t in right], "OR"),
        ], "AND")

    if lexical_pairs > MAX_EXPLICIT_PAIRS:
        raise PubMedConversionError(
            f"{node.value}: {lexical_pairs} Nähepaare sind zu viele "
            "für eine einzelne übersetzte PubMed-Abfrage."
        )
    # Ovid ADJ2: up to 1 intervening word, PubMed :~1.
    pairs = [
        f'"{a} {b}"[{tag}:~{n-1}]'
        for a in lexis_l for b in lexis_r
        for tag in tags if tag in ("tiab", "ti", "ab")
    ]
    if not pairs:
        raise PubMedConversionError(
            f"Für {node.value} kein unterstütztes PubMed-Proximity-Feld."
        )
    notes.append(
        f"{node.value} → PubMed :~{n-1} "
        f"({lexical_pairs} Wortpaare, Wortreihenfolge beliebig)."
    )
    if "tiab" in tags:
        notes.append(
            "PubMed [tiab] umfasst auch Autorenschlagwörter [ot]; "
            "Feldkombinationen und Proximity-Indexierung nicht in jeder "
            "Hinsicht identisch mit Ovid .ti,ab,kf."
        )
    return _join(pairs, "OR")


def _render(node: Node, tags: tuple[str, ...],
            notes: list[str], losses: list[str]) -> str:
    if node.kind == "text":
        return _lexical(node.value, tags, notes)
    if node.kind == "heading":
        return _heading(node.value, notes)
    if node.kind == "field":
        actual = _fields(node.value, notes)
        return _render(node.left, actual, notes, losses)
    if node.kind == "near":
        return _render_near(node, tags, notes, losses)
    if node.kind == "bool":
        if node.value in ("AND", "OR"):
            return _join([
                _render(k, tags, notes, losses)
                for k in _flatten(node, node.value)
            ], node.value)
        if node.value == "NOT":
            return "(" + _render(node.left, tags, notes, losses) + \
                   " NOT " + _render(node.right, tags, notes, losses) + ")"
    raise PubMedConversionError(f"Unbekannter Suchknoten: {node.kind}")


def translate_pubmed(source: str) -> dict:
    tree = parse_ovid(source)
    notes = [
        "PubMed [tiab] beinhaltet Titel, Abstract sowie Autorenschlagwörter "
        "([ot]); nicht vollständig identisch mit Ovid-Indexfeldern.",
        "MeSH-Headings werden als MeSH [mh] (mit Explosion) bzw. "
        "[mh:noexp] (ohne Explosion) übertragen. "
        "Descriptor-Vorzugsbezeichnungen/Indexierungsstand prüfen.",
    ]
    lost: list[str] = []
    query = _render(tree, ("tiab",), notes, lost)
    url = PUBMED_URL + "?" + urlencode({"term": query})
    if len(url.encode("utf-8")) > MAX_BROWSER_URL:
        url = ""
        notes.append(
            "Komplexer Suchstring zu lang für einen zuverlässigen "
            "PubMed-Browser-Link. Die vollständige Suche als Text kopieren "
            "oder über NCBI ESearch per POST prüfen; nichts abgeschnitten."
        )
    return {
        "query": query, "endpoint": "PubMed web / NCBI ESearch (POST for long queries)",
        "url": url,
        "mode": "MeSH + Boolean + conservative PubMed proximity translation",
        "notes": list(dict.fromkeys(notes)),
        "removed": list(dict.fromkeys(lost)),
        "unverified": [],
        "validated_equivalence": False,
        "proximity_preserved": not bool(lost),
        "requires_post": not bool(url),
    }
