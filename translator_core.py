"""Deterministic Ovid MEDLINE -> four target database syntax translator.

No automatic MeSH to PsycINFO/Emtree mapping is attempted. For evidence synthesis,
controlled vocabulary mappings must be supplied and verified by a search expert.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Token:
    typ: str
    value: str
    pos: int


@dataclass(frozen=True)
class Node:
    kind: str
    value: str = ""
    left: Optional['Node'] = None
    right: Optional['Node'] = None


class ParseError(ValueError):
    pass


_OP_RE = re.compile(r"(?:ADJ\d+|AND|OR|NOT)\b", re.I)
_FIELD_RE = re.compile(r"\.(?:ti|ab|kf|kw|tw|id|mp|pt)(?:\s*,\s*(?:ti|ab|kf|kw|tw|id|mp|pt))*\.", re.I)
_HEADING_RE = re.compile(
    r"^(?:(exp)\s+)?(?:\"([^\"]+)\"|'([^']+)'|([^/]+?))\s*/\s*([A-Za-z]{1,5})?$",
    re.I,
)
_TARGETS = ("Ovid PsycINFO", "Elsevier Embase.com", "Web of Science", "Scopus")


def lex(source: str) -> List[Token]:
    """Tokenize operators but never split quoted phrases."""
    result: List[Token] = []
    buf: List[str] = []
    start = 0
    quote: Optional[str] = None
    i = 0

    def flush() -> None:
        nonlocal buf
        value = "".join(buf).strip()
        if value:
            result.append(Token("TEXT", value, start))
        buf = []

    while i < len(source):
        ch = source[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ('"', "'"):
            if not buf:
                start = i
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch in "()":
            flush()
            result.append(Token("LP" if ch == "(" else "RP", ch, i))
            i += 1
            start = i
            continue
        field = _FIELD_RE.match(source, i)
        if field:
            flush()
            result.append(Token("FIELD", field.group(0), i))
            i = field.end()
            start = i
            continue
        op = _OP_RE.match(source, i)
        if op and (i == 0 or not (source[i-1].isalnum() or source[i-1] == '_')):
            flush()
            result.append(Token("OP", op.group(0).upper(), i))
            i = op.end()
            start = i
            continue
        if not buf:
            start = i
        buf.append(ch)
        i += 1
    if quote:
        raise ParseError("Nicht geschlossene Anführungszeichen im Suchstring.")
    flush()
    return result


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.i = 0

    def current(self) -> Optional[Token]:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self) -> Token:
        cur = self.current()
        if cur is None:
            raise ParseError("Unerwartetes Ende der Suchstrategie.")
        self.i += 1
        return cur

    def parse(self) -> Node:
        if not self.tokens:
            raise ParseError("Bitte eine Ovid-Suchstrategie eingeben.")
        tree = self.expression(0)
        if self.current() is not None:
            cur = self.current()
            raise ParseError(f"Unerwartetes Element bei Zeichen {cur.pos + 1}: {cur.value}")
        return tree

    def expression(self, precedence: int) -> Node:
        left = self.primary()
        while True:
            tok = self.current()
            if not tok or tok.typ != "OP":
                break
            prec = 3 if tok.value.startswith("ADJ") else 2 if tok.value in ("AND", "NOT") else 1
            if prec < precedence:
                break
            op = self.take().value
            right = self.expression(prec + 1)
            left = Node("near" if op.startswith("ADJ") else "bool", op, left, right)
        return left

    def primary(self) -> Node:
        tok = self.take()
        if tok.typ == "LP":
            inner = self.expression(0)
            closing = self.take()
            if closing.typ != "RP":
                raise ParseError(f"Fehlende schliessende Klammer vor Zeichen {closing.pos + 1}.")
            node = inner
        elif tok.typ == "TEXT":
            node = Node("heading" if _HEADING_RE.fullmatch(tok.value) else "text", tok.value)
        else:
            raise ParseError(f"Unerwartetes '{tok.value}' bei Zeichen {tok.pos+1}.")
        if self.current() is not None and self.current().typ == "FIELD":
            field = self.take().value
            if node.kind == "heading":
                raise ParseError("Ein kontrollierter Descriptor kann nicht gleichzeitig einen Freitext-Feldcode erhalten.")
            node = Node("field", field, node)
        return node


def parse_ovid(source: str) -> Node:
    if re.search(r"(?m)^\s*\d+\s*\.", source):
        raise ParseError("Nummerierte Ovid-Suchhistories sind in dieser Version nicht unterstützt. Bitte eine kombinierte Abfrage ohne Zeilennummern eingeben.")
    return Parser(lex(source)).parse()


def heading_details(raw: str) -> Tuple[str, bool, str]:
    m = _HEADING_RE.fullmatch(raw)
    if not m:
        raise ParseError(f"Ungültiger Heading-Ausdruck: {raw}")
    name = next(v for v in m.groups()[1:4] if v is not None).strip()
    return name, bool(m.group(1)), (m.group(5) or "").lower()


def simplify(node: Optional[Node], target: str, mappings: Dict[str, str], notes: List[str], removed: List[str], unverified: List[str]) -> Optional[Node]:
    if node is None:
        return None
    if node.kind == "heading":
        name, exploded, subheading = heading_details(node.value)
        key = name.casefold()
        mapped = mappings.get(key, "").strip() if target in ("Ovid PsycINFO", "Elsevier Embase.com") else ""
        if mapped:
            notes.append(f"Mapping manuell vorgegeben: {name} → {mapped}; Hierarchie/Qualifier bitte prüfen.")
            return Node("mapped_heading", ("exp " if exploded else "") + mapped + "/" + (subheading if subheading else ""))
        # Do not silently delete subject headings from databases that support them.
        # Ovid PsycINFO uses the same heading syntax but *not* the same thesaurus;
        # retain the original term only as an explicitly unverified candidate.
        if target == "Ovid PsycINFO":
            unverified.append(node.value)
            return Node("mapped_heading", node.value)
        # In Embase.com, never invent an Emtree descriptor. Retain the concept
        # as explicitly labelled title/abstract/keyword free text instead.
        if target == "Elsevier Embase.com":
            unverified.append(node.value)
            return Node("fallback_heading", name)
        removed.append(node.value)
        return None
    if node.kind in ("text", "mapped_heading", "fallback_heading"):
        return node
    if node.kind == "field":
        child = simplify(node.left, target, mappings, notes, removed, unverified)
        return Node("field", node.value, child) if child is not None else None
    if node.kind in ("near", "bool"):
        left = simplify(node.left, target, mappings, notes, removed, unverified)
        right = simplify(node.right, target, mappings, notes, removed, unverified)
        if left is None and right is None:
            return None
        if left is None or right is None:
            if node.value == "NOT" and left is None:
                notes.append("Ein NOT-Zweig ohne linken Freitextteil wurde entfernt; prüfe die Suchlogik.")
                return None
            notes.append(f"Teil einer {node.value}-Verknüpfung enthielt nur kontrolliertes Vokabular und wurde entfernt. Retrieval kann sich ändern.")
            return left if left is not None else right
        return Node(node.kind, node.value, left, right)
    raise ParseError(f"Interner Knotentyp nicht unterstützt: {node.kind}")


def _term(text: str, target: str, notes: List[str]) -> str:
    txt = text.strip()
    if not txt:
        raise ParseError("Leerer Freitextbegriff.")
    if target == "Elsevier Embase.com":
        if txt.startswith('"') and txt.endswith('"'):
            txt = "'" + txt[1:-1].replace("'", "\\'") + "'"
        elif " " in txt and not (txt.startswith("'") and txt.endswith("'")):
            txt = "'" + txt + "'"
    else:
        if " " in txt and not (txt.startswith('"') and txt.endswith('"')):
            txt = '"' + txt.strip("'") + '"'
    if re.search(r"\$\d+|\#", txt):
        notes.append(f"Ovid-Trunkierung '{txt}' benötigt manuelle Prüfung auf Zieldatenbank-Kompatibilität.")
    return txt


def render(node: Node, target: str, notes: List[str]) -> str:
    if node.kind == "text":
        return _term(node.value, target, notes)
    if node.kind == "fallback_heading":
        # An *explicitly documented* free-text substitute, not an Emtree match.
        return _term(node.value, target, notes) + ":ti,ab,kw"
    if node.kind == "mapped_heading":
        name, exploded, subheading = heading_details(node.value)
        if target == "Ovid PsycINFO":
            label = f'"{name}"' if " " in name else name
            return ("exp " if exploded else "") + label + "/" + subheading
        if target == "Elsevier Embase.com":
            if subheading:
                notes.append(f"MeSH Qualifier '/{subheading}' wurde bei '{name}' nicht übertragen: kein verifiziertes Emtree-Äquivalent.")
            return f"'{name.replace(chr(39), '')}'/{'exp' if exploded else 'de'}"
        raise ParseError("Mapping nur für PsycINFO/Embase vorgesehen")
    if node.kind == "field":
        inner = render(node.left, target, notes)
        if target == "Ovid PsycINFO":
            fs = [x.strip().lower() for x in node.value.strip(".").split(",")]
            fs = ["id" if f in ("kf", "kw") else f for f in fs]
            fs = list(dict.fromkeys(fs))
            if not (inner.startswith("(") and inner.endswith(")")):
                inner = f"({inner})"
            return f"{inner}.{','.join(fs)}."
        if target == "Elsevier Embase.com":
            fs = [x.strip().lower() for x in node.value.strip(".").split(",")]
            fs = ["kw" if f in ("kf", "id") else f for f in fs]
            if "tw" in fs or "mp" in fs or "pt" in fs:
                notes.append(f"Feld {node.value} teilweise nicht exakt in Embase.com übertragen. Bitte Felddefinition prüfen.")
                fs = [x for x in fs if x not in ("tw", "mp", "pt")]
                if not fs:
                    fs = ["ti", "ab", "kw"]
            fs = list(dict.fromkeys(fs))
            if not (inner.startswith("(") and inner.endswith(")")):
                inner = f"({inner})"
            return f"{inner}:{','.join(fs)}"
        if node.value.lower() not in (".ti,ab,kf.", ".ti,ab.", ".ti,ab,kw.", ".ti,ab,id."):
            notes.append(f"Spezifische Feldbegrenzung {node.value} wurde in {target} zu einer allgemeinen Titel/Abstract/Keyword-Suche verbreitert.")
        return inner
    if node.kind == "near":
        n = int(node.value[3:])
        if n < 1:
            raise ParseError("Ovid ADJ muss mindestens 1 sein.")
        op = node.value
        if target == "Elsevier Embase.com":
            op = f"NEAR/{n}"
        elif target == "Web of Science":
            op = f"NEAR/{n-1}"
        elif target == "Scopus":
            if n == 1:
                notes.append("ADJ1 erfordert unmittelbare Nachbarschaft. W/0 in Scopus bitte vor dem Suchen prüfen.")
            op = f"W/{n-1}"
        return f"({render(node.left,target,notes)} {op} {render(node.right,target,notes)})"
    if node.kind == "bool":
        op = "AND NOT" if node.value == "NOT" and target == "Scopus" else node.value
        # Flatten associative OR/AND branches; preserve meaningful grouping.
        if node.value in ("OR", "AND"):
            terms = []
            def collect(cur):
                if cur.kind == "bool" and cur.value == node.value:
                    collect(cur.left)
                    collect(cur.right)
                else:
                    terms.append(render(cur, target, notes))
            collect(node)
            return "(" + f" {op} ".join(terms) + ")"
        return f"({render(node.left, target, notes)} {op} {render(node.right, target, notes)})"
    raise ParseError(f"Unbekannter Knotentyp: {node.kind}")


def translate(source: str, mappings: Optional[Dict[str, Dict[str, str]]] = None) -> dict:
    """Return per-database query and warnings. No network calls or LLMs."""
    tree = parse_ovid(source)
    mapping_sets = mappings or {}
    outputs: Dict[str, dict] = {"Ovid MEDLINE": {"query": source.strip(), "removed": [], "unverified": [], "notes": ["Originale Ovid-MEDLINE-Strategie."]}}
    for target in _TARGETS:
        notes: List[str] = []
        removed: List[str] = []
        unverified: List[str] = []
        manual = {k.casefold(): v for k, v in mapping_sets.get(target, {}).items()}
        reduced = simplify(tree, target, manual, notes, removed, unverified)
        if reduced is None:
            result = ""
            notes.append("Kein Freitext nach Entfernung der kontrollierten Descriptors übrig.")
        else:
            result = render(reduced, target, notes)
            if target == "Web of Science":
                result = f"TS=({result})"
            elif target == "Scopus":
                result = f"TITLE-ABS-KEY({result})"
        if removed:
            notes.insert(0, f"{len(removed)} kontrollierte Heading-Klausel(n) entfernt (nicht als Freitext ersetzt).")
            notes.append("Die Trefferzahl kann abweichen, weil indexierungsbasierte Treffer entfallen.")
        if unverified:
            if target == "Ovid PsycINFO":
                notes.insert(0, f"{len(unverified)} MeSH-Heading(s) als PROVISORISCHE Ovid-PsycINFO-Headings erhalten. APA-Thesaurus-Begriffe, Explosion und Qualifier müssen geprüft werden; nicht als validierte Suche verwenden.")
            else:
                notes.insert(0, f"{len(unverified)} MeSH-Heading(s) als Freitext-Fallback (:ti,ab,kw) erhalten. Dies ist KEINE Emtree-Übersetzung; indexierungsbasierte Treffer können fehlen. Emtree-Mapping erforderlich.")
        outputs[target] = {"query": result, "removed": list(dict.fromkeys(removed)), "unverified": list(dict.fromkeys(unverified)), "notes": list(dict.fromkeys(notes))}
    return outputs
