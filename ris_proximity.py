"""Deterministic Ovid ADJn checks in RIS title, abstract and individual keywords.

No database access, no AI. This is an approximate reconstruction of Ovid tokenization;
correspondence must be assessed against the originating database.
"""
from __future__ import annotations

import csv
import html
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from translator_core import Node, ParseError, parse_ovid

RIS_TAG_RE = re.compile(r"^([A-Z][A-Z0-9])\s{1,2}-\s?(.*)$")
WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass
class RISRecord:
    number: int
    fields: dict[str, list[str]] = field(default_factory=dict)
    raw_lines: list[str] = field(default_factory=list)

    def get(self, *tags: str) -> str:
        for tag in tags:
            values = self.fields.get(tag, [])
            if values:
                return values[0]
        return ""

    @property
    def title(self) -> str:
        return self.get("TI", "T1", "CT", "BT")

    @property
    def abstract(self) -> str:
        texts = [x for tag in ("AB", "N2") for x in self.fields.get(tag, [])]
        return max(texts, key=len, default="")

    @property
    def keywords(self) -> list[str]:
        parts = []
        for value in self.fields.get("KW", []):
            parts.extend(x.strip() for x in value.split(";") if x.strip())
        return parts

    @property
    def doi(self) -> str:
        return self.get("DO", "M3")

    @property
    def identifier(self) -> str:
        return self.get("ID", "AN", "PM") or str(self.number)

    def as_ris(self) -> str:
        lines = list(self.raw_lines)
        if not any(RIS_TAG_RE.match(x) and x.startswith("ER") for x in lines):
            lines.append("ER  -")
        return "\n".join(lines) + "\n\n"


def parse_ris(data: bytes) -> tuple[list[RISRecord], str]:
    """Read RIS exports with UTF-8 fallback; preserve records for exact re-export."""
    try:
        text = data.decode("utf-8-sig")
        encoding = "UTF-8"
    except UnicodeDecodeError:
        text = data.decode("cp1252")
        encoding = "Windows-1252 (fallback)"
    records: list[RISRecord] = []
    current: Optional[RISRecord] = None
    last_tag = ""

    def finish() -> None:
        nonlocal current
        if current and (current.title or current.fields.get("TY")):
            current.number = len(records) + 1
            records.append(current)
        current = None

    for raw in text.splitlines():
        line = raw.rstrip("\r")
        m = RIS_TAG_RE.match(line)
        if m:
            tag, value = m.group(1), m.group(2)
            if tag == "TY":
                finish()
                current = RISRecord(number=0)
            if current is None:
                current = RISRecord(number=0)
            current.raw_lines.append(line)
            if tag == "ER":
                finish()
                last_tag = ""
                continue
            current.fields.setdefault(tag, []).append(value)
            last_tag = tag
        elif current is not None:
            current.raw_lines.append(line)
            if line.strip() and last_tag and current.fields.get(last_tag):
                current.fields[last_tag][-1] += " " + line.strip()
    finish()
    return records, encoding


@dataclass(frozen=True)
class ProximityClause:
    n: int
    left: tuple[str, ...]
    right: tuple[str, ...]
    fields: tuple[str, ...]

    @property
    def label(self) -> str:
        a = " OR ".join(self.left)
        b = " OR ".join(self.right)
        return f"({a}) adj{self.n} ({b}) [" + ",".join(self.fields) + "]"


def _alternatives(node: Node) -> list[str]:
    if node.kind == "field":
        return _alternatives(node.left)
    if node.kind == "text":
        return [node.value.strip().strip('"\'')]
    if node.kind == "bool" and node.value == "OR":
        return _alternatives(node.left) + _alternatives(node.right)
    raise ValueError("A proximity group must consist of terms or OR-connected terms (no AND, NOT, or heading).")


def extract_clauses(ovid_text: str) -> list[ProximityClause]:
    """Find ADJn subexpressions of a combined Ovid query; no guessing on invalid syntax."""
    root = parse_ovid(ovid_text)
    result: list[ProximityClause] = []

    def walk(node: Node, default_fields: tuple[str, ...] = ("ti", "ab", "kf")) -> None:
        if node.kind == "field":
            fields = tuple(x.strip().lower() for x in node.value.strip(".").split(","))
            walk(node.left, fields)
        elif node.kind == "near":
            n = int(node.value[3:])
            if n < 1:
                raise ValueError("adjN erfordert N ≥ 1")
            a, b = _alternatives(node.left), _alternatives(node.right)
            if not a or not b:
                raise ValueError("Leere Proximity-Gruppe.")
            result.append(ProximityClause(n=n, left=tuple(a), right=tuple(b), fields=default_fields))
        elif node.kind == "bool":
            walk(node.left, default_fields)
            walk(node.right, default_fields)
    walk(root)
    if not result:
        raise ValueError("No adjN expression found. Enter an Ovid proximity clause such as '(young OR juvenil*) adj2 (offend* OR delinqu*)'.")
    return result


@dataclass(frozen=True)
class Tok:
    text: str
    start: int
    end: int


def tokenize(text: str) -> list[Tok]:
    return [Tok(m.group().casefold(), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def _word_regex(part: str) -> re.Pattern:
    if re.search(r"[^\w*?]", part, flags=re.UNICODE):
        raise ValueError(f"Invalid term fragment: {part}")
    rx = "".join(".*" if c == "*" else "." if c == "?" else re.escape(c) for c in part.casefold())
    return re.compile(r"^" + rx + r"$", re.I)


def _compile_alternative(term: str) -> tuple[re.Pattern, ...]:
    parts = re.findall(r"[^\W_]+(?:[?*][^\W_]*)*", term, flags=re.UNICODE)
    if not parts:
        raise ValueError(f"Invalid search term: {term}")
    return tuple(_word_regex(x) for x in parts)


def _spans(tokens: list[Tok], pat: tuple[re.Pattern, ...]) -> list[tuple[int, int]]:
    k = len(pat)
    return [(i, i + k - 1) for i in range(len(tokens) - k + 1)
            if all(p.fullmatch(tokens[i+j].text) for j, p in enumerate(pat))]


def _gap(span_a: tuple[int, int], span_b: tuple[int, int]) -> Optional[int]:
    if span_a[1] < span_b[0]:
        return span_b[0] - span_a[1] - 1
    if span_b[1] < span_a[0]:
        return span_a[0] - span_b[1] - 1
    return None


def _highlight(text: str, left: tuple[int, int], right: tuple[int, int], tokens: list[Tok]) -> str:
    intervals = [(tokens[left[0]].start, tokens[left[1]].end),
                 (tokens[right[0]].start, tokens[right[1]].end)]
    a = max(0, min(x[0] for x in intervals) - 100)
    b = min(len(text), max(x[1] for x in intervals) + 120)
    out: list[str] = []
    pos = a
    for lo, hi in sorted(intervals):
        out.append(html.escape(text[pos:lo]))
        out.append("<mark>" + html.escape(text[lo:hi]) + "</mark>")
        pos = hi
    out.append(html.escape(text[pos:b]))
    return ("…" if a > 0 else "") + "".join(out) + ("…" if b < len(text) else "")


def best_in_text(text: str, clause: ProximityClause) -> Optional[dict]:
    if not text.strip():
        return None
    toks = tokenize(text)
    if len(toks) < 2:
        return None
    left_pats = [(x, _compile_alternative(x)) for x in clause.left]
    right_pats = [(x, _compile_alternative(x)) for x in clause.right]
    left = [(term, sp) for term, pat in left_pats for sp in _spans(toks, pat)]
    right = [(term, sp) for term, pat in right_pats for sp in _spans(toks, pat)]
    candidates = []
    for ta, spa in left:
        for tb, spb in right:
            gap = _gap(spa, spb)
            if gap is not None and gap <= clause.n - 1:
                candidates.append((gap, spa[0], ta, tb, spa, spb))
    if not candidates:
        return None
    gap, _, ta, tb, spa, spb = min(candidates)
    return {"left": ta, "right": tb, "gap": gap,
            "evidence_html": _highlight(text, spa, spb, toks),
            "evidence": text[max(0,min(toks[spa[0]].start,toks[spb[0]].start)-100):
                             min(len(text),max(toks[spa[1]].end,toks[spb[1]].end)+120)].replace("\n", " ")}


def check_ris_records(records: list[RISRecord], clause: ProximityClause) -> list[dict]:
    """Return one result per record; each field/keyword is evaluated independently."""
    result = []
    include_ti = any(f in clause.fields for f in ("ti", "tw", "mp"))
    include_ab = any(f in clause.fields for f in ("ab", "tw", "mp"))
    include_kw = any(f in clause.fields for f in ("kf", "kw", "id", "tw", "mp"))
    if not (include_ti or include_ab or include_kw):
        raise ValueError("None of the field codes refer to title, abstract, or keywords. Supported: ti, ab, kf, kw, id, tw, mp.")
    for record in records:
        fields: list[tuple[str, str]] = []
        if include_ti and record.title:
            fields.append(("Title", record.title))
        if include_ab and record.abstract:
            fields.append(("Abstract", record.abstract))
        if include_kw:
            for i, kw in enumerate(record.keywords, 1):
                fields.append((f"Keyword {i}", kw))
        hits = []
        for field_name, text in fields:
            match = best_in_text(text, clause)
            if match:
                hits.append({"field": field_name, **match})
        hits.sort(key=lambda x: (x["gap"], 0 if x["field"] == "Title" else 1 if x["field"] == "Abstract" else 2))
        best = hits[0] if hits else None
        result.append({
            "record_no": record.number, "id": record.identifier, "doi": record.doi,
            "title": record.title, "matched": bool(hits), "field": best["field"] if best else "",
            "term_a": best["left"] if best else "", "term_b": best["right"] if best else "",
            "gap": best["gap"] if best else "", "evidence": best["evidence"] if best else "",
            "evidence_html": best["evidence_html"] if best else "", "matched_fields": ", ".join(h["field"] for h in hits),
            "has_abstract": bool(record.abstract), "has_keywords": bool(record.keywords),
        })
    return result


CSV_COLUMNS = ["record_no", "id", "doi", "title", "matched", "field", "term_a", "term_b", "gap", "evidence", "matched_fields", "has_abstract", "has_keywords"]


def csv_bytes(rows: list[dict]) -> bytes:
    f = io.StringIO()
    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return f.getvalue().encode("utf-8-sig")


def ris_bytes(records: list[RISRecord]) -> bytes:
    return "".join(x.as_ris() for x in records).encode("utf-8")


def html_report(records: list[RISRecord], rows: list[dict], clause: ProximityClause,
                source_file: str = "Uploaded RIS", source_kind: str = "RIS") -> bytes:
    matches = sum(x["matched"] for x in rows)
    field_counts = Counter(x["field"].split()[0] for x in rows if x["matched"])
    no_ab = sum(not x.abstract for x in records)
    no_kw = sum(not x.keywords for x in records)
    pct = (100 * matches / len(rows)) if rows else 0
    lines = []
    for r in rows:
        status = "Match" if r["matched"] else "No match"
        evidence = f"<div class='evidence'>{r['evidence_html']}</div>" if r["matched"] else ""
        details = (f"<div class='details'><b>{html.escape(r['field'])}</b> · "
                   f"{html.escape(str(r['term_a']))} ↔ {html.escape(str(r['term_b']))} · "
                   f"<b>{r['gap']} intervening words</b></div>") if r["matched"] else ""
        badge = "yes" if r["matched"] else "no"
        flags = ("Abstract missing" if not r["has_abstract"] else "") + (" · Keywords missing" if not r["has_keywords"] else "")
        lines.append("<tr data-status='" + badge + "'><td>" + str(r["record_no"]) + "</td>"
                     + "<td><span class='" + badge + "'>" + status + "</span></td><td>"
                     + "<strong>" + html.escape(r["title"] or "(no title)") + "</strong>"
                     + "<div class='meta'>" + html.escape(str(r["doi"] or r["id"])) + "</div>"
                     + "<div class='flags'>" + html.escape(flags.strip(" ·")) + "</div>"
                     + details + evidence + "</td></tr>")
    content = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SearchBridge – {html.escape(source_kind)} proximity report</title><style>
:root{{--bg:#f4f7fb;--ink:#142d40;--muted:#556c7c;--line:#dce4ed;--accent:#0e677c}}*{{box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;background:var(--bg);color:var(--ink)}}
main{{max-width:1080px;margin:0 auto;padding:30px 24px 80px}}header{{background:linear-gradient(115deg,#142b43,#116b75);color:white;padding:32px;border-radius:16px}}
header h1{{margin:0 0 7px}}header p{{margin:0;color:#d1edf0}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:17px 0}}
.stat{{padding:20px;background:#fff;border:1px solid var(--line);border-radius:12px}}.value{{font-size:27px;font-weight:800}}.small{{font-size:12px;color:var(--muted)}}
.box{{background:white;border:1px solid var(--line);padding:16px;border-radius:12px;margin:14px 0}}
code{{word-break:break-word;white-space:normal}}.controls{{margin:20px 0;display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
input,select{{padding:10px 12px;border:1px solid #c9d4e0;border-radius:8px;font-size:14px}}
table{{width:100%;border-collapse:collapse;background:white;border-radius:12px;overflow:hidden}}th,td{{border-bottom:1px solid var(--line);padding:13px;text-align:left;vertical-align:top}}
th{{background:#edf3f8;color:#41576a;font-size:13px}}td:first-child{{width:58px}}td:nth-child(2){{width:100px}}
.yes,.no{{display:inline-block;border-radius:30px;font-size:12px;font-weight:650;padding:4px 10px}}.yes{{background:#d9f2e6;color:#176348}}.no{{background:#fce3e0;color:#a04038}}
.evidence{{margin-top:8px;padding:9px 12px;border-left:3px solid #3d8696;background:#edf5f8;line-height:1.55;white-space:normal}}
mark{{background:#ffe6a2;padding:1px 2px;border-radius:2px}}.meta,.flags,.details{{margin-top:5px;color:var(--muted);font-size:12px}}
.note{{font-size:13px;line-height:1.6;color:#4e6574}}@media(max-width:700px){{.metrics{{grid-template-columns:repeat(2,1fr)}}main{{padding:15px}}header{{padding:20px}}}}
@media print{{.controls{{display:none}}}}
</style></head><body><main><header><h1>SearchBridge</h1><p>{html.escape(source_kind)} proximity verification · Deterministic lexical matching · No AI</p></header>
<div class='metrics'><div class='stat'><div class='value'>{len(rows):,}</div><div class='small'>Records checked</div></div>
<div class='stat'><div class='value'>{matches:,}</div><div class='small'>ADJ{clause.n} matches ({pct:.1f}%)</div></div>
<div class='stat'><div class='value'>{len(rows)-matches:,}</div><div class='small'>No proximity match</div></div>
<div class='stat'><div class='value'>{no_ab:,}</div><div class='small'>Missing abstracts</div></div></div>
<div class='box'><div class='small'>Source: {html.escape(source_file)} · Fields: {html.escape(', '.join(clause.fields))} · Best-match field counts: {html.escape(str(dict(field_counts)))}</div>
<p><b>Ovid expression</b></p><code>{html.escape(clause.label)}</code></div>
<p class='note'>A match requires a word/phrase from each group in the <strong>same title, abstract or individual keyword</strong>.
ADJ{clause.n} allows at most {clause.n-1} intervening words in either order. Missing metadata may cause false negatives.
The candidate dataset is assumed to come from a broad AND search; this report cannot confirm its completeness. Exact Ovid tokenization is not guaranteed.</p>
<div class='controls'><label>Filter <select id='filter'><option value='all'>All records</option><option value='yes'>Matches only</option><option value='no'>Non-matches only</option></select></label>
<label>Find title <input type='search' id='search' placeholder='Search title'></label><span id='visible' class='small'></span></div>
<table><thead><tr><th>#</th><th>ADJ{clause.n}</th><th>Publication &amp; evidence</th></tr></thead><tbody>{''.join(lines)}</tbody></table>
<script>function update(){{const f=document.getElementById('filter').value,q=document.getElementById('search').value.toLowerCase();let n=0;document.querySelectorAll('tbody tr').forEach(tr=>{{let show=(f==='all'||tr.dataset.status===f)&&tr.innerText.toLowerCase().includes(q);tr.style.display=show?'':'none';if(show)n++;}});document.getElementById('visible').textContent=n+' shown';}}document.getElementById('filter').addEventListener('change',update);document.getElementById('search').addEventListener('input',update);update();</script>
</main></body></html>"""
    return content.encode("utf-8")
