"""Conservative automatic PubMed draft from observed seed terms and assigned MeSH."""
from __future__ import annotations
import os
import re
from collections import defaultdict
from simplemma import lemmatize

GENERIC = set("study studies research result results analysis data method methods participant participants effect effects association associations relationship relationships significant group groups sample samples model models level levels factor factors variable variables population populations outcome outcomes findings evidence article articles review reviews".split())
CHECKTAGS = {"humans", "animals", "male", "female", "adult", "middle aged", "aged", "aged, 80 and over", "young adult", "adolescent", "child", "child, preschool", "infant", "pregnancy"}
STOP = GENERIC | set("a an the in of on for from to with and or by among between how what does do is are can which whether".split())

def lemma(word):
    return lemmatize(word.lower(), lang="en") if word.isalpha() else word.lower()

def words(term):
    return re.findall(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", term.lower())

def signature(term):
    return tuple(lemma(w) for w in words(term))

def content_words(term):
    return {lemma(w) for w in words(term) if w not in STOP and len(w)>1}

def suggest_concepts(question):
    """Heuristic question decomposition; never silently impose PICO constraints."""
    q = question.strip().rstrip("?.")
    q = re.sub(r"^(?:what (?:is|are) (?:the )?|how (?:do|does|can) |does |do |is there (?:an? )?)", "", q, flags=re.I)
    q = re.sub(r"^(?:association|relationship|link|correlation) between\s+|^(?:effect|effects|impact|influence) of\s+", "", q, flags=re.I)
    # Relation markers and explicit population qualifiers, not arbitrary OR lists.
    parts = re.split(r"\s+(?:associated with|related to|affect(?:s)?|influence(?:s)?|predict(?:s)?|impact(?:s)?|on|among|versus|vs\.?)\s+", q, flags=re.I)
    if len(parts)==1 and re.search(r"\bbetween\b", question, re.I):
        parts = re.split(r"\s+and\s+", q, maxsplit=1, flags=re.I)
    if re.search(r"\bbetween\b", question, re.I) and parts:
        head = re.split(r"\s+and\s+", parts[0], maxsplit=1, flags=re.I)
        if len(head)==2:
            parts = head + parts[1:]
    result = [p.strip(" ,;") for p in parts if content_words(p)]
    return result[:4] or [question]

def compress_variants(variants):
    """Collapse only observed inflection families into safe, long prefixes."""
    variants = sorted(set(variants))
    if len(variants)<2:
        return variants
    tokens = [words(t) for t in variants]
    if len({len(t) for t in tokens})!=1:
        return variants
    pattern = []
    for column in zip(*tokens):
        unique = set(column)
        if len(unique)==1:
            pattern.append(column[0])
            continue
        prefix = os.path.commonprefix(list(unique))
        # Only regular, observed suffix variation. Irregular forms stay explicit.
        tails = {w[len(prefix):] for w in unique}
        if len(prefix)<4 or not tails <= {"", "s", "es", "e", "ed", "ing", "d", "y", "ies", "ied", "er", "ers"}:
            return variants
        if len(prefix)/max(map(len, unique))<0.65:
            return variants
        pattern.append(prefix+"*")
    return [" ".join(pattern)]

def group_text(rows):
    groups = defaultdict(list)
    for row in rows:
        term = row["Term"].lower()
        sig = signature(term)
        if sig and content_words(term):
            groups[sig].append(row)
    result=[]
    for sig, group in groups.items():
        variants = sorted({r["Term"].lower() for r in group})
        ranks = {int(x) for r in group for x in str(r.get("Consensus ranks","")).split(",") if x.strip().isdigit()}
        representative = max(group, key=lambda r: (float(r["Score"]), r["Term"]))
        result.append(dict(representative, Documents=len(ranks) or max(r["Documents"] for r in group),
                           Variants=variants, Patterns=compress_variants(variants),
                           **{"Consensus ranks":", ".join(map(str,sorted(ranks)))}))
    return sorted(result,key=lambda r:(-float(r["Score"]),-r["Documents"],r["Term"]))

def select_terms(rows, concept, n, mesh=False, ranked=None):
    """Semantic candidates must be above a conservative relevance floor.
    A lexical match is required when local embeddings are unavailable.
    """
    anchor=content_words(concept)
    source=ranked if ranked is not None else rows
    eligible=[]
    for row in source:
        term=row["Term"]
        overlap=anchor & content_words(term)
        if mesh and term.lower() in CHECKTAGS and not overlap:
            continue
        if not mesh and not content_words(term):
            continue
        cosine=row.get("Semantic similarity")
        semantic=isinstance(cosine,(int,float)) and cosine>=0.48
        if not overlap and not semantic:
            continue
        # Avoid generic partial phrases from unrelated concepts.
        if overlap and len(content_words(term))==1 and len(anchor)>1 and not semantic and not mesh:
            continue
        eligible.append(row)
    # Prefer full topic phrases when they score similarly; do not keep redundant
    # narrower phrases already captured by a selected broader text term.
    selected=[]
    for row in eligible:
        if not mesh:
            sig=signature(row["Term"])
            if any(signature(s["Term"])==sig for s in selected):
                continue
            if any(len(signature(s["Term"]))==1 and signature(s["Term"])[0] in sig
                   and len(s.get("Patterns",[]))==1 and s["Patterns"][0].endswith("*")
                   for s in selected):
                continue
        selected.append(row)
        if len(selected)>=n:
            break
    return selected

def format_block(text_rows, mesh_rows):
    parts=[]
    for row in text_rows:
        for term in row.get("Patterns", [row["Term"]]):
            parts.append(f'"{term}"[tiab]')
    parts += [f'"{row["Term"]}"[mh]' for row in mesh_rows]
    return "(" + " OR ".join(dict.fromkeys(parts)) + ")" if parts else ""

def build_query(blocks):
    # Missing concepts must be visible, never silently omitted from an AND draft.
    return "\nAND\n".join(blocks) if blocks and all(blocks) else ""
