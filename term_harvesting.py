"""Deterministic PubMed term harvesting.

MeSH query -> titles: sample-based TF-IDF (log-smoothed IDF, L2 document
normalization, mean TF-IDF over all retrieved titles). Report document frequency
and raw counts separately: TF-IDF is NOT a frequency metric.

TIAB query -> assigned MeSH: percentage among MeSH-indexed records, counts,
major-topic counts and percentage among all downloaded records.

No LLMs, no external NLP package, no automatic mapping/inferred synonyms.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import io
import math
import re
import unicodedata

from ris_proximity import RISRecord


ENGLISH_STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been
before being below between both but by can cannot could did do does doing down
during each few for from further had has have having he her here hers herself
him himself his how i if in into is it its itself just me more most my myself
no nor not now of off on once only or other our ours ourselves out over own
same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were
what when where which while who whom why will with would you your yours yourself
yourselves via per versus vs within without among amongs across using use used
also et al one two three new based associated study studies research article
articles results review reviews systematic method methods evidence findings
case cases effect effects impact analysis analyses investigation investigations
""".split())
# Standard indexing "check tags" are valid terms; users can choose to include.
COMMON_CHECK_TAGS = frozenset({
    "humans", "animals", "male", "female", "adult", "aged", "child",
    "adolescent", "infant", "infant, newborn", "young adult", "middle aged",
    "aged, 80 and over", "pregnancy", "female", "male", "aged, 80 and over",
})
TOKEN_RE = re.compile(r"[^\W\d_][^\W\d_'-]*", re.UNICODE)


def _ascii_normalized(word: str) -> str:
    norm = unicodedata.normalize("NFKD", word.casefold())
    return "".join(c for c in norm if not unicodedata.combining(c))


def word_tokens(text: str) -> list[str]:
    """Break punctuation/hyphenated title expressions into alphabetic words."""
    return [_ascii_normalized(m.group(0)) for m in TOKEN_RE.finditer(text)
            if len(m.group(0)) >= 2]


def vocabulary_exclusions(query: str) -> tuple[set[str], tuple[str, ...]]:
    """Exclude exact query words and prefix-truncated words, not whole expressions."""
    normalized = re.sub(r"\[[^\]]+\]", " ", query)
    normalized = re.sub(r"\b(?:AND|OR|NOT|mesh|mh|tiab|noexp)\b", " ", normalized, flags=re.I)
    words = set()
    wildcards = []
    for match in re.finditer(r"[A-Za-z][A-Za-z0-9]*\*?", normalized):
        word = _ascii_normalized(match.group(0))
        if len(word) < 3:
            continue
        if word.endswith("*") and len(word) >= 4:
            wildcards.append(word[:-1])
        else:
            words.add(word)
    return words, tuple(wildcards)


def _make_ngrams(words: list[str], ngram_max: int) -> list[str]:
    result = []
    for size in range(1, ngram_max + 1):
        for i in range(len(words) - size + 1):
            result.append(" ".join(words[i:i + size]))
    return result


@dataclass
class HarvestResult:
    rows: list[dict]
    included_docs: int
    missing_docs: int
    source_docs: int
    method: str
    note: str


def title_tfidf(
    records: list[RISRecord],
    *,
    min_df: int = 2,
    max_ngram: int = 1,
    query: str = "",
    remove_query_terms: bool = False,
    extra_stopwords: str = "",
    top_n: int = 100,
) -> HarvestResult:
    """Rank terms by mean L2-normalized TF-IDF across nonempty titles.

    IDF = log((N+1)/(df+1))+1; score = sum_d tfidf(d, term) / N.
    These scores describe the SELECTED retrieval sample, not all PubMed records.
    """
    if min_df < 1 or max_ngram not in (1, 2) or top_n < 1:
        raise ValueError("min_df >=1, max_ngram in {1,2}, top_n >=1 erforderlich.")
    titles = [r.title for r in records if r.title.strip()]
    n = len(titles)
    if n == 0:
        return HarvestResult([], 0, len(records), len(records), "TF-IDF", "No titles in sample.")
    if min_df > n:
        raise ValueError(f"min_df={min_df} ist grösser als die Zahl vorhandener Titel ({n}).")
    stops = ENGLISH_STOPWORDS | set(word_tokens(extra_stopwords))
    exclude, prefixes = vocabulary_exclusions(query) if remove_query_terms else (set(), ())
    corpora = []
    for title in titles:
        # Preserve original title adjacency: never form artificial bigrams
        # across removed stopwords or excluded query words.
        words = word_tokens(title)
        phrases = []
        for size in range(1, max_ngram + 1):
            for i in range(len(words) - size + 1):
                part = words[i:i + size]
                if all(
                    len(w) >= 3 and w not in stops and w not in exclude
                    and not any(w.startswith(prefix) for prefix in prefixes)
                    for w in part
                ):
                    phrases.append(" ".join(part))
        corpora.append(Counter(phrases))
    df = Counter()
    count = Counter()
    for terms in corpora:
        df.update(terms.keys())
        count.update(terms)
    selected = {term for term, documents in df.items() if documents >= min_df}
    idf = {term: math.log((n + 1) / (df[term] + 1)) + 1 for term in selected}
    scores = defaultdict(float)
    for terms in corpora:
        raw_weights = {term: freq * idf[term] for term, freq in terms.items() if term in selected}
        norm = math.sqrt(sum(weight ** 2 for weight in raw_weights.values()))
        if norm == 0:
            continue
        for term, weight in raw_weights.items():
            scores[term] += weight / norm / n
    ranks = sorted(scores, key=lambda x: (-scores[x], -df[x], x))[:top_n]
    rows = [{
        "Rank": i + 1,
        "Title term": term,
        "Mean TF-IDF": round(scores[term], 6),
        "Document frequency": df[term],
        "% titles": round(100 * df[term] / n, 2),
        "Occurrences": count[term],
    } for i, term in enumerate(ranks)]
    return HarvestResult(
        rows, n, len(records) - n, len(records), "title_tfidf",
        "Smoothed IDF log((N+1)/(df+1))+1, L2 document normalization, "
        "mean TF-IDF across all sampled titles. Document frequency is distinct "
        "from TF-IDF. Results depend on the retrieved sample and sorting."
    )


def ranked_mesh(
    records: list[RISRecord], *,
    top_n: int = 100,
    exclude_check_tags: bool = True,
) -> HarvestResult:
    """Count descriptor assignments per PMID; major-topic flags separately.

    Percentages use the MeSH-indexed subset as denominator; also include %
    of all retrieved records so missing indexation is not hidden.
    """
    if top_n < 1:
        raise ValueError("top_n muss >= 1 sein.")
    indexed = [r for r in records if r.fields.get("MH")]
    n = len(indexed)
    df = Counter()
    majors = Counter()
    for rec in indexed:
        terms = {t.strip() for t in rec.fields.get("MH", []) if t.strip()}
        major = {t.strip() for t in rec.fields.get("M1", []) if t.strip()}
        if exclude_check_tags:
            terms = {t for t in terms if t.casefold() not in COMMON_CHECK_TAGS}
            major = {t for t in major if t.casefold() not in COMMON_CHECK_TAGS}
        df.update(terms)
        majors.update(major & terms)
    rows = [{
        "Rank": i + 1,
        "MeSH descriptor": term,
        "Indexed records": df[term],
        "% indexed records": round(100 * df[term] / n, 2) if n else 0,
        "Major topic records": majors[term],
        "% all retrieved": round(100 * df[term] / len(records), 2) if records else 0,
    } for i, term in enumerate(sorted(df, key=lambda t: (-df[t], -majors[t], t))[:top_n])]
    return HarvestResult(
        rows, n, len(records) - n, len(records), "mesh_frequency",
        "Counts of the actual NLM MeSH descriptor assignments (one count per "
        "PMID). Major topic counts come from the MeSH DescriptorName "
        "MajorTopicYN flag. Percentages refer to MeSH-indexed records; "
        "the full-sample percentage is also reported. Unindexed records are "
        "not treated as MeSH-negative articles."
    )


def rows_to_csv(rows: list[dict], columns: list[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")
