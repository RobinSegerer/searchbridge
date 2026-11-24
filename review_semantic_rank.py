"""Two-stage semantic prioritization for systematic-review title/abstract screening.

This only reorders records; it never excludes records or makes eligibility
decisions. Embeddings and cross-encoder inference run locally with FastEmbed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

from review_core import ReviewRecord

BI_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
MAX_CROSS_CANDIDATES = 2000


@dataclass
class SemanticRanking:
    records: list[ReviewRecord]
    details: dict[str, dict]
    candidate_count: int
    method: str = "MiniLM bi-encoder cosine; MS MARCO MiniLM cross-encoder reranking"


def _cosine(a, b) -> float:
    x = [float(v) for v in a]
    y = [float(v) for v in b]
    if len(x) != len(y) or not x:
        raise ValueError("Incompatible embedding dimensions.")
    length = math.sqrt(sum(v * v for v in x) * sum(v * v for v in y))
    if not length:
        return 0.0
    result = sum(u * v for u, v in zip(x, y)) / length
    if not math.isfinite(result):
        raise ValueError("Non-finite semantic similarity.")
    return result


def semantic_rerank(
    records: list[ReviewRecord],
    focus: str,
    *,
    embedding_model,
    cross_encoder,
    cross_top_n: int = 200,
    progress: Callable[[str, float], None] | None = None,
) -> SemanticRanking:
    """Rank every record semantically, then rerank top candidates jointly.

    The remaining records are retained in semantic order after the cross-encoder
    group. No threshold or eligibility classification is applied.
    The two model instances are injectable for deterministic offline tests.
    """
    if not focus.strip():
        raise ValueError("A research question or topic is required for semantic ranking.")
    if not 1 <= cross_top_n <= MAX_CROSS_CANDIDATES:
        raise ValueError("Cross-encoder candidates must be between 1 and 2,000.")
    if not records:
        return SemanticRanking([], {}, 0)
    if len({r.uid for r in records}) != len(records):
        raise ValueError("Duplicate record identifiers cannot be ranked reliably.")
    documents = [
        ("Title: " + (r.title or "") + "\nAbstract: " + (r.abstract or ""))[:5000]
        for r in records
    ]
    if progress:
        progress("Embedding research focus and all records", 0.05)
    vectors = list(embedding_model.embed([focus.strip()] + documents, batch_size=32))
    if len(vectors) != len(records) + 1:
        raise ValueError("Embedding model returned an unexpected number of vectors.")
    similarities = {
        record.uid: _cosine(vectors[0], vector)
        for record, vector in zip(records, vectors[1:])
    }
    semantic = sorted(records, key=lambda r: (-similarities[r.uid], r.uid))
    limit = min(cross_top_n, len(semantic))
    if progress:
        progress("Cross-encoder reranking of top candidates", 0.70)
    candidates = semantic[:limit]
    cross_documents = [
        ("Title: " + (r.title or "") + "\nAbstract: " + (r.abstract or ""))[:3500]
        for r in candidates
    ]
    scores = list(cross_encoder.rerank(focus.strip(), cross_documents))
    if len(scores) != limit:
        raise ValueError("Cross-encoder returned an unexpected number of scores.")
    cross_scores = {r.uid: float(s) for r, s in zip(candidates, scores)}
    if any(not math.isfinite(s) for s in cross_scores.values()):
        raise ValueError("Cross-encoder returned a non-finite relevance score.")
    top = sorted(candidates, key=lambda r: (-cross_scores[r.uid], r.uid))
    ordered = top + semantic[limit:]
    for record in ordered:
        # Expose the bi-encoder cosine in existing ranking-score exports.
        record.rank_score = round(similarities[record.uid], 6)
    if progress:
        progress("Finalizing semantic ranking and provenance", 0.96)
    details = {
        record.uid: {
            "Bi-encoder similarity": round(similarities[record.uid], 6),
            "Cross-encoder score": (
                round(cross_scores[record.uid], 6)
                if record.uid in cross_scores else None
            ),
            "Final rank": i,
            "Ranking stage": (
                "Cross-encoder" if record.uid in cross_scores else "Bi-encoder"
            ),
        }
        for i, record in enumerate(ordered, 1)
    }
    if progress:
        progress("Semantic reranking complete", 1.0)
    return SemanticRanking(ordered, details, limit)
