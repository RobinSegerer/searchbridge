"""Two-stage local semantic ranking for Review Pipeline records.

Bi-encoder scores ALL deduplicated records; cross-encoder rescoring is
restricted to its highest-ranking candidates. No records are discarded.
The existing TF-IDF score remains an independent auditable baseline.
Both ONNX models run on the Streamlit server via FastEmbed.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np

from review_core import ReviewRecord

BI_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_MODEL_ID = "Xenova/ms-marco-MiniLM-L-6-v2"
MAX_CROSS_CANDIDATES = 2000


class RerankError(RuntimeError):
    """Semantic ranking was not completed; keep the lexical results."""


@dataclass
class RerankResult:
    ordered_uids: list[str]
    scores: dict[str, dict]
    cross_count: int
    bi_model: str = BI_MODEL_ID
    cross_model: str = CROSS_MODEL_ID


@lru_cache(maxsize=1)
def _bi_encoder():
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=BI_MODEL_ID)


@lru_cache(maxsize=1)
def _cross_encoder():
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(model_name=CROSS_MODEL_ID)


def _document(record: ReviewRecord) -> str:
    # Avoid sending duplicate records or overly long material to the models.
    # A leading title puts the topic at the start of truncated model inputs.
    title = (record.title or "").strip()
    abstract = (record.abstract or "").strip()[:2400]
    return (title + ". " + abstract).strip() or "No bibliographic text available"


def rerank_records(
    records: list[ReviewRecord], focus: str, *,
    cross_top_k: int = 200,
    bi_encoder=None, cross_encoder=None,
    progress: Callable[[str, int, int], None] | None = None,
) -> RerankResult:
    """Rank without a lexical shortlist and without excluding records.

    The cross-encoder operates on the top K *semantic* candidates; the
    remainder retain their bi-encoder order. Cross scores are not
    comparable with lexical or bi-encoder scores.
    """
    if not focus.strip():
        raise ValueError("Enter a research question or target concept first.")
    if not 1 <= cross_top_k <= MAX_CROSS_CANDIDATES:
        raise ValueError("Cross-encoder candidate count must be 1–2,000.")
    if not records:
        return RerankResult([], {}, 0)
    try:
        bi = bi_encoder if bi_encoder is not None else _bi_encoder()
        docs = [_document(r) for r in records]
        query_vector = np.asarray(list(bi.embed([focus.strip()]))[0], dtype=np.float32)
        norm = float(np.linalg.norm(query_vector))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Invalid research-focus embedding")
        query_vector /= norm

        similarities = []
        # Batches keep memory bounded and make progress observable.
        for start in range(0, len(records), 128):
            chunk = docs[start:start + 128]
            vectors = np.asarray(list(bi.embed(chunk, batch_size=32)), dtype=np.float32)
            if vectors.shape != (len(chunk), len(query_vector)):
                raise ValueError("Unexpected embedding dimensions or count")
            norms = np.linalg.norm(vectors, axis=1)
            norms[norms == 0] = 1
            sims = (vectors @ query_vector) / norms
            if not np.all(np.isfinite(sims)):
                raise ValueError("Non-finite semantic similarity")
            similarities.extend(float(x) for x in sims)
            if progress:
                progress("Bi-encoder", min(start + len(chunk), len(records)), len(records))

        indices = sorted(range(len(records)),
                         key=lambda i: (-similarities[i], records[i].uid))
        top = indices[:min(cross_top_k, len(indices))]
        cross = cross_encoder if cross_encoder is not None else _cross_encoder()
        rerank_values = list(cross.rerank(focus.strip(), [docs[i] for i in top]))
        if len(rerank_values) != len(top):
            raise ValueError("Unexpected cross-encoder result count")
        cross_scores = {i: float(s) for i, s in zip(top, rerank_values)}
        if not np.all(np.isfinite(list(cross_scores.values()))):
            raise ValueError("Invalid cross-encoder scores")
        if progress:
            progress("Cross-encoder", len(top), len(top))
        top.sort(key=lambda i: (-cross_scores[i], -similarities[i], records[i].uid))
        order = top + indices[len(top):]
        scores = {
            rec.uid: {
                "lexical_tfidf": rec.rank_score,
                "bi_encoder_cosine": round(similarities[i], 6),
                "cross_encoder_score": (
                    round(cross_scores[i], 6) if i in cross_scores else None
                ),
                "ranking_stage": "cross_encoder" if i in cross_scores else "bi_encoder",
            }
            for i, rec in enumerate(records)
        }
        return RerankResult([records[i].uid for i in order], scores, len(top))
    except Exception as exc:
        raise RerankError(
            "Semantic ranking could not finish. The lexical ranking is still "
            "available; retry on the server or reduce the cross-encoder pool. "
            "Details: " + str(exc)
        ) from exc
