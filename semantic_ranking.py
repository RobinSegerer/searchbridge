"""Semantic and hybrid reranking for SearchBridge title-term harvesting.

Candidates ALWAYS originate in the observed PubMed titles and their recorded
TF-IDF scores. FastEmbed runs the original all-MiniLM-L6-v2 sentence-embedding
model through ONNX on CPU, without requiring PyTorch or a generative LLM.

Ranking is either cosine similarity to an explicit human-supplied target
concept or weighted reciprocal-rank fusion (RRF) of cosine and TF-IDF ranks.
Optional MMR reorders only the displayed top-N to reduce redundant candidates.

No external API or embeddings are sent to a third-party inference service.
The model weights are downloaded from the Hugging Face hub upon first use.
"""
from __future__ import annotations

import math
import re
from typing import Any, Sequence

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60
MAX_EMBED_CANDIDATES = 1500


class EmbeddingError(RuntimeError):
    """Embedding model or its result could not be used safely."""


def load_fastembed_model():
    """Lazy import so TF-IDF and TIAB→MeSH work without the model installed."""
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise EmbeddingError(
            "FastEmbed is not installed. Install requirements.txt or "
            "select the TF-IDF mode."
        ) from exc
    try:
        return TextEmbedding(model_name=MODEL_ID)
    except Exception as exc:
        raise EmbeddingError(
            "Could not load the MiniLM sentence-embedding model. "
            "The first run requires an internet connection to Hugging Face."
        ) from exc


def _unit(vec: Any) -> tuple[float, ...]:
    """Work with numpy vectors or test vectors, without importing numpy."""
    try:
        v = tuple(float(x) for x in vec)
    except (TypeError, ValueError) as exc:
        raise EmbeddingError("Embedding is not a numeric one-dimensional vector.") from exc
    if not v or any(not math.isfinite(x) for x in v):
        raise EmbeddingError("Embedding contains missing, nonfinite, or empty data.")
    size = math.sqrt(sum(x * x for x in v))
    if size <= 1e-12:
        raise EmbeddingError("Embedding has zero norm.")
    return tuple(x / size for x in v)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _rrf(rank: int) -> float:
    return 1.0 / (RRF_K + rank)


def default_semantic_focus(query: str) -> str:
    """Offer a suggested concept without claiming this is the user's intent."""
    q = re.sub(r"\[[^\]]+\]", " ", query)
    q = re.sub(r"\b(?:AND|OR|NOT)\b", ", ", q, flags=re.I)
    q = re.sub(r'[(){}"\'\[\]]', " ", q)
    q = re.sub(r"\s+", " ", q).strip(" ,")
    return q[:240]


def rerank_title_terms(
    tfidf_rows: list[dict],
    *,
    focus: str,
    encoder: Any,
    mode: str = "hybrid",
    semantic_weight: float = 0.60,
    top_n: int = 100,
    diversify: bool = False,
    diversity_weight: float = 0.15,
) -> list[dict]:
    """Rank a TF-IDF-derived candidate pool; no invented lexical candidates.

    Weighted RRF uses rank positions rather than mixing incommensurable raw
    TF-IDF and cosine scales. MMR uses that base ranking and max observed
    embedding similarity among already selected output terms, with a fixed
    documented penalty (only if diversify=True).
    """
    if mode not in ("semantic", "hybrid"):
        raise ValueError("mode must be semantic or hybrid.")
    if not (0.0 <= semantic_weight <= 1.0):
        raise ValueError("semantic_weight must be between 0 and 1.")
    if not (0.0 <= diversity_weight < 1.0):
        raise ValueError("diversity_weight must be between 0 and 1.")
    if top_n < 1:
        raise ValueError("top_n must be positive.")
    if not focus or len(focus.strip()) < 3:
        raise ValueError("Bitte ein aussagekräftiges semantisches Zielkonzept eingeben.")
    if len(tfidf_rows) > MAX_EMBED_CANDIDATES:
        raise ValueError("Too many candidates for a single semantic embedding batch.")
    if not tfidf_rows:
        return []

    terms = [str(row["Title term"]) for row in tfidf_rows]
    if len(set(terms)) != len(terms):
        raise ValueError("Candidate list contains duplicate terms.")

    # FastEmbed API: embedding vectors produced lazily as a generator.
    # Always materialize once and check dimensions/number of results.
    try:
        raw = list(encoder.embed([focus.strip()] + terms, batch_size=32))
    except Exception as exc:
        raise EmbeddingError(
            "Embedding failed. Check the model download and runtime memory."
        ) from exc
    if len(raw) != len(terms) + 1:
        raise EmbeddingError(
            "Embedding backend returned fewer/more vectors than requested."
        )
    vectors = [_unit(v) for v in raw]
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise EmbeddingError("Embedding dimensions differ.")

    focus_vec, term_vectors = vectors[0], vectors[1:]
    similarity = [_dot(focus_vec, v) for v in term_vectors]
    indices = list(range(len(terms)))
    semantic_order = sorted(indices, key=lambda i: (-similarity[i], terms[i]))
    semantic_ranks = [0] * len(terms)
    for rank, ix in enumerate(semantic_order, 1):
        semantic_ranks[ix] = rank

    # Re-rank the original TF-IDF rows by score, not their input order.
    tfidf_order = sorted(indices, key=lambda i: (
        -float(tfidf_rows[i]["Mean TF-IDF"]), terms[i]))
    tfidf_ranks = [0] * len(terms)
    for rank, ix in enumerate(tfidf_order, 1):
        tfidf_ranks[ix] = rank

    # Normalize weighted RRF by its theoretical maximum (rank=1).
    rrf_scale = _rrf(1)
    if mode == "semantic":
        combined = [float(cos) for cos in similarity]
    else:
        combined = [
            (semantic_weight * _rrf(semantic_ranks[i])
             + (1 - semantic_weight) * _rrf(tfidf_ranks[i])) / rrf_scale
            for i in indices
        ]
    base_order = sorted(indices, key=lambda i: (
        -combined[i], -similarity[i], -float(tfidf_rows[i]["Mean TF-IDF"]),
        terms[i]
    ))
    limit = min(top_n, len(terms))

    if not diversify or limit < 2:
        final_order = base_order[:limit]
        mmr_scores = {i: combined[i] for i in final_order}
    else:
        # Greedy Maximal Marginal Relevance on title-term embeddings;
        # no full O(n*n) similarity matrix is materialized.
        final_order = []
        available = set(indices)
        mmr_scores = {}
        # Maintain incremental redundancy; O(n*k*d) rather than O(n*k^2*d).
        max_redundancy = [0.0] * len(terms)
        for _ in range(limit):
            best_ix = None
            best_val = float("-inf")
            for i in available:
                val = ((1 - diversity_weight) * combined[i]
                       - diversity_weight * max_redundancy[i])
                if best_ix is None or val > best_val or (
                    val == best_val and terms[i] < terms[best_ix]
                ):
                    best_ix = i
                    best_val = val
            final_order.append(best_ix)
            mmr_scores[best_ix] = best_val
            available.remove(best_ix)
            for i in available:
                max_redundancy[i] = max(
                    max_redundancy[i],
                    max(0.0, _dot(term_vectors[i], term_vectors[best_ix]))
                )

    results: list[dict] = []
    for rank, i in enumerate(final_order, 1):
        original = tfidf_rows[i]
        results.append({
            "Rank": rank,
            "Title term": terms[i],
            "Semantic similarity": round(similarity[i], 6),
            "Semantic rank": semantic_ranks[i],
            "TF-IDF rank": tfidf_ranks[i],
            "Mean TF-IDF": original["Mean TF-IDF"],
            "Document frequency": original["Document frequency"],
            "% titles": original["% titles"],
            "Occurrences": original["Occurrences"],
            "Hybrid RRF": round(combined[i], 6) if mode == "hybrid" else "",
            "MMR score": round(mmr_scores[i], 6) if diversify else "",
        })
    return results


def compare_top_k(
    tfidf_rows: list[dict],
    ranked_rows: list[dict],
    k: int = 20,
) -> dict:
    """Describe ranking overlap only, not unverified retrieval performance."""
    if k < 1:
        raise ValueError("k must be >= 1.")
    original = {str(x["Title term"]) for x in tfidf_rows[:k]}
    revised = {str(x["Title term"]) for x in ranked_rows[:k]}
    return {
        "Top K": k,
        "Shared terms": len(original & revised),
        "New terms": sorted(revised - original),
        "Dropped terms": sorted(original - revised),
    }
