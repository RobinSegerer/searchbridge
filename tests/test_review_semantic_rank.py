"""Offline regression tests for local two-stage ranking (no model download)."""
from __future__ import annotations

import unittest

from review_core import ReviewRecord
from review_semantic_rank import semantic_rerank


class FakeEmbedding:
    def embed(self, texts, batch_size=32):
        for text in texts:
            if text == "target topic":
                yield [1.0, 0.0]
            elif "alpha" in text:
                yield [0.99, 0.01]
            elif "beta" in text:
                yield [0.90, 0.10]
            else:
                yield [0.01, 0.99]


class FakeCrossEncoder:
    def rerank(self, query, documents):
        assert query == "target topic"
        return [3.0 if "beta" in document else 1.0 for document in documents]


class SemanticReviewRankingTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            ReviewRecord(uid="alpha", title="alpha paper"),
            ReviewRecord(uid="beta", title="beta paper"),
            ReviewRecord(uid="gamma", title="gamma paper"),
        ]

    def test_cross_encoder_reorders_top_candidates_without_exclusions(self):
        out = semantic_rerank(
            self.records, "target topic",
            embedding_model=FakeEmbedding(),
            cross_encoder=FakeCrossEncoder(),
            cross_top_n=2,
        )
        self.assertEqual([r.uid for r in out.records],
                         ["beta", "alpha", "gamma"])
        self.assertEqual(out.candidate_count, 2)
        self.assertEqual(out.details["beta"]["Final rank"], 1)
        self.assertEqual(out.details["gamma"]["Ranking stage"], "Bi-encoder")
        self.assertEqual(len(out.details), 3)

    def test_top_one_does_not_drop_other_studies(self):
        out = semantic_rerank(
            self.records, "target topic",
            embedding_model=FakeEmbedding(),
            cross_encoder=FakeCrossEncoder(),
            cross_top_n=1,
        )
        self.assertEqual(len(out.records), 3)
        self.assertEqual(set(r.uid for r in out.records),
                         {"alpha", "beta", "gamma"})

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            semantic_rerank(
                self.records, " ", embedding_model=FakeEmbedding(),
                cross_encoder=FakeCrossEncoder(),
            )


if __name__ == "__main__":
    unittest.main()
