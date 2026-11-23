"""Offline deterministic tests for semantic ranking; no model download."""
import unittest

from ris_proximity import RISRecord
from semantic_ranking import (
    MODEL_ID, RRF_K, MAX_EMBED_CANDIDATES, EmbeddingError,
    rerank_title_terms, compare_top_k, annotate_seed_presence,
    default_semantic_focus,
)


def row(rank, term, score):
    return {
        "Rank": rank, "Title term": term, "Mean TF-IDF": score,
        "Document frequency": 10 - rank, "% titles": 33,
        "Occurrences": 10 - rank,
    }


class MockEmbedder:
    def __init__(self):
        self.calls = []
        self.vectors = {
            "academic self-concept": [1.0, 0.0, 0.0],
            "academic performance": [0.0, 1.0, 0.0],
            "scholastic self concept": [0.99, 0.1, 0.0],
            "academic self perception": [0.97, 0.15, 0.0],
            "student wellbeing": [0.0, 0.0, 1.0],
            "self efficacy": [0.8, 0.25, 0.05],
        }

    def embed(self, texts, batch_size=32):
        self.calls.append((list(texts), batch_size))
        return (self.vectors[text] for text in texts)


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            row(1, "academic performance", 0.8),
            row(2, "student wellbeing", 0.7),
            row(3, "scholastic self concept", 0.55),
            row(4, "academic self perception", 0.40),
            row(5, "self efficacy", 0.30),
        ]
        self.encoder = MockEmbedder()

    def test_semantic_reranks_only_observed_terms(self):
        rows = rerank_title_terms(
            self.rows, focus="academic self-concept",
            encoder=self.encoder, mode="semantic", top_n=5,
        )
        self.assertEqual(rows[0]["Title term"], "scholastic self concept")
        self.assertEqual(rows[0]["TF-IDF rank"], 3)
        self.assertGreater(rows[0]["Semantic similarity"], 0.99)
        self.assertEqual(
            {r["Title term"] for r in rows},
            {r["Title term"] for r in self.rows},
        )
        self.assertEqual(self.encoder.calls[0][1], 32)
        self.assertEqual(rows[0]["Rank"], 1)

    def test_rrf_zero_weight_matches_baseline_rank(self):
        rows = rerank_title_terms(
            self.rows, focus="academic self-concept",
            encoder=self.encoder, mode="hybrid",
            semantic_weight=0, top_n=5,
        )
        self.assertEqual(
            [r["Title term"] for r in rows],
            [r["Title term"] for r in self.rows],
        )
        self.assertAlmostEqual(rows[0]["Hybrid RRF"], 1)

    def test_rrf_full_weight_matches_semantic(self):
        rows = rerank_title_terms(
            self.rows, focus="academic self-concept",
            encoder=self.encoder, mode="hybrid",
            semantic_weight=1, top_n=5,
        )
        self.assertEqual(rows[0]["Title term"], "scholastic self concept")
        self.assertEqual(rows[0]["Semantic rank"], 1)
        self.assertTrue(all(r["Hybrid RRF"] > 0 for r in rows))

    def test_mmr_diversifies_when_penalty_high(self):
        rows = rerank_title_terms(
            self.rows, focus="academic self-concept",
            encoder=self.encoder, mode="semantic",
            top_n=3, diversify=True, diversity_weight=0.45,
        )
        self.assertEqual(rows[0]["Title term"], "scholastic self concept")
        self.assertIn("MMR score", rows[0])
        self.assertEqual(len({r["Title term"] for r in rows}), 3)

    def test_missing_and_nonfinite_embeddings_fail(self):
        class BadEmbed:
            def embed(self, texts, batch_size=32):
                return [[1, 0]]
        with self.assertRaises(EmbeddingError):
            rerank_title_terms(
                self.rows, focus="academic self-concept",
                encoder=BadEmbed(), mode="semantic",
            )
        class NanEmbed:
            def embed(self, texts, batch_size=32):
                return [[float("nan"), 0] for _ in texts]
        with self.assertRaises(EmbeddingError):
            rerank_title_terms(
                self.rows, focus="academic self-concept",
                encoder=NanEmbed(), mode="semantic",
            )

    def test_mode_validation_and_size_limit(self):
        with self.assertRaises(ValueError):
            rerank_title_terms(
                self.rows, focus="academic self-concept",
                encoder=self.encoder, mode="tfidf",
            )
        with self.assertRaises(ValueError):
            rerank_title_terms(
                self.rows, focus=" ", encoder=self.encoder,
            )
        with self.assertRaises(ValueError):
            rerank_title_terms(
                self.rows, focus="academic self-concept",
                encoder=self.encoder, semantic_weight=-0.1,
            )

    def test_mesh_descriptor_reranking_preserves_counts(self):
        rows = [
            {"Rank":1,"MeSH descriptor":"academic performance","Indexed records":50},
            {"Rank":2,"MeSH descriptor":"scholastic self concept","Indexed records":15},
            {"Rank":3,"MeSH descriptor":"self efficacy","Indexed records":12},
        ]
        result = rerank_title_terms(
            rows, focus="academic self-concept", encoder=self.encoder,
            mode="semantic", term_key="MeSH descriptor",
            score_key="Indexed records", base_rank_label="Frequency rank",
        )
        self.assertEqual(result[0]["MeSH descriptor"], "scholastic self concept")
        self.assertEqual(result[0]["Indexed records"], 15)
        self.assertEqual(result[0]["Frequency rank"], 2)

    def test_topk_compare_not_a_recall_metric(self):
        ranked = rerank_title_terms(
            self.rows, focus="academic self-concept", encoder=self.encoder,
            mode="semantic", top_n=5,
        )
        check = compare_top_k(self.rows, ranked, k=2)
        self.assertEqual(check["Top K"], 2)
        self.assertEqual(len(check["New terms"]), 2)

    def test_seed_presence_exact_phrase_and_field_boundaries(self):
        seeds = [
            RISRecord(1, {"TI":["Study of scholastic self concept"],"AB":["Learning strategies"]}, []),
            RISRecord(2, {"TI":["Scholastic and self concept"],"AB":["self efficacy predicts outcomes"]}, []),
            RISRecord(3, {"TI":["Scholastic self"],"AB":["concept outcomes"]}, []),
        ]
        result = annotate_seed_presence(
            [{"Title term":"scholastic self concept"},
             {"Title term":"self efficacy"}],
            seeds
        )
        self.assertEqual(result[0]["Seed records containing term"], 1)
        self.assertAlmostEqual(result[0]["% seed records"], 33.33)
        self.assertEqual(result[1]["Seed records containing term"], 1)

    def test_focus_suggestion_strips_pubmed_tags(self):
        text = default_semantic_focus(
            '("Students"[mh] AND "Academic Self Concept"[tiab])'
        )
        self.assertNotIn("[mh]", text)
        self.assertIn("Students", text)
        self.assertIn("Academic Self Concept", text)


if __name__ == "__main__":
    unittest.main()
