"""Offline regression tests: semantic reranking and strict LLM screening."""
from __future__ import annotations

import unittest

from review_core import ReviewRecord
from review_reranking import rerank_records
from openai_review_screening import (
    _apply_safeguards, _evidence_present, _validate_decision, _manual_decision,
)


class FakeBiEncoder:
    def embed(self, documents, batch_size=32):
        for item in documents:
            if "sleep" in item.casefold() or "insomnia" in item.casefold():
                yield [1.0, 0.0]
            else:
                yield [0.0, 1.0]


class FakeCrossEncoder:
    def rerank(self, query, documents):
        return [1.0 if "insomnia" in x.casefold() else 0.2 for x in documents]


class SemanticTests(unittest.TestCase):
    def test_all_records_retained_and_cross_only_scores_top_k(self):
        records = [
            ReviewRecord(uid="a", title="Sleep quality", abstract="Healthy volunteers"),
            ReviewRecord(uid="b", title="Insomnia treatment", abstract="Clinical effects"),
            ReviewRecord(uid="c", title="Housing prices", abstract="Inflation patterns"),
        ]
        result = rerank_records(
            records, "sleep intervention", cross_top_k=2,
            bi_encoder=FakeBiEncoder(), cross_encoder=FakeCrossEncoder(),
        )
        self.assertEqual(result.ordered_uids, ["b", "a", "c"])
        self.assertEqual(set(result.ordered_uids), {"a", "b", "c"})
        self.assertIsNone(result.scores["c"]["cross_encoder_score"])
        self.assertEqual(result.cross_count, 2)

    def test_invalid_focus_is_rejected(self):
        with self.assertRaises(ValueError):
            rerank_records([ReviewRecord(uid="a")], "", bi_encoder=FakeBiEncoder())


class StrictScreeningTests(unittest.TestCase):
    def setUp(self):
        self.record = ReviewRecord(
            uid="r1", title="Sleep training in children",
            abstract="A randomized trial with improved sleep outcomes."
        )
        self.base = {
            "decision": "include", "reason": "Relevant empirical trial",
            "evidence": "A randomized trial",
            "uncertainty": "low", "potential_relevance": "high",
            "criterion_assessments": [
                {"criterion": "Randomized trial", "status": "met",
                 "evidence": "A randomized trial"}
            ],
        }

    def test_supported_include_is_not_downgraded(self):
        result = _apply_safeguards(_validate_decision(dict(self.base)), self.record)
        self.assertEqual(result["decision"], "include")
        self.assertEqual(result["raw_model_decision"], "include")
        self.assertEqual(result["safeguard_reasons"], [])

    def test_unclear_criterion_never_passes_as_include(self):
        raw = {**self.base, "criterion_assessments": [
            {"criterion": "Population", "status": "unclear", "evidence": ""}
        ]}
        result = _apply_safeguards(_validate_decision(raw), self.record)
        self.assertEqual(result["decision"], "manual_review")
        self.assertEqual(result["raw_model_decision"], "include")
        self.assertTrue(result["safeguard_reasons"])
        self.assertEqual(result["potential_relevance"], "high")

    def test_exclude_without_verifiable_unmet_criterion_is_manual(self):
        raw = {**self.base, "decision": "exclude",
               "criterion_assessments": [
                   {"criterion": "Adult population", "status": "not_met",
                    "evidence": "Nonexistent quotation"}
               ]}
        result = _apply_safeguards(_validate_decision(raw), self.record)
        self.assertEqual(result["decision"], "manual_review")

    def test_newlines_are_not_fabricated_evidence(self):
        self.assertTrue(_evidence_present(
            '"A randomized trial"', self.record.abstract
        ))
        self.assertFalse(_evidence_present("an invented result", self.record.abstract))

    def test_missing_abstract_records_origin_and_relevance(self):
        result = _manual_decision("Abstract missing", "test-model",
                                  "missing_metadata_rule")
        self.assertEqual(result["decision"], "manual_review")
        self.assertEqual(result["potential_relevance"], "unknown")
        self.assertEqual(result["origin"], "missing_metadata_rule")


if __name__ == "__main__":
    unittest.main()
