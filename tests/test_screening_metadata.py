"""Regression tests for bibliographic fields in SearchBridge screening output."""
from __future__ import annotations

import csv
import io
import unittest
import zipfile

from review_core import MergeResult, ReviewRecord
from review_pipeline_ui import _bundle, _record_fields, _screening_row


class ScreeningMetadataTests(unittest.TestCase):
    def setUp(self):
        self.record = ReviewRecord(
            uid="example-1",
            title="Example study",
            abstract="A complete study abstract.",
            authors="First Author; Second Author",
            year="2025",
            journal="Journal of Examples",
            doi="10.1234/example",
            source="PubMed",
            sources=["PubMed", "Scopus"],
        )
        self.ai = {
            self.record.uid: {
                "decision": "manual_review",
                "reason": "Insufficient population detail",
                "evidence": "Population not specified",
                "uncertainty": "high",
            }
        }
        self.human = {
            self.record.uid: {
                "decision": "include",
                "reason": "Meets inclusion criteria",
                "reviewer": "RS",
                "timestamp_utc": "2026-11-23T00:00:00+00:00",
            }
        }

    def test_display_fields_include_all_seven_metadata_columns(self):
        row = _record_fields(self.record)
        self.assertEqual(
            list(row),
            ["Title", "Abstract", "Authors", "Year", "Journal", "DOI", "Database"],
        )
        self.assertEqual(row["Database"], "PubMed; Scopus")
        self.assertEqual(row["Abstract"], "A complete study abstract.")

    def test_ai_and_human_decisions_remain_separate(self):
        row = _screening_row(self.record, self.ai, self.human)
        self.assertEqual(row["AI suggestion"], "manual_review")
        self.assertEqual(row["Human decision"], "include")
        self.assertEqual(row["Human reason"], "Meets inclusion criteria")
        self.assertEqual(row["Authors"], "First Author; Second Author")

    def test_bundle_exports_self_contained_screening_csv_files(self):
        merged = MergeResult(
            records=[self.record], counts={"PubMed": 1, "Scopus": 1},
            duplicates_removed=0, audit=[], fuzzy_review=[], input_count=1,
        )
        bundle = _bundle(merged, [self.record], [], self.ai, self.human, [])
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            for filename in (
                "screening_results.csv",
                "human_decisions.csv",
                "ai_screening_suggestions.csv",
            ):
                rows = list(csv.DictReader(
                    io.StringIO(archive.read(filename).decode("utf-8-sig"))
                ))
                self.assertEqual(len(rows), 1, filename)
                for field in (
                    "Title", "Abstract", "Authors", "Year",
                    "Journal", "DOI", "Database",
                ):
                    self.assertIn(field, rows[0], filename)
                self.assertEqual(rows[0]["Database"], "PubMed; Scopus")


if __name__ == "__main__":
    unittest.main()
