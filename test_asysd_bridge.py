"""Offline tests for optional genuine ASySD R bridge and audit invariants."""
import csv
import io
import unittest
from unittest.mock import patch

from asysd_bridge import (
    asysd_csv, _reconstruct, _uids_from_asysd,
    run_asysd, ASySDError
)
from review_core import ReviewRecord


def record(uid, source, doi="", abstract=""):
    return ReviewRecord(
        uid=uid, source=source, source_id=uid,
        title="Digital media and student achievement",
        year="2025", doi=doi, abstract=abstract,
    )


class ASySDBridgeTests(unittest.TestCase):
    def setUp(self):
        self.originals = [
            record("a", "PubMed", "10.1234/abc", "Long complete abstract text"),
            record("b", "OpenAlex", "10.1234/abc", ""),
            record("c", "Scopus", "10.1234/other", "Another abstract"),
        ]

    def test_input_csv_has_original_uids(self):
        rows = list(csv.DictReader(
            io.StringIO(asysd_csv(self.originals).decode("utf-8-sig"))))
        self.assertEqual([x["record_id"] for x in rows], ["a", "b", "c"])
        self.assertIn("source", rows[0])
        self.assertEqual(rows[0]["abstract"], "Long complete abstract text")

    def test_asysd_merged_groups_preserve_sources_and_abstract(self):
        unique = [
            {"record_ids": "a, b"},
            {"record_ids": "c"},
        ]
        manual = [{"record_id.x": "a", "record_id.y": "c"}]
        result = _reconstruct(self.originals, unique, manual)
        self.assertEqual(result.duplicates_removed, 1)
        self.assertEqual(len(result.records), 2)
        joined = next(x for x in result.records if x.doi == "10.1234/abc")
        self.assertEqual(set(joined.sources), {"PubMed", "OpenAlex"})
        self.assertEqual(joined.abstract, "Long complete abstract text")
        self.assertEqual(len(result.audit), 1)
        self.assertIn("ASySD", result.audit[0]["matching_evidence"])
        self.assertEqual(result.fuzzy_review[0]["status"],
                         "MANUAL REVIEW - ASySD candidate NOT merged")

    def test_missing_or_unknown_uid_fail_closed(self):
        with self.assertRaises(ASySDError):
            _reconstruct(self.originals, [{"record_ids": "a,b"}], [])
        with self.assertRaises(ASySDError):
            _reconstruct(self.originals,
                         [{"record_ids": "a,b"},{"record_ids": "not-a-uid"}], [])

    def test_conflicting_dois_fail_closed(self):
        with self.assertRaises(ASySDError):
            _reconstruct(self.originals,
                         [{"record_ids": "a,b,c"}], [])

    def test_duplicate_assignment_fail_closed(self):
        with self.assertRaises(ASySDError):
            _reconstruct(self.originals,
                         [{"record_ids": "a,b"}, {"record_ids": "b,c"}], [])

    def test_cannot_claim_asysd_without_r(self):
        with patch("asysd_bridge.asysd_available", return_value=False):
            with self.assertRaises(ASySDError):
                run_asysd(self.originals)

    def test_identifier_parser_for_merged_and_single(self):
        self.assertEqual(
            _uids_from_asysd({"record_ids": "a, b, a"}), ["a", "b"])
        self.assertEqual(
            _uids_from_asysd({"record_id": "c"}), ["c"])


if __name__ == "__main__":
    unittest.main()
