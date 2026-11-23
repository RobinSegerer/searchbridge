"""Offline regression tests for federated review retrieval, merge and screening."""
from __future__ import annotations

import csv
import io
import json
import unittest
from unittest.mock import patch

from ris_proximity import parse_ris
from review_core import (
    ReviewRecord, from_openalex_results, from_scopus_results, from_ris,
    merge_records, lexical_rank, csv_bytes, ris_bytes, norm_doi,
)
from review_retrieval import openalex_search, scopus_search, pubmed_search, RetrievalError
from openai_review_screening import (
    OpenAIScreenError, screen_one, screen_batch,
)


def make(source, ident, title, year="2025", doi="", pmid="", abstract=""):
    return ReviewRecord(
        uid=f"{source}-{ident}", source=source, source_id=ident,
        title=title, year=year, doi=doi, pmid=pmid,
        abstract=abstract,
    )


class TestReviewCore(unittest.TestCase):
    def test_cross_source_duplicate_doi_and_provenance(self):
        a = make("PubMed", "1", "Social media use by students",
                 doi="https://doi.org/10.1234/ABC", pmid="123",
                 abstract="This abstract is detailed.")
        b = make("OpenAlex", "W2", "Social-media use by students",
                 doi="10.1234/abc", abstract="")
        c = make("Scopus", "3", "Other study", doi="10.5678/xyz")
        m = merge_records([a, b, c])
        self.assertEqual((m.input_count, len(m.records), m.duplicates_removed),
                         (3, 2, 1))
        common = next(x for x in m.records if x.doi == "10.1234/abc")
        self.assertEqual(set(common.sources), {"PubMed", "OpenAlex"})
        self.assertEqual(len(common.identifiers), 2)
        self.assertIn("detailed", common.abstract)
        self.assertEqual(len(m.audit), 1)
        self.assertEqual(m.counts, {"OpenAlex": 1, "PubMed": 1, "Scopus": 1})

    def test_duplicate_pmid_without_doi(self):
        a = make("PubMed", "1", "An interesting study", pmid="12345")
        b = make("OpenAlex", "2", "An interesting study re-titled", pmid="12345")
        self.assertEqual(len(merge_records([a, b]).records), 1)

    def test_title_year_duplicate_without_doi(self):
        a = make("RIS-A", "1", "Adolescent wellbeing and digital health 2019")
        b = make("RIS-B", "2", "Adolescent well-being and digital health 2019")
        # punctuation normalization differs here: "wellbeing" and "well being"
        m = merge_records([a, b])
        self.assertEqual(len(m.records), 2)

    def test_exact_normalized_title_year(self):
        a = make("RIS-A", "1", "Adolescent wellbeing: social media use")
        b = make("RIS-B", "2", "Adolescent Wellbeing – Social Media Use!")
        self.assertEqual(len(merge_records([a, b]).records), 1)

    def test_distinct_doi_prevents_wrong_title_merge(self):
        a = make("RIS-A", "1", "Adolescent wellbeing: social media use",
                 doi="10.1111/aaa")
        b = make("RIS-B", "2", "Adolescent Wellbeing – Social Media Use!",
                 doi="10.2222/bbb")
        m = merge_records([a, b])
        self.assertEqual(len(m.records), 2)

    def test_near_duplicate_only_manual(self):
        a = make("PubMed", "1", "Social media and academic achievement in students")
        b = make("Scopus", "2", "Social media and academic achievement in student")
        m = merge_records([a, b])
        self.assertEqual(len(m.records), 2)
        self.assertEqual(len(m.fuzzy_review), 1)
        self.assertEqual(m.fuzzy_review[0]["status"], "MANUAL REVIEW - NOT merged")

    def test_dedup_reports_stage_progress_and_preserves_groups(self):
        calls = []
        source = [
            make("PubMed", "1", "Social media use in college students",
                 doi="10.1234/abc"),
            make("OpenAlex", "2", "Social media use in college students",
                 doi="10.1234/abc"),
            make("Scopus", "3", "Completely different title on education",
                 doi="10.5678/xyz"),
        ]
        out = merge_records(
            source,
            progress=lambda stage, done, total: calls.append(
                (stage, done, total)),
        )
        self.assertEqual(out.duplicates_removed, 1)
        for stage in (
            "Identifier abgleichen",
            "Metadaten zusammenführen",
            "Ähnliche Titel prüfen",
        ):
            self.assertTrue(
                any(name == stage and done == total for name, done, total in calls),
                stage,
            )
        self.assertFalse(out.fuzzy_skipped)

    def test_large_year_group_skipped_with_visible_metadata(self):
        studies = [
            make("RIS", str(n), f"Unique study title of adolescent behavior {n}")
            for n in range(255)
        ]
        out = merge_records(studies)
        self.assertEqual(len(out.records), len(studies))
        self.assertEqual(out.fuzzy_skipped["2025"], 255)

    def test_ris_import_roundtrip_and_source(self):
        raw = b'TY  - JOUR\nTI  - Academic achievement and social media\nAB  - Observational analysis.\nDO  - 10.1000/abc\nPY  - 2022\nER  -\n'
        ris, _ = parse_ris(raw)
        parsed = from_ris(ris, "RIS: Web of Science")
        self.assertEqual(parsed[0].doi, "10.1000/abc")
        self.assertEqual(parsed[0].sources, ["RIS: Web of Science"])
        exported = ris_bytes(parsed)
        self.assertIn(b"N1  - SearchBridge sources: RIS: Web of Science", exported)
        self.assertIn(b"DO  - 10.1000/abc", exported)

    def test_lexical_rank_does_not_discard_records(self):
        a = make("RIS", "1", "University students and social media use",
                 abstract="Students' academic outcomes and self concept.")
        b = make("RIS", "2", "Plant biology investigation",
                 abstract="Cytoplasmic proteins in algae.")
        ranked = lexical_rank([b, a], "social media university students")
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0].uid, a.uid)
        self.assertGreater(ranked[0].rank_score, ranked[1].rank_score)
        self.assertEqual(ranked[1].rank_score, 0)

    def test_openalex_inverted_abstract(self):
        items = [{"id": "https://openalex.org/W1", "title": "A title",
                  "publication_year": 2024, "doi": "https://doi.org/10.3333/abc",
                  "abstract_inverted_index": {"social": [0], "media": [1]},
                  "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/123456/"}}]
        records = from_openalex_results(items)
        self.assertEqual(records[0].abstract, "social media")
        self.assertEqual(records[0].pmid, "123456")
        self.assertEqual(records[0].doi, "10.3333/abc")

    def test_scopus_standard_no_abstract(self):
        records = from_scopus_results([{
            "dc:identifier": "SCOPUS_ID:1", "dc:title": "Digital literacy",
            "prism:doi": "10.1000/abc", "prism:coverDate": "2021-05-10",
            "dc:creator": "Smith J",
        }])
        self.assertEqual(records[0].year, "2021")
        self.assertEqual(records[0].abstract, "")
        self.assertEqual(records[0].authors, "Smith J")

    def test_csv_with_provenance(self):
        row = make("PubMed", "1", "Title of study")
        output = csv_bytes([row], {
            row.uid: {"decision":"manual_review", "reason":"not enough detail"}})
        self.assertTrue(output.startswith(b"\xef\xbb\xbf"))
        rows = list(csv.DictReader(io.StringIO(output.decode("utf-8-sig"))))
        self.assertEqual(rows[0]["sources"], "PubMed")
        self.assertEqual(rows[0]["screen_decision"], "manual_review")


class MockResponse:
    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("Fake HTTP error")


class TestRetrieval(unittest.TestCase):
    def test_openalex_oql_limit_and_metadata(self):
        class Session:
            def post(self, *args, **kwargs):
                self.payload = kwargs["json"]
                return MockResponse({
                    "meta": {"count": 3},
                    "results": [
                        {"id": "W1", "title": "Paper A"},
                        {"id": "W2", "title": "Paper B"},
                    ],
                })
        session = Session()
        with patch("review_retrieval._http", return_value=session):
            result = openalex_search("get works about science", limit=2)
        self.assertEqual(result.total, 3)
        self.assertEqual(len(result.records), 2)
        self.assertFalse(result.complete)
        self.assertEqual(session.payload["per_page"], 2)
        with self.assertRaises(ValueError):
            openalex_search("get works about science", limit=101)

    def test_pubmed_progress_forwarded_to_fetcher(self):
        from types import SimpleNamespace

        def fake_fetch(query, limit, email, api_key, progress):
            progress(8, 20)
            return SimpleNamespace(records=[], candidate_count=20)

        observed = []
        with patch("review_retrieval.fetch_pubmed_candidates",
                   side_effect=fake_fetch):
            out = pubmed_search(
                "Students[mh]", limit=20,
                progress=lambda done, total: observed.append((done, total)),
            )
        self.assertEqual(observed, [(8, 20)])
        self.assertFalse(out.complete)

    def test_openalex_page_count_progress(self):
        class Session:
            def post(self, *args, **kwargs):
                return MockResponse({
                    "meta": {"count": 5},
                    "results": [
                        {"id": "W1", "title": "Paper A"},
                        {"id": "W2", "title": "Paper B"},
                    ],
                })

        observed = []
        with patch("review_retrieval._http", return_value=Session()):
            out = openalex_search(
                "get works about science", limit=2,
                progress=lambda done, total: observed.append((done, total)),
            )
        self.assertEqual(len(out.records), 2)
        self.assertEqual(observed, [(2, 2)])

    def test_scopus_key_and_metadata(self):
        with self.assertRaises(ValueError):
            scopus_search("TITLE(education)", limit=10, api_key="")
        class Session:
            def get(self, *args, **kwargs):
                self.headers = kwargs["headers"]
                return MockResponse({"search-results": {
                    "opensearch:totalResults": "1",
                    "entry": [{"dc:identifier":"SCOPUS_ID:1",
                               "dc:title":"Education outcomes"}]}})
        session = Session()
        with patch("review_retrieval._http", return_value=session):
            r = scopus_search("TITLE(education)", limit=10,
                              api_key="test-scopus-key")
        self.assertEqual(r.total, 1)
        self.assertEqual(len(r.records), 1)
        self.assertEqual(session.headers["X-ELS-APIKey"], "test-scopus-key")


    def test_scopus_page_count_progress(self):
        class Session:
            def get(self, *args, **kwargs):
                return MockResponse({
                    "search-results": {
                        "opensearch:totalResults": "1",
                        "entry": [{
                            "dc:identifier": "SCOPUS_ID:77",
                            "dc:title": "Learning and attention",
                        }],
                    },
                })

        observed = []
        with patch("review_retrieval._http", return_value=Session()):
            result = scopus_search(
                "TITLE(attention)", limit=3, api_key="test",
                progress=lambda done, total: observed.append((done, total)),
            )
        self.assertEqual(result.total, 1)
        self.assertEqual(observed, [(1, 1)])


class TestOpenAIScreen(unittest.TestCase):
    def test_missing_abstract_always_manual_without_network(self):
        r = make("RIS", "1", "Academic social media report")
        out = screen_one(
            r, criteria="Include empirical studies of university students",
            key="not-a-real-key",
        )
        self.assertEqual(out["decision"], "manual_review")
        self.assertEqual(out["origin"], "missing_metadata_rule")

    def test_structured_output_and_store_false(self):
        class Session:
            def post(self, *args, **kwargs):
                self.sent = kwargs["json"]
                answer = {
                    "decision": "include",
                    "reason": "All criteria explicitly supported.",
                    "evidence": "survey of university students",
                    "uncertainty": "low",
                }
                return MockResponse({"output": [{
                    "type":"message", "content":[{
                        "type":"output_text", "text": json.dumps(answer)
                    }]
                }]})
        s = Session()
        record = make(
            "RIS", "1", "University study",
            abstract="A survey of university students assessed social media.")
        out = screen_one(
            record, criteria="Include empirical studies of university students",
            key="fake", session=s,
        )
        self.assertEqual(out["decision"], "include")
        self.assertFalse(s.sent["store"])
        self.assertEqual(s.sent["text"]["format"]["strict"], True)
        self.assertEqual(s.sent["model"], "gpt-6.1-terra")
        self.assertEqual(s.sent["reasoning"]["effort"], "low")

    def test_nonverbatim_model_evidence_requires_manual_review(self):
        class Session:
            def post(self, *args, **kwargs):
                answer = {
                    "decision": "exclude",
                    "reason": "Wrong population",
                    "evidence": "Children were enrolled",
                    "uncertainty": "low",
                }
                return MockResponse({"output": [{
                    "type": "message", "content": [{
                        "type": "output_text", "text": json.dumps(answer)
                    }]
                }]})
        study = make("RIS", "1", "University investigation",
                     abstract="University students completed questionnaires.")
        result = screen_one(
            study, criteria="Include university student investigations",
            key="fake", session=Session())
        self.assertEqual(result["decision"], "manual_review")
        self.assertEqual(result["uncertainty"], "high")

    def test_refusal_not_false_exclusion(self):
        class Session:
            def post(self, *args, **kwargs):
                return MockResponse({"output": [{
                    "type":"message", "content":[{
                        "type":"refusal", "refusal": "Cannot help"
                    }]
                }]})
        record = make("RIS", "1", "Study", abstract="Some abstract text.")
        with self.assertRaises(OpenAIScreenError):
            screen_one(record, criteria="Empirical studies of university students",
                       key="fake", session=Session())

    def test_screen_batch_failures_manual_review(self):
        a = make("RIS", "1", "A", abstract="Some abstract text.")
        b = make("RIS", "2", "B")
        def fake_one(r, **kwargs):
            if r.uid == a.uid:
                raise OpenAIScreenError("API error")
            return {"decision": "manual_review", "reason":"No abstract"}
        with patch("openai_review_screening.screen_one", side_effect=fake_one):
            result = screen_batch([a, b], key="fake",
                                   criteria="Empirical studies of university students")
        self.assertEqual(result.attempted, 2)
        self.assertEqual(result.decisions[a.uid]["decision"], "manual_review")
        self.assertEqual(len(result.errors), 1)


if __name__ == "__main__":
    unittest.main()
