"""Mocked unit tests for MeSH RDF EXP resolution (no network calls)."""
import csv
import io
import unittest
from unittest.mock import Mock

from mesh_expansion import (
    ExplodedHeading, MeSHError, find_exp_headings, lookup_exact_descriptor,
    expand_mesh_heading, labels_as_text, csv_export, display_rows
)

PREFIX = "http://id.nlm.nih.gov/mesh/"


def b(value):
    return {"type": "uri" if str(value).startswith("http") else "literal", "value": value}


def hit(mesh_id: str, label: str, root: str, child: str):
    return {"descriptor": b(PREFIX + mesh_id), "label": b(label),
            "rootTree": b(PREFIX + root), "childTree": b(PREFIX + child)}


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class MockNLM:
    def __init__(self, *page_rows):
        self.requests = []
        self.pages = page_rows

    def get(self, url, params, timeout):
        self.requests.append((url, dict(params), timeout))
        if "lookup/descriptor" in url:
            return Response([{"resource": PREFIX + "D012345",
                              "label": "Students"}])
        if "sparql" in url:
            if "SELECT DISTINCT ?rootTree" in params["query"]:
                return Response({"results": {"bindings": [
                    {"rootTree": b(PREFIX + "M01.750")},
                    {"rootTree": b(PREFIX + "M02.100")},
                ]}})
            query = params["query"]
            import re
            offset = int(re.search(r"OFFSET (\d+)", query).group(1))
            page_size = int(re.search(r"LIMIT (\d+)", query).group(1))
            all_rows = self.pages[0] if self.pages else []
            return Response({"results": {"bindings": all_rows[offset:offset+page_size]}})
        raise AssertionError(url)


class MeSHExpansionTests(unittest.TestCase):
    def test_read_exp_headings_from_complex_ovid_query(self):
        text = """((students adj2 achievement).ti,ab,kf. OR exp Students/
            OR exp "Self Concept"/ OR exp Education/ OR exp Social Media/
            OR exp students/) AND (academic performance/ OR social media.ti.)"""
        labels = [h.name for h in find_exp_headings(text)]
        self.assertEqual(labels, ["Students", "Self Concept", "Education", "Social Media"])
        self.assertEqual(find_exp_headings("(exp Students/ OR study.ti.)")[0].label,
                         "exp Students/")

    def test_exact_lookup_uses_preferred_label(self):
        mock = MockNLM()
        self.assertEqual(lookup_exact_descriptor("Students", mock),
                         ("D012345", "Students"))
        url, params, timeout = mock.requests[0]
        self.assertIn("lookup/descriptor", url)
        self.assertEqual(params["match"], "exact")
        self.assertEqual(params["year"], "current")
        self.assertGreaterEqual(timeout[1], 60)

    def test_full_hierarchy_and_deduplication(self):
        # Descriptor D100001 appears in both MeSH tree paths:
        data = [
            hit("D100001", "College Students", "M01.750", "M01.750.101"),
            hit("D100001", "College Students", "M02.100", "M02.100.512"),
            hit("D100002", "Graduate Students", "M01.750", "M01.750.101.102"),
        ]
        mock = MockNLM(data)
        exp = expand_mesh_heading(ExplodedHeading("Students"), session=mock, page_size=2)
        self.assertEqual(exp.mesh_id, "D012345")
        self.assertEqual(exp.children_count, 2)
        self.assertEqual(exp.descriptor_count, 3)
        self.assertEqual(exp.root_trees, ("M01.750", "M02.100"))
        self.assertEqual(len(exp.positions), 5)
        self.assertEqual(sorted(x.level for x in exp.positions), [0, 0, 1, 1, 2])
        self.assertTrue(all(p.mesh_url.startswith("https://id.nlm.nih.gov/mesh/D")
                            for p in exp.positions))
        self.assertEqual(len(mock.requests), 4)  # lookup + 2 paginated queries + roots
        self.assertEqual(len(display_rows(exp)), 5)
        unique = labels_as_text(exp).splitlines()
        self.assertEqual(unique[0], "Students")
        self.assertEqual(set(unique), {"Students", "College Students", "Graduate Students"})
        out = list(csv.DictReader(io.StringIO(csv_export(exp).decode("utf-8-sig"))))
        self.assertEqual(len(out), 5)
        self.assertEqual(out[0]["Relation"], "Source descriptor")

    def test_heading_without_descendants(self):
        mock = MockNLM([])
        result = expand_mesh_heading(ExplodedHeading("Students"), session=mock)
        self.assertEqual(result.children_count, 0)
        self.assertEqual(result.descriptor_count, 1)
        self.assertEqual(len(result.positions), 2)  # one row per root tree number

    def test_fail_closed_if_sparql_truncated(self):
        data = [hit("D100001", "A", "M01.750", "M01.750.100"),
                hit("D100002", "B", "M01.750", "M01.750.200")]
        with self.assertRaises(MeSHError) as captured:
            expand_mesh_heading(ExplodedHeading("Students"), session=MockNLM(data),
                                page_size=2, max_positions=2)
        self.assertIn("incomplete hierarchy", str(captured.exception))

    def test_no_fuzzy_match_substitution(self):
        class WrongLookup(MockNLM):
            def get(self, url, params, timeout):
                if "lookup" in url:
                    return Response([{"resource": PREFIX + "D012345",
                                      "label": "College Students"}])
                raise AssertionError("SPARQL must not be called after a failed exact lookup")
        with self.assertRaises(MeSHError):
            lookup_exact_descriptor("Students", WrongLookup())

    def test_unexpected_tree_response_fails(self):
        mock = MockNLM([hit("D100001", "Something",
                           "M01.750", "M02.999")])
        with self.assertRaises(MeSHError):
            expand_mesh_heading(ExplodedHeading("Students"), session=mock)


if __name__ == "__main__":
    unittest.main()
