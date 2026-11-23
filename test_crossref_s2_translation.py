"""Regression tests for honest Crossref discovery and S2 Bulk Boolean conversion."""
import unittest
from urllib.parse import urlparse, parse_qs

from crossref_translation import translate_crossref
from semantic_scholar_translation import translate_semantic_scholar


def balanced(s):
    depth = 0
    in_quotes = False
    for c in s:
        if c == '"':
            in_quotes = not in_quotes
        if not in_quotes:
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            if depth < 0:
                return False
    return depth == 0 and not in_quotes


class CrossrefAndSemanticScholarTests(unittest.TestCase):
    def test_crossref_short_anchors_no_fake_boolean(self):
        q = """(((College OR university) adj2 (student* OR graduate*)).ti,ab,kf.)
        AND (((academic OR scholarly) adj2 (performance OR achievement)).ti,ab,kf.)
        AND (((social OR online) adj3 (platform* OR network*)).ti,ab,kf.)"""
        result = translate_crossref(q)
        self.assertEqual(result["query"], "College student academic performance social platform")
        self.assertEqual(len(result["anchors"]), 3)
        self.assertLess(len(result["query"].split()), 11)
        self.assertGreater(result["source_term_count"], 6)
        self.assertNotIn("AND", result["query"])
        self.assertNotIn("OR", result["query"])
        self.assertIn("query.bibliographic", result["url"])
        self.assertEqual(
            parse_qs(urlparse(result["url"]).query)["query.bibliographic"][0],
            result["query"],
        )
        self.assertTrue(all(a["url"].startswith("https://api.crossref.org")
                            for a in result["anchors"]))
        self.assertFalse(result["validated_equivalence"])
        self.assertTrue(any("20%" in n for n in result["notes"]))

    def test_crossref_not_never_appears_in_positive_probe(self):
        result = translate_crossref('(students.ti,ab. NOT (children OR animals).ti,ab.)')
        self.assertEqual(result["query"], "students")
        self.assertEqual(set(result["excluded_not_terms"]), {"children", "animals"})
        self.assertNotIn("children", result["query"])
        self.assertTrue(any("NOT" in n for n in result["notes"]))

    def test_crossref_large_or_does_not_encode_entire_synonym_list(self):
        q = "(" + " OR ".join(f"psychology{i}" for i in range(140)) + ").ti,ab."
        r = translate_crossref(q)
        self.assertEqual(r["query"], "psychology0")
        self.assertEqual(r["source_term_count"], 140)
        self.assertLess(len(r["url"]), 350)

    def test_semantic_scholar_flat_or_and_and(self):
        q = """(((College OR university) adj2 (student* OR graduate*)).ti,ab,kf.)
        AND ((academic OR scholarly) adj3 (performance OR achievement)).ti,ab,kf.
        AND (social OR online OR digital OR cyber* OR internet).ti,ab,kf."""
        r = translate_semantic_scholar(q)
        self.assertTrue(balanced(r["query"]))
        self.assertEqual(r["query"].count(" | "), 1 + 1 + 1 + 1 + 4)
        self.assertTrue(r["query"].startswith("(+("))
        self.assertIn("College | university", r["query"])
        self.assertIn("student* | graduate*", r["query"])
        self.assertNotIn("ADJ2", r["query"])
        self.assertTrue(any("ADJ2" in s for s in r["removed"]))
        self.assertFalse(r["validated_equivalence"])
        self.assertFalse(r["proximity_preserved"])
        self.assertIn("paper/search/bulk", r["url"])
        self.assertNotIn("(((((((", r["query"])

    def test_semantic_scholar_question_wildcard_and_hyphen(self):
        q = '((wom?n OR self-concept* OR "drop out") AND child*).ti,ab,kf.'
        r = translate_semantic_scholar(q)
        self.assertIn("wom*", r["query"])
        self.assertNotIn("wom?n", r["query"])
        self.assertIn("(+self +concept*)", r["query"])
        self.assertIn('"drop out"', r["query"])
        self.assertTrue(any("Bindestrich" in n for n in r["notes"]))
        self.assertTrue(balanced(r["query"]))

    def test_semantic_scholar_not(self):
        q = '(student* NOT child*).ti,ab.'
        r = translate_semantic_scholar(q)
        self.assertIn(" -(child*)", r["query"])
        self.assertTrue(balanced(r["query"]))

    def test_semantic_scholar_long_url_suppressed_not_truncated(self):
        q = "(" + " OR ".join("psychologystudies"+str(i)+"longprefix" for i in range(330)) + ").ti,ab."
        r = translate_semantic_scholar(q)
        self.assertEqual(r["url"], "")
        self.assertIn("psychologystudies329longprefix", r["query"])
        self.assertTrue(any("GET-URL" in n for n in r["notes"]))
        self.assertTrue(balanced(r["query"]))


if __name__ == "__main__":
    unittest.main()
