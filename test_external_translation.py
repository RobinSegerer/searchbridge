"""Offline regression tests for Ovid → OpenAlex, Crossref and S2 Bulk translation."""
import unittest
from urllib.parse import urlparse, parse_qs

from external_translation import translate_external
from translator_core import ParseError


class ExternalTranslationTests(unittest.TestCase):
    def test_basic_boolean_grouping_and_scopes(self):
        source = '((young OR juvenile) adj2 (offend* OR delinqu*)).ti,ab,kf. AND exp Students/'
        result = translate_external(source)
        oa = result["OpenAlex"]
        s2 = result["Semantic Scholar"]
        cr = result["Crossref"]
        self.assertEqual(len(result), 3)
        self.assertIn("OQL", oa["endpoint"])
        self.assertIn("young", oa["query"])
        self.assertIn("AND", oa["query"])
        self.assertIn("within 1 words of each other", oa["query"])
        self.assertIn("exp Students/", oa["unverified"])
        self.assertTrue(any("ADJ2" in x for x in oa["notes"]))
        self.assertIn("bulk", s2["endpoint"])
        self.assertIn("+((", s2["query"])
        self.assertIn(" | ", s2["query"])
        self.assertNotIn("ADJ2", s2["query"])
        self.assertTrue(any("Feldbeschränkung" in x for x in s2["removed"]))
        self.assertIn("query.bibliographic", cr["url"])
        self.assertNotIn(" AND ", cr["query"])
        self.assertIn("exploratory", cr["mode"])
        for v in result.values():
            self.assertFalse(v["validated_equivalence"])
            self.assertTrue(v["notes"])
            self.assertTrue(v["url"].startswith("https://"))

    def test_no_wildcard_scoped_title_only(self):
        q = '(mental OR health).ti.'
        result = translate_external(q)
        oa = result["OpenAlex"]
        self.assertEqual(oa["endpoint"], "GET /?oql= (OQL)")
        parsed = parse_qs(urlparse(oa["url"]).query)
        self.assertEqual(parsed["oql"][0], "get works where title has (mental OR health)")

    def test_mixed_fields_warns_and_broadens(self):
        q = '(education.ti. AND student*.ti,ab,kf.)'
        oa = translate_external(q)["OpenAlex"]
        self.assertIn("OQL", oa["endpoint"])
        self.assertTrue(any("Unterschiedliche Suchfeldgrenzen" in x for x in oa["notes"]))

    def test_me_sh_expansion_is_just_free_text(self):
        q = '(exp Students/ OR exp "Self Concept"/)'
        oa = translate_external(q, explosions={
            "students": ["Students", "Student Dropouts", "Students, Nursing"]
        })["OpenAlex"]
        self.assertIn('"Student Dropouts"', oa["query"])
        self.assertIn('"Students, Nursing"', oa["query"])
        self.assertIn('"Self Concept"', oa["query"])
        self.assertTrue(any("FREITEXT" in x for x in oa["notes"]))
        self.assertIn('exp "Self Concept"/', oa["unverified"])
        self.assertNotIn("mesh:", oa["query"].lower())

    def test_semantic_scholar_question_mark_broadens_and_notes(self):
        q = '(young adj2 wom?n).ti,ab.'
        s2 = translate_external(q)["Semantic Scholar"]
        self.assertIn("wom*", s2["query"])
        self.assertNotIn("wom?n", s2["query"])
        self.assertTrue(any("Wildcard" in x for x in s2["notes"]))
        self.assertIn("bulk", s2["url"])

    def test_crossref_flatten_is_not_boolean(self):
        q = '(juvenil* OR adolescent*) AND (offend* OR delinqu*)'
        cr = translate_external(q)["Crossref"]
        self.assertIn("juvenil", cr["query"])
        self.assertIn("adolescent", cr["query"])
        self.assertIn("offend", cr["query"])
        self.assertTrue(any("Boolesche" in x for x in cr["removed"]))
        self.assertTrue(any("20%" in x for x in cr["notes"]))
        self.assertNotIn(" OR ", cr["query"])
        self.assertNotIn(" AND ", cr["query"])

    def test_ovid_heading_subheading_warning(self):
        q = '(exp "Self Concept"/px AND internet.ti,ab.)'
        result = translate_external(q)
        self.assertIn("Self Concept", result["OpenAlex"]["query"])
        self.assertTrue(any("Qualifier" in x for x in result["OpenAlex"]["removed"]))

    def test_o_vid_boolean_not_translates(self):
        q = '(student* NOT child*).ti,ab.'
        result = translate_external(q)
        self.assertIn(" NOT ", result["OpenAlex"]["query"])
        self.assertIn(" -(", result["Semantic Scholar"]["query"])

    def test_openalex_url_limit_does_not_produce_clickable_url(self):
        q = "(" + " OR ".join("word" + str(i) + "pseudopsychology" for i in range(260)) + ").ti,ab,kf."
        result = translate_external(q)
        oa = result["OpenAlex"]
        self.assertEqual(oa["url"], "")
        self.assertTrue(oa["uses_post"])
        self.assertTrue(any("POST" in x for x in oa["notes"]))

    def test_existing_complex_query_support(self):
        q = """(((College or university) adj2 (student* or graduate* or undergraduate* or postgraduate*)).ti,ab,kf. or ((young adj2 (adult* or person* or individual* or people or population* or man or men or wom?n)).ti,ab,kf. or young adult/ or exp Students/))
And
(((Academic or university or scholastic or study or scholarly or college) adj2 (achievement or performance or success or career or drop out or grade* or fail* or Self-percept* or self-image* or self-worth or self-assessment* or self-belie* or self-confiden* or self-evaluati* or self-regard* or self-concept*)).ti,ab,kf. or achievement/ or student dropouts/ or underachievement/ or academic performance/ or academic success/ or college admission test/ or professional competence/ or test taking skills/ or (exp Self Concept/ and exp Education/))
And
(((Online or Digital or social or Web* or communicat* or Internet* or virtual or sharing or cyber*) adj3 (Platform* or network* or communit* or forum)).ti,ab,kf. or (social media or Facebook or Twitter or Instagram or LinkedIn or YouTube or Pinterest or Snapchat or TikTok or Reddit or WhatsApp or Tumblr or Flickr or Quora or WeChat or Vine or Periscope or Vimeo).ti,ab,kf. or exp Social Media/ or social networking/ or online social networking/)"""
        result = translate_external(q)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result["OpenAlex"]["unverified"]), 15)
        self.assertEqual(len(result["Semantic Scholar"]["unverified"]), 15)
        self.assertIn("Wom", result["OpenAlex"]["query"].title())
        self.assertTrue(result["OpenAlex"]["request_json"].startswith("{"))
        self.assertFalse(result["OpenAlex"]["proximity_preserved"])
        self.assertTrue(any("breites AND" in x for x in result["OpenAlex"]["removed"]))
        self.assertGreater(len(result["Crossref"]["query"]), 300)
        self.assertIn("student*", result["Semantic Scholar"]["query"])
        self.assertTrue(any("ADJ" in x for x in result["OpenAlex"]["notes"]))


if __name__ == "__main__":
    unittest.main()
