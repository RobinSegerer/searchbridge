"""PubMed translation regression: MeSH explosion, fields, proximity and wildcards."""
import unittest
from urllib.parse import parse_qs, urlparse

from pubmed_translation import translate_pubmed, PubMedConversionError


FULL = """(((College or university) adj2 (student* or graduate* or undergraduate* or postgraduate*)).ti,ab,kf. or ((young adj2 (adult* or person* or individual* or people or population* or man or men or wom?n)).ti,ab,kf. or young adult/ or exp Students/))
And
(((Academic or university or scholastic or study or scholarly or college) adj2 (achievement or performance or success or career or drop out or grade* or fail* or Self-percept* or self-image* or self-worth or self-assessment* or self-belie* or self-confiden* or self-evaluati* or self-regard* or self-concept*)).ti,ab,kf. or achievement/ or student dropouts/ or underachievement/ or academic performance/ or academic success/ or college admission test/ or professional competence/ or test taking skills/ or (exp Self Concept/ and exp Education/))
And
(((Online or Digital or social or Web* or communicat* or Internet* or virtual or sharing or cyber*) adj3 (Platform* or network* or communit* or forum)).ti,ab,kf. or (social media or Facebook or Twitter or Instagram or LinkedIn or YouTube or Pinterest or Snapchat or TikTok or Reddit or WhatsApp or Tumblr or Flickr or Quora or WeChat or Vine or Periscope or Vimeo).ti,ab,kf. or exp Social Media/ or social networking/ or online social networking/)"""


class TestPubMedTranslation(unittest.TestCase):
    def test_exp_vs_noexp_mesh(self):
        result = translate_pubmed('(exp Students/ OR "Self Concept"/ OR young adult/)')
        q = result["query"]
        self.assertIn('"Students"[mh]', q)
        self.assertIn('"Self Concept"[mh:noexp]', q)
        self.assertIn('"young adult"[mh:noexp]', q)
        self.assertNotIn("exp Students/", q)
        self.assertEqual(result["removed"], [])

    def test_explicit_field_mapping_and_no_atm(self):
        a = translate_pubmed('(academic OR performance).ti,ab,kf.')["query"]
        self.assertIn("academic[tiab]", a)
        self.assertIn("performance[tiab]", a)
        b = translate_pubmed('(academic OR performance).ti,ab.')["query"]
        self.assertIn("academic[ti]", b)
        self.assertIn("academic[ab]", b)
        self.assertNotIn("academic[tiab]", b)
        c = translate_pubmed('(student OR college).kf.')["query"]
        self.assertIn("student[ot]", c)
        self.assertNotIn("student[tiab]", c)

    def test_true_proximity_for_nonwildcard_pairs(self):
        r = translate_pubmed('((college OR university) adj2 (student OR graduate)).ti,ab,kf.')
        q = r["query"]
        self.assertIn('"college student"[tiab:~1]', q)
        self.assertIn('"university graduate"[tiab:~1]', q)
        self.assertEqual(q.count('[tiab:~1]'), 4)
        self.assertTrue(r["proximity_preserved"])
        self.assertTrue(any("ADJ2" in n for n in r["notes"]))

    def test_adj1_and_ti_ab_precision(self):
        r = translate_pubmed('(university adj1 students).ti,ab.')
        self.assertEqual(
            r["query"],
            '("university students"[ti:~0] OR "university students"[ab:~0])'
        )

    def test_wildcards_in_proximity_fall_back_to_broad_candidate(self):
        r = translate_pubmed('((college OR university) adj2 (student* OR graduate*)).ti,ab,kf.')
        q = r["query"]
        self.assertNotIn("[tiab:~", q)
        self.assertIn(" AND ", q)
        self.assertIn("student*[tiab]", q)
        self.assertFalse(r["proximity_preserved"])
        self.assertTrue(any("ADJ2" in x for x in r["removed"]))

    def test_explicit_woman_women_from_ovid_singlechar(self):
        r = translate_pubmed('(young adj2 wom?n).ti,ab,kf.')
        self.assertIn('"young woman"[tiab:~1]', r["query"])
        self.assertIn('"young women"[tiab:~1]', r["query"])
        self.assertEqual(r["query"].count('[tiab:~1]'), 2)
        self.assertIn("wom?n", " ".join(r["notes"]))

    def test_short_prefix_web_is_not_illegal_pubmed_wildcard(self):
        r = translate_pubmed('(Web* OR Internet* OR social).ti,ab,kf.')
        self.assertNotIn('Web*[tiab]', r["query"])
        self.assertIn('web[tiab]', r["query"])
        self.assertIn('website*[tiab]', r["query"])
        self.assertIn('Internet*[tiab]', r["query"])
        self.assertTrue(any("NICHT vollständig äquivalent" in x for x in r["notes"]))

    def test_unknown_short_prefix_requires_human_action(self):
        with self.assertRaises(PubMedConversionError):
            translate_pubmed('(eco* AND university).ti,ab,kf.')

    def test_unsupported_qualifier_requires_mapping_not_loss(self):
        with self.assertRaises(PubMedConversionError):
            translate_pubmed('(exp "Self Concept"/px AND academic.ti,ab.)')

    def test_full_original_query(self):
        r = translate_pubmed(FULL)
        q = r["query"]
        self.assertGreater(len(q), 500)
        self.assertEqual(q.count('[mh]'), 4)  # Students, Self Concept, Education, Social Media
        self.assertEqual(q.count('[mh:noexp]'), 11)
        self.assertIn('"young adult"[mh:noexp]', q)
        self.assertIn('"Self Concept"[mh]', q)
        self.assertIn('"Social Media"[mh]', q)
        self.assertIn("woman[tiab]", q)
        self.assertIn("women[tiab]", q)
        self.assertIn("website*[tiab]", q)
        self.assertNotIn("wom?n", q)
        self.assertNotIn("Web*[tiab]", q)
        self.assertNotIn("(((((((", q)
        self.assertTrue(r["requires_post"])
        self.assertEqual(r["url"], "")
        self.assertFalse(r["proximity_preserved"])
        self.assertTrue(len(r["removed"]) >= 4)
        self.assertFalse(r["validated_equivalence"])

    def test_short_url_roundtrip(self):
        r = translate_pubmed('(student OR college).ti,ab,kf.')
        self.assertTrue(r["url"].startswith("https://pubmed.ncbi.nlm.nih.gov"))
        self.assertEqual(parse_qs(urlparse(r["url"]).query)["term"][0], r["query"])


if __name__ == "__main__":
    unittest.main()
