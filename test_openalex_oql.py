"""Regression checks for compact, native Ovid→OpenAlex OQL translation."""
import json
import re
import unittest
from openalex_oql import translate_openalex_oql, OQLConversionError


class OpenAlexOQLTests(unittest.TestCase):
    def test_adj2_cartesian_four_pairs(self):
        q = '((college OR university) adj2 (student* OR graduate*)).ti,ab,kf.'
        result = translate_openalex_oql(q)
        text = result["query"]
        self.assertEqual(text.count("within 1 words of each other"), 4)
        self.assertIn('("college" and "student*" within 1 words of each other)', text)
        self.assertIn('("university" and "graduate*" within 1 words of each other)', text)
        self.assertIn("title-abstract-keywords has", text)
        self.assertEqual(result["removed"], ["Ovid field codes without exact mappings: ['ab', 'kf', 'ti']."])
        self.assertFalse(result["validated_equivalence"])

    def test_adj3_wildcard_and_exact_phrase(self):
        q='(("social media" OR web*) adj3 (platform* OR wom?n)).ti,ab.'
        out=translate_openalex_oql(q)
        self.assertEqual(out["query"].count("within 2 words of each other"), 2)
        self.assertIn('("social media" and "platform*" within 2 words of each other)',out["query"])
        self.assertIn('("web*" AND "wom?n")',out["query"])
        self.assertIn("title-abstract has",out["query"])

    def test_flatten_long_or_chain(self):
        words=[f"concept{i}" for i in range(30)]
        q="("+" OR ".join(words)+").ti,ab."
        out=translate_openalex_oql(q)
        self.assertEqual(out["query"].count(" OR "),29)
        self.assertNotIn("((((((",out["query"])

    def test_openalex_two_wildcard_minimum_prefix_rule(self):
        q = '((Web* OR social) adj3 (platform* OR network*)).ti,ab,kf.'
        out = translate_openalex_oql(q)
        self.assertEqual(out["query"].count("within 2 words of each other"), 2)
        self.assertIn('("Web*" AND "platform*")', out["query"])
        self.assertTrue(any("Web*" in loss and "broad AND" in loss for loss in out["removed"]))
        self.assertFalse(out["validated_equivalence"])

    def test_heading_kept_as_unverified_freetext(self):
        out=translate_openalex_oql(
            '(exp Students/ OR exp "Self Concept"/px) AND social.ti,ab.',
            explosions={"students":["Students","Student Dropouts"]})
        self.assertIn('"Student Dropouts"',out["query"])
        self.assertIn('"Self Concept"',out["query"])
        self.assertIn("exp Students/",out["unverified"])
        self.assertTrue(any("Qualifier" in x for x in out["removed"]))

    def test_reject_untranslatable_complex_proximity(self):
        with self.assertRaises(OQLConversionError):
            translate_openalex_oql('((a AND b) adj3 c).ti,ab.')

    def test_post_when_url_long_preserves_every_term(self):
        terms=["psychology"+str(i) for i in range(160)]
        q='((young OR juvenile) adj2 ('+" OR ".join(terms)+')).ti,ab,kf.'
        out=translate_openalex_oql(q)
        self.assertTrue(out["uses_post"])
        self.assertEqual(out["url"],"")
        self.assertEqual(json.loads(out["request_json"])["oql"],out["query"])
        self.assertIn(terms[-1],out["query"])
        self.assertGreater(out["query"].count("within 1 words"),300-1)

    def test_complete_user_strategy(self):
        q="""(((College or university) adj2 (student* or graduate* or undergraduate* or postgraduate*)).ti,ab,kf. or ((young adj2 (adult* or person* or individual* or people or population* or man or men or wom?n)).ti,ab,kf. or young adult/ or exp Students/))
And
(((Academic or university or scholastic or study or scholarly or college) adj2 (achievement or performance or success or career or drop out or grade* or fail* or Self-percept* or self-image* or self-worth or self-assessment* or self-belie* or self-confiden* or self-evaluati* or self-regard* or self-concept*)).ti,ab,kf. or achievement/ or student dropouts/ or underachievement/ or academic performance/ or academic success/ or college admission test/ or professional competence/ or test taking skills/ or (exp Self Concept/ and exp Education/))
And
(((Online or Digital or social or Web* or communicat* or Internet* or virtual or sharing or cyber*) adj3 (Platform* or network* or communit* or forum)).ti,ab,kf. or (social media or Facebook or Twitter or Instagram or LinkedIn or YouTube or Pinterest or Snapchat or TikTok or Reddit or WhatsApp or Tumblr or Flickr or Quora or WeChat or Vine or Periscope or Vimeo).ti,ab,kf. or exp Social Media/ or social networking/ or online social networking/)"""
        out=translate_openalex_oql(q)
        self.assertIn("get works where title-abstract-keywords has",out["query"])
        self.assertEqual(out["query"].count("within 1 words of each other"),8+8+96)
        self.assertEqual(out["query"].count("within 2 words of each other"),33)
        self.assertEqual(out["query"].count('"Web*" AND '),3)
        self.assertFalse(out["proximity_preserved"])
        self.assertTrue(out["uses_post"])
        self.assertIn('"wom?n"',out["query"])
        self.assertEqual(len(out["unverified"]),15)
        self.assertNotIn("(((((((",out["query"])
        self.assertGreater(len(out["query"]),8190)


if __name__=="__main__":
    unittest.main()
