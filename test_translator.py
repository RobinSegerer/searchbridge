import unittest
from translator_core import parse_ovid, translate, ParseError

class TranslationTests(unittest.TestCase):
    def test_example_explode_removed_preserves_boolean(self):
        q = '(exp Self Concept/ AND exp Education/) OR (academic adj2 self-concept*).ti,ab,kf.'
        s = translate(q)
        a = s['Scopus']['query']
        self.assertIn('W/1', a)
        self.assertIn('academic', a)
        self.assertNotIn('Self Concept', a)
        self.assertNotIn(' OR )', a)
        self.assertNotIn('AND AND', a)
        self.assertTrue(a.startswith('TITLE-ABS-KEY('))
        self.assertIn('NEAR/1', s['Web of Science']['query'])
        self.assertIn('NEAR/2', s['Elsevier Embase.com']['query'])
    def test_psycinfo_field(self):
        q = '((college OR university) adj2 student*).ti,ab,kf. OR Students/'
        r = translate(q)
        self.assertIn('.ti,ab,id.', r['Ovid PsycINFO']['query'])
        self.assertIn(':ti,ab,kw', r['Elsevier Embase.com']['query'])
        self.assertEqual(r['Ovid PsycINFO']['removed'], [])
        self.assertIn('Students/', r['Ovid PsycINFO']['query'])
        self.assertEqual(r['Ovid PsycINFO']['unverified'], ['Students/'])
        self.assertIn("('Students'):ti,ab,kw", r['Elsevier Embase.com']['query'])
    def test_custom_mapping(self):
        r = translate('exp Students/ OR youth*.ti,ab,kf.', {'Ovid PsycINFO': {'Students': 'College Students'}})
        self.assertIn('exp "College Students"/', r['Ovid PsycINFO']['query'])
    def test_source_example(self):
        q = '''(((College or university) adj2 (student* or graduate* or undergraduate* or postgraduate*)).ti,ab,kf. or (young adult/ or exp Students/)) AND (((academic or university) adj2 (achievement or performance or self-concept*)).ti,ab,kf. or (exp Self Concept/ and exp Education/)) AND (((online or social) adj3 (platform* or network*)).ti,ab,kf. or exp Social Media/)'''
        r=translate(q)
        for target in ('Scopus','Web of Science','Ovid PsycINFO','Elsevier Embase.com'):
            self.assertTrue(r[target]['query'])
            self.assertNotIn(' AND AND ', r[target]['query'])
        self.assertEqual(len(r['Scopus']['removed']), 5)
    def test_invalid(self):
        with self.assertRaises(ParseError):
            parse_ovid('(student* OR (adolescent* AND crime*)')
    def test_phrase_and_quoted(self):
        r=translate('(social media or "young adult" or TikTok).ti,ab,kf.')
        self.assertIn('"social media"', r['Scopus']['query'])
        self.assertIn("'social media'", r['Elsevier Embase.com']['query'])


    def test_full_user_strategy_retains_subject_headings(self):
        q = "(((College or university) adj2 (student* or graduate* or undergraduate* or postgraduate*)).ti,ab,kf. or ((young adj2 (adult* or person* or individual* or people or population* or man or men or wom?n)).ti,ab,kf. or young adult/ or exp Students/)) \nAnd\n(((Academic or university or scholastic or study or scholarly or college) adj2 (achievement or performance or success or career or drop out or grade* or fail* or Self-percept* or self-image* or self-worth or self-assessment* or self-belie* or self-confiden* or self-evaluati* or self-regard* or self-concept*)).ti,ab,kf. or achievement/ or student dropouts/ or underachievement/ or academic performance/ or academic success/ or college admission test/ or professional competence/ or test taking skills/ or (exp Self Concept/ and exp Education/)) \nAnd\n(((Online or Digital or social or Web* or communicat* or Internet* or virtual or sharing or cyber*) adj3 (Platform* or network* or communit* or forum)).ti,ab,kf. or (social media or Facebook or Twitter or Instagram or LinkedIn or YouTube or Pinterest or Snapchat or TikTok or Reddit or WhatsApp or Tumblr or Flickr or Quora or WeChat or Vine or Periscope or Vimeo).ti,ab,kf. or exp Social Media/ or social networking/ or online social networking/)"
        results = translate(q)
        psy = results["Ovid PsycINFO"]
        embase = results["Elsevier Embase.com"]
        scopus = results["Scopus"]
        wos = results["Web of Science"]
        self.assertEqual(len(psy["unverified"]), 15)
        self.assertEqual(len(embase["unverified"]), 15)
        self.assertEqual(len(scopus["removed"]), 15)
        self.assertEqual(len(wos["removed"]), 15)
        self.assertEqual(psy["removed"], [])
        self.assertIn("young adult/", psy["query"])
        self.assertIn("exp Students/", psy["query"])
        self.assertIn("student dropouts/", psy["query"])
        self.assertIn("exp Self Concept/", psy["query"])
        self.assertIn("exp Education/", psy["query"])
        self.assertIn("exp Social Media/", psy["query"])
        self.assertIn(".ti,ab,id.", psy["query"])
        self.assertIn("('social networking'):ti,ab,kw", embase["query"])
        self.assertIn("('academic performance'):ti,ab,kw", embase["query"])
        self.assertIn("NEAR/2", embase["query"])
        self.assertIn("NEAR/1", wos["query"])
        self.assertIn("W/1", scopus["query"])
        self.assertIn("TITLE-ABS-KEY(", scopus["query"])
        self.assertIn("TS=(", wos["query"])
        self.assertNotIn("young adult/", scopus["query"])
        self.assertNotIn("Social Media/", wos["query"])
        # OR-group depth should be linear rather than an ever-growing parenthesis tower.
        self.assertNotIn("(((((((", psy["query"])
        self.assertNotIn("(((((((", scopus["query"])

if __name__ == '__main__':
    unittest.main()
