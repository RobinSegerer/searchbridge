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
        self.assertEqual(r['Ovid PsycINFO']['removed'], ['Students/'])
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

if __name__ == '__main__':
    unittest.main()
