import unittest
from ris_proximity import parse_ris, extract_clauses, check_ris_records, best_in_text, csv_bytes, html_report, ris_bytes

QUERY = '((juvenil* OR young* OR adolescent* OR youth*) adj2 (perpetr* OR offend* OR crimin* OR delinqu* OR justice* OR convict* OR prison*)).ti,ab,kf.'

class RISProximityTests(unittest.TestCase):
    def test_parse_ris_and_three_adjacency_matches(self):
        raw = b'''TY  - JOUR\nTI  - Juvenile delinquency and family context\nAB  - Examines young adult offenders.\nKW  - Juvenile justice\nER  -\nTY  - JOUR\nTI  - Children of young parents\nAB  - Criminal justice in different places.\nKW  - social work\nER  -\nTY  - JOUR\nTI  - Interventions for adolescents\nAB  - Police officers and young adults work together.\nKW  - adolescent offenders\nER  -\n'''
        recs, enc = parse_ris(raw)
        self.assertEqual((len(recs), enc), (3, 'UTF-8'))
        c = extract_clauses(QUERY)[0]
        r = check_ris_records(recs, c)
        self.assertTrue(r[0]['matched'])
        self.assertEqual(r[0]['gap'], 0)
        self.assertFalse(r[1]['matched'])
        self.assertTrue(r[2]['matched'])
        self.assertIn('Keyword', r[2]['matched_fields'])
        self.assertTrue(csv_bytes(r).startswith(b'\xef\xbb\xbf'))
        self.assertIn(b'<mark>', html_report(recs, r, c))
        self.assertIn(b'TY  - JOUR', ris_bytes(recs))

    def test_adj2_one_gap_any_order(self):
        c = extract_clauses('(young adj2 offend*)')[0]
        self.assertIsNotNone(best_in_text('young adult offenders', c))
        self.assertIsNotNone(best_in_text('offenders among young', c))
        self.assertIsNone(best_in_text('young university adult offenders', c))
        self.assertEqual(best_in_text('young adult offenders', c)['gap'], 1)

    def test_adj3_two_gaps(self):
        c = extract_clauses('(adolescent adj3 delinqu*)')[0]
        self.assertEqual(best_in_text('adolescent and persistent delinquency', c)['gap'], 2)
        self.assertIsNone(best_in_text('adolescent and often persistent delinquency', c))

    def test_title_and_abstract_cannot_combine(self):
        raw = b'TY  - JOUR\nTI  - Young adulthood\nAB  - Offenders were surveyed.\nER  -\n'
        recs, _ = parse_ris(raw)
        r = check_ris_records(recs, extract_clauses('(young adj2 offend*).ti,ab,kf.')[0])
        self.assertFalse(r[0]['matched'])

    def test_separate_keyword_lines_cannot_combine(self):
        raw = b'TY  - JOUR\nTI  - Other subject\nKW  - young\nKW  - offender\nER  -\n'
        recs, _ = parse_ris(raw)
        r = check_ris_records(recs, extract_clauses('(young adj2 offend*).ti,ab,kf.')[0])
        self.assertFalse(r[0]['matched'])

    def test_wildcard_and_phrase(self):
        a = extract_clauses('(wom?n adj2 "justice system")')[0]
        self.assertEqual(best_in_text('women and justice system', a)['gap'], 1)
        self.assertEqual(best_in_text('justice system for woman', a)['gap'], 1)
        self.assertIsNone(best_in_text('women participate in the justice system', a))

    def test_extraction_inside_complex_boolean(self):
        c = extract_clauses('((young adj2 offend*).ti,ab,kf. OR exp Students/) AND ((online adj3 network*).ti,ab,kf. OR Social Media/)')
        self.assertEqual([x.n for x in c], [2, 3])

    def test_escape_html_in_user_content(self):
        raw = b'TY  - JOUR\nTI  - Juvenile offenders and <script>alert(1)</script>\nER  -\n'
        recs,_=parse_ris(raw)
        rows=check_ris_records(recs,extract_clauses('(juvenil* adj2 offend*)')[0])
        report=html_report(recs,rows,extract_clauses('(juvenil* adj2 offend*)')[0]).decode('utf-8')
        self.assertIn('&lt;script&gt;',report)
        self.assertNotIn('<script>alert(1)</script>',report)

if __name__ == '__main__':
    unittest.main()
