"""Offline tests using synthetic, explicitly fake PubMed E-utilities responses."""
import unittest
from unittest.mock import patch
import json

from pubmed_proximity import (parsed_terms, build_pubmed_and_query, pubmed_xml_to_ris,
                              fetch_pubmed_candidates)
from ris_proximity import check_ris_records, ProximityClause, ris_bytes, parse_ris, html_report

XML = b'''<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
<PubmedArticle><MedlineCitation><PMID>111</PMID><Article><ArticleTitle>Young adult offenders in rehabilitation</ArticleTitle>
<Abstract><AbstractText Label="Purpose">Study of young adult offenders and their lives.</AbstractText></Abstract>
<Journal><Title>Journal A</Title><JournalIssue><PubDate><Year>2025</Year></PubDate></JournalIssue></Journal>
<AuthorList><Author><LastName>Smith</LastName><ForeName>John</ForeName></Author></AuthorList>
</Article><KeywordList><Keyword>young offenders</Keyword></KeywordList></MedlineCitation>
<PubmedData><ArticleIdList><ArticleId IdType="doi">10.0000/example1</ArticleId></ArticleIdList></PubmedData></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID>222</PMID><Article><ArticleTitle>Young people facing complex barriers</ArticleTitle>
<Abstract><AbstractText>Offenders in prison were surveyed separately.</AbstractText></Abstract>
</Article></MedlineCitation><PubmedData><ArticleIdList /></PubmedData></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID>333</PMID><Article><ArticleTitle>Youth health in a large cohort</ArticleTitle>
<Abstract><AbstractText>Criminal offenders were contacted.</AbstractText></Abstract>
<KeywordList><Keyword>adolescent intervention</Keyword><Keyword>adult prison</Keyword></KeywordList>
</Article></MedlineCitation><PubmedData><ArticleIdList /></PubmedData></PubmedArticle>
</PubmedArticleSet>'''


class PubMedTests(unittest.TestCase):
    def test_terms_and_broad_query(self):
        a = parsed_terms('juvenil* OR young* OR adolescent* OR youth*')
        b = parsed_terms('perpetr* OR offend* OR delinqu*')
        self.assertEqual(len(a), 4)
        q = build_pubmed_and_query(a, b)
        self.assertIn('juvenil*[tiab]', q)
        self.assertIn('perpetr*[tiab]', q)
        self.assertIn(') AND (', q)
        self.assertNotIn('adj2', q)

    def test_question_mark_is_broadened(self):
        q = build_pubmed_and_query(parsed_terms('wom?n'), parsed_terms('offend*'))
        self.assertIn('wom*[tiab]', q)

    def test_complex_term_expands_to_and_superset(self):
        q = build_pubmed_and_query(parsed_terms('young adult OR juvenile'), parsed_terms('self-concept*'))
        self.assertIn('(young[tiab] AND adult[tiab])', q)
        self.assertIn('(self[tiab] AND concept*[tiab])', q)

    def test_invalid_nested_logic_rejected(self):
        with self.assertRaises(ValueError):
            parsed_terms('juvenile AND youth')

    def test_xml_and_local_matching(self):
        recs = pubmed_xml_to_ris(XML)
        self.assertEqual(len(recs), 3)
        self.assertEqual(recs[0].identifier, '111')
        self.assertEqual(recs[0].doi, '10.0000/example1')
        self.assertIn('Purpose:', recs[0].abstract)
        self.assertIn('young offenders', recs[0].keywords)
        c = ProximityClause(n=2, left=('young*', 'adolescent*', 'youth*'),
                            right=('offend*', 'prison*'), fields=('ti', 'ab', 'kf'))
        rows = check_ris_records(recs, c)
        self.assertTrue(rows[0]['matched'])
        self.assertEqual(rows[0]['gap'], 0)
        self.assertFalse(rows[1]['matched'])
        self.assertFalse(rows[2]['matched'])
        self.assertIn(b'<mark>', html_report(recs, rows, c))
        roundtrip, _ = parse_ris(ris_bytes(recs))
        self.assertEqual(len(roundtrip), 3)
        self.assertEqual(roundtrip[0].doi, '10.0000/example1')

    def test_mocked_esearch_efetch(self):
        class FakeResponse:
            def __init__(self, content):
                self.content = content
                self.text = content.decode('utf-8')
            def json(self):
                return json.loads(self.text)
        search = FakeResponse(json.dumps({'esearchresult': {
            'count': '3', 'webenv': 'dummy', 'querykey': '1',
            'querytranslation': 'query verified'
        }}).encode())
        progress = []
        with patch('pubmed_proximity.NCBIClient.get', side_effect=[search, FakeResponse(XML)]) as get:
            result = fetch_pubmed_candidates('juvenil*[tiab] AND offend*[tiab]', limit=3,
                                              progress=lambda i,n: progress.append((i,n)))
        self.assertEqual(result.candidate_count, 3)
        self.assertEqual(result.fetched, 3)
        self.assertEqual(result.translation, 'query verified')
        self.assertEqual(progress, [(3,3)])
        self.assertEqual(get.call_count, 2)

if __name__ == '__main__':
    unittest.main()
