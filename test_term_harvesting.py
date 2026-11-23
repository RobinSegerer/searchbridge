"""Deterministic title-TFIDF and MeSH descriptor harvesting regression tests."""
import csv
import io
import unittest

from ris_proximity import RISRecord
from term_harvesting import (
    word_tokens, vocabulary_exclusions, title_tfidf, ranked_mesh, rows_to_csv,
)
from pubmed_proximity import pubmed_xml_to_ris


def rec(n, title, mesh=None, major=None):
    d = {"TY": ["JOUR"], "ID": [str(n)], "TI": [title]}
    if mesh:
        d["MH"] = list(mesh)
    if major:
        d["M1"] = list(major)
    return RISRecord(n, d, [])


class HarvestingTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            rec(1, "Social media engagement and academic performance",
                ["Students", "Social Media", "Humans"], ["Social Media"]),
            rec(2, "Social media use predicts academic achievement",
                ["Students", "Male"], ["Students"]),
            rec(3, "Digital platforms and academic performance"),
            rec(4, "Social media engagement and university life",
                ["Social Media", "Students"], ["Social Media"]),
        ]

    def test_tfidf_scores_and_doc_counts(self):
        result = title_tfidf(self.records, min_df=2, top_n=100)
        words = {x["Title term"]: x for x in result.rows}
        self.assertEqual(result.included_docs, 4)
        self.assertEqual(result.missing_docs, 0)
        self.assertEqual(words["media"]["Document frequency"], 3)
        self.assertEqual(words["academic"]["Document frequency"], 3)
        self.assertEqual(words["performance"]["Occurrences"], 2)
        self.assertGreater(words["media"]["Mean TF-IDF"], 0)
        self.assertAlmostEqual(words["social"]["% titles"], 75.0)
        self.assertNotIn("digital", words)   # df=1
        self.assertNotIn("and", words)       # stop word
        self.assertEqual(result.rows[0]["Rank"], 1)
        self.assertEqual(sorted((x["Rank"] for x in result.rows)),
                         list(range(1, len(result.rows) + 1)))

    def test_query_term_removal_and_truncation(self):
        excluded, stems = vocabulary_exclusions("students[mh] AND soci*[tiab]")
        self.assertIn("students", excluded)
        self.assertIn("soci", stems)
        result = title_tfidf(
            self.records, query="soci*[tiab] AND university[mh]",
            remove_query_terms=True, min_df=1)
        terms = {r["Title term"] for r in result.rows}
        self.assertNotIn("social", terms)
        self.assertNotIn("university", terms)
        self.assertIn("media", terms)

    def test_bigrams_preserve_real_adjacency_after_stopwords(self):
        records = [
            rec(1, "Social and media engagement"),
            rec(2, "Social media engagement"),
        ]
        result = title_tfidf(records, max_ngram=2, min_df=1)
        terms = {r["Title term"]: r for r in result.rows}
        self.assertEqual(terms["social media"]["Document frequency"], 1)
        self.assertEqual(terms["media engagement"]["Document frequency"], 2)
        self.assertNotIn("social engagement", terms)

    def test_mesh_frequency_indexed_denominator_and_major_topics(self):
        result = ranked_mesh(self.records)
        mesh = {r["MeSH descriptor"]: r for r in result.rows}
        self.assertEqual(result.included_docs, 3)
        self.assertEqual(result.missing_docs, 1)
        self.assertNotIn("Humans", mesh)
        self.assertNotIn("Male", mesh)
        self.assertEqual(mesh["Students"]["Indexed records"], 3)
        self.assertEqual(mesh["Students"]["Major topic records"], 1)
        self.assertEqual(mesh["Students"]["% indexed records"], 100.0)
        self.assertEqual(mesh["Students"]["% all retrieved"], 75.0)
        self.assertEqual(mesh["Social Media"]["Indexed records"], 2)
        self.assertEqual(mesh["Social Media"]["Major topic records"], 2)
        self.assertEqual(mesh["Social Media"]["% indexed records"], 66.67)

    def test_optional_check_tag_inclusion(self):
        result = ranked_mesh(self.records, exclude_check_tags=False)
        mesh = {r["MeSH descriptor"]: r for r in result.rows}
        self.assertEqual(mesh["Humans"]["Indexed records"], 1)
        self.assertEqual(mesh["Male"]["Indexed records"], 1)

    def test_missing_mesh_is_not_treated_as_negative(self):
        result = ranked_mesh([rec(1, "Unindexed article")])
        self.assertEqual(result.included_docs, 0)
        self.assertEqual(result.missing_docs, 1)
        self.assertEqual(result.rows, [])

    def test_csv_export_with_bom(self):
        result = title_tfidf(self.records, min_df=2)
        columns = ["Rank", "Title term", "Mean TF-IDF",
                   "Document frequency", "% titles", "Occurrences"]
        raw = rows_to_csv(result.rows, columns)
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        self.assertEqual(len(rows), len(result.rows))

    def test_pubmed_efetch_xml_has_mesh_and_major_not_qualifiers(self):
        xml = b'''<PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <PMID>100</PMID><Article><ArticleTitle>Academic outcomes</ArticleTitle></Article>
              <MeshHeadingList>
                <MeshHeading><DescriptorName MajorTopicYN="Y" UI="D013334">Students</DescriptorName>
                  <QualifierName MajorTopicYN="N">psychology</QualifierName></MeshHeading>
                <MeshHeading><DescriptorName MajorTopicYN="N" UI="D061108">Social Media</DescriptorName></MeshHeading>
              </MeshHeadingList>
            </MedlineCitation>
            <PubmedData><ArticleIdList /></PubmedData>
          </PubmedArticle>
          <PubmedArticle><MedlineCitation><PMID>200</PMID>
            <Article><ArticleTitle>New study</ArticleTitle></Article>
            </MedlineCitation><PubmedData><ArticleIdList /></PubmedData>
          </PubmedArticle>
          </PubmedArticleSet>'''
        records = pubmed_xml_to_ris(xml)
        self.assertEqual(records[0].fields["MH"], ["Students", "Social Media"])
        self.assertEqual(records[0].fields["M1"], ["Students"])
        self.assertNotIn("psychology", records[0].fields["MH"])
        self.assertNotIn("MH", records[1].fields)
        self.assertEqual(ranked_mesh(records).included_docs, 1)


if __name__ == "__main__":
    unittest.main()
