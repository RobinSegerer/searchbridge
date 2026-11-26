import unittest
from auto_query import suggest_concepts, compress_variants, group_text, select_terms, format_block, build_query

class TestAutoQuery(unittest.TestCase):
    def test_concepts(self):
        self.assertEqual(suggest_concepts("What is the association between sleep quality and depression among university students?"),
                         ["sleep quality", "depression", "university students"])
    def test_compact_plural(self):
        self.assertEqual(compress_variants(["student", "students"]), ["student*"])
        self.assertEqual(compress_variants(["child", "children"]), ["child", "children"])
        self.assertEqual(compress_variants(["risk", "risky"]), ["risk*"])
        self.assertEqual(compress_variants(["man", "men"]), ["man", "men"])
    def test_phrase(self):
        self.assertEqual(compress_variants(["university student", "university students"]), ["university student*"])
    def test_lemma_group_counts_unique_papers(self):
        rows = [{"Term":"student","Score":1,"Documents":2,"Consensus ranks":"1, 2"},
                {"Term":"students","Score":2,"Documents":2,"Consensus ranks":"2, 3"}]
        groups=group_text(rows)
        self.assertEqual(len(groups),1)
        self.assertEqual(groups[0]["Documents"],3)
        self.assertEqual(groups[0]["Patterns"],["student*"])
    def test_no_unrelated_mesh(self):
        rows=[{"Term":"Humans","Documents":20},{"Term":"Depression","Documents":10}]
        self.assertEqual([x["Term"] for x in select_terms(rows,"depression",3,mesh=True)],["Depression"])
    def test_query_syntax_and_missing_concept(self):
        self.assertEqual(format_block([{"Term":"students","Patterns":["student*"]}],[{"Term":"Students"}]),
                         '("student*"[tiab] OR "Students"[mh])')
        self.assertEqual(build_query(["(A)","(B)"]),"(A)\nAND\n(B)")
        self.assertEqual(build_query(["(A)",""]),"")
if __name__=="__main__":
    unittest.main()
