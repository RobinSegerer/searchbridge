# SearchBridge

SearchBridge is a Streamlit application for rule-based search strategy translation and local proximity checks.

## Features

- Translate Ovid MEDLINE strategies into Ovid PsycINFO, Embase.com, Web of Science, and Scopus query syntax.
- **PsycINFO:** translate `.kf.` to `.id.`; preserve unmapped source subject headings as explicitly **unverified candidates**, not verified APA terms.
- **Embase.com:** use only manually verified Emtree mappings; otherwise retain concepts as explicitly documented title/abstract/keyword free-text fallbacks (**not** Emtree equivalents).
- **Web of Science / Scopus:** remove controlled-vocabulary-only clauses while preserving the remaining Boolean structure; those results cannot be assumed equivalent to the original MeSH-indexed strategy.
- **PubMed Term Harvesting:** two complementary, no-LLM workflows. A PubMed **MeSH query** retrieves bibliographic records, extracts publication titles and ranks English words (optionally adjacent two-word phrases) by within-sample, L2-normalized TF-IDF. The output includes raw word occurrences and title document frequencies. Conversely, a **TIAB query** ranks the *actually assigned* MeSH descriptors by document count, prevalence among MeSH-indexed records and MajorTopicYN counts. Both workflows include a capped PubMed ESearch/EFetch sample, explicit completeness/indexing warnings, CSV export, RIS source records and an audit JSON in the ZIP report. The retrieval is ordered by PubMed relevance and not representative of the full dataset when truncated; harvested terms are suggestions for expert review, not validated replacements. No API key required for basic use.
- **MeSH EXP explorer:** identifies each `exp .../` in the Ovid MEDLINE source and resolves the *exact* NLM MeSH descriptor with its narrower descendants through [NLM MeSH RDF](https://id.nlm.nih.gov/mesh/). Shows hierarchy depth, tree numbers, descriptor IDs, and a direct NLM link. Download CSV (all tree positions) or TXT (unique descriptor labels). Uses a 24-hour Streamlit cache; requires internet but no key. **These are MeSH expansions only**, not verified PsycINFO or Emtree exploded vocabularies.
- Flatten consecutive AND/OR groups for readability while preserving proximity parentheses.
- Replace Ovid PsycINFO `.kf.` with `.id.` and log removed controlled-vocabulary clauses.
- Check `adj1` through `adj10` using records downloaded from PubMed or uploaded as RIS.
- Export HTML evidence reports and bibliographic files.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy

Connect this GitHub repository to [Streamlit Community Cloud](https://share.streamlit.io/) and choose `app.py` as the entry point.

**Limitations:** Proximity post-filtering approximates Ovid search behavior. Complete retrieval requires downloading every AND-query candidate. Mapping between MeSH and Emtree is not inferred automatically.

No LLM or paid API key is required. The MeSH expansion feature needs a working connection to the public NLM MeSH RDF API; if it is unavailable, SearchBridge reports the error and does not invent hierarchy terms.