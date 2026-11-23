# SearchBridge

SearchBridge is a Streamlit application for rule-based search strategy translation and local proximity checks.

## Features

- Translate Ovid MEDLINE strategies into Ovid PsycINFO, Embase.com, Web of Science, and Scopus query syntax.
- **PsycINFO:** translate `.kf.` to `.id.`; preserve unmapped source subject headings as explicitly **unverified candidates**, not verified APA terms.
- **Embase.com:** use only manually verified Emtree mappings; otherwise retain concepts as explicitly documented title/abstract/keyword free-text fallbacks (**not** Emtree equivalents).
- **Web of Science / Scopus:** remove controlled-vocabulary-only clauses while preserving the remaining Boolean structure; those results cannot be assumed equivalent to the original MeSH-indexed strategy.
- **External API translators (new):** Ovid MEDLINE → **OpenAlex**, **Crossref**, **Semantic Scholar (Academic Graph Bulk API)**. OpenAlex: native **OpenAlex Query Language (OQL)**, `AND/OR/NOT`, and actual proximity expressions with `adjN` mapped to `within N-1 words of each other`. OR-word groups are expanded into each valid term pair (the complete Ovid strategy can therefore become much longer). The translator outputs `GET /?oql=` for short queries and a **POST / JSON payload** for long queries without truncating them. OpenAlex's stricter double-wildcard proximity rules can make individual pair expressions invalid; these are retained with their original wildcards as *explicitly flagged broader AND candidates*, not silently narrowed or claimed equivalent. MeSH indexing and `.kf.` fields are not equivalent to OpenAlex keywords. Semantic Scholar: Graph `paper/search/bulk` Boolean `+` (AND), `|` (OR), `-` (NOT); unlike the site search and `/paper/search`, with no native Ovid field mappings; a free API key may be needed because anonymous traffic is rate limited. Crossref: exploratory `query.bibliographic` only—**NOT a Boolean translation**, with its approximate 20% term-match threshold. Long OpenAlex URLs beyond 8190 bytes are suppressed with a warning. Every tab shows the exact API parameter, execution link when short enough, and a loss report. Loaded MeSH EXP descendants can optionally be included **as text concepts only**, not as true indexed terms.
- **Semantic + Hybrid Term Harvesting:** the new ranking selector works for both MeSH-to-title candidates and TIAB-to-MeSH descriptor frequencies. Semantic uses the original English [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) embedding model via [FastEmbed CPU ONNX](https://github.com/qdrant/fastembed), without PyTorch or a generative LLM. The semantic target is editable; the model ranks **only existing title n-grams (1–3 words) or actual MeSH descriptor labels**, never invents terms. **Hybrid** uses weighted reciprocal rank fusion (RRF, k=60) over semantic and original statistical ranks. Optional MMR reorders suggestions to reduce semantic redundancy. The UI limits the candidate embedding pool to 1,500 terms and caches the model. A first-use model download and additional memory are required. The statistical mode needs no model. You may upload **seed papers (RIS)** for a clearly labelled *literal occurrence in title/abstract* check; it is **not** a test of retrieval recall or a systematic review search validation. All scores, mode, focus, model, filters and counts are documented in the ZIP JSON export.
- **PubMed Term Harvesting:** two complementary, no-LLM workflows. A PubMed **MeSH query** retrieves bibliographic records, extracts publication titles and ranks English words (optionally adjacent two-word phrases) by within-sample, L2-normalized TF-IDF. The output includes raw word occurrences and title document frequencies. Conversely, a **TIAB query** ranks the *actually assigned* MeSH descriptors by document count, prevalence among MeSH-indexed records and MajorTopicYN counts. Both workflows include a capped PubMed ESearch/EFetch sample, explicit completeness/indexing warnings, CSV export, RIS source records and an audit JSON in the ZIP report. The retrieval is ordered by PubMed relevance and not representative of the full dataset when truncated; harvested terms are suggestions for expert review, not validated replacements. No API key required for basic use.
- **MeSH EXP explorer:** identifies each `exp .../` in the Ovid MEDLINE source and resolves the *exact* NLM MeSH descriptor with its narrower descendants through [NLM MeSH RDF](https://id.nlm.nih.gov/mesh/). Shows hierarchy depth, tree numbers, descriptor IDs, and a direct NLM link. Download CSV (all tree positions) or TXT (unique descriptor labels). Uses a 24-hour Streamlit cache; requires internet but no key. **These are MeSH expansions only**, not verified PsycINFO or Emtree exploded vocabularies.
- Flatten consecutive AND/OR groups for readability while preserving proximity parentheses.
- Replace Ovid PsycINFO `.kf.` with `.id.` and log removed controlled-vocabulary clauses.
- Check `adj1` through `adj10` using records downloaded from PubMed or uploaded as RIS.
- Export HTML evidence reports and bibliographic files.

## Semantic ranking and online deployment

The Streamlit default remains the fast, transparent statistical ranking. To use
**Semantic** or **Hybrid** select that mode in the **Term Harvesting** tab and
enter a focused English concept (e.g. `academic self-concept among university students`).
The app downloads model weights on first use; embeddings and retrieved titles are
processed on the Streamlit server. On Streamlit Community Cloud, free-tier CPU/RAM
limits may slow or interrupt initial loading; reduce the candidate pool if needed.

The embeddings **do not establish construct equivalence** (e.g. academic
self-efficacy is not automatically an academic self-concept synonym). Terms
must still be checked by a search expert and validated against known relevant
papers and actual databases. The seed RIS display checks only occurrence of
whole contiguous words/phrases inside titles/abstracts, not whether PubMed or
Ovid would retrieve the record.

For a literal original Ovid search with complex adjacency and indexed MeSH, use the
OpenAlex website's native [cross-database strategy importer](https://help.openalex.org/how-to/search-strategies-from-other-databases/) as an additional comparison: unlike SearchBridge's conservative API fallback, OpenAlex can translate some proximity expressions directly into its native OQL form. However, index-based retrieval semantics still differ. Treat all database-specific strategies as candidates for expert verification.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy

Connect this GitHub repository to [Streamlit Community Cloud](https://share.streamlit.io/) and choose `app.py` as the entry point.

**Limitations:** Proximity post-filtering approximates Ovid search behavior. Complete retrieval requires downloading every AND-query candidate. Mapping between MeSH and Emtree is not inferred automatically.

No generative LLM or paid API key is required. The MeSH expansion feature needs a working connection to the public NLM MeSH RDF API; if it is unavailable, SearchBridge reports the error and does not invent hierarchy terms.