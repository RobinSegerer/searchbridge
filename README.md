# SearchBridge

SearchBridge is a Streamlit application for rule-based search strategy translation and local proximity checks.

## Features

- Translate Ovid MEDLINE strategies into Ovid PsycINFO, Embase.com, Web of Science, and Scopus query syntax.
- **PsycINFO:** translate `.kf.` to `.id.`; preserve unmapped source subject headings as explicitly **unverified candidates**, not verified APA terms.
- **Embase.com:** use only manually verified Emtree mappings; otherwise retain concepts as explicitly documented title/abstract/keyword free-text fallbacks (**not** Emtree equivalents).
- **Web of Science / Scopus:** remove controlled-vocabulary-only clauses while preserving the remaining Boolean structure; those results cannot be assumed equivalent to the original MeSH-indexed strategy.
- **PubMed target in Search String Translator (new):** translates Ovid MEDLINE MeSH `exp Descriptor/` to `"Descriptor"[mh]` and nonexploded `Descriptor/` to `"Descriptor"[mh:noexp]`. `.ti,ab,kf.` becomes PubMed `[tiab]` (which includes author keywords / Other Term). Individual title/abstract limits are preserved as `[ti]` OR `[ab]`. Untruncated Ovid `adj2` / `adj3` are rendered as genuine unordered PubMed proximity `"word1 word2"[tiab:~1]` / `[tiab:~2]`, with all OR-word alternatives expanded. PubMed **ignores proximity with wildcard terms**, so affected clauses use broadened AND candidate conditions with a prominent warning and optional later local proximity verification. The Ovid `wom?n` is explicitly expanded to `woman OR women`; short `Web*` is approximated using a documented, **non-exhaustive** list of PubMed-valid web prefixes, since PubMed requires at least four letters before `*`. The full query can be validated using **NCBI ESearch POST** (also when the browser GET URL would be too long), displaying actual NCBI Search Details, count and warnings. These are reproducible candidate strategies, not guaranteed retrieval-identical to Ovid MEDLINE. The new PubMed strategy also prepopulates the **Review Pipeline** PubMed search field.
- **External API translators (new):** Ovid MEDLINE → **OpenAlex**, **Crossref**, **Semantic Scholar (Academic Graph Bulk API)**. OpenAlex: native **OpenAlex Query Language (OQL)**, `AND/OR/NOT`, and actual proximity expressions with `adjN` mapped to `within N-1 words of each other`. OR-word groups are expanded into each valid term pair (the complete Ovid strategy can therefore become much longer). The translator outputs `GET /?oql=` for short queries and a **POST / JSON payload** for long queries without truncating them. OpenAlex's stricter double-wildcard proximity rules can make individual pair expressions invalid; these are retained with their original wildcards as *explicitly flagged broader AND candidates*, not silently narrowed or claimed equivalent. MeSH indexing and `.kf.` fields are not equivalent to OpenAlex keywords. Semantic Scholar: Graph `paper/search/bulk` Boolean `+` (AND), `|` (OR), `-` (NOT); unlike the site search and `/paper/search`, with no native Ovid field mappings; a free API key may be needed because anonymous traffic is rate limited. Crossref: exploratory `query.bibliographic` only—**NOT a Boolean translation**, with its approximate 20% term-match threshold. Long OpenAlex URLs beyond 8190 bytes are suppressed with a warning. Every tab shows the exact API parameter, execution link when short enough, and a loss report. Loaded MeSH EXP descendants can optionally be included **as text concepts only**, not as true indexed terms.
- **Crossref + Semantic Scholar audit (2026-11-23):** Crossref does **not** support Boolean/proximity search. Instead of the misleading concatenation of all OR synonyms, SearchBridge generates a short, **editable** bibliographic discovery query plus a separate API link per top-level AND concept. Each concept's default anchor comes from its first Ovid adjacency pair, not automatic semantic relevance scoring. The complete list of positive source-term candidates and the omitted NOT terms are reported. Live Crossref `query.bibliographic` probes returned HTTP 200 but extremely broad result counts: **these are NOT systematic search results** and cannot be used for sensitivity claims. Semantic Scholar's API **Bulk** endpoint supports `+` required, `|` OR, `-` excluded, grouped parentheses, quoted phrases and prefix `*`. The implementation flattens repeated OR/AND groups, broadens Ovid `?` wildcard with an explicit warning, and replaces Ovid `adjN` with a **broad AND candidate condition** (S2 phrase slop differs from unordered Ovid proximity). Unrepresentable long GET URLs are suppressed. A test button in the UI optionally supplies an S2 API key as an HTTP header and checks the actual endpoint. Anonymous live testing yielded **HTTP 429**, so **the generated S2 query has not yet been accepted by the production parser**. Parsed and syntactic regressions are covered by unit tests, but do not establish equivalence.
- **Faster retrieval and auditable human screening (updated):** the Review Pipeline shows a real-time per-database progress indicator (actual retrieved record counts vs the configured cap) and then stage-specific progress for DOI/PMID/title-year matching, metadata merging and fuzzy-pair review. The Python dedup uses incremental component DOI tracking instead of repeatedly scanning all input records; large skipped fuzzy groups are explicitly reported rather than silently treated as checked. Automatic merges remain conservative and maintain source IDs. The final human screening view displays the **full abstract** and metadata; independent Include/Exclude/Manual Review decisions require a rationale for excludes and are saved separately from AI suggestions. The optional OpenAI model selector defaults to GPT-6.1 Terra with Sol and Luna alternatives. **ASySD is not integrated into the deployed app**: its R dependencies and full round trip were not verified; the incomplete interface has deliberately been removed. R can in principle be installed in Streamlit Community Cloud with packages.txt, but that alone does not install or validate the ASySD package. For actual ASySD use its separate official R/Shiny workflow and export the merged citations as RIS for that tool.
- **Review Pipeline (new):** one Streamlit workflow for **PubMed** (NCBI ESearch/EFetch), **OpenAlex** (native OQL POST), **Scopus** (Elsevier Search API + institutional entitlements) and multiple **RIS imports** from PsycINFO, Web of Science, Embase, etc. Every source is logged with its exact search query, hit count, retrieved count and completeness flag. SearchBridge preserves source identifiers and merges exact DOI/PMID or non-contradictory normalized title+year matches. Close but nonidentical titles are displayed as **manual fuzzy duplicate candidates** and are not deleted automatically. The merged corpus is ranked using a **lexical, title-weighted TF-IDF similarity** to an editable research focus. No record is filtered or excluded by ranking. All original and merged provenance is retained. An optional OpenAI **Responses API** title/abstract prescreen uses an explicit user key and consent, a mandatory user-entered PICO/PCC/eligibility protocol, structured `include/exclude/manual_review` suggestions, rationale, evidence spans and uncertainty labels. Missing abstracts and failed API calls default to **manual_review**. The UI retains **human decisions separately**, never auto-accepting AI recommendations. Outputs include RIS/CSV/JSON, query-source logs, dedup audit, fuzzy review candidates, human decisions, and model/criteria/run logs. **No fully automatic PRISMA flow or systematic review screening is claimed**; all AI classifications and dedup decisions require expert verification. API search limits can leave an incomplete retrieval set. Scopus STANDARD metadata can omit abstracts. Institutional reuse or AI-sharing of licensed Scopus records requires a licence check. Never commit or export API keys.
- **Semantic + Hybrid Term Harvesting:** the new ranking selector works for both MeSH-to-title candidates and TIAB-to-MeSH descriptor frequencies. Semantic uses the original English [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) embedding model via [FastEmbed CPU ONNX](https://github.com/qdrant/fastembed), without PyTorch or a generative LLM. The semantic target is editable; the model ranks **only existing title n-grams (1–3 words) or actual MeSH descriptor labels**, never invents terms. **Hybrid** uses weighted reciprocal rank fusion (RRF, k=60) over semantic and original statistical ranks. Optional MMR reorders suggestions to reduce semantic redundancy. The UI limits the candidate embedding pool to 1,500 terms and caches the model. A first-use model download and additional memory are required. The statistical mode needs no model. You may upload **seed papers (RIS)** for a clearly labelled *literal occurrence in title/abstract* check; it is **not** a test of retrieval recall or a systematic review search validation. All scores, mode, focus, model, filters and counts are documented in the ZIP JSON export.
- **PubMed Term Harvesting:** two complementary, no-LLM workflows. A PubMed **MeSH query** retrieves bibliographic records, extracts publication titles and ranks English words (optionally adjacent two-word phrases) by within-sample, L2-normalized TF-IDF. The output includes raw word occurrences and title document frequencies. Conversely, a **TIAB query** ranks the *actually assigned* MeSH descriptors by document count, prevalence among MeSH-indexed records and MajorTopicYN counts. Both workflows include a capped PubMed ESearch/EFetch sample, explicit completeness/indexing warnings, CSV export, RIS source records and an audit JSON in the ZIP report. The retrieval is ordered by PubMed relevance and not representative of the full dataset when truncated; harvested terms are suggestions for expert review, not validated replacements. No API key required for basic use.
- **MeSH EXP explorer:** identifies each `exp .../` in the Ovid MEDLINE source and resolves the *exact* NLM MeSH descriptor with its narrower descendants through [NLM MeSH RDF](https://id.nlm.nih.gov/mesh/). Shows hierarchy depth, tree numbers, descriptor IDs, and a direct NLM link. Download CSV (all tree positions) or TXT (unique descriptor labels). Uses a 24-hour Streamlit cache; requires internet but no key. **These are MeSH expansions only**, not verified PsycINFO or Emtree exploded vocabularies.
- Flatten consecutive AND/OR groups for readability while preserving proximity parentheses.
- Replace Ovid PsycINFO `.kf.` with `.id.` and log removed controlled-vocabulary clauses.
- Check `adj1` through `adj10` using records downloaded from PubMed or uploaded as RIS.
- Export HTML evidence reports and bibliographic files.

## Review Pipeline quick start

1. Open the **Review Pipeline** tab and tick PubMed, OpenAlex and/or Scopus.
2. Enter a query in **each source's native syntax** (the Search String Translator
   tab helps generate OpenAlex OQL and Scopus expressions). OpenAlex
   multiword OQL phrases need quotes, e.g. `get works about "social media"`.
3. Enter any source API keys *in the running UI*, not the repository.
   OpenAlex requires a key for pagination after the first 100 hits;
   Scopus requires a key with appropriate institution permissions. For
   unsupported sources, upload RIS files.
4. Click **Search databases and merge records**. Follow the two
   progress indicators for each source and the subsequent three deduplication
   stages. Inspect the source completeness report and fuzzy duplicate
   candidates requiring manual review. The app uses its clearly named
   conservative Python DOI/PMID/title-year algorithm; **not ASySD**.
5. Set a focused ranking topic. Ranking prioritizes human screening; it
   does **not** exclude studies.
6. Optional: enter explicit PICO/PCC eligibility criteria and your **OpenAI
   API key**; consent to sending title/abstract metadata to the provider;
   prescreen at most 60 prioritized records per run (default: GPT-6.1 Terra,
   alternatives: GPT-6.2 Sol / GPT-6 Luna). **Review the complete abstract**
   and record a separate human decision with rationale. The API requests
   set `store=false`, but you must still assess
   licensing, consent, privacy and retention conditions for your source data.
7. Download the ZIP audit bundle before ending the Streamlit session.
   **Session state is not a durable database** and does not substitute
   for Covidence/Rayyan/ASReview or project storage.

### Screening metadata and English-language interface

The Streamlit interface, search translation warnings, retrieval diagnostics,
term harvesting controls, and review workflow labels are in English.

Every review table uses the same bibliographic fields: **Title, Abstract,
Authors, Year, Journal, DOI, Database**. The Database field preserves all
source databases for records merged during deduplication (including RIS
source filenames). The human screening detail view displays the full abstract
and the same citation metadata.

The **Screening results: CSV** download includes these fields alongside
AI suggestions, evidence, uncertainty, human decisions, reasons, and reviewer
information. The ZIP also contains self-contained `screening_results.csv`,
`ai_screening_suggestions.csv`, and `human_decisions.csv` tables in addition
to deduplicated RIS/JSON and the audit log. Missing metadata remain empty
rather than being invented.

### What is and is not complete

- **Working / tested:** PubMed API and OpenAlex API live retrieval (5 records
  from each on 2026-11-23), offline Scopus response parsing and key handling,
  conservative deduplication, ranking and mocked structured OpenAI responses.
- **ASySD:** no executable ASySD integration is offered in this Streamlit app; use the official R/Shiny package separately if required. Installing R alone would not be sufficient to claim integration.
- **Not yet verified with real credentials:** institution-entitled Scopus
  requests and paid OpenAI screening. Their operation depends on your keys
  and entitlement. No real OpenAI screening has been run.
- **Not included:** systematic capture of all >10,000 results, a full
  PRISMA-2020 report, record-level dual independent human screening,
  full-text extraction, calibrated risk-of-bias assessment and automated
  meta-analysis. These should be later, separately tested workstreams.

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