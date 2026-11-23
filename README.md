# SearchBridge

SearchBridge is a Streamlit application for rule-based search strategy translation and local proximity checks.

## Features

- Translate Ovid MEDLINE strategies into Ovid PsycINFO, Embase.com, Web of Science, and Scopus query syntax.
- **PsycINFO:** translate `.kf.` to `.id.`; preserve unmapped source subject headings as explicitly **unverified candidates**, not verified APA terms.
- **Embase.com:** use only manually verified Emtree mappings; otherwise retain concepts as explicitly documented title/abstract/keyword free-text fallbacks (**not** Emtree equivalents).
- **Web of Science / Scopus:** remove controlled-vocabulary-only clauses while preserving the remaining Boolean structure; those results cannot be assumed equivalent to the original MeSH-indexed strategy.
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

No LLM or paid API key is required.