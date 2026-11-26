"""Read credentials only on the server; never put them into widgets."""
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

def secret(name):
    try:
        return str(st.secrets.get(name, "") or "").strip()
    except (FileNotFoundError, StreamlitSecretNotFoundError):
        return ""

def status(name):
    return "Configured in server secrets" if secret(name) else "Not configured (server secrets)"

def redact(value):
    text = str(value)
    from urllib.parse import quote, quote_plus
    for name in ("NCBI_API_KEY", "CONSENSUS_API_KEY", "OPENALEX_API_KEY",
                 "SCOPUS_API_KEY", "SCOPUS_INST_TOKEN", "SEMANTIC_SCHOLAR_API_KEY"):
        key = secret(name)
        if key:
            for variant in {key, quote(key, safe=""), quote_plus(key)}:
                text = text.replace(variant, "[redacted]")
    return text
