"""Consensus seed retrieval and evidence-based term harvesting."""
import csv
import io
import math
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import requests
import streamlit as st

STOP = set("""a an the and or of in on for to with by from as at is are was were be been being
this that these those it its their our we they have has had can could may might
not but than then such also using used use study studies research results
 findings method methods participants data analysis significant significantly
 between among across into during after before which who what how whether""".split())
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

def norm(s):
    return re.sub(r"[\W_]+", "", s.casefold())

def doi_norm(s):
    return re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", (s or "").strip(), flags=re.I).lower()

def terms(text):
    out = Counter()
    for segment in re.split(r"[.!?;:\n,()]", text.lower()):
        tokens = re.findall(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", segment)
        for n in (1, 2, 3):
            for i in range(len(tokens)-n+1):
                words = tokens[i:i+n]
                if any(w in STOP or len(w)<2 for w in words):
                    continue
                out[" ".join(words)] += 1
    return out

def retrieve(query, key, limit):
    papers, seen, page, calls = [], set(), 0, 0
    while len(papers)<limit and calls<10:
        if calls:
            time.sleep(1.1)
        r = requests.get("https://api.consensus.app/v1/search",
            headers={"x-api-key": key}, params={"query":query,"page":page,
            "page_size":min(100,limit-len(papers))}, timeout=45)
        if not r.ok:
            raise RuntimeError(f"Consensus HTTP {r.status_code}")
        data = r.json()
        batch = data["results"]
        calls += 1
        for p in batch:
            ident = doi_norm(p.get("doi")) or norm(p.get("title") or "") or p.get("paper_id")
            if ident and ident not in seen:
                seen.add(ident)
                papers.append(p)
        nxt = data.get("next_page")
        if not batch or data.get("is_end") or nxt is None or nxt<=page:
            break
        page = nxt
    return papers[:limit], calls

def rank_terms(papers, field, title_weight, min_df):
    df, tf, score, sources = Counter(), Counter(), Counter(), defaultdict(list)
    for i,p in enumerate(papers,1):
        t = terms(p.get("title") or "") if field!="Abstract" else Counter()
        a = terms(p.get("abstract") or "") if field!="Title" else Counter()
        combined = t+a
        for term,count in combined.items():
            df[term]+=1
            tf[term]+=count
            score[term]+=(1+math.log(t[term]))*title_weight if t[term] else 0
            score[term]+=(1+math.log(a[term])) if a[term] else 0
            sources[term].append(i)
    n=len(papers)
    rows=[{"Term":t,"Documents":d,"Occurrences":tf[t],
           "Score":round(score[t]*(1+math.log((n+1)/(d+1)))/max(n,1),4),
           "Consensus ranks":", ".join(map(str,sources[t]))}
          for t,d in df.items() if d>=min_df]
    return sorted(rows,key=lambda r:(-r["Score"],-r["Documents"],r["Term"]))

def pubmed_request(session, endpoint, params):
    time.sleep(.36)
    r=session.post(BASE+endpoint,data={"db":"pubmed","tool":"searchbridge",**params},timeout=35)
    r.raise_for_status()
    return r

def mesh_match(papers, progress, email=""):
    matches, headings, major, sources = [], Counter(), Counter(), defaultdict(list)
    indexed=0
    with requests.Session() as session:
        common={"email":email} if email else {}
        for i,p in enumerate(papers,1):
            title=p.get("title") or ""
            doi=doi_norm(p.get("doi"))
            match={"Rank":i,"Title":title,"DOI":doi,"PMID":"","Status":"Not found"}
            try:
                probes=[]
                if doi: probes.append((f'"{doi}"[aid]',"DOI"))
                if title: probes.append(('"'+title.replace('"',' ')+'"[ti]',"Title"))
                article=None
                for query,mode in probes:
                    ids=pubmed_request(session,"esearch.fcgi",{**common,"term":query,"retmode":"json","retmax":10}).json()["esearchresult"]["idlist"]
                    if not ids: continue
                    root=ET.fromstring(pubmed_request(session,"efetch.fcgi",{**common,"id":",".join(ids),"retmode":"xml"}).content)
                    candidates=[]
                    for record in root.findall("PubmedArticle"):
                        rt=record.find(".//ArticleTitle")
                        exact_title=rt is not None and norm("".join(rt.itertext()))==norm(title)
                        dois=[doi_norm(x.text) for x in record.findall(".//ArticleId") if x.get("IdType")=="doi"]
                        if (mode=="DOI" and doi in dois) or (mode=="Title" and exact_title and (not doi or not dois or doi in dois)):
                            candidates.append(record)
                    if len(candidates)==1:
                        article=candidates[0]
                        break
                    if len(candidates)>1:
                        match["Status"]="Ambiguous"
                        break
                if article is not None:
                    match["PMID"]=article.findtext(".//MedlineCitation/PMID","")
                    descriptors=article.findall(".//MeshHeading/DescriptorName")
                    match["Status"]="Matched via "+mode if descriptors else "Matched; no MeSH"
                    if descriptors:
                        indexed+=1
                    unique={d.text for d in descriptors if d.text}
                    maj={d.text for d in descriptors if d.text and d.get("MajorTopicYN")=="Y"}
                    headings.update(unique)
                    major.update(maj)
                    for t in unique: sources[t].append(i)
            except (requests.RequestException, ValueError, KeyError, ET.ParseError) as exc:
                match["Status"]="Lookup error: "+type(exc).__name__
            matches.append(match)
            progress(i/len(papers))
    rows=[{"Term":t,"Documents":n,"Major topic":major[t],
           "% indexed":round(100*n/max(indexed,1),1),
           "Consensus ranks":", ".join(map(str,sources[t]))}
          for t,n in headings.items()]
    return sorted(rows,key=lambda r:(-r["Documents"],-r["Major topic"],r["Term"])),matches,indexed

def download(label, rows, name):
    if rows:
        buf=io.StringIO()
        writer=csv.DictWriter(buf,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        st.download_button(label,buf.getvalue(),name,"text/csv",key="consensus_dl_"+name)

def render():
    st.subheader("Consensus-assisted search strategy development")
    st.caption("Exploratory seed corpus; no LLM generation. Official MeSH requires a separate, free PubMed lookup.")
    with st.form("consensus_seed"):
        query=st.text_area("Research question",height=90)
        key=st.text_input("Consensus API key",type="password")
        limit=st.number_input("Maximum seed papers",min_value=5,max_value=100,value=50)
        submitted=st.form_submit_button("Retrieve Consensus papers",type="primary")
    if submitted:
        if not query.strip() or not key.strip():
            st.warning("Enter a research question and API key.")
        else:
            try:
                with st.spinner("Retrieving Consensus papers…"):
                    papers,calls=retrieve(query,key,int(limit))
                st.session_state["consensus_corpus"]=(query,papers,calls)
                st.session_state.pop("consensus_mesh",None)
                for i in range(6):
                    st.session_state.pop(f"consensus_free_{i}",None)
                    st.session_state.pop(f"consensus_mesh_{i}",None)
            except (requests.RequestException,RuntimeError,ValueError,KeyError) as exc:
                st.error("Retrieval failed: "+(str(exc) if isinstance(exc,RuntimeError) else type(exc).__name__))
    if "consensus_corpus" not in st.session_state: return
    q,papers,calls=st.session_state["consensus_corpus"]
    st.write(f"{len(papers)} seed papers · {calls} API calls · {sum(bool(p.get('abstract')) for p in papers)} abstracts")
    st.caption("Retrieved for: "+q)
    if not papers: return
    with st.expander("Inspect seed papers"):
        records=[{"Rank":i,"Title":p.get("title"),"Abstract":p.get("abstract"),"DOI":p.get("doi"),"Year":p.get("publish_year")} for i,p in enumerate(papers,1)]
        st.dataframe(records,use_container_width=True)
        download("Export seed corpus",records,"consensus_seeds.csv")
    field=st.radio("Text source",["Title + abstract","Title","Abstract"],horizontal=True,key="consensus_field")
    weight=st.slider("Title weight",1.0,4.0,2.0,key="consensus_weight")
    min_df=st.slider("Minimum document frequency",1,max(2,min(10,len(papers))),min(2,len(papers)),key="consensus_df")
    rows=rank_terms(papers,field,weight,min_df)
    st.markdown("#### Free-text candidates")
    st.dataframe(rows[:200],use_container_width=True)
    download("Export all ranked text terms",rows,"consensus_terms.csv")
    st.markdown("#### Assigned MeSH descriptors")
    email=st.text_input("NCBI email (optional)",key="consensus_email")
    if st.button("Match seed papers in PubMed",key="consensus_match"):
        bar=st.progress(0.0)
        try:
            st.session_state["consensus_mesh"]=mesh_match(papers,bar.progress,email)
        finally:
            bar.empty()
    mesh=[]
    if "consensus_mesh" in st.session_state:
        mesh,matches,indexed=st.session_state["consensus_mesh"]
        st.write(f"{sum(bool(m['PMID']) for m in matches)}/{len(papers)} matched; {indexed} with MeSH.")
        st.dataframe(mesh,use_container_width=True)
        with st.expander("Matching audit"):
            st.dataframe(matches,use_container_width=True)
        download("Export MeSH ranking",mesh,"consensus_mesh.csv")
        download("Export matching audit",matches,"consensus_matches.csv")
    st.markdown("#### Build an initial PubMed query")
    count=st.number_input("Number of concept blocks",1,6,2,key="consensus_block_count")
    blocks=[]
    for i in range(int(count)):
        st.markdown(f"**Concept {i+1}**")
        free_options=[r["Term"] for r in rows[:200]]
        free_key=f"consensus_free_{i}"
        if free_key in st.session_state:
            st.session_state[free_key]=[t for t in st.session_state[free_key] if t in free_options]
        free=st.multiselect("Text terms",free_options,key=free_key)
        mesh_options=[r["Term"] for r in mesh]
        mesh_key=f"consensus_mesh_{i}"
        if mesh_key in st.session_state:
            st.session_state[mesh_key]=[t for t in st.session_state[mesh_key] if t in mesh_options]
        controlled=st.multiselect("MeSH descriptors",mesh_options,key=mesh_key)
        parts=[f'"{t}"[tiab]' for t in free]+[f'"{t}"[mh]' for t in controlled]
        if parts: blocks.append("("+" OR ".join(parts)+")")
    if blocks:
        if len(blocks)<count: st.warning("Some concept blocks are empty and omitted.")
        result="\nAND\n".join(blocks)
        st.code(result,language="text")
        st.download_button("Download draft query",result,"consensus_query.txt",key="consensus_query_dl")
    st.caption("Ranking uses log term frequency, smoothed corpus IDF and adjustable title weight. No stemming or semantic reranking. Validate the draft against known relevant studies.")
