#!/usr/bin/env python3
"""
B4 point-in-time causal-theme discovery.

Purpose
-------
Identify common economic drivers behind institutional co-buying without using
future market returns or present-day narrative labels.

Information firewall
--------------------
- Inputs: B2 blind corporate-action-filtered cluster/member outputs.
- Point-in-time text: SEC filings filed on/before each quarter's legal 13F cutoff.
- No price, SPY, or forward-return file is imported or read.
- No human chooses a theme or beneficiary.
- Candidate drivers come only from contemporaneous causal/demand sentences.
- A driver must have out-of-seed-cluster support before it can become a theme.
- Beneficiaries are selected by a frozen deterministic score.

This is "causal-evidence" rather than causal proof: it requires contemporaneous,
directional, falsifiable common-driver evidence but does not claim structural
causal identification in the econometric sense.
"""
from pathlib import Path
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib, json, math, re, threading, time
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

R = Path("results")
CLUSTERS = R / "b2_blind_clusters.csv"
MEMBERS = R / "b2_blind_cluster_members.csv"

# ---------- pre-registered universe / thresholds ----------
TOP_B2_CLUSTERS = 5
MAX_MEMBERS_PER_CLUSTER = 12
MAX_CAUSAL_SENTENCES_PER_COMPANY = 12
MIN_DRIVER_COMPANIES = 5
MIN_DRIVER_B2_CLUSTERS = 2          # out-of-seed falsification
MIN_DRIVER_SENTENCE_COMPANIES = 3   # seed sentence-cluster support
SENTENCE_CLUSTER_DISTANCE = 0.72    # cosine distance => approx similarity >= .28
EXPANSION_SIMILARITY = 0.18
MIN_THEME_SCORE = 0.55
PORTFOLIO_SIZE = 5
PAST_NOVELTY_QUARTERS = 4

# Theme score:
# 30% evidence breadth, 20% directional consistency, 20% abnormal accumulation,
# 20% external/out-of-seed validation, 10% novelty.
THEME_W = {
    "breadth": 0.30,
    "direction": 0.20,
    "abnormal": 0.20,
    "external": 0.20,
    "novelty": 0.10,
}
# Beneficiary score, all percentiles within the accepted theme's supporters.
BEN_W = {
    "abnormal": 0.35,
    "driver_exposure": 0.30,
    "holders": 0.15,
    "buyer_ratio": 0.10,
    "evidence_count": 0.10,
}

UA = {
    "User-Agent": "13f-backtest academic research contact@example.com",
    "Accept-Encoding": "gzip, deflate",
}
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_SUB = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_ARCH = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{primary}"

SEC_LOCK = threading.Lock()
SEC_LAST = [0.0]
SUB_CACHE = {}
OLD_SUB_CACHE = {}
DOC_CACHE = {}

BAD_ISSUER = re.compile(
    r"(NUVEEN|POWERSHS|POWERSHARES|ISHARES|SPDR|VANGUARD|ETF|ETN|"
    r"MUNICIPAL|MUNCP|BLACKROCK.*FUND|PIMCO.*FUND|FIRST TR.*FUND|"
    r"INVESCO.*TRUST|PROSHARES|WISDOMTREE|GLOBAL X|GRAYSCALE)", re.I
)
CORP_WORDS = {
    "INC","INCORPORATED","CORP","CORPORATION","CO","COMPANY","LTD","LIMITED",
    "PLC","LLC","LP","HLDGS","HOLDINGS","NEW","DEL","THE","NV","SA","AG"
}

GENERIC = {
    "company","companies","business","businesses","customer","customers","product",
    "products","service","services","market","markets","year","years","quarter",
    "quarters","fiscal","annual","report","results","operations","operating",
    "revenue","revenues","sales","management","including","primarily","certain",
    "approximately","million","billion","item","form","registrant","securities",
    "financial","statement","statements","common","stock","shares","forward",
    "looking","forward-looking","believe","believes","expect","expects","expected",
    "may","could","would","general","various","related","increase","increased",
    "growth","demand","higher","strong","significant","primarily","due","result",
    "results","impact","impacts","benefit","benefits","benefited","supported",
    "driven","reflecting","based","period","ended","respectively"
}
STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) | GENERIC)

# Sentences have to contain a causal/demand mechanism AND a positive business
# direction. Risk-factor hypotheticals are explicitly downweighted/excluded.
CAUSAL_PATTERNS = [
    r"driven by", r"due to", r"because of", r"as a result of", r"resulting from",
    r"benefit(?:ed|ing|s)? from", r"supported by", r"fueled by", r"led by",
    r"in response to", r"reflecting", r"attributable to", r"demand for",
    r"adoption of", r"shift(?:ing)? (?:to|toward)", r"transition(?:ing)? to",
    r"investment in", r"spending on", r"capital expenditures?", r"capacity expansion",
    r"supply constraints?", r"shortage(?:s)? of", r"secular (?:growth|trend)",
]
POSITIVE_PATTERNS = [
    r"\bstrong(?:er)? demand\b", r"\bhigher demand\b", r"\bincreased demand\b",
    r"\bdemand (?:growth|increased|rose|grew)\b", r"\bgrowth\b",
    r"\bbenefit(?:ed|ing|s)?\b", r"\bexpansion\b", r"\badoption\b",
    r"\bopportunit(?:y|ies)\b", r"\btailwind\b", r"\bincreased spending\b",
    r"\bhigher (?:sales|revenue|volume|orders)\b", r"\bbacklog\b",
]
NEGATIVE_PATTERNS = [
    r"\bdeclin", r"\bdecreas", r"\blower demand\b", r"\bweak(?:er|ness)?\b",
    r"\bheadwind", r"\badverse", r"\bpressure", r"\breduced", r"\bslowdown",
    r"\bcould adversely\b", r"\bmay adversely\b", r"\bmaterially adverse\b",
]
MECHANICAL_PATTERNS = [
    r"\bacquisition", r"\bmerger", r"\bdivest", r"\brestructur", r"\bspin[- ]?off",
    r"\bforeign exchange\b", r"\bcurrency", r"\btax rate\b", r"\bshare repurchase",
    r"\bdebt refinancing\b", r"\baccounting", r"\blitigation",
]

def sec_get(url, timeout=35):
    err=None
    for attempt in range(3):
        with SEC_LOCK:
            wait=0.13-(time.monotonic()-SEC_LAST[0])  # <8 requests/sec globally
            if wait>0:
                time.sleep(wait)
            SEC_LAST[0]=time.monotonic()
        try:
            r=requests.get(url,headers=UA,timeout=timeout)
            if r.status_code==200:
                return r
            err=RuntimeError(f"HTTP {r.status_code} {url}")
            if r.status_code not in (429,500,502,503,504):
                return r
        except Exception as e:
            err=e
        time.sleep(0.6*(attempt+1))
    raise err or RuntimeError("SEC request failed")

def q_end(q):
    y=int(q[:4]); n=int(q[-1])
    return {1:date(y,3,31),2:date(y,6,30),3:date(y,9,30),4:date(y,12,31)}[n]

def legal_cutoff(q):
    d=q_end(q)+timedelta(days=45)
    while d.weekday()>=5:
        d+=timedelta(days=1)
    return d

def norm_tokens(s):
    toks=re.findall(r"[A-Z0-9]+",str(s or "").upper().replace("&"," AND "))
    return [x for x in toks if x not in CORP_WORDS and len(x)>1]

def load_company_index():
    r=sec_get(SEC_TICKERS,60); r.raise_for_status()
    out=[]
    for _,v in r.json().items():
        out.append({
            "cik":int(v["cik_str"]),
            "ticker":str(v.get("ticker","")).upper().strip(),
            "title":str(v.get("title","")).upper().strip(),
            "tokens":set(norm_tokens(v.get("title",""))),
        })
    return out

def resolve_cik(ticker,issuer,index):
    t=str(ticker or "").upper().strip().replace(".","-")
    exact=[x for x in index if x["ticker"]==t]
    if exact:
        return exact[0]["cik"],"ticker_exact"
    it=set(norm_tokens(issuer))
    if len(it)<2:
        return None,"unresolved"
    best=None; score=0.0
    for x in index:
        xt=x["tokens"]
        if not xt: continue
        inter=len(it & xt)
        s=inter/max(1,min(len(it),len(xt)))
        if inter>=2 and s>score:
            best=x; score=s
    if best and score>=0.80:
        return best["cik"],f"title_match_{score:.2f}"
    return None,"unresolved"

def filing_rows(obj):
    recent=obj.get("filings",{}).get("recent",{})
    n=len(recent.get("accessionNumber",[]))
    rows=[]
    for i in range(n):
        rows.append({
            "accessionNumber": recent.get("accessionNumber",[""]*n)[i],
            "filingDate": recent.get("filingDate",[""]*n)[i],
            "form": recent.get("form",[""]*n)[i],
            "primaryDocument": recent.get("primaryDocument",[""]*n)[i],
        })
    return rows

def get_submissions(cik):
    cik=int(cik)
    if cik not in SUB_CACHE:
        r=sec_get(SEC_SUB.format(cik=cik),45)
        if r.status_code!=200:
            return None
        SUB_CACHE[cik]=r.json()
    return SUB_CACHE[cik]

def latest_filings(cik,cutoff,max_docs=2):
    obj=get_submissions(cik)
    if not obj: return []
    forms={"10-K","10-Q","20-F"}
    rows=filing_rows(obj)
    candidates=[x for x in rows if x["form"] in forms and x["filingDate"] and x["filingDate"]<=cutoff.isoformat()]
    if len(candidates)<max_docs:
        for f in obj.get("filings",{}).get("files",[])[:4]:
            name=f.get("name")
            if not name: continue
            key=(int(cik),name)
            if key not in OLD_SUB_CACHE:
                rr=sec_get("https://data.sec.gov/submissions/"+name,35)
                if rr.status_code!=200: continue
                OLD_SUB_CACHE[key]=rr.json()
            old=OLD_SUB_CACHE[key]
            n=len(old.get("accessionNumber",[]))
            for i in range(n):
                row={k:(old.get(k,[""]*n)[i] if i<len(old.get(k,[])) else "") for k in
                     ["accessionNumber","filingDate","form","primaryDocument"]}
                if row["form"] in forms and row["filingDate"] and row["filingDate"]<=cutoff.isoformat():
                    candidates.append(row)
            if len(candidates)>=max_docs: break
    candidates=sorted(candidates,key=lambda x:(x["filingDate"],x["accessionNumber"]),reverse=True)
    # Prefer one latest interim/current filing plus the latest annual if distinct.
    chosen=[]
    for x in candidates:
        if x not in chosen:
            chosen.append(x)
        if len(chosen)>=max_docs: break
    return chosen

def html_text(html):
    soup=BeautifulSoup(html,"html.parser")
    for tag in soup(["script","style","noscript"]):
        tag.decompose()
    return re.sub(r"\s+"," ",soup.get_text(" "))

def fetch_doc(cik,filing):
    acc=filing["accessionNumber"].replace("-","")
    primary=filing.get("primaryDocument") or ""
    key=(int(cik),acc,primary)
    if key in DOC_CACHE:
        return DOC_CACHE[key]
    if not primary:
        DOC_CACHE[key]=""; return ""
    r=sec_get(SEC_ARCH.format(cik=int(cik),acc=acc,primary=primary),45)
    if r.status_code!=200:
        DOC_CACHE[key]=""; return ""
    txt=html_text(r.text)
    DOC_CACHE[key]=txt
    return txt

def split_sentences(text):
    # SEC filing HTML is flattened; this conservative splitter works better than
    # paragraph similarity and keeps the evidence human-auditable.
    text=re.sub(r"\s+"," ",text)
    parts=re.split(r"(?<=[.!?])\s+(?=[A-Z])",text)
    return parts

def mask_issuer(s,issuer,ticker):
    out=s
    for tok in sorted(set(norm_tokens(issuer)),key=len,reverse=True):
        if len(tok)>=4:
            out=re.sub(rf"\b{re.escape(tok)}\b"," ENTITY ",out,flags=re.I)
    if ticker:
        out=re.sub(rf"\b{re.escape(str(ticker))}\b"," ENTITY ",out,flags=re.I)
    return re.sub(r"\s+"," ",out).strip()

def sentence_score(s):
    low=s.lower()
    causal=sum(bool(re.search(p,low)) for p in CAUSAL_PATTERNS)
    pos=sum(bool(re.search(p,low)) for p in POSITIVE_PATTERNS)
    neg=sum(bool(re.search(p,low)) for p in NEGATIVE_PATTERNS)
    mech=sum(bool(re.search(p,low)) for p in MECHANICAL_PATTERNS)
    # Reject hypothetical risk-factor language and mechanical corporate events.
    modal=len(re.findall(r"\b(may|could|might)\b",low))
    if causal==0 or pos==0 or mech>0 or neg>pos or modal>=3:
        return None
    words=re.findall(r"[A-Za-z][A-Za-z\-]+",s)
    if len(words)<14 or len(words)>90:
        return None
    digits=len(re.findall(r"\d",s))
    if digits/max(1,len(s))>0.10:
        return None
    return 2*causal+pos-neg-2*mech

def causal_sentences(text,issuer,ticker):
    scored=[]
    seen=set()
    for s in split_sentences(text):
        sc=sentence_score(s)
        if sc is None: continue
        s=mask_issuer(s,issuer,ticker)
        key=re.sub(r"[^a-z]+"," ",s.lower()).strip()
        if len(key)<40 or key in seen: continue
        seen.add(key)
        scored.append((sc,s))
    scored=sorted(scored,key=lambda x:(x[0],len(x[1])),reverse=True)
    return [s for _,s in scored[:MAX_CAUSAL_SENTENCES_PER_COMPANY]]

def rank_pct(s):
    return pd.Series(s).rank(pct=True,method="average").to_numpy()

def quarter_seed(q):
    return int(hashlib.sha256(q.encode()).hexdigest()[:8],16)

def past_novelty(top_terms,past_quarter_company_docs):
    if not past_quarter_company_docs or not top_terms:
        return 1.0
    docs=[]
    for qdocs in past_quarter_company_docs[-PAST_NOVELTY_QUARTERS:]:
        docs.extend(qdocs)
    if not docs:
        return 1.0
    terms=[t.lower() for t in top_terms[:5]]
    hit=sum(any(t in d.lower() for t in terms) for d in docs)
    rate=hit/max(1,len(docs))
    return float(np.clip(1.0-rate,0,1))

def main():
    clusters=pd.read_csv(CLUSTERS)
    members=pd.read_csv(MEMBERS)
    clusters=clusters[clusters.cluster_rank<=TOP_B2_CLUSTERS].copy()
    members=members[
        (members.cluster_rank<=TOP_B2_CLUSTERS) &
        (members.member_rank<=MAX_MEMBERS_PER_CLUSTER) &
        (~members.issuer.fillna("").str.contains(BAD_ISSUER))
    ].copy()

    sec_index=load_company_index()
    doc_audit=[]
    evidence_rows=[]
    driver_rows=[]
    pick_rows=[]
    past_docs=[]

    quarters=sorted(clusters.quarter.unique(),key=lambda q:(int(q[:4]),int(q[-1])))
    for q in quarters:
        print("B4 quarter",q,flush=True)
        cutoff=legal_cutoff(q)
        mq=members[members.quarter.eq(q)].copy()
        if mq.empty:
            pick_rows.append({"quarter":q,"decision":"NO_TRADE","reason":"empty B2 causal candidate pool"})
            past_docs.append([])
            continue

        # One row per security; if duplicated across structural clusters retain the
        # strongest (lowest) cluster rank, then best member rank.
        mq=mq.sort_values(["cluster_rank","member_rank"]).drop_duplicates("cusip").copy()
        rows=list(mq.itertuples(index=False))

        def load_one(r):
            cik,method=resolve_cik(r.ticker,r.issuer,sec_index)
            filings=[]; text=""
            if cik:
                try:
                    filings=latest_filings(cik,cutoff,max_docs=2)
                    texts=[fetch_doc(cik,f) for f in filings]
                    text=" ".join(x for x in texts if x)
                except Exception as e:
                    print("SEC_ERROR",q,r.ticker,repr(e),flush=True)
            sents=causal_sentences(text,r.issuer,r.ticker) if text else []
            audit={
                "quarter":q,"ticker":r.ticker,"issuer":r.issuer,"cusip":r.cusip,
                "cluster_id":r.cluster_id,"cluster_rank":r.cluster_rank,"member_rank":r.member_rank,
                "cik":cik,"resolve_method":method,"cutoff":cutoff.isoformat(),
                "filings":"|".join(f"{f.get('form')}:{f.get('filingDate')}:{f.get('accessionNumber')}" for f in filings),
                "causal_sentence_count":len(sents),
            }
            return str(r.cusip),sents,audit

        loaded=[]
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs=[ex.submit(load_one,r) for r in rows]
            for fut in as_completed(futs):
                loaded.append(fut.result())
        loaded.sort(key=lambda x:x[0])
        sent_map={}
        for cusip,sents,audit in loaded:
            sent_map[cusip]=sents
            doc_audit.append(audit)

        qdocs=[" ".join(v) for v in sent_map.values() if v]
        # Build a sentence-level corpus with company metadata.
        sent_records=[]
        meta_by_cusip=mq.set_index(mq.cusip.astype(str),drop=False)
        for cusip,sents in sent_map.items():
            if cusip not in meta_by_cusip.index: continue
            rr=meta_by_cusip.loc[cusip]
            if isinstance(rr,pd.DataFrame): rr=rr.iloc[0]
            for s in sents:
                sent_records.append({
                    "cusip":cusip,"ticker":rr.ticker,"issuer":rr.issuer,
                    "cluster_id":rr.cluster_id,"abnormal_score":float(rr.abnormal_score),
                    "current_holders":float(rr.current_holders),
                    "buyer_ratio":float(rr.buyer_ratio) if pd.notna(rr.buyer_ratio) else np.nan,
                    "sentence":s,
                })
        if len(sent_records)<12:
            pick_rows.append({"quarter":q,"decision":"NO_TRADE","reason":"insufficient contemporaneous causal sentences"})
            past_docs.append(qdocs)
            continue

        sdf=pd.DataFrame(sent_records)
        try:
            vec=TfidfVectorizer(
                stop_words=STOP_WORDS,lowercase=True,ngram_range=(1,2),
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
                min_df=2,max_df=0.65,max_features=9000,sublinear_tf=True,
            )
            X=vec.fit_transform(sdf.sentence.tolist())
        except Exception:
            pick_rows.append({"quarter":q,"decision":"NO_TRADE","reason":"causal TF-IDF vocabulary failure"})
            past_docs.append(qdocs)
            continue
        if X.shape[0]<MIN_DRIVER_SENTENCE_COMPANIES or X.shape[1]<5:
            pick_rows.append({"quarter":q,"decision":"NO_TRADE","reason":"insufficient causal feature space"})
            past_docs.append(qdocs)
            continue

        # Deterministic agglomerative clustering of causal sentences.
        model=AgglomerativeClustering(
            n_clusters=None,distance_threshold=SENTENCE_CLUSTER_DISTANCE,
            metric="cosine",linkage="average",compute_full_tree=True
        )
        labels=model.fit_predict(X.toarray())
        sdf["driver_seed"]=labels

        names=np.array(vec.get_feature_names_out())
        global_centroid=np.asarray(X.mean(axis=0)).ravel()
        candidates=[]
        for lab,g in sdf.groupby("driver_seed"):
            seed_companies=g.cusip.nunique()
            if seed_companies<MIN_DRIVER_SENTENCE_COMPANIES:
                continue
            idx=g.index.to_numpy()
            centroid=np.asarray(X[idx].mean(axis=0))
            discr=centroid.ravel()-global_centroid
            order=np.argsort(discr)[::-1]
            terms=[]
            for j in order:
                if discr[j]<=0: break
                term=str(names[j])
                if any(w in GENERIC for w in term.split()): continue
                if re.search(r"\b(entity|filing|agreement|transaction)\b",term): continue
                terms.append(term)
                if len(terms)>=8: break
            if len(terms)<2:
                continue

            # Expand from seed sentences to all candidate companies using only
            # contemporaneous causal-sentence similarity to the driver centroid.
            sim=cosine_similarity(X,centroid).ravel()
            exposure={}
            ev_count={}
            evidence={}
            for cusip,gg in sdf.assign(_sim=sim).groupby("cusip"):
                best=float(gg._sim.max())
                if best>=EXPANSION_SIMILARITY:
                    exposure[cusip]=best
                    ev_count[cusip]=int((gg._sim>=EXPANSION_SIMILARITY).sum())
                    bestrow=gg.sort_values("_sim",ascending=False).iloc[0]
                    evidence[cusip]=str(bestrow.sentence)[:800]
            supporters=list(exposure)
            if len(supporters)<MIN_DRIVER_COMPANIES:
                continue
            sm=mq[mq.cusip.astype(str).isin(supporters)].copy()
            if sm.empty: continue
            # Seed cluster = cluster contributing most supporters. Require support
            # from at least one other B2 co-buy community (falsification check).
            counts=sm.groupby("cluster_id").size().sort_values(ascending=False)
            seed_cluster=str(counts.index[0])
            ext=sm[~sm.cluster_id.eq(seed_cluster)]
            n_clusters=sm.cluster_id.nunique()
            if n_clusters<MIN_DRIVER_B2_CLUSTERS or len(ext)<1:
                continue

            # Directional consistency: all input sentences passed an explicitly
            # positive mechanism gate; penalize companies with only a single weak
            # piece of evidence.
            direction=float(np.mean([min(1.0,0.65+0.15*ev_count[c]) for c in supporters]))
            breadth=float(min(1.0,len(supporters)/8.0))

            # Cross-sectional abnormal percentile within the quarter's causal pool.
            pool=mq.copy()
            pool["_ab_pct"]=pool.abnormal_score.rank(pct=True,method="average")
            sm=sm.merge(pool[["cusip","_ab_pct"]],on="cusip",how="left")
            abnormal=float(sm._ab_pct.mean())

            ext_ab=float(ext.abnormal_score.rank(pct=True,method="average").mean()) if len(ext) else 0.0
            external=float(0.5*min(1.0,len(ext)/2.0)+0.5*ext_ab)
            novelty=past_novelty(terms,past_docs)

            theme_score=(
                THEME_W["breadth"]*breadth+
                THEME_W["direction"]*direction+
                THEME_W["abnormal"]*abnormal+
                THEME_W["external"]*external+
                THEME_W["novelty"]*novelty
            )
            candidates.append({
                "driver_seed":int(lab),"driver_terms":" | ".join(terms),
                "seed_cluster":seed_cluster,"supporters":supporters,
                "support_count":len(supporters),"support_clusters":int(n_clusters),
                "external_support_count":int(len(ext)),"breadth":breadth,
                "directional_consistency":direction,"mean_abnormal_pct":abnormal,
                "external_validation":external,"novelty":novelty,
                "theme_score":float(theme_score),"exposure":exposure,
                "evidence_count":ev_count,"evidence":evidence,
            })

        # De-duplicate near-identical driver candidates by supporter set.
        candidates=sorted(candidates,key=lambda x:(x["theme_score"],x["support_count"]),reverse=True)
        dedup=[]
        for c in candidates:
            ss=set(c["supporters"])
            if any(len(ss & set(d["supporters"]))/max(1,len(ss | set(d["supporters"])))>=0.70 for d in dedup):
                continue
            dedup.append(c)
        candidates=dedup

        for rank,c in enumerate(candidates,1):
            driver_rows.append({
                "quarter":q,"driver_rank":rank,"driver_terms":c["driver_terms"],
                "seed_cluster":c["seed_cluster"],"support_count":c["support_count"],
                "support_clusters":c["support_clusters"],"external_support_count":c["external_support_count"],
                "breadth":c["breadth"],"directional_consistency":c["directional_consistency"],
                "mean_abnormal_pct":c["mean_abnormal_pct"],"external_validation":c["external_validation"],
                "novelty":c["novelty"],"theme_score":c["theme_score"],
                "support_tickers":"|".join(mq[mq.cusip.astype(str).isin(c["supporters"])].ticker.astype(str)),
            })
            for cusip in c["supporters"]:
                rr=mq[mq.cusip.astype(str).eq(cusip)].iloc[0]
                evidence_rows.append({
                    "quarter":q,"driver_rank":rank,"driver_terms":c["driver_terms"],
                    "ticker":rr.ticker,"issuer":rr.issuer,"cusip":cusip,"cluster_id":rr.cluster_id,
                    "driver_exposure":c["exposure"][cusip],"evidence_count":c["evidence_count"][cusip],
                    "evidence_sentence":c["evidence"][cusip],
                    "abnormal_score":rr.abnormal_score,"current_holders":rr.current_holders,
                    "buyer_ratio":rr.buyer_ratio,
                })

        eligible=[c for c in candidates if c["theme_score"]>=MIN_THEME_SCORE]
        if not eligible:
            pick_rows.append({"quarter":q,"decision":"NO_TRADE","reason":"no causal driver passed frozen B4 threshold"})
            past_docs.append(qdocs)
            continue

        chosen=eligible[0]
        sm=mq[mq.cusip.astype(str).isin(chosen["supporters"])].copy()
        sm["driver_exposure"]=sm.cusip.astype(str).map(chosen["exposure"])
        sm["evidence_count"]=sm.cusip.astype(str).map(chosen["evidence_count"])
        for col in ["abnormal_score","driver_exposure","current_holders","buyer_ratio","evidence_count"]:
            sm[col+"_pct"]=sm[col].rank(pct=True,method="average")
        sm["beneficiary_score"]=(
            BEN_W["abnormal"]*sm["abnormal_score_pct"]+
            BEN_W["driver_exposure"]*sm["driver_exposure_pct"]+
            BEN_W["holders"]*sm["current_holders_pct"]+
            BEN_W["buyer_ratio"]*sm["buyer_ratio_pct"].fillna(0.5)+
            BEN_W["evidence_count"]*sm["evidence_count_pct"]
        )
        sel=sm.sort_values(["beneficiary_score","abnormal_score"],ascending=[False,False]).head(PORTFOLIO_SIZE)
        if len(sel)<PORTFOLIO_SIZE:
            pick_rows.append({"quarter":q,"decision":"NO_TRADE","reason":"fewer than five deterministic beneficiaries"})
            past_docs.append(qdocs)
            continue
        pick_rows.append({
            "quarter":q,"decision":"INVEST","driver_terms":chosen["driver_terms"],
            "seed_cluster":chosen["seed_cluster"],"theme_score":chosen["theme_score"],
            "support_count":chosen["support_count"],"external_support_count":chosen["external_support_count"],
            "selected_tickers":"|".join(sel.ticker.astype(str)),
            "selected_cusips":"|".join(sel.cusip.astype(str)),
            "beneficiary_scores":"|".join(f"{x:.6f}" for x in sel.beneficiary_score),
            "reason":"B4 deterministic causal-evidence driver + out-of-seed validation + deterministic Top5",
        })
        past_docs.append(qdocs)

    pd.DataFrame(doc_audit).to_csv(R/"b4_document_audit.csv",index=False)
    pd.DataFrame(driver_rows).to_csv(R/"b4_driver_scores.csv",index=False)
    pd.DataFrame(evidence_rows).to_csv(R/"b4_driver_evidence.csv",index=False)
    picks=pd.DataFrame(pick_rows)
    picks.to_csv(R/"b4_picks_frozen.csv",index=False)

    meta={
        "experiment":"B4 point-in-time causal-theme engine",
        "forward_returns_used":False,
        "prices_read":False,
        "input":"B2 corporate-action-filtered co-buy clusters/members",
        "point_in_time_text":"latest two full SEC 10-K/10-Q/20-F filings filed on or before legal 13F cutoff",
        "candidate_pool":f"top {TOP_B2_CLUSTERS} B2 clusters x top {MAX_MEMBERS_PER_CLUSTER} members, deduped by CUSIP",
        "causal_evidence_gate":"sentence must contain a causal/demand mechanism and positive business-direction marker; mechanical corporate-action and hypothetical adverse-risk sentences excluded",
        "driver_discovery":f"TF-IDF on causal sentences only; agglomerative cosine distance threshold {SENTENCE_CLUSTER_DISTANCE}; expand company support at cosine >= {EXPANSION_SIMILARITY}",
        "driver_minimums":{
            "companies":MIN_DRIVER_COMPANIES,
            "B2_clusters":MIN_DRIVER_B2_CLUSTERS,
            "seed_sentence_companies":MIN_DRIVER_SENTENCE_COMPANIES,
        },
        "out_of_seed_falsification":"accepted driver must be supported by at least one company outside the B2 cluster contributing the most supporters",
        "theme_score_weights":THEME_W,
        "minimum_theme_score":MIN_THEME_SCORE,
        "novelty":f"1 minus prior support rate of top driver terms over previous {PAST_NOVELTY_QUARTERS} quarters only",
        "beneficiary_score_weights":BEN_W,
        "portfolio_size":PORTFOLIO_SIZE,
        "manual_theme_selection":False,
        "manual_stock_replacement":False,
        "known_limitations":[
            "Current SEC ticker/title index is used only for CIK resolution; unresolved historical/delisted issuers remain missing.",
            "Causal-evidence language is observational textual evidence, not econometric causal identification.",
            "Only SEC filings are used in this first B4 implementation; earnings-call/news evidence is not yet included.",
        ],
    }
    (R/"b4_methodology.json").write_text(json.dumps(meta,indent=2))
    print(picks.to_string(index=False),flush=True)
    print("B4 INVEST",int((picks.decision=="INVEST").sum()),"of",len(picks),flush=True)

if __name__=="__main__":
    main()