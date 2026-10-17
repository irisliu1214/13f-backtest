#!/usr/bin/env python3
"""
B3-v2 deterministic theme discovery with strict business-section semantics.

Goal: prevent future knowledge from affecting portfolio membership.

Firewall:
- Inputs are B2 blind 13F cluster/member outputs plus SEC filings that were public
  before the quarter's legal 13F cutoff/entry date.
- No market prices or forward returns are read.
- No human chooses themes or beneficiaries.
- Company/ticker identity is not used in scoring. Identity is carried only as an
  audit key and for SEC document lookup.
- Economic coherence comes from TF-IDF similarity of point-in-time SEC business
  text, not a present-day hand-authored taxonomy or LLM world knowledge.
- Picks are written before any return evaluator is run.

Important limitation:
SEC CIK resolution uses SEC's current company-ticker/title index. Historical or
delisted issuers that cannot be resolved are treated as missing text; we never
fill them with present-day descriptions. This reduces coverage and can create
sample-selection bias, so coverage is emitted explicitly.
"""
from pathlib import Path
from datetime import date, timedelta
import hashlib, io, json, math, re, time
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

R = Path("results")
CLUSTERS = R / "b2_blind_clusters.csv"
MEMBERS = R / "b2_blind_cluster_members.csv"

TOP_STRUCTURAL_CLUSTERS = 3
MAX_TEXT_MEMBERS = 10
MIN_TEXT_DOCS = 5
MIN_TEXT_COVERAGE = 0.60
MIN_SEMANTIC_PERCENTILE = 0.65
PORTFOLIO_SIZE = 5
RANDOM_DRAWS = 250

UA = {
    "User-Agent": "13f-backtest academic research contact@example.com",
    "Accept-Encoding": "gzip, deflate",
}
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_SUB = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_ARCH = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no}/{primary}"
SUBMISSION_CACHE = {}
OLD_SUBMISSION_CACHE = {}
FILING_CACHE = {}

BAD_ISSUER = re.compile(
    r"(NUVEEN|POWERSHS|POWERSHARES|ISHARES|SPDR|VANGUARD|ETF|ETN|"
    r"MUNICIPAL|MUNCP|BLACKROCK.*FUND|PIMCO.*FUND|FIRST TR.*FUND|"
    r"INVESCO.*TRUST|PROSHARES|WISDOMTREE|GLOBAL X|GRAYSCALE)", re.I
)
CORP_WORDS = {
    "INC","INCORPORATED","CORP","CORPORATION","CO","COMPANY","LTD","LIMITED",
    "PLC","LLC","LP","HLDGS","HOLDINGS","NEW","DEL","THE","NV","SA","AG"
}
GENERIC_TERMS = {
    "company","companies","business","businesses","products","services","service",
    "customers","customer","market","markets","including","operations","operating",
    "provide","provides","provided","based","primarily","approximately","year",
    "years","use","uses","used","technology","technologies","solutions","system",
    "systems","development","develop","developed","sales","revenue","management",
    "registrant","securities","financial","statement","statements","fiscal","annual",
    "report","reports","item","section","act","exchange","commission","common","stock",
    "shares","share","risk","risks","may","could","would","believe","certain",
    "million","percent","december","january","table","contents","gaap","form",
    "filed","filing","results","material","adversely","affect","officer","president"
}
STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) | GENERIC_TERMS)

def q_end(q):
    y=int(q[:4]); n=int(q[-1])
    return {1:date(y,3,31),2:date(y,6,30),3:date(y,9,30),4:date(y,12,31)}[n]

def legal_cutoff(q):
    d=q_end(q)+timedelta(days=45)
    while d.weekday()>=5:
        d += timedelta(days=1)
    return d

def norm_name(s):
    toks=re.findall(r"[A-Z0-9]+", str(s or "").upper().replace("&"," AND "))
    return [t for t in toks if t not in CORP_WORDS and len(t)>1]

def load_company_index():
    r=requests.get(SEC_TICKERS,headers=UA,timeout=60); r.raise_for_status()
    obj=r.json()
    rows=[]
    for _,v in obj.items():
        rows.append({
            "cik":int(v["cik_str"]),
            "ticker":str(v.get("ticker","")).upper().strip(),
            "title":str(v.get("title","")).upper().strip(),
            "tokens":set(norm_name(v.get("title",""))),
        })
    return rows

def resolve_cik(ticker, issuer, idx):
    t=str(ticker or "").upper().strip().replace(".","-")
    exact=[x for x in idx if x["ticker"]==t]
    if exact:
        return exact[0]["cik"], "ticker_exact"
    it=set(norm_name(issuer))
    if len(it)<2:
        return None, "unresolved"
    best=None; best_score=0.0
    for x in idx:
        xt=x["tokens"]
        if not xt: continue
        inter=len(it & xt)
        if inter==0: continue
        score=inter/max(1,min(len(it),len(xt)))
        if inter>=2 and score>best_score:
            best_score=score; best=x
    if best is not None and best_score>=0.80:
        return best["cik"], f"title_match_{best_score:.2f}"
    return None, "unresolved"

def filing_rows_from_submission(obj):
    out=[]
    recent=obj.get("filings",{}).get("recent",{})
    if recent:
        keys=["accessionNumber","filingDate","form","primaryDocument"]
        n=len(recent.get("accessionNumber",[]))
        for i in range(n):
            out.append({k:(recent.get(k,[""]*n)[i] if i < len(recent.get(k,[])) else "") for k in keys})
    return out

def choose_filing(cik, cutoff):
    cache_key=(int(cik),cutoff.isoformat())
    if cache_key in FILING_CACHE:
        return FILING_CACHE[cache_key]
    if int(cik) in SUBMISSION_CACHE:
        obj=SUBMISSION_CACHE[int(cik)]
    else:
        url=SEC_SUB.format(cik=cik)
        r=requests.get(url,headers=UA,timeout=30)
        if r.status_code!=200:
            FILING_CACHE[cache_key]=None
            return None
        obj=r.json()
        SUBMISSION_CACHE[int(cik)]=obj
    rows=filing_rows_from_submission(obj)
    forms={"10-K","20-F","40-F"}
    candidates=[x for x in rows if x.get("form") in forms and x.get("filingDate") and x["filingDate"]<=cutoff.isoformat()]
    if not candidates:
        for f in obj.get("filings",{}).get("files",[])[:4]:
            name=f.get("name")
            if not name: continue
            old_key=(int(cik),name)
            if old_key in OLD_SUBMISSION_CACHE:
                old_obj=OLD_SUBMISSION_CACHE[old_key]
            else:
                rr=requests.get("https://data.sec.gov/submissions/"+name,headers=UA,timeout=30)
                if rr.status_code!=200: continue
                old_obj=rr.json(); OLD_SUBMISSION_CACHE[old_key]=old_obj
            rows2=filing_rows_from_submission({"filings":{"recent":old_obj}})
            candidates.extend([x for x in rows2 if x.get("form") in forms and x.get("filingDate") and x["filingDate"]<=cutoff.isoformat()])
            time.sleep(0.11)
    if not candidates:
        FILING_CACHE[cache_key]=None
        return None
    candidates=sorted(candidates,key=lambda x:(x["filingDate"],x["accessionNumber"]),reverse=True)
    FILING_CACHE[cache_key]=candidates[0]
    return FILING_CACHE[cache_key]

def clean_html_text(html):
    soup=BeautifulSoup(html,"html.parser")
    for tag in soup(["script","style","noscript"]):
        tag.decompose()
    text=soup.get_text(" ")
    text=re.sub(r"\s+"," ",text)
    return text

def _strict_section(text, start_patterns, end_patterns, min_chars=3000, max_chars=140000):
    """Return the longest plausible real section; never fall back to whole filing."""
    u=re.sub(r"\\s+"," ",text.upper())
    starts=[]
    for pat in start_patterns:
        starts.extend(m.start() for m in re.finditer(pat,u,re.I))
    candidates=[]
    for s in starts:
        ends=[]
        tail=u[s+100:]
        for pat in end_patterns:
            m=re.search(pat,tail,re.I)
            if m: ends.append(s+100+m.start())
        if not ends:
            continue
        e=min(ends)
        if e<=s:
            continue
        chunk=text[s:e]
        if min_chars<=len(chunk)<=max_chars:
            candidates.append(chunk)
    return max(candidates,key=len) if candidates else ""

def extract_business(text, form):
    if form=="10-K":
        return _strict_section(
            text,
            [r"ITEM\\s*1[\\.\\-—:]?\\s*BUSINESS\\b"],
            [r"ITEM\\s*1A[\\.\\-—:]?\\s*RISK\\s*FACTORS", r"ITEM\\s*2[\\.\\-—:]?\\s*PROPERTIES"]
        )
    if form=="20-F":
        return _strict_section(
            text,
            [r"ITEM\\s*4[\\.\\-—:]?\\s*INFORMATION\\s+ON\\s+THE\\s+COMPANY"],
            [r"ITEM\\s*4A[\\.\\-—:]?", r"ITEM\\s*5[\\.\\-—:]?"]
        )
    return ""
def fetch_business_text(cik, filing, cache):
    acc=filing["accessionNumber"].replace("-","")
    primary=filing.get("primaryDocument") or ""
    key=(cik,acc,primary)
    if key in cache: return cache[key]
    if not primary:
        cache[key]=""; return ""
    url=SEC_ARCH.format(cik=cik,acc_no=acc,primary=primary)
    try:
        r=requests.get(url,headers=UA,timeout=60)
        if r.status_code!=200:
            cache[key]=""; return ""
        txt=extract_business(clean_html_text(r.text),filing.get("form",""))
        cache[key]=txt
        time.sleep(0.12)
        return txt
    except Exception:
        cache[key]=""; return ""

def percentile_rank(s):
    return pd.Series(s).rank(pct=True,method="average").to_numpy()

def pair_mean(mat, idxs):
    if len(idxs)<2: return np.nan
    sub=mat[np.ix_(idxs,idxs)]
    tri=sub[np.triu_indices(len(idxs),1)]
    return float(np.mean(tri)) if len(tri) else np.nan

def stable_seed(q):
    return int(hashlib.sha256(q.encode()).hexdigest()[:8],16)

def main():
    clusters=pd.read_csv(CLUSTERS)
    members=pd.read_csv(MEMBERS)
    members=members[~members.issuer.fillna("").str.contains(BAD_ISSUER)].copy()
    clusters=clusters[clusters.cluster_rank<=TOP_STRUCTURAL_CLUSTERS].copy()

    idx=load_company_index()
    doc_cache={}
    audit=[]
    pick_rows=[]
    cluster_rows=[]

    for q in sorted(clusters.quarter.unique(), key=lambda x:(int(x[:4]),int(x[-1]))):
        print("B3 quarter",q,flush=True)
        cq=clusters[clusters.quarter.eq(q)].sort_values("cluster_rank")
        mq=members[members.quarter.eq(q) & members.cluster_id.isin(cq.cluster_id)].copy()
        if mq.empty: continue
        cutoff=legal_cutoff(q)

        # Fetch only top structural members. The ranking was generated without prices.
        wanted=(mq.sort_values(["cluster_rank","member_rank"])
                  .groupby("cluster_id",group_keys=False).head(MAX_TEXT_MEMBERS).copy())
        docs={}
        resolve_meta={}
        for r in wanted.itertuples(index=False):
            key=(r.cluster_id,r.ticker)
            cik,method=resolve_cik(r.ticker,r.issuer,idx)
            filing=None; txt=""
            if cik:
                filing=choose_filing(cik,cutoff)
                if filing:
                    txt=fetch_business_text(cik,filing,doc_cache)
            docs[key]=txt
            resolve_meta[key]=(cik,method,filing)
            audit.append({
                "quarter":q,"cluster_id":r.cluster_id,"ticker":r.ticker,"issuer":r.issuer,
                "cik":cik,"resolve_method":method,
                "cutoff":cutoff.isoformat(),
                "filing_date":filing.get("filingDate") if filing else None,
                "form":filing.get("form") if filing else None,
                "accession":filing.get("accessionNumber") if filing else None,
                "text_chars":len(txt),
                "usable_text":bool(len(txt)>=2000),
            })

        usable=wanted[[len(docs.get((r.cluster_id,r.ticker),""))>=2000 for r in wanted.itertuples(index=False)]].copy()
        if len(usable)<MIN_TEXT_DOCS:
            for cr in cq.itertuples(index=False):
                cluster_rows.append({"quarter":q,"cluster_id":cr.cluster_id,"status":"insufficient_quarter_text"})
            pick_rows.append({"quarter":q,"decision":"NO_TRADE","cluster_id":"","auto_terms":"","selected_tickers":"","reason":"insufficient point-in-time SEC text"})
            continue

        corpus=[docs[(r.cluster_id,r.ticker)] for r in usable.itertuples(index=False)]
        vec=TfidfVectorizer(
            stop_words=STOP_WORDS,lowercase=True,ngram_range=(1,2),
            token_pattern=r"(?u)\\b[a-zA-Z][a-zA-Z\\-]{2,}\\b",
            min_df=2,max_df=0.75,max_features=7000,sublinear_tf=True
        )
        try:
            X=vec.fit_transform(corpus)
        except Exception:
            pick_rows.append({"quarter":q,"decision":"NO_TRADE","cluster_id":"","auto_terms":"","selected_tickers":"","reason":"TF-IDF vocabulary failure"})
            continue
        S=cosine_similarity(X)
        pos={(r.cluster_id,r.ticker):i for i,r in enumerate(usable.itertuples(index=False))}
        rng=np.random.default_rng(stable_seed(q))

        qmetrics=[]
        for cr in cq.itertuples(index=False):
            gm=wanted[wanted.cluster_id.eq(cr.cluster_id)].copy()
            gu=usable[usable.cluster_id.eq(cr.cluster_id)].copy()
            idxs=[pos[(r.cluster_id,r.ticker)] for r in gu.itertuples(index=False)]
            coverage=len(gu)/max(1,len(gm))
            sem=pair_mean(S,idxs)
            draws=[]
            if len(idxs)>=2 and len(usable)>=len(idxs):
                allidx=np.arange(len(usable))
                for _ in range(RANDOM_DRAWS):
                    sample=rng.choice(allidx,size=len(idxs),replace=False)
                    draws.append(pair_mean(S,list(sample)))
            pct=float(np.mean(np.array(draws)<=sem)) if draws and np.isfinite(sem) else np.nan

            # Automatic label terms: cluster-specific business vocabulary.
            terms=[]
            if idxs:
                centroid=np.asarray(X[idxs].mean(axis=0)).ravel()
                global_centroid=np.asarray(X.mean(axis=0)).ravel()
                discr=centroid-global_centroid
                names=np.array(vec.get_feature_names_out())
                order=np.argsort(discr)[::-1]
                for j in order:
                    term=str(names[j])
                    if discr[j]<=0:
                        break
                    if any(w in GENERIC_TERMS for w in term.split()):
                        continue
                    if re.search(r"\\b(inc|corp|ltd|plc)\\b",term):
                        continue
                    terms.append(term)
                    if len(terms)>=8:
                        break

            qmetrics.append({
                "quarter":q,"cluster_id":cr.cluster_id,"cluster_rank":int(cr.cluster_rank),
                "cluster_size":int(cr.cluster_size),"cluster_score_raw":float(cr.cluster_score),
                "mean_abnormal":float(cr.mean_abnormal),"mean_similarity_13f":float(cr.mean_similarity),
                "edge_density":float(cr.edge_density),"text_n":len(gu),"text_coverage":coverage,
                "semantic_similarity":sem,"semantic_percentile":pct,
                "auto_terms":" | ".join(terms),
            })

        qm=pd.DataFrame(qmetrics)
        # All normalization is within the quarter, before returns exist.
        for col in ["cluster_score_raw","mean_abnormal","edge_density","semantic_percentile"]:
            qm[col+"_pct"]=qm[col].rank(pct=True,method="average")
        qm["theme_score"]=(
            0.35*qm["mean_abnormal_pct"]+
            0.25*qm["edge_density_pct"]+
            0.25*qm["semantic_percentile"].fillna(0.0)+
            0.15*qm["cluster_score_raw_pct"]
        )
        qm["eligible_theme"]=(
            (qm.text_n>=MIN_TEXT_DOCS)&
            (qm.text_coverage>=MIN_TEXT_COVERAGE)&
            (qm.semantic_percentile>=MIN_SEMANTIC_PERCENTILE)
        )
        cluster_rows.extend(qm.to_dict("records"))
        eligible=qm[qm.eligible_theme].sort_values(["theme_score","cluster_rank"],ascending=[False,True])
        if eligible.empty:
            pick_rows.append({"quarter":q,"decision":"NO_TRADE","cluster_id":"","auto_terms":"","selected_tickers":"","reason":"no cluster passed pre-registered semantic coherence threshold"})
            continue

        chosen=eligible.iloc[0]
        cid=chosen.cluster_id
        gm=usable[usable.cluster_id.eq(cid)].copy()
        if len(gm)<PORTFOLIO_SIZE:
            pick_rows.append({"quarter":q,"decision":"NO_TRADE","cluster_id":cid,"auto_terms":chosen.auto_terms,"selected_tickers":"","reason":"fewer than five members with point-in-time SEC text"})
            continue

        # Theme-fit = similarity of each member's historical business text to
        # the cluster centroid. Identity/ticker does not enter this score.
        idxs=[pos[(r.cluster_id,r.ticker)] for r in gm.itertuples(index=False)]
        centroid=np.asarray(X[idxs].mean(axis=0))
        fit=cosine_similarity(X[idxs],centroid).ravel()
        gm=gm.copy()
        gm["theme_fit"]=fit
        for col in ["abnormal_score","institutional_ownership_increase","buyer_ratio","holder_abs_change","theme_fit"]:
            gm[col+"_pct"]=gm[col].rank(pct=True,method="average")
        gm["beneficiary_score"]=(
            0.45*gm["abnormal_score_pct"]+
            0.20*gm["institutional_ownership_increase_pct"]+
            0.15*gm["buyer_ratio_pct"]+
            0.10*gm["holder_abs_change_pct"]+
            0.10*gm["theme_fit_pct"]
        )
        sel=gm.sort_values(["beneficiary_score","member_rank"],ascending=[False,True]).head(PORTFOLIO_SIZE)
        pick_rows.append({
            "quarter":q,"decision":"INVEST","cluster_id":cid,
            "auto_terms":chosen.auto_terms,
            "theme_score":float(chosen.theme_score),
            "semantic_percentile":float(chosen.semantic_percentile),
            "selected_tickers":"|".join(sel.ticker.astype(str)),
            "selected_cusips":"|".join(sel.cusip.astype(str)),
            "beneficiary_scores":"|".join(f"{x:.6f}" for x in sel.beneficiary_score),
            "reason":"deterministic top theme + deterministic top5; no forward returns read",
        })

    pd.DataFrame(audit).to_csv(R/"b3v2_document_audit.csv",index=False)
    pd.DataFrame(cluster_rows).to_csv(R/"b3v2_cluster_scores.csv",index=False)
    picks=pd.DataFrame(pick_rows)
    picks.to_csv(R/"b3v2_picks_frozen.csv",index=False)

    meta={
        "experiment":"B3-v2",
        "forward_returns_used":False,
        "input_clusters":"B2 blind corporate-action-filtered co-buy clusters",
        "semantic_source":"strict Item 1 Business from latest full SEC 10-K, or Item 4 from 20-F, filed on or before each quarter legal cutoff; amendments and whole-filing fallbacks prohibited",
        "semantic_model":"TF-IDF over strict historical business sections only; alphabetic tokens; SEC/accounting boilerplate stopwords; no LLM; no present-day hand-authored industry taxonomy",
        "cluster_candidates":TOP_STRUCTURAL_CLUSTERS,
        "max_text_members_per_cluster":MAX_TEXT_MEMBERS,
        "minimum_text_docs":MIN_TEXT_DOCS,
        "minimum_text_coverage":MIN_TEXT_COVERAGE,
        "minimum_semantic_percentile":MIN_SEMANTIC_PERCENTILE,
        "theme_score":"0.35 abnormal-percentile + 0.25 edge-density-percentile + 0.25 semantic-percentile + 0.15 structural-cluster-score-percentile",
        "beneficiary_score":"0.45 abnormal + 0.20 ownership-growth + 0.15 buyer-ratio + 0.10 holder-abs + 0.10 historical-text theme-fit; each percentile within chosen cluster",
        "portfolio_size":PORTFOLIO_SIZE,
        "manual_replacement_allowed":False,
        "ticker_or_identity_used_in_scoring":False,
        "known_limitation":"SEC CIK resolution begins from the current SEC company ticker/title index. Unresolved historical/delisted issuers are missing rather than backfilled with current descriptions; this can create sample-selection bias.",
    }
    (R/"b3v2_methodology.json").write_text(json.dumps(meta,indent=2))
    print(picks.to_string(index=False))
    print("INVEST quarters",int((picks.decision=="INVEST").sum()),"of",len(picks))

if __name__=="__main__":
    main()