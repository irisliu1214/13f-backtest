#!/usr/bin/env python3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pandas as pd
import numpy as np
import b4_causal as core

YEAR=int(os.environ["B4_YEAR"])
OUT=Path("results/b4_cache_v2")
OUT.mkdir(parents=True,exist_ok=True)

members=pd.read_csv(core.MEMBERS)
members=members[
    (members.cluster_rank<=core.TOP_B2_CLUSTERS) &
    (members.member_rank<=core.MAX_MEMBERS_PER_CLUSTER) &
    (~members.issuer.fillna("").str.contains(core.BAD_ISSUER))
].copy()
members=members[members.quarter.astype(str).str.startswith(str(YEAR))].copy()
sec_index=core.load_company_index()

audit_rows=[]
sentence_rows=[]

for q in sorted(members.quarter.unique(), key=lambda x:int(str(x)[-1])):
    print("CACHE",q,flush=True)
    cutoff=core.legal_cutoff(q)
    mq=members[members.quarter.eq(q)].sort_values(["cluster_rank","member_rank"]).drop_duplicates("cusip").copy()
    rows=list(mq.itertuples(index=False))

    def load_one(r):
        cik,method=core.resolve_cik(r.ticker,r.issuer,sec_index)
        filings=[]; text=""
        if cik:
            try:
                filings=core.latest_filings(cik,cutoff,max_docs=2)
                texts=[core.fetch_doc(cik,f) for f in filings]
                text=" ".join(x for x in texts if x)
            except Exception as e:
                print("SEC_ERROR",q,r.ticker,repr(e),flush=True)
        sents=core.causal_sentences(text,r.issuer,r.ticker) if text else []
        audit={
            "quarter":q,"ticker":r.ticker,"issuer":r.issuer,"cusip":str(r.cusip),
            "cluster_id":r.cluster_id,"cluster_rank":r.cluster_rank,"member_rank":r.member_rank,
            "cik":cik,"resolve_method":method,"cutoff":cutoff.isoformat(),
            "filings":"|".join(f"{f.get('form')}:{f.get('filingDate')}:{f.get('accessionNumber')}" for f in filings),
            "causal_sentence_count":len(sents),
        }
        ss=[{
            "quarter":q,"ticker":r.ticker,"issuer":r.issuer,"cusip":str(r.cusip),
            "cluster_id":r.cluster_id,"sentence_rank":i+1,"sentence":s
        } for i,s in enumerate(sents)]
        return audit,ss

    loaded=[]
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs=[ex.submit(load_one,r) for r in rows]
        for fut in as_completed(futs):
            loaded.append(fut.result())
    loaded.sort(key=lambda x:(str(x[0]["quarter"]),str(x[0]["cusip"])))
    for audit,ss in loaded:
        audit_rows.append(audit)
        sentence_rows.extend(ss)

AUDIT_COLS=["quarter","ticker","issuer","cusip","cluster_id","cluster_rank","member_rank","cik","resolve_method","cutoff","filings","causal_sentence_count"]
SENT_COLS=["quarter","ticker","issuer","cusip","cluster_id","sentence_rank","sentence"]
pd.DataFrame(audit_rows,columns=AUDIT_COLS).to_csv(OUT/f"b4_audit_{YEAR}.csv",index=False)
pd.DataFrame(sentence_rows,columns=SENT_COLS).to_csv(OUT/f"b4_sentences_{YEAR}.csv",index=False)
print("YEAR",YEAR,"audit",len(audit_rows),"sentences",len(sentence_rows),flush=True)