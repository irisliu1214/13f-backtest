#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, re
import numpy as np
import pandas as pd
import b4v3_1_firewall as fw
import b4_causal as core

R=Path("results")
CACHE=R/"b4_cache_v2"
OUT=R/"b4v4_packets"
OUT.mkdir(parents=True,exist_ok=True)

TOP_CLUSTERS=5
MAX_MEMBERS=10
MAX_EVIDENCE_PER_COMPANY=4
MIN_SENTENCE_SCORE=0.70

members=pd.read_csv(core.MEMBERS,dtype={"cusip":str})
members=members[
    (members.cluster_rank<=TOP_CLUSTERS)&
    (members.member_rank<=MAX_MEMBERS)&
    (~members.issuer.fillna("").str.contains(core.BAD_ISSUER))
].copy()
members["cusip"]=members.cusip.astype(str)

sents=pd.concat(
    [pd.read_csv(x,dtype={"cusip":str})[["quarter","cusip","sentence"]] for x in sorted(CACHE.glob("b4_sentences_*.csv"))],
    ignore_index=True
)

def realized_score(s):
    s=str(s)
    if fw.ACCOUNTING_INTERNAL.search(s): return 0.0
    if fw.RISK_HYPOTHETICAL.search(s): return 0.0
    trans=len(fw.TRANSMISSION.findall(s))
    pos=1 if fw.POSITIVE_DIRECTION.search(s) else 0
    ext=len(fw.EXTERNAL_ANCHORS.findall(s))
    # Keep a broad evidence pool for LLM reasoning: realized business effect plus
    # either an external context or explicit causal connector.
    conn=1 if re.search(r"\b(due to|driven by|because of|as a result of|resulting from|reflecting|attributable to|benefited from|in response to)\b",s,re.I) else 0
    if trans==0 or pos==0 or (ext==0 and conn==0): return 0.0
    return min(1.0,0.50+0.12*min(trans,2)+0.12*min(ext,2)+0.14*conn)

def mask_proper_nouns(s):
    s=str(s)
    # Existing cache already masks issuer/ticker. This adds best-effort masking of
    # multiword proper names while preserving economic nouns and acronyms like AI.
    s=re.sub(r"\b(?:[A-Z][a-z]{2,})(?:\s+[A-Z][a-z]{2,})+\b","ENTITY",s)
    s=re.sub(r"\b(?:Mr|Ms|Dr)\.\s+[A-Z][A-Za-z-]+\b","PERSON",s)
    return re.sub(r"\s+"," ",s).strip()

def alias_order(q,perm,cusips):
    def key(c):
        return hashlib.sha256(f"B4V4|{q}|P{perm}|{c}".encode()).hexdigest()
    ordered=sorted(cusips,key=key)
    return {c:f"C{i+1:02d}" for i,c in enumerate(ordered)}

quarters=sorted(members.quarter.unique(),key=lambda q:(int(str(q)[:4]),int(str(q)[-1])))
mapping_rows=[]
packets=[]
for q in quarters:
    mq=members[members.quarter.eq(q)].sort_values(["cluster_rank","member_rank"]).drop_duplicates("cusip").copy()
    qs=sents[sents.quarter.eq(q)].merge(
        mq[["cusip","ticker","issuer","cluster_id","cluster_rank","member_rank"]],
        on="cusip",how="inner"
    )
    scored=[]
    for r in qs.itertuples(index=False):
        sc=realized_score(r.sentence)
        if sc>=MIN_SENTENCE_SCORE:
            scored.append({
                "cusip":str(r.cusip),"ticker":r.ticker,"issuer":r.issuer,
                "cluster_id":r.cluster_id,"cluster_rank":int(r.cluster_rank),
                "member_rank":int(r.member_rank),"score":float(sc),
                "sentence":mask_proper_nouns(r.sentence)
            })
    ev=pd.DataFrame(scored)
    if ev.empty:
        base_companies=[]
    else:
        ev=ev.sort_values(["cusip","score"],ascending=[True,False]).groupby("cusip").head(MAX_EVIDENCE_PER_COMPANY)
        base_companies=sorted(ev.cusip.unique())

    for perm in [1,2]:
        amap=alias_order(q,perm,base_companies)
        companies=[]
        eid=0
        for cusip in base_companies:
            g=ev[ev.cusip.eq(cusip)].sort_values("score",ascending=False)
            rr=g.iloc[0]
            evid=[]
            for _,x in g.iterrows():
                eid+=1
                evid.append({
                    "evidence_id":f"{q}_P{perm}_E{eid:03d}",
                    "text":x.sentence
                })
            companies.append({
                "alias":amap[cusip],
                "b2_cluster_alias":f"G{int(rr.cluster_rank)}",
                "evidence":evid
            })
            mapping_rows.append({
                "quarter":q,"permutation":perm,"alias":amap[cusip],
                "cusip":cusip,"ticker":rr.ticker,"issuer":rr.issuer,
                "cluster_id":rr.cluster_id,"cluster_rank":int(rr.cluster_rank)
            })
        packet={
            "quarter":q,
            "permutation":perm,
            "cutoff_rule":"quarter end +45 days, contemporaneous SEC filings only",
            "instructions":{
                "task":"Identify zero or more shared upstream external economic drivers. Different wording may belong to the same causal chain. Do not require lexical similarity.",
                "no_outside_knowledge":"Use only the supplied evidence. Do not infer company identity. Do not recommend securities.",
                "edge_requirement":"Every accepted company-driver edge must cite evidence_id(s).",
                "driver_requirement":"Driver must be external to company bookkeeping/internal administration and must transmit into demand/orders/usage/pricing/revenue/volume/capacity/capex/customer activity.",
                "minimums":"A passed driver needs >=4 companies, >=2 B2 cluster aliases, >=2 companies outside its dominant cluster, and mean edge confidence >=0.70.",
                "output":"Return structured driver objects with driver_name, causal_chain, member_aliases, per_member evidence_ids/edge_confidence/evidence_directness, counterevidence, pass."
            },
            "companies":companies
        }
        packets.append(packet)
        (OUT/f"{q}_perm{perm}.json").write_text(json.dumps(packet,indent=2))

pd.DataFrame(mapping_rows).to_csv(R/"b4v4_identity_map_private.csv",index=False)
(R/"b4v4_packet_index.json").write_text(json.dumps([
    {"quarter":p["quarter"],"permutation":p["permutation"],"n_companies":len(p["companies"])}
    for p in packets
],indent=2))
print("quarters",len(quarters),"packets",len(packets),"mapped rows",len(mapping_rows))
print(pd.DataFrame([
    {"quarter":p["quarter"],"perm":p["permutation"],"n_companies":len(p["companies"])}
    for p in packets
]).to_string(index=False))