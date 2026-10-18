#!/usr/bin/env python3
"""
Finalize B4-v4 after the blind causal graph is already frozen.
This step reveals identity to code only and mechanically selects Top5.
No price/return files are read.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import b4_causal as core

R=Path("results")
graph=json.load(open(R/"b4v4_final_graph_blind.json"))
imap=pd.read_csv(R/"b4v4_identity_map_private.csv",dtype={"cusip":str})
imap=imap[imap.permutation.eq(1)].copy()
members=pd.read_csv(core.MEMBERS,dtype={"cusip":str})
members["cusip"]=members.cusip.astype(str)

W={"abnormal_score_pct":0.40,"llm_edge_confidence":0.25,"buyer_ratio_pct":0.15,"current_holders_pct":0.10,"evidence_directness":0.10}

reveal=[]
picks=[]
for q in graph["quarters"]:
    quarter=q["quarter"]
    if q["decision"]!="PASS_BLIND_GRAPH":
        picks.append({"quarter":quarter,"decision":"NO_TRADE","reason":q.get("reason",q.get("stage","blind graph rejected"))})
        continue

    rows=[]
    for e in q["members"]:
        alias=e["alias"]
        mm=imap[(imap.quarter.eq(quarter))&(imap.alias.eq(alias))]
        if len(mm)!=1:
            raise RuntimeError((quarter,alias,"identity map",len(mm)))
        ident=mm.iloc[0]
        bm=members[
            (members.quarter.eq(quarter))&
            (members.cusip.eq(str(ident.cusip)))&
            (members.cluster_id.astype(str).eq(str(ident.cluster_id)))
        ]
        if bm.empty:
            # Fallback only to quarter+CUSIP; packet creation already deduped CUSIP.
            bm=members[(members.quarter.eq(quarter))&(members.cusip.eq(str(ident.cusip)))]
        if bm.empty:
            raise RuntimeError((quarter,alias,ident.cusip,"missing B2 member"))
        b=bm.sort_values(["cluster_rank","member_rank"]).iloc[0]
        row={
            "quarter":quarter,"alias":alias,"ticker":ident.ticker,"issuer":ident.issuer,
            "cusip":str(ident.cusip),"cluster_id":ident.cluster_id,
            "driver_name":q["driver_name"],"causal_chain":q["causal_chain"],
            "edge_confidence":float(e["edge_confidence"]),
            "evidence_directness":float(e["evidence_directness"]),
            "evidence_ids":"|".join(e["evidence_ids"]),
            "abnormal_score":float(b.abnormal_score),
            "buyer_ratio":float(b.buyer_ratio) if pd.notna(b.buyer_ratio) else np.nan,
            "current_holders":float(b.current_holders),
        }
        rows.append(row)
        reveal.append(row.copy())

    df=pd.DataFrame(rows)
    if len(df)<5:
        picks.append({
            "quarter":quarter,"decision":"NO_TRADE","driver_name":q["driver_name"],
            "support_count":len(df),
            "reason":"Blind causal graph passed audits but has fewer than five accepted companies, so deterministic Top5 cannot be formed."
        })
        continue

    df["abnormal_score_pct"]=df.abnormal_score.rank(pct=True,method="average")
    df["buyer_ratio_pct"]=df.buyer_ratio.rank(pct=True,method="average").fillna(0.5)
    df["current_holders_pct"]=df.current_holders.rank(pct=True,method="average")
    df["beneficiary_score"]=(
        W["abnormal_score_pct"]*df.abnormal_score_pct+
        W["llm_edge_confidence"]*df.edge_confidence+
        W["buyer_ratio_pct"]*df.buyer_ratio_pct+
        W["current_holders_pct"]*df.current_holders_pct+
        W["evidence_directness"]*df.evidence_directness
    )
    df=df.sort_values(["beneficiary_score","abnormal_score","ticker"],ascending=[False,False,True])
    sel=df.head(5)
    picks.append({
        "quarter":quarter,"decision":"INVEST","driver_name":q["driver_name"],
        "causal_chain":q["causal_chain"],"support_count":len(df),
        "dependence_pass_rate":q["dependence_pass_rate"],
        "selected_tickers":"|".join(sel.ticker.astype(str)),
        "selected_cusips":"|".join(sel.cusip.astype(str)),
        "beneficiary_scores":"|".join(f"{x:.6f}" for x in sel.beneficiary_score),
        "reason":"B4-v4 frozen blind causal graph + pre-registered mechanical Top5."
    })

pd.DataFrame(reveal).to_csv(R/"b4v4_identity_reveal_audit.csv",index=False)
p=pd.DataFrame(picks)
p.to_csv(R/"b4v4_picks_frozen.csv",index=False)
meta={
    "experiment":"B4-v4 finalization",
    "forward_returns_used":False,"prices_read":False,
    "blind_graph_source":"results/b4v4_final_graph_blind.json",
    "identity_reveal_after_blind_graph_freeze":True,
    "mechanical_top5_weights":W,
    "minimum_graph_members_for_trade":5,
    "manual_stock_replacement":False,
}
(R/"b4v4_finalization_methodology.json").write_text(json.dumps(meta,indent=2))
print(p.to_string(index=False))
print("INVEST",int((p.decision=="INVEST").sum()),"NO_TRADE",int((p.decision=="NO_TRADE").sum()))
