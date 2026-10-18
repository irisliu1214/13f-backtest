#!/usr/bin/env python3
"""
Build B4-v4 evidence-ablation packets from frozen permutation-1 adjudication.
No identities, prices or returns are read.

For each accepted edge, remove ALL evidence IDs cited by the LLM for that edge.
The ablation packet contains only the remaining contemporaneous evidence for
that alias. Later adjudication asks whether the edge can still be supported.
"""
from pathlib import Path
import json

R=Path("results")
P=R/"b4v4_packets"
OUT=R/"b4v4_ablation_packets"
OUT.mkdir(parents=True,exist_ok=True)

adj=json.load(open(R/"b4v4_llm_adjudication_perm1.json"))
index=[]

for qobj in adj["quarters"]:
    if qobj["decision"]!="PASS_PROVISIONAL":
        continue
    q=qobj["quarter"]
    packet=json.load(open(P/f"{q}_perm1.json"))
    by_alias={c["alias"]:c for c in packet["companies"]}
    edges=[]
    for m in qobj["members"]:
        c=by_alias[m["alias"]]
        cited=set(m["evidence_ids"])
        remaining=[e for e in c["evidence"] if e["evidence_id"] not in cited]
        edge={
            "alias":m["alias"],
            "b2_cluster_alias":m["b2_cluster_alias"],
            "baseline_edge_confidence":m["edge_confidence"],
            "baseline_evidence_directness":m["evidence_directness"],
            "removed_evidence_ids":sorted(cited),
            "remaining_evidence":remaining,
            "question":"Using ONLY remaining_evidence, does this alias still have evidence for the frozen driver/causal chain? Do not use memory of removed text or company identity."
        }
        edges.append(edge)
        index.append({
            "quarter":q,"alias":m["alias"],
            "baseline_edge_confidence":m["edge_confidence"],
            "removed_count":len(cited),"remaining_count":len(remaining)
        })
    out={
        "quarter":q,
        "driver_name":qobj["driver_name"],
        "frozen_causal_chain":qobj["causal_chain"],
        "instructions":{
            "identity_blind":True,
            "outside_knowledge_forbidden":True,
            "task":"Re-score each company-driver edge using only remaining_evidence after all originally cited evidence was removed.",
            "pass_dependence_rule":"Evidence dependence passes if new confidence drops >=0.20, OR new confidence <0.70, OR edge is removed."
        },
        "edges":edges
    }
    (OUT/f"{q}.json").write_text(json.dumps(out,indent=2))

(R/"b4v4_ablation_index.json").write_text(json.dumps(index,indent=2))
print("ablation quarters",len(list(OUT.glob("*.json"))),"edges",len(index))
for r in index:
    print(r["quarter"],r["alias"],"base",r["baseline_edge_confidence"],"remaining",r["remaining_count"])
