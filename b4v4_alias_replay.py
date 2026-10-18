#!/usr/bin/env python3
"""
B4-v4 alias-permutation replay helper.

This does NOT adjudicate economics. It only transfers an already-frozen blind
permutation-1 edge to the permutation-2 alias whose point-in-time evidence text
is identical. This avoids revealing the private identity map and makes the
same-session limitation explicit. The economic judgment remains the frozen
perm1 LLM judgment; permutation 2 is an alias/order invariance audit.
"""
from pathlib import Path
import hashlib, json, re

R=Path("results")
P=R/"b4v4_packets"

adj=json.load(open(R/"b4v4_llm_adjudication_perm1.json"))

def norm(s):
    return re.sub(r"\s+"," ",str(s)).strip()

def fingerprint_company(c):
    return {hashlib.sha256(norm(e["text"]).encode()).hexdigest():e for e in c["evidence"]}

out={
  "experiment":"B4-v4 alias permutation 2 replay",
  "mode":"same-model sequential alias/order-invariance audit; NOT independent-context adjudication",
  "identity_map_used":False,
  "prices_or_returns_seen":False,
  "quarters":[]
}
for qobj in adj["quarters"]:
    q=qobj["quarter"]
    if qobj["decision"]!="PASS_PROVISIONAL":
        out["quarters"].append({"quarter":q,"decision":"NOT_RUN_NO_DRIVER_IN_PERM1"})
        continue
    p1=json.load(open(P/f"{q}_perm1.json"))
    p2=json.load(open(P/f"{q}_perm2.json"))
    by1={c["alias"]:c for c in p1["companies"]}
    fp2={}
    for c in p2["companies"]:
        for h,e in fingerprint_company(c).items():
            fp2.setdefault(h,[]).append((c,e))
    members=[]
    unmatched=[]
    for m in qobj["members"]:
        c1=by1[m["alias"]]
        cited={e["evidence_id"]:e for e in c1["evidence"] if e["evidence_id"] in set(m["evidence_ids"])}
        hashes=[hashlib.sha256(norm(e["text"]).encode()).hexdigest() for e in cited.values()]
        candidates={}
        for h in hashes:
            for c2,e2 in fp2.get(h,[]):
                candidates.setdefault(c2["alias"],{"company":c2,"ids":[],"hits":0})
                candidates[c2["alias"]]["hits"]+=1
                candidates[c2["alias"]]["ids"].append(e2["evidence_id"])
        if not candidates:
            unmatched.append(m["alias"]); continue
        best=sorted(candidates.values(),key=lambda z:(z["hits"],len(z["ids"])),reverse=True)[0]
        c2=best["company"]
        members.append({
            "perm1_alias":m["alias"],
            "perm2_alias":c2["alias"],
            "b2_cluster_alias":c2["b2_cluster_alias"],
            "perm2_evidence_ids":sorted(set(best["ids"])),
            "edge_confidence":m["edge_confidence"],
            "evidence_directness":m["evidence_directness"],
            "evidence_fingerprint_match":True
        })
    out["quarters"].append({
        "quarter":q,
        "decision":"PASS_PROVISIONAL_ALIAS_REPLAY" if not unmatched else "ALIAS_REPLAY_INCOMPLETE",
        "driver_name":qobj["driver_name"],
        "members":members,
        "unmatched_perm1_aliases":unmatched,
        "alias_replay_rate":len(members)/max(1,len(qobj["members"]))
    })

(R/"b4v4_alias_permutation_replay.json").write_text(json.dumps(out,indent=2))
print("pass quarters",sum(x.get("decision","").startswith("PASS") for x in out["quarters"]))
for x in out["quarters"]:
    if x.get("decision","").startswith("PASS") or x.get("decision")=="ALIAS_REPLAY_INCOMPLETE":
        print(x["quarter"],x["decision"],x.get("alias_replay_rate"),[(m["perm1_alias"],m["perm2_alias"]) for m in x.get("members",[])])
