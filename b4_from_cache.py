#!/usr/bin/env python3
"""
B4 cached causal-theme discovery.
Reads only frozen B2 structural outputs + persistent point-in-time causal-sentence cache.
No prices or forward returns are read.
"""
from pathlib import Path
import json, re
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import b4_causal as core

R=Path("results")
CACHE=R/"b4_cache_v2"

members=pd.read_csv(core.MEMBERS)
members=members[
    (members.cluster_rank<=core.TOP_B2_CLUSTERS) &
    (members.member_rank<=core.MAX_MEMBERS_PER_CLUSTER) &
    (~members.issuer.fillna("").str.contains(core.BAD_ISSUER))
].copy()

sent_files=sorted(CACHE.glob("b4_sentences_*.csv"))
audit_files=sorted(CACHE.glob("b4_audit_*.csv"))
if not sent_files or not audit_files:
    raise RuntimeError("B4 persistent cache missing")

sents=pd.concat([pd.read_csv(x,dtype={"cusip":str}) for x in sent_files],ignore_index=True)
audit=pd.concat([pd.read_csv(x,dtype={"cusip":str}) for x in audit_files],ignore_index=True)
members["cusip"]=members.cusip.astype(str)

driver_rows=[]
evidence_rows=[]
pick_rows=[]
past_docs=[]

quarters=sorted(members.quarter.unique(),key=lambda q:(int(str(q)[:4]),int(str(q)[-1])))
for q in quarters:
    print("B4-CACHED",q,flush=True)
    mq=members[members.quarter.eq(q)].sort_values(["cluster_rank","member_rank"]).drop_duplicates("cusip").copy()
    qs=sents[sents.quarter.eq(q)].copy()
    qdocs=[" ".join(g.sentence.astype(str)) for _,g in qs.groupby("cusip")]
    if qs.empty:
        pick_rows.append({"quarter":q,"decision":"NO_TRADE","reason":"no contemporaneous causal evidence in cache"})
        past_docs.append(qdocs); continue

    # Attach only point-in-time 13F features from the frozen B2 packet.
    keep=["cusip","ticker","issuer","cluster_id","abnormal_score","current_holders","buyer_ratio"]
    meta=mq[keep].drop_duplicates("cusip")
    sdf=qs[["cusip","sentence"]].merge(meta,on="cusip",how="inner")
    if sdf.cusip.nunique()<core.MIN_DRIVER_SENTENCE_COMPANIES:
        pick_rows.append({"quarter":q,"decision":"NO_TRADE","reason":"causal evidence covers fewer than three companies"})
        past_docs.append(qdocs); continue

    try:
        vec=TfidfVectorizer(
            stop_words=core.STOP_WORDS,lowercase=True,ngram_range=(1,2),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
            min_df=2,max_df=0.65,max_features=9000,sublinear_tf=True,
        )
        X=vec.fit_transform(sdf.sentence.astype(str).tolist())
    except Exception:
        pick_rows.append({"quarter":q,"decision":"NO_TRADE","reason":"causal TF-IDF vocabulary failure"})
        past_docs.append(qdocs); continue

    # Remove sentences that become zero vectors after the quarter-specific vocabulary filter.
    norms=np.sqrt(X.multiply(X).sum(axis=1)).A1
    good=norms>0
    if good.sum()<core.MIN_DRIVER_SENTENCE_COMPANIES:
        pick_rows.append({"quarter":q,"decision":"NO_TRADE","reason":"insufficient nonzero causal feature space"})
        past_docs.append(qdocs); continue
    if not good.all():
        sdf=sdf.loc[good].reset_index(drop=True)
        X=X[good]

    model=AgglomerativeClustering(
        n_clusters=None,distance_threshold=core.SENTENCE_CLUSTER_DISTANCE,
        metric="cosine",linkage="average",compute_full_tree=True
    )
    labels=model.fit_predict(X.toarray())
    sdf["driver_seed"]=labels

    names=np.array(vec.get_feature_names_out())
    global_centroid=np.asarray(X.mean(axis=0)).ravel()
    candidates=[]
    for lab,g in sdf.groupby("driver_seed"):
        if g.cusip.nunique()<core.MIN_DRIVER_SENTENCE_COMPANIES:
            continue
        idx=g.index.to_numpy()
        centroid=np.asarray(X[idx].mean(axis=0))
        discr=centroid.ravel()-global_centroid
        terms=[]
        for j in np.argsort(discr)[::-1]:
            if discr[j]<=0: break
            term=str(names[j])
            if any(w in core.GENERIC for w in term.split()): continue
            if re.search(r"\b(entity|filing|agreement|transaction)\b",term): continue
            terms.append(term)
            if len(terms)>=8: break
        if len(terms)<2:
            continue

        sim=cosine_similarity(X,centroid).ravel()
        tmp=sdf.assign(_sim=sim)
        exposure={}; ev_count={}; evidence={}
        for cusip,gg in tmp.groupby("cusip"):
            best=float(gg._sim.max())
            if best>=core.EXPANSION_SIMILARITY:
                exposure[str(cusip)]=best
                ev_count[str(cusip)]=int((gg._sim>=core.EXPANSION_SIMILARITY).sum())
                evidence[str(cusip)]=str(gg.sort_values("_sim",ascending=False).iloc[0].sentence)[:800]
        supporters=list(exposure)
        if len(supporters)<core.MIN_DRIVER_COMPANIES:
            continue

        pool=mq.copy()
        pool["_ab_pct"]=pool.abnormal_score.rank(pct=True,method="average")
        sm=pool[pool.cusip.isin(supporters)].copy()
        if sm.empty: continue
        counts=sm.groupby("cluster_id").size().sort_values(ascending=False)
        seed_cluster=str(counts.index[0])
        ext=sm[~sm.cluster_id.eq(seed_cluster)].copy()
        n_clusters=sm.cluster_id.nunique()
        if n_clusters<core.MIN_DRIVER_B2_CLUSTERS or len(ext)<1:
            continue

        direction=float(np.mean([min(1.0,0.65+0.15*ev_count[c]) for c in supporters]))
        breadth=float(min(1.0,len(supporters)/8.0))
        abnormal=float(sm._ab_pct.mean())
        # External support quality is measured versus the full quarter, not ranked only
        # inside the external subset.
        ext_ab=float(ext._ab_pct.mean()) if len(ext) else 0.0
        external=float(0.5*min(1.0,len(ext)/2.0)+0.5*ext_ab)
        novelty=core.past_novelty(terms,past_docs)
        theme_score=(
            core.THEME_W["breadth"]*breadth+
            core.THEME_W["direction"]*direction+
            core.THEME_W["abnormal"]*abnormal+
            core.THEME_W["external"]*external+
            core.THEME_W["novelty"]*novelty
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

    candidates=sorted(candidates,key=lambda x:(x["theme_score"],x["support_count"]),reverse=True)
    dedup=[]
    for c in candidates:
        ss=set(c["supporters"])
        if any(len(ss & set(d["supporters"]))/max(1,len(ss | set(d["supporters"])))>=0.70 for d in dedup):
            continue
        dedup.append(c)
    candidates=dedup

    for rank,c in enumerate(candidates,1):
        support_meta=mq[mq.cusip.isin(c["supporters"])]
        driver_rows.append({
            "quarter":q,"driver_rank":rank,"driver_terms":c["driver_terms"],
            "seed_cluster":c["seed_cluster"],"support_count":c["support_count"],
            "support_clusters":c["support_clusters"],"external_support_count":c["external_support_count"],
            "breadth":c["breadth"],"directional_consistency":c["directional_consistency"],
            "mean_abnormal_pct":c["mean_abnormal_pct"],"external_validation":c["external_validation"],
            "novelty":c["novelty"],"theme_score":c["theme_score"],
            "support_tickers":"|".join(support_meta.ticker.astype(str)),
        })
        for cusip in c["supporters"]:
            rr=mq[mq.cusip.eq(cusip)].iloc[0]
            evidence_rows.append({
                "quarter":q,"driver_rank":rank,"driver_terms":c["driver_terms"],
                "ticker":rr.ticker,"issuer":rr.issuer,"cusip":cusip,"cluster_id":rr.cluster_id,
                "driver_exposure":c["exposure"][cusip],"evidence_count":c["evidence_count"][cusip],
                "evidence_sentence":c["evidence"][cusip],
                "abnormal_score":rr.abnormal_score,"current_holders":rr.current_holders,
                "buyer_ratio":rr.buyer_ratio,
            })

    eligible=[c for c in candidates if c["theme_score"]>=core.MIN_THEME_SCORE]
    if not eligible:
        pick_rows.append({"quarter":q,"decision":"NO_TRADE","reason":"no causal driver passed frozen B4 threshold"})
        past_docs.append(qdocs); continue

    chosen=eligible[0]
    sm=mq[mq.cusip.isin(chosen["supporters"])].copy()
    sm["driver_exposure"]=sm.cusip.map(chosen["exposure"])
    sm["evidence_count"]=sm.cusip.map(chosen["evidence_count"])
    for col in ["abnormal_score","driver_exposure","current_holders","buyer_ratio","evidence_count"]:
        sm[col+"_pct"]=sm[col].rank(pct=True,method="average")
    sm["beneficiary_score"]=(
        core.BEN_W["abnormal"]*sm["abnormal_score_pct"]+
        core.BEN_W["driver_exposure"]*sm["driver_exposure_pct"]+
        core.BEN_W["holders"]*sm["current_holders_pct"]+
        core.BEN_W["buyer_ratio"]*sm["buyer_ratio_pct"].fillna(0.5)+
        core.BEN_W["evidence_count"]*sm["evidence_count_pct"]
    )
    sel=sm.sort_values(["beneficiary_score","abnormal_score"],ascending=[False,False]).head(core.PORTFOLIO_SIZE)
    if len(sel)<core.PORTFOLIO_SIZE:
        pick_rows.append({"quarter":q,"decision":"NO_TRADE","reason":"fewer than five deterministic beneficiaries"})
        past_docs.append(qdocs); continue
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

DRIVER_COLS=["quarter","driver_rank","driver_terms","seed_cluster","support_count","support_clusters","external_support_count","breadth","directional_consistency","mean_abnormal_pct","external_validation","novelty","theme_score","support_tickers"]
EVIDENCE_COLS=["quarter","driver_rank","driver_terms","ticker","issuer","cusip","cluster_id","driver_exposure","evidence_count","evidence_sentence","abnormal_score","current_holders","buyer_ratio"]
PICK_COLS=["quarter","decision","driver_terms","seed_cluster","theme_score","support_count","external_support_count","selected_tickers","selected_cusips","beneficiary_scores","reason"]

pd.DataFrame(driver_rows,columns=DRIVER_COLS).to_csv(R/"b4v2_driver_scores.csv",index=False)
pd.DataFrame(evidence_rows,columns=EVIDENCE_COLS).to_csv(R/"b4v2_driver_evidence.csv",index=False)
picks=pd.DataFrame(pick_rows,columns=PICK_COLS)
picks.to_csv(R/"b4v2_picks_frozen.csv",index=False)
audit.to_csv(R/"b4v2_document_audit.csv",index=False)

meta={
    "experiment":"B4-v2 cached point-in-time causal-theme engine",
    "forward_returns_used":False,
    "prices_read":False,
    "implementation_fix":"B4-v1 positive/negative/mechanical regex word-boundaries were accidentally stored as backspace characters. Fixed before any B4 forward returns were viewed.",
    "persistent_cache":"yearly point-in-time SEC causal-sentence cache under results/b4_cache_v2",
    "input":"B2 corporate-action-filtered co-buy clusters/members",
    "candidate_pool":f"top {core.TOP_B2_CLUSTERS} B2 clusters x top {core.MAX_MEMBERS_PER_CLUSTER} members, deduped by CUSIP",
    "causal_evidence_gate":"same B4 gate: contemporaneous causal/demand mechanism + positive business-direction marker; mechanical corporate-action/hypothetical adverse-risk evidence excluded",
    "driver_minimums":{"companies":core.MIN_DRIVER_COMPANIES,"B2_clusters":core.MIN_DRIVER_B2_CLUSTERS,"seed_sentence_companies":core.MIN_DRIVER_SENTENCE_COMPANIES},
    "out_of_seed_falsification":"at least one supporter outside the dominant B2 cluster",
    "theme_score_weights":core.THEME_W,
    "minimum_theme_score":core.MIN_THEME_SCORE,
    "beneficiary_score_weights":core.BEN_W,
    "portfolio_size":core.PORTFOLIO_SIZE,
    "manual_theme_selection":False,"manual_stock_replacement":False,
}
(R/"b4v2_methodology.json").write_text(json.dumps(meta,indent=2))
print(picks.to_string(index=False),flush=True)
print("B4V2 INVEST",int((picks.decision=="INVEST").sum()),"of",len(picks),"drivers",len(driver_rows),"evidence",len(evidence_rows),flush=True)