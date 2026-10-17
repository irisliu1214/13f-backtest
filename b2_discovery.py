#!/usr/bin/env python3
"""
Experiment B2 discovery engine.

Information firewall:
  - No price or forward-return data are read here.
  - Corporate-action-like holder discontinuities are removed before clustering.
  - Clusters are discovered from common institutional buyers, not from a
    present-day theme taxonomy.
  - Economic labels and beneficiary picks are added only after this script
    produces the blind cluster packet.
"""
from pathlib import Path
from datetime import timedelta
import json, math
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse

import backtest as bt

OUT = Path("results")
MIN_HOLDERS = 20
MIN_VALUE_USD = 25_000_000
NETWORK_CANDIDATES = 250
EDGE_THRESHOLD = 0.08
MAX_NEIGHBORS = 10
MIN_CLUSTER_SIZE = 3
TOP_CLUSTERS_PER_Q = 8
TOP_MEMBERS_PER_CLUSTER = 18

def pct_rank(s):
    x = pd.to_numeric(s, errors="coerce")
    return x.rank(pct=True, method="average")

def corporate_action_flags(m):
    prev = pd.to_numeric(m["prev_holders"], errors="coerce").fillna(0)
    cur = pd.to_numeric(m["cur_holders"], errors="coerce").fillna(0)
    pct = pd.to_numeric(m["holder_pct_change"], errors="coerce")
    habs = pd.to_numeric(m["holder_abs_change"], errors="coerce").fillna(0)
    active = pd.to_numeric(m["active_changers"], errors="coerce").fillna(0)
    br = pd.to_numeric(m["buyer_ratio"], errors="coerce")

    zero_base = prev <= 0
    near_zero_reset = (prev <= 5) & (cur >= 50) & (pct >= 5)
    mass_holder_reset = (cur >= 100) & (pct >= 8)
    mechanical_distribution = (
        (cur >= 100) & (habs >= 75) & (pct >= 1.5) &
        (active >= 5) & br.notna() & (br <= 0.25)
    )
    return zero_base, near_zero_reset, mass_holder_reset, mechanical_distribution

def manager_buy_matrix(db, q, metrics, candidates):
    """Return stock x manager weighted buyer matrix for candidate CUSIPs."""
    if candidates.empty:
        return None, [], []

    pq = bt.prev_quarter(q)
    cutoff = bt.cutoff_for(q)
    pcutoff = bt.cutoff_for(pq)

    ctbl = candidates[["cusip","ticker"]].drop_duplicates("cusip").copy()
    stbl = metrics[["cusip","split_factor"]].drop_duplicates("cusip").copy()
    db.register("b2_cands", ctbl[["cusip"]])
    db.register("b2_splits", stbl)

    pos = db.execute("""
      WITH cf AS (
        SELECT accession,cik,
               ROW_NUMBER() OVER(PARTITION BY cik ORDER BY available_at DESC, accession DESC) rn
        FROM filings
        WHERE period=? AND CAST(available_at AS DATE)<=?
          AND cik IN (SELECT cik FROM matched_mgrs)
      ), pf AS (
        SELECT accession,cik,
               ROW_NUMBER() OVER(PARTITION BY cik ORDER BY available_at DESC, accession DESC) rn
        FROM filings
        WHERE period=? AND CAST(available_at AS DATE)<=?
          AND cik IN (SELECT cik FROM matched_mgrs)
      ), cpos AS (
        SELECT f.cik,h.cusip,SUM(h.shares) shares
        FROM cf f JOIN holdings h USING(accession)
        WHERE f.rn=1 AND h.cusip IN (SELECT cusip FROM b2_cands)
        GROUP BY f.cik,h.cusip
      ), ppos AS (
        SELECT f.cik,h.cusip,SUM(h.shares) shares
        FROM pf f JOIN holdings h USING(accession)
        WHERE f.rn=1 AND h.cusip IN (SELECT cusip FROM b2_cands)
        GROUP BY f.cik,h.cusip
      )
      SELECT c.cik,c.cusip,c.shares cur_shares,
             COALESCE(p.shares,0) prev_shares,
             s.split_factor
      FROM cpos c
      LEFT JOIN ppos p ON c.cik=p.cik AND c.cusip=p.cusip
      JOIN b2_splits s ON c.cusip=s.cusip
    """, [q, cutoff, pq, pcutoff]).df()
    if pos.empty:
        return None, [], []

    pos["is_buy"] = (
        (pos["prev_shares"] <= 0) |
        (pos["cur_shares"] > pos["prev_shares"] * pos["split_factor"] * (1.0 + bt.BUYER_EPS))
    )
    buys = pos[pos.is_buy].copy()
    if buys.empty:
        return None, [], []

    # Downweight managers that bought many candidate stocks this quarter. This
    # reduces broad index/rebalance style co-movement.
    activity = buys.groupby("cik").size().rename("n_buys")
    buys = buys.merge(activity, on="cik", how="left")
    buys["weight"] = 1.0 / np.sqrt(buys["n_buys"].clip(lower=1))

    cusips = ctbl.cusip.astype(str).tolist()
    managers = sorted(buys.cik.astype(str).unique().tolist())
    ri = {c:i for i,c in enumerate(cusips)}
    ci = {c:i for i,c in enumerate(managers)}
    rows=[]; cols=[]; vals=[]
    for r in buys.itertuples(index=False):
        c=str(r.cusip); k=str(r.cik)
        if c in ri and k in ci:
            rows.append(ri[c]); cols.append(ci[k]); vals.append(float(r.weight))
    X = sparse.csr_matrix((vals,(rows,cols)), shape=(len(cusips),len(managers)), dtype=float)
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
    norms[norms==0]=1.0
    X = sparse.diags(1.0/norms) @ X
    return X, cusips, managers

def discover_clusters(db, q, qm, full_metrics):
    cand = qm.sort_values("abnormal_score", ascending=False).head(NETWORK_CANDIDATES).copy()
    X, cusips, managers = manager_buy_matrix(db, q, full_metrics, cand)
    if X is None or len(cusips) < MIN_CLUSTER_SIZE:
        return []

    sim = (X @ X.T).toarray()
    np.fill_diagonal(sim, 0.0)
    G = nx.Graph()
    G.add_nodes_from(range(len(cusips)))

    for i in range(len(cusips)):
        order = np.argsort(sim[i])[::-1][:MAX_NEIGHBORS]
        for j in order:
            if j <= i:
                continue
            w=float(sim[i,j])
            if w >= EDGE_THRESHOLD:
                G.add_edge(i,j,weight=w)

    clusters=[]
    for comp in nx.connected_components(G):
        if len(comp) < MIN_CLUSTER_SIZE:
            continue
        sub=G.subgraph(comp)
        if sub.number_of_edges()==0:
            continue
        try:
            comms=list(nx.algorithms.community.greedy_modularity_communities(sub, weight="weight"))
        except Exception:
            comms=[set(comp)]
        for comm in comms:
            if len(comm) < MIN_CLUSTER_SIZE:
                continue
            idx=sorted(comm)
            cs=[cusips[i] for i in idx]
            mm=cand[cand.cusip.astype(str).isin(cs)].copy()
            if len(mm) < MIN_CLUSTER_SIZE:
                continue
            pair_vals=[]
            for a in range(len(idx)):
                for b in range(a+1,len(idx)):
                    pair_vals.append(float(sim[idx[a],idx[b]]))
            mean_sim=float(np.mean(pair_vals)) if pair_vals else 0.0
            edge_density=float(np.mean(np.array(pair_vals)>=EDGE_THRESHOLD)) if pair_vals else 0.0
            mean_ab=float(mm.abnormal_score.mean())
            cluster_score = mean_ab * (1.0 + 0.18*math.log1p(len(mm))) * (0.70 + mean_sim)
            clusters.append({
                "size":int(len(mm)),
                "mean_abnormal":mean_ab,
                "mean_similarity":mean_sim,
                "edge_density":edge_density,
                "cluster_score":float(cluster_score),
                "members":mm.sort_values("abnormal_score",ascending=False),
            })
    clusters=sorted(clusters,key=lambda x:(x["cluster_score"],x["size"]),reverse=True)
    return clusters[:TOP_CLUSTERS_PER_Q]

def main():
    bt.SEC_FUND_TICKERS = bt.load_sec_fund_tickers()
    bt.SEC_FUND_TICKERS_BY_YEAR = bt.load_sec_annual_fund_tickers()
    bt.SEC_CEF_NAMES_BY_YEAR = bt.load_sec_cef_names_by_year()
    snap = bt.load_snapshot()
    paths = bt.cache_dataset_files(snap)
    db = bt.build_db(paths)

    allq=[]
    q_meta=[]
    quarters=bt.quarter_ends(bt.START_YEAR, bt.END_YEAR)
    for q in quarters:
        print("B2 metrics", bt.qlabel(q), flush=True)
        m, meta=bt.aggregate_quarter(db,q)
        q_meta.append({"quarter":bt.qlabel(q),"period_end":q.isoformat(),**meta})
        if m.empty:
            continue
        m=m.copy()
        m["quarter"]=bt.qlabel(q)
        m["period_end"]=q.isoformat()
        zero,near,mass,dist=corporate_action_flags(m)
        m["corp_zero_base"]=zero
        m["corp_near_zero_reset"]=near
        m["corp_mass_holder_reset"]=mass
        m["corp_mechanical_distribution"]=dist
        m["corporate_action_filter"] = zero | near | mass | dist
        allq.append(m)

    full=pd.concat(allq,ignore_index=True)
    full["current_holders"]=full["cur_holders"]
    full["prior_holders"]=full["prev_holders"]
    full["current_value_usd"]=full["cur_value"]

    # Base research universe. We intentionally require a nontrivial existing
    # institutional base; this avoids tiny microcaps dominating percent-growth.
    eligible = full[
        (~full.corporate_action_filter)
        & (full.prev_holders > 0)
        & (full.cur_holders >= MIN_HOLDERS)
        & (full.cur_value >= MIN_VALUE_USD)
        & (full.holder_abs_change > 0)
    ].copy()

    # Cross-sectional abnormal accumulation: breadth, composite, buyer conviction,
    # and split-adjusted position-depth growth. Percentile ranks are robust to the
    # long tails in raw percentage changes.
    for col,new in [
        ("holder_abs_change","p_holder_abs"),
        ("composite_accumulation_score","p_composite"),
        ("institutional_ownership_increase","p_ownership"),
        ("buyer_ratio","p_buyer"),
    ]:
        eligible[new]=eligible.groupby("quarter")[col].transform(pct_rank)

    eligible["p_buyer"]=eligible["p_buyer"].where(eligible.active_changers>=5)
    parts=eligible[["p_holder_abs","p_composite","p_ownership","p_buyer"]].copy()
    weights=np.array([0.35,0.25,0.20,0.20])
    arr=parts.to_numpy(dtype=float)
    valid=np.isfinite(arr)
    num=np.nansum(arr*weights,axis=1)
    den=np.sum(valid*weights,axis=1)
    eligible["cross_score"]=np.where(den>0,num/den,np.nan)

    # Own-history surprise: compare current cross score with the same stock's
    # prior eight available quarters. No future quarter enters this feature.
    eligible["_qidx"]=eligible["quarter"].map({bt.qlabel(q):i for i,q in enumerate(quarters)})
    eligible=eligible.sort_values(["ticker","_qidx"])
    hist_mean=[]; hist_std=[]; hist_n=[]
    for _,g in eligible.groupby("ticker",sort=False):
        vals=g.cross_score.astype(float)
        hm=vals.shift(1).rolling(8,min_periods=4).mean()
        hs=vals.shift(1).rolling(8,min_periods=4).std(ddof=0)
        hn=vals.shift(1).rolling(8,min_periods=1).count()
        hist_mean.extend(hm.tolist()); hist_std.extend(hs.tolist()); hist_n.extend(hn.tolist())
    eligible["hist_mean"]=hist_mean
    eligible["hist_std"]=hist_std
    eligible["hist_n"]=hist_n
    eligible["hist_z"]=(eligible.cross_score-eligible.hist_mean)/eligible.hist_std.replace(0,np.nan)
    eligible["hist_strength"]=0.5+0.5*np.tanh(eligible.hist_z.fillna(0.0)/2.0)
    eligible["abnormal_score"]=0.80*eligible.cross_score+0.20*eligible.hist_strength
    eligible=eligible.sort_values(["_qidx","abnormal_score"],ascending=[True,False])

    audit_cols=[
        "quarter","period_end","ticker","cusip","issuer","cur_holders","prev_holders",
        "holder_abs_change","holder_pct_change","institutional_ownership_increase",
        "buyers","sellers","active_changers","buyer_ratio","composite_accumulation_score",
        "cur_value","is_leader","cross_score","hist_n","hist_z","hist_strength","abnormal_score"
    ]
    eligible[audit_cols].to_csv(OUT/"b2_abnormal_accumulation.csv",index=False)

    excluded=full[full.corporate_action_filter].copy()
    excluded=excluded.sort_values(["quarter","holder_abs_change"],ascending=[True,False])
    excluded_cols=[
        "quarter","period_end","ticker","cusip","issuer","cur_holders","prev_holders",
        "holder_abs_change","holder_pct_change","buyer_ratio","active_changers",
        "corp_zero_base","corp_near_zero_reset","corp_mass_holder_reset",
        "corp_mechanical_distribution"
    ]
    excluded[excluded_cols].to_csv(OUT/"b2_corporate_action_exclusions.csv",index=False)

    cluster_rows=[]
    member_rows=[]
    q_to_date={bt.qlabel(q):q for q in quarters}
    for qlabel,qm in eligible.groupby("quarter",sort=False):
        q=q_to_date[qlabel]
        fm=full[full.quarter.eq(qlabel)].copy()
        print("B2 network",qlabel,"eligible",len(qm),flush=True)
        clusters=discover_clusters(db,q,qm,fm)
        for rank,c in enumerate(clusters,1):
            members=c["members"].sort_values("abnormal_score",ascending=False).head(TOP_MEMBERS_PER_CLUSTER)
            tickers="|".join(members.ticker.astype(str))
            issuers=" | ".join(members.issuer.astype(str))
            cid=f"{qlabel}_C{rank}"
            cluster_rows.append({
                "quarter":qlabel,"period_end":str(members.period_end.iloc[0]),"cluster_id":cid,
                "cluster_rank":rank,"cluster_size":c["size"],
                "cluster_score":c["cluster_score"],"mean_abnormal":c["mean_abnormal"],
                "mean_similarity":c["mean_similarity"],"edge_density":c["edge_density"],
                "top_tickers":tickers,"top_issuers":issuers,
            })
            for mrank,r in enumerate(members.itertuples(index=False),1):
                member_rows.append({
                    "quarter":qlabel,"cluster_id":cid,"cluster_rank":rank,"member_rank":mrank,
                    "ticker":r.ticker,"issuer":r.issuer,"cusip":r.cusip,
                    "abnormal_score":r.abnormal_score,"cross_score":r.cross_score,
                    "hist_z":r.hist_z,"holder_abs_change":r.holder_abs_change,
                    "holder_pct_change":r.holder_pct_change,
                    "institutional_ownership_increase":r.institutional_ownership_increase,
                    "buyer_ratio":r.buyer_ratio,"current_holders":r.cur_holders,
                    "current_value_usd":r.cur_value,"is_leader":bool(r.is_leader),
                })

    cdf=pd.DataFrame(cluster_rows)
    mdf=pd.DataFrame(member_rows)
    cdf.to_csv(OUT/"b2_blind_clusters.csv",index=False)
    mdf.to_csv(OUT/"b2_blind_cluster_members.csv",index=False)
    pd.DataFrame(q_meta).to_csv(OUT/"b2_data_quality.csv",index=False)

    meta={
        "experiment":"B2",
        "forward_returns_used":False,
        "years":f"{bt.START_YEAR}-{bt.END_YEAR}",
        "corporate_action_filter":{
            "zero_base":"prior_holders <= 0",
            "near_zero_reset":"prior_holders <= 5 and current_holders >= 50 and holder_pct_change >= 5x",
            "mass_holder_reset":"current_holders >= 100 and holder_pct_change >= 8x",
            "mechanical_distribution":"current_holders >=100, holder_abs>=75, pct>=1.5x, active_changers>=5, buyer_ratio<=0.25",
        },
        "base_universe":f"current_holders >= {MIN_HOLDERS}, current_13F_value >= {MIN_VALUE_USD}, positive holder change",
        "abnormal_score":"0.80 * cross-sectional weighted percentile score + 0.20 * trailing-own-history surprise",
        "cross_score_weights":{"holder_abs":0.35,"composite":0.25,"ownership_growth":0.20,"buyer_ratio":0.20},
        "own_history":"prior 8 available quarters only, minimum 4; tanh-scaled z score",
        "cluster_method":"co-buy network among top 250 abnormal stocks; manager weights inverse sqrt(number of candidate buys); cosine similarity; top-10 neighbors; edge>=0.08; greedy modularity communities",
        "minimum_cluster_size":MIN_CLUSTER_SIZE,
        "information_firewall":"No market prices or forward returns are read or written by b2_discovery.py",
    }
    (OUT/"b2_methodology.json").write_text(json.dumps(meta,indent=2))
    print("B2 wrote",len(cdf),"clusters and",len(mdf),"member rows",flush=True)

if __name__=="__main__":
    main()
