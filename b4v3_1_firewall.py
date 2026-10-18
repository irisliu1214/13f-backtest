#!/usr/bin/env python3
"""
B4-v3.1 Economic Driver Firewall -- discovery only.

This script is deliberately return-blind. It reads:
  * B2 corporate-action-filtered co-buy member data
  * persistent point-in-time causal-sentence cache built before any B4-v3 returns

It does NOT read prices, SPY, B3/B4 return outputs, or present-day company descriptions.

Concept
-------
A candidate theme must pass three independent gates:
1) EXTERNALITY: the extracted causal antecedent must describe a market/customer/
   industry/regulatory/supply/price/technology/infrastructure condition, not an
   accounting, reporting, financing, tax, compensation, or internal bookkeeping event.
2) BUSINESS TRANSMISSION: the same contemporaneous sentence must explicitly connect
   the antecedent to a business outcome such as demand, usage, orders, backlog,
   revenue/sales/volume, pricing, customer growth, capacity, or capex.
3) CROSS-COMPANY VALIDATION: substantially similar extracted driver phrases must
   appear independently across >=5 companies, span >=2 B2 co-buy communities, and
   include >=2 supporters outside the dominant seed community.

Novelty is measured only against prior-quarter cached text. No future text is used.

Important: this is causal-evidence screening, not econometric causal identification.
"""
from pathlib import Path
import json, math, re
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import b4_causal as core

R=Path("results")
CACHE=R/"b4_cache_v2"
OUT_PREFIX="b4v3_1_pre_return"

TOP_B2_CLUSTERS=5
MAX_MEMBERS_PER_CLUSTER=12
PORTFOLIO_SIZE=5

# Pre-return frozen gates.
MIN_SEED_COMPANIES=3
MIN_SUPPORT_COMPANIES=5
MIN_SUPPORT_CLUSTERS=2
MIN_EXTERNAL_SUPPORT=2
DRIVER_CLUSTER_DISTANCE=0.66       # cosine similarity roughly >= .34
EXPANSION_SIMILARITY=0.22
MIN_COMPANY_NOVELTY=0.35
MIN_THEME_NOVELTY=0.40
MIN_EXTERNALITY_RATE=0.80
MIN_TRANSMISSION_RATE=0.80
MIN_THEME_SCORE=0.70
TRAILING_QUARTERS=4

THEME_W={
    "breadth":0.20,
    "externality":0.20,
    "transmission":0.20,
    "novelty":0.15,
    "out_of_seed":0.15,
    "abnormal":0.10,
}
BEN_W={
    "abnormal":0.35,
    "driver_exposure":0.25,
    "novel_evidence":0.15,
    "holders":0.15,
    "buyer_ratio":0.10,
}

# These are semantic *classes*, not historical theme labels. They define what an
# external economic antecedent may look like without naming winners/sectors.
EXTERNAL_ANCHORS=re.compile(
    r"\\b(customer|customers|consumer|consumers|market|markets|industry|industries|"
    r"regulation|regulatory|government|federal|state|policy|policies|economic|economy|macroeconomic|"
    r"inflation|interest rate|interest rates|commodity|commodities|raw material|raw materials|"
    r"supply|shortage|shortages|constraint|constraints|infrastructure|technology|technologies|digital|online|"
    r"energy|power|healthcare|housing|labor market|competition|competitive|international|global|"
    r"pandemic|outbreak|hurricane|weather|geopolitical|tariff|tariffs|trade|environmental|emission|emissions)\\b",
    re.I
)

TRANSMISSION=re.compile(
    r"\\b(demand|usage|orders?|backlog|bookings?|revenue|revenues|sales|volume|volumes|"
    r"pricing|price realization|market share|customers?|subscribers?|shipments?|production|"
    r"capacity|utilization|capital expenditures?|capex|gross profit|operating income|margin|margins)\\b",
    re.I
)

POSITIVE_DIRECTION=re.compile(
    r"\\b(increased|grew|rose|rising|higher|stronger|strong demand|growth|expanded|expanding|"
    r"accelerated|benefited|benefiting|improved|record|surged|additional demand|higher demand|"
    r"increased demand|increased usage|increased orders|increased sales|increased revenue)\\b",
    re.I
)

# Hard firewall: if the extracted antecedent is dominated by these concepts, it is
# not an economic driver. The list is intentionally category-level accounting /
# internal-corporate vocabulary rather than hand-picking historical losers.
ACCOUNTING_INTERNAL=re.compile(
    r"\b(asu|asc|fasb|gaap|accounting|accounted|standard|standards|adoption guidance|"
    r"deferred tax|income tax|income taxes|tax expense|tax benefit|tax benefits|tax asset|"
    r"tax assets|tax liability|tax liabilities|valuation allowance|uncertain tax|"
    r"retained earnings|accumulated deficit|cumulative effect|right-of-use|lease liability|"
    r"balance sheet|statement of cash flows|cash flow statement|consolidated statements?|"
    r"condensed consolidated|fair value|impairment|amortization|depreciation|"
    r"stock-based compensation|share-based compensation|compensation expense|payroll|salaries|bonuses|"
    r"issuance costs?|debt issuance|debt refinancing|credit facility|public offering|"
    r"earnings per share|eps|goodwill|intangible assets?|working capital|"
    r"table of contents|three months ended|six months ended|nine months ended|year ended|"
    r"reclassification|presentation|disclosure|disclosures|effective date|reporting period|"
    r"legal settlement|litigation|acquisition accounting|purchase accounting)\b",
    re.I
)

PURELY_INTERNAL=re.compile(
    r"\b(our hiring|headcount|employee|employees|personnel|marketing expense|selling expense|"
    r"general and administrative|research and development expense|r&d expense|"
    r"internal control|corporate overhead|professional services|insurance expense)\b",
    re.I
)

# Causal connector extraction. For 'X due to Y' / 'X driven by Y', take Y.
POST_CONNECTORS=[
    r"primarily due to",r"due to",r"because of",r"driven by",r"attributable to",
    r"resulting from",r"benefit(?:ed|ing|s)? from",r"supported by",r"fueled by",
    r"in response to",r"reflecting"
]
# These constructions themselves often encode the antecedent phrase.
DIRECT_DRIVER_PATTERNS=[
    r"\bdemand for\s+([^.;:]{3,140})",
    r"\badoption of\s+([^.;:]{3,140})",
    r"\bspending on\s+([^.;:]{3,140})",
    r"\binvestment in\s+([^.;:]{3,140})",
    r"\binvestments in\s+([^.;:]{3,140})",
    r"\bshortage of\s+([^.;:]{3,140})",
    r"\bshortages of\s+([^.;:]{3,140})",
    r"\bsupply constraints?\s+([^.;:]{0,120})",
    r"\bmarket growth\s+([^.;:]{0,120})",
    r"\bindustry growth\s+([^.;:]{0,120})",
]

DRIVER_STOP=set(ENGLISH_STOP_WORDS)|DRIVER_GENERIC|{
    "entity","company","companies","business","businesses","result","results","period","quarter",
    "year","years","three","six","nine","months","ended","approximately","million","billion",
    "primarily","including","include","included","certain","respectively","compared","prior",
    "current","new","continued","continue","also","related","due","driven","reflecting",
    "increase","increased","increasing","higher","growth","strong","additional","support",
    "supported","benefit","benefited","benefiting","resulting","based","impact","impacts"
}

def clean_clause(x):
    x=re.sub(r"\s+"," ",str(x)).strip(" ,;:-")
    # Stop at common contrast / accounting tails.
    x=re.split(r"\b(?:partially offset by|offset by|while|although|however|which included|including a reduction in)\b",x,1,flags=re.I)[0]
    words=x.split()
    return " ".join(words[:28]).strip(" ,;:-")

def extract_driver_phrases(sentence):
    s=str(sentence)
    out=[]
    low=s.lower()
    for pat in POST_CONNECTORS:
        m=re.search(pat+r"\s+([^.;:]{3,220})",s,re.I)
        if m:
            out.append(clean_clause(m.group(1)))
    for pat in DIRECT_DRIVER_PATTERNS:
        for m in re.finditer(pat,s,re.I):
            lead=re.search(r"\b(demand for|adoption of|spending on|investment in|investments in|shortage of|shortages of|supply constraints?|market growth|industry growth)\b",m.group(0),re.I)
            phrase=(lead.group(0)+" "+m.group(1)) if lead else m.group(1)
            out.append(clean_clause(phrase))
    # Deduplicate normalized.
    seen=set(); good=[]
    for p in out:
        key=re.sub(r"[^a-z]+"," ",p.lower()).strip()
        if len(key)<12 or key in seen: continue
        seen.add(key); good.append(p)
    return good

def content_tokens(x):
    toks=re.findall(r"[a-z][a-z\-]{2,}",str(x).lower())
    return {t for t in toks if t not in DRIVER_STOP}

def externality_score(phrase,sentence):
    p=str(phrase); s=str(sentence)
    toks=content_tokens(p)
    if len(toks)<2:
        return 0.0
    # Any accounting/reporting contamination in the evidence sentence is disqualifying.
    if ACCOUNTING_INTERNAL.search(s):
        return 0.0
    if PURELY_INTERNAL.search(p):
        return 0.0
    # Must contain an external-market context in the antecedent or surrounding sentence.
    ext_phrase=len(EXTERNAL_ANCHORS.findall(p))
    ext_sentence=len(EXTERNAL_ANCHORS.findall(s))
    if ext_phrase==0 and ext_sentence==0:
        return 0.0
    # At least two specific content tokens must survive the generic-economic stoplist.
    specific={t for t in toks if t not in DRIVER_GENERIC}
    if len(specific)<2:
        return 0.0
    return float(np.clip(0.62+0.08*min(ext_phrase+ext_sentence,3)+0.04*min(len(specific),4),0,1))

def transmission_score(sentence,phrase):
    s=str(sentence)
    # Reject hypothetical/risk-factor language. v3.1 uses realized/current evidence only.
    if RISK_HYPOTHETICAL.search(s):
        return 0.0
    if ACCOUNTING_INTERNAL.search(s):
        return 0.0
    if not POSITIVE_DIRECTION.search(s):
        return 0.0
    hits=len(TRANSMISSION.findall(s))
    if hits==0:
        return 0.0
    outside=s.lower().replace(str(phrase).lower()," ")
    out_hits=len(TRANSMISSION.findall(outside))
    if out_hits==0:
        return 0.0
    return float(np.clip(0.72+0.10*min(out_hits,2)+0.08*min(hits,2),0,1))

def jaccard(a,b):
    if not a or not b: return 0.0
    return len(a&b)/len(a|b)

def company_novelty(phrase,cusip,q,history):
    toks=content_tokens(phrase)
    prev=history.get(str(cusip),[])
    if not prev: return 1.0
    # prev is list[(quarter, token-set)], already only prior quarters retained.
    sims=[jaccard(toks,t) for pq,t in prev[-80:]]
    return float(1.0-max(sims,default=0.0))

def rank_pct(s):
    return pd.Series(s).rank(pct=True,method="average")

def main():
    members=pd.read_csv(core.MEMBERS,dtype={"cusip":str})
    members=members[
        (members.cluster_rank<=TOP_B2_CLUSTERS)&
        (members.member_rank<=MAX_MEMBERS_PER_CLUSTER)&
        (~members.issuer.fillna("").str.contains(core.BAD_ISSUER))
    ].copy()
    members["cusip"]=members.cusip.astype(str)

    sf=sorted(CACHE.glob("b4_sentences_*.csv"))
    af=sorted(CACHE.glob("b4_audit_*.csv"))
    if len(sf)!=11 or len(af)!=11:
        raise RuntimeError("Expected complete 2016-2026 persistent B4 cache")
    sents=pd.concat([pd.read_csv(x,dtype={"cusip":str}) for x in sf],ignore_index=True)
    audit=pd.concat([pd.read_csv(x,dtype={"cusip":str}) for x in af],ignore_index=True)

    quarters=sorted(members.quarter.unique(),key=lambda q:(int(str(q)[:4]),int(str(q)[-1])))
    history={}
    audit_rows=[]
    candidate_rows=[]
    evidence_rows=[]
    pick_rows=[]

    for q in quarters:
        print("B4V3",q,flush=True)
        mq=members[members.quarter.eq(q)].sort_values(["cluster_rank","member_rank"]).drop_duplicates("cusip").copy()
        qs=sents[sents.quarter.eq(q)].copy()
        qs=qs.merge(mq[["cusip","ticker","issuer","cluster_id","abnormal_score","current_holders","buyer_ratio"]],
                    on="cusip",how="inner",suffixes=("","_m"))

        # Stage 1: extract antecedent phrases and apply externality/transmission firewall.
        erows=[]
        for r in qs.itertuples(index=False):
            sent=str(r.sentence)
            phrases=extract_driver_phrases(sent)
            for phrase in phrases:
                ext=externality_score(phrase,sent)
                trans=transmission_score(sent,phrase)
                nov=company_novelty(phrase,r.cusip,q,history)
                pass_fw=(ext>=0.70 and trans>=0.80 and nov>=MIN_COMPANY_NOVELTY)
                audit_rows.append({
                    "quarter":q,"ticker":r.ticker,"cusip":r.cusip,"cluster_id":r.cluster_id,
                    "sentence":sent,"driver_phrase":phrase,"externality_score":ext,
                    "transmission_score":trans,"company_novelty":nov,"pass_firewall":pass_fw,
                    "accounting_hit":bool(ACCOUNTING_INTERNAL.search(phrase)),
                    "internal_hit":bool(PURELY_INTERNAL.search(phrase)),
                })
                if pass_fw:
                    erows.append({
                        "quarter":q,"ticker":r.ticker,"issuer":r.issuer,"cusip":str(r.cusip),
                        "cluster_id":r.cluster_id,"abnormal_score":float(r.abnormal_score),
                        "current_holders":float(r.current_holders),
                        "buyer_ratio":float(r.buyer_ratio) if pd.notna(r.buyer_ratio) else np.nan,
                        "sentence":sent,"driver_phrase":phrase,"externality":ext,
                        "transmission":trans,"novelty":nov,
                    })

        # Update history only AFTER current-quarter novelty has been scored.
        for r in qs.itertuples(index=False):
            for phrase in extract_driver_phrases(str(r.sentence)):
                history.setdefault(str(r.cusip),[]).append((q,content_tokens(phrase)))

        if len(erows)<MIN_SUPPORT_COMPANIES:
            pick_rows.append({"quarter":q,"decision":"NO_TRADE","reason":"economic-driver firewall left insufficient evidence"})
            continue

        e=pd.DataFrame(erows).drop_duplicates(["cusip","driver_phrase"]).reset_index(drop=True)
        if e.cusip.nunique()<MIN_SUPPORT_COMPANIES:
            pick_rows.append({"quarter":q,"decision":"NO_TRADE","reason":"economic-driver firewall covers fewer than five companies"})
            continue

        # Cluster extracted antecedent phrases, not full filing sentences.
        try:
            vec=TfidfVectorizer(
                lowercase=True,ngram_range=(1,2),stop_words=sorted(DRIVER_STOP),
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
                min_df=2,max_df=0.70,max_features=7000,sublinear_tf=True
            )
            X=vec.fit_transform(e.driver_phrase.tolist())
        except Exception:
            pick_rows.append({"quarter":q,"decision":"NO_TRADE","reason":"driver phrase vocabulary failure"})
            continue
        norms=np.sqrt(X.multiply(X).sum(axis=1)).A1
        good=norms>0
        e=e.loc[good].reset_index(drop=True); X=X[good]
        if len(e)<MIN_SUPPORT_COMPANIES:
            pick_rows.append({"quarter":q,"decision":"NO_TRADE","reason":"insufficient nonzero economic-driver phrases"})
            continue

        labels=AgglomerativeClustering(
            n_clusters=None,distance_threshold=DRIVER_CLUSTER_DISTANCE,
            metric="cosine",linkage="average",compute_full_tree=True
        ).fit_predict(X.toarray())
        e["driver_seed"]=labels

        names=np.array(vec.get_feature_names_out())
        global_cent=np.asarray(X.mean(axis=0)).ravel()
        candidates=[]
        for lab,g in e.groupby("driver_seed"):
            if g.cusip.nunique()<MIN_SEED_COMPANIES:
                continue
            idx=g.index.to_numpy()
            cent=np.asarray(X[idx].mean(axis=0))
            sim=cosine_similarity(X,cent).ravel()
            tmp=e.assign(_sim=sim)
            exposure={}; best_evidence={}; ev_count={}; company_nov={}; company_ext={}; company_trans={}
            for cusip,gg in tmp.groupby("cusip"):
                best=float(gg._sim.max())
                if best>=EXPANSION_SIMILARITY:
                    br=gg.sort_values(["_sim","novelty","externality","transmission"],ascending=False).iloc[0]
                    exposure[str(cusip)]=best
                    best_evidence[str(cusip)]=str(br.sentence)[:1000]
                    ev_count[str(cusip)]=int((gg._sim>=EXPANSION_SIMILARITY).sum())
                    company_nov[str(cusip)]=float(gg.loc[gg._sim>=EXPANSION_SIMILARITY,"novelty"].max())
                    company_ext[str(cusip)]=float(gg.loc[gg._sim>=EXPANSION_SIMILARITY,"externality"].mean())
                    company_trans[str(cusip)]=float(gg.loc[gg._sim>=EXPANSION_SIMILARITY,"transmission"].mean())
            supporters=list(exposure)
            if len(supporters)<MIN_SUPPORT_COMPANIES: continue

            sm=mq[mq.cusip.isin(supporters)].copy()
            if sm.empty: continue
            counts=sm.groupby("cluster_id").size().sort_values(ascending=False)
            seed_cluster=str(counts.index[0])
            ext_sm=sm[~sm.cluster_id.eq(seed_cluster)]
            nclusters=sm.cluster_id.nunique()
            if nclusters<MIN_SUPPORT_CLUSTERS or len(ext_sm)<MIN_EXTERNAL_SUPPORT:
                continue

            ext_rate=float(np.mean([company_ext[c]>=0.70 for c in supporters]))
            trans_rate=float(np.mean([company_trans[c]>=0.80 for c in supporters]))
            theme_nov=float(np.mean([company_nov[c] for c in supporters]))
            if ext_rate<MIN_EXTERNALITY_RATE or trans_rate<MIN_TRANSMISSION_RATE or theme_nov<MIN_THEME_NOVELTY:
                continue

            discr=cent.ravel()-global_cent
            terms=[]
            for j in np.argsort(discr)[::-1]:
                if discr[j]<=0: break
                term=str(names[j])
                if ACCOUNTING_INTERNAL.search(term) or PURELY_INTERNAL.search(term): continue
                if len(content_tokens(term))==0: continue
                terms.append(term)
                if len(terms)>=8: break
            if len(terms)<2: continue
            specific_terms=[t for t in terms if any(tok not in DRIVER_GENERIC for tok in content_tokens(t))]
            if len(specific_terms)<2: continue
            terms=specific_terms[:8]

            pool=mq.copy(); pool["_ab_pct"]=rank_pct(pool.abnormal_score)
            sm=sm.merge(pool[["cusip","_ab_pct"]],on="cusip",how="left")
            abnormal=float(sm._ab_pct.mean())
            breadth=float(min(1.0,len(supporters)/8.0))
            outseed=float(min(1.0,len(ext_sm)/4.0))
            score=(
                THEME_W["breadth"]*breadth+
                THEME_W["externality"]*float(np.mean([company_ext[c] for c in supporters]))+
                THEME_W["transmission"]*float(np.mean([company_trans[c] for c in supporters]))+
                THEME_W["novelty"]*theme_nov+
                THEME_W["out_of_seed"]*outseed+
                THEME_W["abnormal"]*abnormal
            )
            candidates.append({
                "driver_seed":int(lab),"driver_terms":" | ".join(terms),
                "seed_cluster":seed_cluster,"supporters":supporters,
                "support_count":len(supporters),"support_clusters":int(nclusters),
                "external_support_count":int(len(ext_sm)),
                "externality_rate":ext_rate,"transmission_rate":trans_rate,
                "mean_externality":float(np.mean([company_ext[c] for c in supporters])),
                "mean_transmission":float(np.mean([company_trans[c] for c in supporters])),
                "mean_novelty":theme_nov,"breadth":breadth,"out_of_seed":outseed,
                "mean_abnormal_pct":abnormal,"theme_score":float(score),
                "exposure":exposure,"evidence":best_evidence,"evidence_count":ev_count,
                "company_novelty":company_nov
            })

        # De-duplicate near-identical supporter sets.
        candidates=sorted(candidates,key=lambda x:(x["theme_score"],x["support_count"]),reverse=True)
        dedup=[]
        for c in candidates:
            ss=set(c["supporters"])
            if any(len(ss&set(d["supporters"]))/max(1,len(ss|set(d["supporters"])))>=0.70 for d in dedup):
                continue
            dedup.append(c)
        candidates=dedup

        for rank,c in enumerate(candidates,1):
            sm=mq[mq.cusip.isin(c["supporters"])]
            candidate_rows.append({
                "quarter":q,"driver_rank":rank,"driver_terms":c["driver_terms"],
                "seed_cluster":c["seed_cluster"],"support_count":c["support_count"],
                "support_clusters":c["support_clusters"],"external_support_count":c["external_support_count"],
                "externality_rate":c["externality_rate"],"transmission_rate":c["transmission_rate"],
                "mean_externality":c["mean_externality"],"mean_transmission":c["mean_transmission"],
                "mean_novelty":c["mean_novelty"],"mean_abnormal_pct":c["mean_abnormal_pct"],
                "theme_score":c["theme_score"],"support_tickers":"|".join(sm.ticker.astype(str))
            })
            for cusip in c["supporters"]:
                rr=mq[mq.cusip.eq(cusip)].iloc[0]
                evidence_rows.append({
                    "quarter":q,"driver_rank":rank,"driver_terms":c["driver_terms"],
                    "ticker":rr.ticker,"issuer":rr.issuer,"cusip":cusip,"cluster_id":rr.cluster_id,
                    "driver_exposure":c["exposure"][cusip],"evidence_count":c["evidence_count"][cusip],
                    "company_novelty":c["company_novelty"][cusip],
                    "evidence_sentence":c["evidence"][cusip],
                    "abnormal_score":rr.abnormal_score,"current_holders":rr.current_holders,
                    "buyer_ratio":rr.buyer_ratio,
                })

        eligible=[c for c in candidates if c["theme_score"]>=MIN_THEME_SCORE]
        if not eligible:
            pick_rows.append({"quarter":q,"decision":"NO_TRADE","reason":"no external economic driver passed v3 firewall"})
            continue

        chosen=eligible[0]
        sm=mq[mq.cusip.isin(chosen["supporters"])].copy()
        sm["driver_exposure"]=sm.cusip.map(chosen["exposure"])
        sm["novel_evidence"]=sm.cusip.map(chosen["company_novelty"])
        sm["evidence_count"]=sm.cusip.map(chosen["evidence_count"])
        for col in ["abnormal_score","driver_exposure","novel_evidence","current_holders","buyer_ratio"]:
            sm[col+"_pct"]=sm[col].rank(pct=True,method="average")
        sm["beneficiary_score"]=(
            BEN_W["abnormal"]*sm.abnormal_score_pct+
            BEN_W["driver_exposure"]*sm.driver_exposure_pct+
            BEN_W["novel_evidence"]*sm.novel_evidence_pct+
            BEN_W["holders"]*sm.current_holders_pct+
            BEN_W["buyer_ratio"]*sm.buyer_ratio_pct.fillna(0.5)
        )
        sel=sm.sort_values(["beneficiary_score","abnormal_score"],ascending=False).head(PORTFOLIO_SIZE)
        if len(sel)<PORTFOLIO_SIZE:
            pick_rows.append({"quarter":q,"decision":"NO_TRADE","reason":"fewer than five deterministic beneficiaries"})
            continue
        pick_rows.append({
            "quarter":q,"decision":"INVEST","driver_terms":chosen["driver_terms"],
            "seed_cluster":chosen["seed_cluster"],"theme_score":chosen["theme_score"],
            "support_count":chosen["support_count"],"external_support_count":chosen["external_support_count"],
            "mean_novelty":chosen["mean_novelty"],
            "selected_tickers":"|".join(sel.ticker.astype(str)),
            "selected_cusips":"|".join(sel.cusip.astype(str)),
            "beneficiary_scores":"|".join(f"{x:.6f}" for x in sel.beneficiary_score),
            "reason":"B4-v3.1 realized external economic driver firewall + cross-company validation + deterministic Top5",
        })

    pd.DataFrame(audit_rows).to_csv(R/f"{OUT_PREFIX}_firewall_audit.csv",index=False)
    pd.DataFrame(candidate_rows).to_csv(R/f"{OUT_PREFIX}_driver_scores.csv",index=False)
    pd.DataFrame(evidence_rows).to_csv(R/f"{OUT_PREFIX}_driver_evidence.csv",index=False)
    picks=pd.DataFrame(pick_rows)
    picks.to_csv(R/f"{OUT_PREFIX}_picks.csv",index=False)

    meta={
        "experiment":"B4-v3.1 Economic Driver Firewall, pre-return discovery",
        "forward_returns_used":False,"prices_read":False,
        "source_cache":"results/b4_cache_v2, built from point-in-time SEC filings",
        "designed_after_observing_b4v2_returns":True,
        "v3_1_revision_basis":"Semantic sanity audit only; no B4-v3 forward returns were opened. Tightened factual-vs-hypothetical evidence, sentence-level accounting exclusion, and removed generic economic words from driver clustering.",
        "interpretation":"historical robustness experiment, not untouched out-of-sample evidence",
        "gates":{
            "min_seed_companies":MIN_SEED_COMPANIES,
            "min_support_companies":MIN_SUPPORT_COMPANIES,
            "min_support_clusters":MIN_SUPPORT_CLUSTERS,
            "min_external_support":MIN_EXTERNAL_SUPPORT,
            "driver_cluster_distance":DRIVER_CLUSTER_DISTANCE,
            "expansion_similarity":EXPANSION_SIMILARITY,
            "min_company_novelty":MIN_COMPANY_NOVELTY,
            "min_theme_novelty":MIN_THEME_NOVELTY,
            "min_externality_rate":MIN_EXTERNALITY_RATE,
            "min_transmission_rate":MIN_TRANSMISSION_RATE,
            "min_theme_score":MIN_THEME_SCORE,
        },
        "theme_weights":THEME_W,"beneficiary_weights":BEN_W,
        "manual_theme_selection":False,"manual_stock_replacement":False,
        "prohibited_inputs":["prices","forward returns","SPY returns","present-day company descriptions"],
    }
    (R/f"{OUT_PREFIX}_methodology.json").write_text(json.dumps(meta,indent=2))
    print(picks.to_string(index=False),flush=True)
    print("B4V3.1 PRE-RETURN INVEST",int((picks.decision=="INVEST").sum()),"of",len(picks),
          "candidates",len(candidate_rows),"evidence",len(evidence_rows),flush=True)

if __name__=="__main__":
    main()