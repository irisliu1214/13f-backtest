#!/usr/bin/env python3
from pathlib import Path
import math
import json
import numpy as np
import pandas as pd

RESULTS = Path("results")
SRC = RESULTS / "accumulation_candidate_pool.csv"
OUT = RESULTS
TOP_PACKET = 18
PORTFOLIO_SIZE = 5

# Fixed, ex-ante-style industry taxonomy. These are broad business buckets, not
# "winner" labels; the purpose is to aggregate contemporaneous 13F clustering.
TICKER_BUCKETS = {
    "Memory / Storage": set("MU WDC STX SNDK PSTG NTAP PURE".split()),
    "Semiconductor Equipment": set("AMAT LRCX KLAC ASML MKSI TER ACLS COHU ENTG ICHR UCTT CAMT".split()),
    "Semiconductors / Compute": set("NVDA AMD INTC AVGO QCOM TXN NXPI MCHP ADI ON MPWR ARM TSM ASX MRVL SWKS QRVO LSCC MTSI".split()),
    "Networking / Connectivity": set("ANET CSCO JNPR CIEN CRDO LITE COHR AAOI GLW INFN COMM ARW AVT".split()),
    "Data Center / Hardware": set("DELL HPE SMCI IBM HPQ NTAP PSTG VRT".split()),
    "Power / Grid / Electrical": set("GEV ETN HUBB PWR EME NVT VST CEG NRG BE POWL AES NEE DUK SO AEP EXC XEL PEG PCG ED".split()),
    "Cooling / Building Systems": set("VRT MOD TT JCI CARR IR FIX WSO".split()),
    "Cloud / Enterprise Software": set("MSFT ORCL CRM NOW WDAY TEAM SNOW DDOG MDB HUBS SAP ADBE INTU PLTR PATH ESTC DT".split()),
    "Cybersecurity": set("PANW CRWD ZS FTNT OKTA CYBR CHKP TENB S VRNS GEN RPD".split()),
    "Internet / Digital Platforms": set("GOOGL GOOG META NFLX UBER ABNB DASH RDDT SNAP PINS SPOT".split()),
    "E-commerce": set("AMZN SHOP MELI SE ETSY EBAY BABA JD PDD CPNG".split()),
    "Payments / Fintech": set("V MA PYPL SQ XYZ FI FIS GPN AFRM SOFI NU COIN HOOD ADYEY".split()),
    "Biotech / Pharma": set("LLY NVO MRK PFE BMY ABBV AMGN GILD REGN VRTX BIIB MRNA BNTX ALNY NBIX".split()),
    "Medical Devices / Tools": set("ISRG SYK MDT BSX EW TMO DHR ABT DXCM PODD RMD BDX".split()),
    "EV / Auto / Mobility": set("TSLA RIVN LCID GM F MBLY APTV BWA LEA ALB QS CHPT".split()),
    "Clean Energy": set("FSLR ENPH SEDG RUN NOVA ARRY CSIQ NEE BE PLUG STEM".split()),
    "Oil / Gas / Services": set("XOM CVX COP EOG OXY DVN FANG MRO APA SLB HAL BKR BHGE FTI LNG EQT CTRA".split()),
    "Defense / Aerospace": set("LMT NOC RTX GD HII LHX TDG HEI AVAV KTOS RKLB ACHR JOBY".split()),
    "Housing / Construction": set("DHI LEN PHM TOL NVR KBH MTH BLDR BLD TREX OC MAS".split()),
    "Banks / Capital Markets": set("JPM BAC C WFC GS MS SCHW BK STT COF AXP SPGI MCO CME ICE NDAQ".split()),
    "Insurance": set("CB AIG PGR ALL TRV MET PRU AFL ACGL WRB AJG MMC AON".split()),
    "Consumer Brands / Retail": set("WMT COST TGT HD LOW NKE LULU TPR RL EL ULTA SBUX MCD YUM YUMC".split()),
    "Media / Telecom": set("CMCSA T VZ CHTR TMUS DIS PARA WBD CTL LUMN".split()),
    "Industrials / Automation": set("HON GE MMM EMR ROK PH ITW CAT DE CMI URI GWW FAST".split()),
    "Materials / Mining": set("FCX NEM GOLD NUE STLD CLF AA ARNC NTR MOS CF".split()),
    "Real Estate": set("AMT CCI EQIX DLR PLD SPG O VICI WELL AVB EQR JBGS CLNY".split()),
}

KEYWORD_BUCKETS = [
    ("Semiconductor Equipment", ["SEMICONDUCTOR EQUIPMENT","LITHOGRAPH","WAFER","MATERIALS INC","PROCESS SYSTEMS"]),
    ("Semiconductors / Compute", ["SEMICONDUCTOR","MICRO DEVICES","MICROCHIP","NVIDIA","BROADCOM","QUALCOMM","NXP","ANALOG DEVICES"]),
    ("Memory / Storage", ["MICRON","SEAGATE","WESTERN DIGITAL","SANDISK","STORAGE"]),
    ("Networking / Connectivity", ["NETWORKS","NETWORKING","COMMUNICATION SYSTEMS","CIENA","ARISTA","OPTICAL","FIBER"]),
    ("Power / Grid / Electrical", ["ELECTRIC","POWER","ENERGY CORP","UTILITIES","EATON","HUBBELL","GRID"]),
    ("Cooling / Building Systems", ["AIR CONDITION","CLIMATE","CARRIER","TRANE","JOHNSON CTLS","BUILDING SYSTEM"]),
    ("Cloud / Enterprise Software", ["SOFTWARE","SALESFORCE","SERVICENOW","WORKDAY","ORACLE","ADOBE"]),
    ("Cybersecurity", ["CYBER"]),
    ("Payments / Fintech", ["PAYMENT","PAYPAL","MASTERCARD","VISA","FINTECH"]),
    ("Biotech / Pharma", ["PHARMA","THERAPEUT","BIOPHARMA","BIOTECH","BIOSCI"]),
    ("Medical Devices / Tools", ["MEDICAL","SURGICAL","DIAGNOSTIC","LABORATOR"]),
    ("Oil / Gas / Services", ["OIL","GAS","PETROLEUM","ENERGY SVCS","BAKER HUGHES","TECHNIP","SCHLUMBERGER"]),
    ("Defense / Aerospace", ["AEROSPACE","DEFENSE","DEFENCE"]),
    ("Housing / Construction", ["HOMES","HOMEBUILDER","BUILDERS","CONSTRUCTION"]),
    ("Banks / Capital Markets", ["BANK","BANCORP","FINANCIAL","CAPITAL MARKETS","EXCHANGE"]),
    ("Insurance", ["INSURANCE","ASSURANCE"]),
    ("Real Estate", ["REIT","REALTY","PROPERTIES","PPTYS","REAL ESTATE"]),
]

def bucket_for(ticker, issuer):
    t = str(ticker or "").upper().strip()
    for b, tickers in TICKER_BUCKETS.items():
        if t in tickers:
            return b
    u = str(issuer or "").upper()
    for b, kws in KEYWORD_BUCKETS:
        if any(k in u for k in kws):
            return b
    return "Other / Unclassified"

def rank_strength(rank):
    try:
        r = float(rank)
        return max(0.0, min(1.0, (31.0-r)/30.0))
    except Exception:
        return np.nan

def main():
    df = pd.read_csv(SRC)
    # Theme discovery uses the whole market; opportunistic selection later excludes leaders.
    allu = df[df["universe"].eq("AllStocks")].copy()
    allu["rank_strength"] = allu["signal_rank"].map(rank_strength)

    # One contemporaneous stock-level consensus record per quarter/ticker.
    agg = allu.groupby(["quarter","period_end","ticker","issuer"], as_index=False).agg(
        method_hits=("method","nunique"),
        mean_rank_strength=("rank_strength","mean"),
        best_rank=("signal_rank","min"),
        holder_abs_change=("holder_abs_change","max"),
        holder_pct_change=("holder_pct_change","max"),
        institutional_ownership_increase=("institutional_ownership_increase","max"),
        buyer_ratio=("buyer_ratio","max"),
        composite_accumulation_score=("composite_accumulation_score","max"),
        current_holders=("current_holders","max"),
        is_leader=("is_leader","max"),
        current_value_usd=("current_value_usd","max"),
    )
    agg["consensus_score"] = agg["mean_rank_strength"] * (0.50 + 0.50*agg["method_hits"]/4.0)
    agg["bucket"] = [bucket_for(t,i) for t,i in zip(agg.ticker, agg.issuer)]

    # Percentile of holder growth inside each quarter: keeps units comparable through time.
    agg["holder_strength"] = agg.groupby("quarter")["holder_abs_change"].rank(pct=True).fillna(0.0)

    # Blind packets: intentionally no forward-return columns.
    packets = (agg.sort_values(["quarter","consensus_score","method_hits","holder_abs_change"],
                               ascending=[True,False,False,False])
                 .groupby("quarter", as_index=False, group_keys=False)
                 .head(TOP_PACKET)
                 [["quarter","period_end","ticker","issuer","bucket","method_hits","best_rank",
                   "consensus_score","holder_abs_change","holder_pct_change",
                   "institutional_ownership_increase","buyer_ratio","current_holders",
                   "is_leader","current_value_usd"]])
    packets.to_csv(OUT/"theme_blind_packets.csv", index=False)

    # Quant-first theme aggregation.
    theme = agg[agg.bucket.ne("Other / Unclassified")].groupby(
        ["quarter","period_end","bucket"], as_index=False
    ).agg(
        n_names=("ticker","nunique"),
        mean_consensus=("consensus_score","mean"),
        max_consensus=("consensus_score","max"),
        mean_method_coverage=("method_hits", lambda s: float(np.mean(s/4.0))),
        mean_holder_strength=("holder_strength","mean"),
    )
    theme["breadth_score"] = np.minimum(theme["n_names"]/5.0, 1.0)
    theme["raw_theme_score"] = (
        0.45*theme["mean_consensus"] +
        0.25*theme["breadth_score"] +
        0.20*theme["mean_method_coverage"] +
        0.10*theme["mean_holder_strength"]
    )
    qorder = {q:i for i,q in enumerate(sorted(theme.quarter.unique(), key=lambda x:(int(x[:4]),int(x[-1]))))}
    theme["qidx"] = theme.quarter.map(qorder)
    prev = theme[["qidx","bucket","raw_theme_score"]].copy()
    prev["qidx"] += 1
    prev = prev.rename(columns={"raw_theme_score":"prev_theme_score"})
    theme = theme.merge(prev, on=["qidx","bucket"], how="left")
    theme["acceleration"] = theme["raw_theme_score"] - theme["prev_theme_score"].fillna(theme["raw_theme_score"])
    theme["theme_score"] = theme["raw_theme_score"] + 0.15*theme["acceleration"].clip(-0.5,0.5)
    # Need breadth or exceptional strength; avoids one-stock "themes".
    eligible = theme[(theme.n_names >= 2) | (theme.max_consensus >= 0.82)].copy()
    eligible["theme_rank"] = eligible.groupby("quarter")["theme_score"].rank(method="first", ascending=False)
    top_themes = eligible[eligible.theme_rank <= 3].sort_values(["quarter","theme_rank"])
    top_themes.to_csv(OUT/"quant_theme_rankings.csv", index=False)

    # Quant stock selection: top names inside top themes, favor non-leaders to find second-order beneficiaries.
    top_keys = set(zip(top_themes.quarter, top_themes.bucket))
    cand = agg[[ (q,b) in top_keys for q,b in zip(agg.quarter, agg.bucket) ]].copy()
    cand = cand.sort_values(["quarter","consensus_score","method_hits","holder_abs_change"],
                            ascending=[True,False,False,False])
    selected_rows = []
    for q, g in cand.groupby("quarter", sort=True):
        # First pass: opportunistic/non-leader, max 2 names per theme.
        chosen=[]; counts={}
        for _,r in g.iterrows():
            if bool(r.is_leader):
                continue
            b=r.bucket
            if counts.get(b,0)>=2: continue
            chosen.append(r)
            counts[b]=counts.get(b,0)+1
            if len(chosen)>=PORTFOLIO_SIZE: break
        # Fill from leaders if theme breadth is too small.
        if len(chosen)<PORTFOLIO_SIZE:
            used={str(r.ticker) for r in chosen}
            for _,r in g.iterrows():
                if str(r.ticker) in used: continue
                b=r.bucket
                if counts.get(b,0)>=2: continue
                chosen.append(r); counts[b]=counts.get(b,0)+1
                if len(chosen)>=PORTFOLIO_SIZE: break
        for j,r in enumerate(chosen,1):
            d=r.to_dict(); d["portfolio_rank"]=j; selected_rows.append(d)
    sel = pd.DataFrame(selected_rows)

    # Forward returns are in the point-in-time candidate pool from the prior backtest.
    # Keep them OUT of the blind packet and merge only into the evaluation output.
    ret = allu.groupby(["quarter","ticker"], as_index=False).agg(
        entry_date=("entry_date","first"),
        exit_date=("exit_date","first"),
        return_90d=("return_90d","first"),
        spy_return_90d=("spy_return_90d","first"),
        spy_excess_return=("spy_excess_return","first"),
    )
    if not sel.empty:
        sel = sel.merge(ret, on=["quarter","ticker"], how="left")
    sel.to_csv(OUT/"quant_theme_constituents.csv", index=False)

    qrows=[]
    if not sel.empty:
        for q,g in sel.groupby("quarter",sort=True):
            z=g.dropna(subset=["return_90d"])
            complete=len(g)==PORTFOLIO_SIZE and len(z)==PORTFOLIO_SIZE
            themes = top_themes[top_themes.quarter.eq(q)].sort_values("theme_rank")
            qrows.append({
                "quarter":q,
                "top_themes":" | ".join(themes.bucket.tolist()),
                "tickers":" | ".join(g.sort_values("portfolio_rank").ticker.tolist()),
                "complete_top5":complete,
                "portfolio_return_90d":float(z.return_90d.mean()) if complete else np.nan,
                "spy_return_90d":float(z.spy_return_90d.mean()) if complete else np.nan,
                "spy_excess_return":float(z.spy_excess_return.mean()) if complete else np.nan,
            })
    qdf=pd.DataFrame(qrows)
    qdf.to_csv(OUT/"quant_theme_quarterly.csv",index=False)
    if not qdf.empty:
        z=qdf.dropna(subset=["portfolio_return_90d"])
        summary={
            "complete_quarters":int(len(z)),
            "avg_90d_return":float(z.portfolio_return_90d.mean()) if len(z) else None,
            "median_90d_return":float(z.portfolio_return_90d.median()) if len(z) else None,
            "positive_rate":float((z.portfolio_return_90d>0).mean()) if len(z) else None,
            "avg_spy_excess":float(z.spy_excess_return.mean()) if len(z) else None,
            "spy_outperformance_rate":float((z.spy_excess_return>0).mean()) if len(z) else None,
        }
    else:
        summary={}
    (OUT/"quant_theme_summary.json").write_text(json.dumps(summary,indent=2))

    # Separate forward-return lookup for the blind packet. Do not use until
    # qualitative selections are frozen.
    fwd = packets[["quarter","ticker"]].merge(ret,on=["quarter","ticker"],how="left")
    fwd.to_csv(OUT/"theme_packet_forward_returns.csv",index=False)

    meta={
        "source":"results/accumulation_candidate_pool.csv",
        "information_firewall":"theme_blind_packets.csv excludes all forward-return fields; forward returns are stored separately",
        "quant_method":"fixed industry taxonomy + rank consensus + breadth + method coverage + holder-growth percentile + quarter-over-quarter acceleration",
        "theme_score_formula":"0.45 mean_consensus + 0.25 breadth + 0.20 method_coverage + 0.10 holder_strength + 0.15 acceleration",
        "portfolio_rule":"top 3 themes; choose up to 5 stocks, favor non-leaders, max 2 per theme; equal-weight evaluation",
        "residual_hindsight_risk":"taxonomy is authored today; broad industry labels are intended to minimize but cannot fully eliminate retrospective classification bias",
    }
    (OUT/"theme_pilot_metadata.json").write_text(json.dumps(meta,indent=2))
    print("Wrote theme pilot outputs")
    print(summary)

if __name__=="__main__":
    main()
