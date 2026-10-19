#!/usr/bin/env python3
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
import backtest as bt

OUT = Path("results")
OUT.mkdir(exist_ok=True)

START_Q = date(2016,3,31)
END_Q = date(2026,3,31)
TOP_N = 30

def quarter_ends():
    out=[]
    for y in range(2016, 2027):
        for m,d in [(3,31),(6,30),(9,30),(12,31)]:
            q=date(y,m,d)
            if START_Q <= q <= END_Q:
                out.append(q)
    return out

def compute_new_holders(db, q):
    pq = bt.prev_quarter(q)
    cutoff = bt.cutoff_for(q)
    pcutoff = bt.cutoff_for(pq)

    managers = db.execute("""
      WITH c AS (
        SELECT DISTINCT cik FROM filings
        WHERE period=? AND CAST(available_at AS DATE)<=?
      ), p AS (
        SELECT DISTINCT cik FROM filings
        WHERE period=? AND CAST(available_at AS DATE)<=?
      )
      SELECT c.cik FROM c INNER JOIN p USING(cik)
    """, [q, cutoff, pq, pcutoff]).df()

    if managers.empty:
        return pd.DataFrame(), 0

    db.register("nh_matched_mgrs", managers)

    x = db.execute("""
      WITH cf AS (
        SELECT accession,cik,
               ROW_NUMBER() OVER(PARTITION BY cik ORDER BY available_at DESC, accession DESC) rn
        FROM filings
        WHERE period=? AND CAST(available_at AS DATE)<=?
          AND cik IN (SELECT cik FROM nh_matched_mgrs)
      ),
      pf AS (
        SELECT accession,cik,
               ROW_NUMBER() OVER(PARTITION BY cik ORDER BY available_at DESC, accession DESC) rn
        FROM filings
        WHERE period=? AND CAST(available_at AS DATE)<=?
          AND cik IN (SELECT cik FROM nh_matched_mgrs)
      ),
      cpos AS (
        SELECT f.cik,h.cusip,
               MODE(h.ticker) ticker,
               MODE(h.issuerName) issuer,
               MODE(h.titleOfClass) class_name,
               SUM(h.shares) shares,
               SUM(h.value_usd) value_usd
        FROM cf f JOIN holdings h USING(accession)
        WHERE f.rn=1
        GROUP BY f.cik,h.cusip
        HAVING SUM(h.shares) > 0
      ),
      ppos AS (
        SELECT f.cik,h.cusip,SUM(h.shares) shares
        FROM pf f JOIN holdings h USING(accession)
        WHERE f.rn=1
        GROUP BY f.cik,h.cusip
        HAVING SUM(h.shares) > 0
      ),
      stats AS (
        SELECT
          c.cusip,
          MODE(c.ticker) ticker,
          MODE(c.issuer) issuer,
          MODE(c.class_name) class_name,
          COUNT(DISTINCT c.cik) current_holders,
          COUNT(DISTINCT CASE WHEN p.cik IS NULL THEN c.cik END) new_holders,
          COUNT(DISTINCT CASE WHEN p.cik IS NOT NULL THEN c.cik END) continuing_holders,
          SUM(c.value_usd) current_value_usd
        FROM cpos c
        LEFT JOIN ppos p ON c.cik=p.cik AND c.cusip=p.cusip
        GROUP BY c.cusip
      ),
      prevstats AS (
        SELECT cusip, COUNT(DISTINCT cik) prior_holders
        FROM ppos
        GROUP BY cusip
      )
      SELECT
        s.cusip,s.ticker,s.issuer,s.class_name,
        s.new_holders,s.current_holders,s.continuing_holders,
        COALESCE(p.prior_holders,0) prior_holders,
        s.current_value_usd
      FROM stats s
      LEFT JOIN prevstats p USING(cusip)
    """, [q, cutoff, pq, pcutoff]).df()

    if x.empty:
        return x, len(managers)

    x = bt.clean_equity(x, year=q.year)
    x = x[x["new_holders"] > 0].copy()
    x["new_holder_share_of_current"] = np.where(
        x["current_holders"] > 0,
        x["new_holders"] / x["current_holders"],
        np.nan
    )
    x["quarter"] = bt.qlabel(q)
    x["period_end"] = q.isoformat()
    x["filing_cutoff"] = cutoff.isoformat()
    x["matched_managers"] = int(len(managers))

    # Pure ranking by new-holder count. The next two fields only break ties.
    x = x.sort_values(
        ["new_holders","current_holders","current_value_usd","ticker"],
        ascending=[False,False,False,True]
    )
    x = x.drop_duplicates(subset=["ticker"], keep="first")
    x = x.drop_duplicates(subset=["issuer"], keep="first")
    x = x.head(TOP_N).copy()
    x["rank"] = range(1, len(x)+1)
    return x, len(managers)

def main():
    # Reuse the exact point-in-time source/filtering stack from the existing backtest.
    bt.START_YEAR = 2016
    bt.END_YEAR = 2026
    bt.SEC_FUND_TICKERS = bt.load_sec_fund_tickers()
    bt.SEC_FUND_TICKERS_BY_YEAR = bt.load_sec_annual_fund_tickers()
    bt.SEC_CEF_NAMES_BY_YEAR = bt.load_sec_cef_names_by_year()

    snap = bt.load_snapshot()
    paths = bt.cache_dataset_files(snap)
    db = bt.build_db(paths)

    rows=[]
    quality=[]
    for q in quarter_ends():
        print("new-holder", bt.qlabel(q), flush=True)
        top, n_mgr = compute_new_holders(db,q)
        quality.append({
            "quarter": bt.qlabel(q),
            "period_end": q.isoformat(),
            "matched_managers": n_mgr,
            "top_rows": len(top)
        })
        if not top.empty:
            rows.append(top)

    out = pd.concat(rows, ignore_index=True)
    cols = [
        "quarter","rank","ticker","issuer","cusip","new_holders",
        "current_holders","prior_holders","continuing_holders",
        "new_holder_share_of_current","current_value_usd",
        "matched_managers","period_end","filing_cutoff"
    ]
    out = out[cols]
    csv_path = OUT/"new_holder_top30_2016Q1_2026Q1.csv"
    out.to_csv(csv_path,index=False)
    pd.DataFrame(quality).to_csv(OUT/"new_holder_top30_data_quality.csv",index=False)

    md=[]
    md.append("# 13F New-Holder Top 30 — 2016Q1 to 2026Q1")
    md.append("")
    md.append("Primary signal: count of matched 13F managers with no position in the same CUSIP in the prior quarter and a positive position in the current quarter.")
    md.append("")
    md.append("Managers must have original 13F-HR filings in both adjacent quarters by the point-in-time statutory filing cutoffs. Ranking is by new_holders only; current_holders and current_value_usd break ties.")
    md.append("")
    for q, g in out.groupby("quarter", sort=False):
        md.append(f"## {q}")
        md.append("")
        md.append("| Rank | Ticker | Company | New holders | Current holders | Prior holders |")
        md.append("|---:|---|---|---:|---:|---:|")
        for r in g.itertuples(index=False):
            issuer=str(r.issuer).replace("|","/")
            md.append(f"| {r.rank} | {r.ticker} | {issuer} | {int(r.new_holders)} | {int(r.current_holders)} | {int(r.prior_holders)} |")
        md.append("")
    (OUT/"new_holder_top30_2016Q1_2026Q1.md").write_text("\n".join(md), encoding="utf-8")
    print(out.groupby("quarter").size().to_string())
    print("wrote", len(out), "rows")

if __name__=="__main__":
    main()
