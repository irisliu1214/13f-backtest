#!/usr/bin/env python3
import os, re, json, math, time, shutil, hashlib
from pathlib import Path
from datetime import date, datetime, timedelta

import requests
import duckdb
import pandas as pd
import numpy as np
import yfinance as yf

DATASET = "austin-starks/sec-ownership-disclosures"
HF_ROOT = f"https://huggingface.co/datasets/{DATASET}/resolve/main"
CACHE = Path("data_cache")
RESULTS = Path("results")
CACHE.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

START_YEAR = int(os.getenv("START_YEAR", "2016"))
END_YEAR = int(os.getenv("END_YEAR", "2026"))
PRIMARY_FLOOR_SHARES = float(os.getenv("PRIMARY_FLOOR_SHARES", "1000000"))
FLOORS = [float(x) for x in os.getenv(
    "FLOOR_GRID", "100000,500000,1000000,5000000,10000000,25000000,50000000,100000000"
).split(",")]

EQUITY_BAD = re.compile(
    r"(NOTE|BOND|DEBT|PRFD|PREFERRED|PFD|WARRANT|UNIT|RIGHT|CVT|CONVERTIBLE|"
    r"ETF|ETN|INDEX|MUTUAL|ISHARES|VANGUARD|SPDR|PROSHARES|KRANESHARES|"
    r"GLOBAL X|FIDELITY COVINGTON|LISTED FD|INVESCO.*TRUST|WISDOMTREE)", re.I
)
TICKER_OK = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")

def download(url: str, out: Path):
    if out.exists() and out.stat().st_size > 0:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(out)

def load_snapshot():
    r = requests.get(f"{HF_ROOT}/snapshot.json", timeout=60)
    r.raise_for_status()
    return r.json()

def files_for(snapshot, table, years):
    ans = []
    t = snapshot["tables"][table]
    for m in t["manifests"]:
        if int(m["year"]) in years:
            ans.extend(m["files"])
    return ans

def cache_dataset_files(snapshot):
    # Availability-year shards. Include 2015 for 2016Q1 prior-quarter filings and 2027 is not available/needed.
    years = set(range(max(2015, START_YEAR - 1), END_YEAR + 1))
    paths = {"institutional_filings": [], "institutional_holdings": []}
    for table in paths:
        selected = files_for(snapshot, table, years)
        print(f"{table}: {len(selected)} parquet file(s)")
        for i, f in enumerate(selected, 1):
            p = CACHE / f["publicPath"]
            print(f"  [{i}/{len(selected)}] {f['publicPath']} ({f['size']/1e6:.1f} MB)")
            download(f"{HF_ROOT}/{f['publicPath']}", p)
            paths[table].append(p)
    return paths

def sql_list(paths):
    return "[" + ",".join("'" + str(p).replace("'", "''") + "'" for p in paths) + "]"

def build_db(paths):
    db = duckdb.connect(str(CACHE / "backtest.duckdb"))
    db.execute("PRAGMA threads=4")
    db.execute("PRAGMA memory_limit='6GB'")
    db.execute("DROP TABLE IF EXISTS filings")
    db.execute("DROP TABLE IF EXISTS holdings")

    db.execute(f"""
      CREATE TABLE filings AS
      SELECT
        accession,
        CAST(cik AS VARCHAR) AS cik,
        managerName,
        CAST(periodOfReport AS DATE) AS period,
        CAST(availableAt AS TIMESTAMP) AS available_at
      FROM read_parquet({sql_list(paths['institutional_filings'])}, union_by_name=true)
      WHERE submissionType='13F-HR'
        AND COALESCE(isAmendment, false)=false
        AND periodOfReport IS NOT NULL
        AND CAST(periodOfReport AS DATE) BETWEEN DATE '{START_YEAR-1}-12-31' AND DATE '{END_YEAR}-12-31'
    """)

    # Only keep columns needed for common-equity accumulation.
    db.execute(f"""
      CREATE TABLE holdings AS
      SELECT
        accession,
        CAST(cusip AS VARCHAR) AS cusip,
        UPPER(TRIM(CAST(resolvedTicker AS VARCHAR))) AS ticker,
        issuerName,
        titleOfClass,
        UPPER(TRIM(CAST(sharesType AS VARCHAR))) AS shares_type,
        CAST(sharesAmount AS DOUBLE) AS shares,
        CAST(value AS DOUBLE) AS value_usd,
        putCall
      FROM read_parquet({sql_list(paths['institutional_holdings'])}, union_by_name=true)
      WHERE sharesAmount IS NOT NULL
        AND CAST(sharesAmount AS DOUBLE) >= 0
        AND CAST(value AS DOUBLE) > 0
        AND putCall IS NULL
        AND UPPER(TRIM(CAST(sharesType AS VARCHAR)))='SH'
        AND resolvedTicker IS NOT NULL
    """)
    db.execute("CREATE INDEX IF NOT EXISTS filings_period ON filings(period)")
    db.execute("CREATE INDEX IF NOT EXISTS holdings_accession ON holdings(accession)")
    return db

def quarter_ends(start_year, end_year):
    out = []
    for y in range(start_year, end_year + 1):
        for m, d in [(3,31),(6,30),(9,30),(12,31)]:
            q = date(y,m,d)
            if q <= date.today():
                out.append(q)
    return out

def cutoff_for(q: date) -> date:
    # Statutory 13F deadline is 45 days after quarter end. Using one fixed cutoff
    # avoids cherry-picking early filers and eliminates period-end look-ahead.
    return q + timedelta(days=45)

def prev_quarter(q: date) -> date:
    if q.month == 3: return date(q.year-1,12,31)
    if q.month == 6: return date(q.year,3,31)
    if q.month == 9: return date(q.year,6,30)
    return date(q.year,9,30)

def qlabel(q): return f"{q.year}Q{(q.month-1)//3+1}"

def clean_equity(df):
    if df.empty: return df
    text = (df["issuer"].fillna("") + " " + df["class_name"].fillna(""))
    good = ~text.str.contains(EQUITY_BAD)
    good &= ~text.str.contains(r"(FUND|PORTFOLIO|BETABUILDERS|ISHARES|VANGUARD|SPDR|PROSHARES|ETF|INDEX|ETN|KRANESHARES|GLOBAL X)", case=False, regex=True)
    good &= df["ticker"].fillna("").str.match(TICKER_OK)
    return df.loc[good].copy()

def aggregate_quarter(db, q: date):
    pq = prev_quarter(q)
    cutoff = cutoff_for(q)
    pcutoff = cutoff_for(pq)

    # A manager must have an original filing in both periods by each fixed information cutoff.
    managers = db.execute("""
      WITH c AS (
        SELECT DISTINCT cik FROM filings WHERE period=? AND CAST(available_at AS DATE)<=?
      ), p AS (
        SELECT DISTINCT cik FROM filings WHERE period=? AND CAST(available_at AS DATE)<=?
      )
      SELECT c.cik FROM c INNER JOIN p USING(cik)
    """, [q, cutoff, pq, pcutoff]).df()
    if managers.empty:
        return pd.DataFrame(), {"matched_managers": 0}

    db.register("matched_mgrs", managers)
    def one(period, asof):
        x = db.execute("""
          WITH f AS (
            SELECT accession,cik,
                   ROW_NUMBER() OVER(PARTITION BY cik ORDER BY available_at DESC, accession DESC) rn
            FROM filings
            WHERE period=? AND CAST(available_at AS DATE)<=?
              AND cik IN (SELECT cik FROM matched_mgrs)
          )
          SELECT
            h.cusip,
            MODE(h.ticker) ticker,
            MODE(h.issuerName) issuer,
            MODE(h.titleOfClass) AS class_name,
            SUM(h.shares) shares,
            SUM(h.value_usd) value_usd,
            COUNT(DISTINCT f.cik) holders
          FROM f JOIN holdings h USING(accession)
          WHERE f.rn=1
          GROUP BY h.cusip
        """, [period, asof]).df()
        return clean_equity(x)

    cur = one(q, cutoff).rename(columns={
        "shares":"cur_shares","value_usd":"cur_value","holders":"cur_holders",
        "issuer":"cur_issuer","class_name":"cur_class"
    })
    prv = one(pq, pcutoff).rename(columns={
        "shares":"prev_shares","value_usd":"prev_value","holders":"prev_holders",
        "issuer":"prev_issuer","class_name":"prev_class"
    })
    m = cur.merge(prv, on="cusip", how="outer", suffixes=("", "_prev"))
    m["ticker"] = m["ticker"].fillna(m.get("ticker_prev"))
    for c in ["cur_shares","prev_shares","cur_value","prev_value","cur_holders","prev_holders"]:
        m[c] = m[c].fillna(0.0)
    m["issuer"] = m["cur_issuer"].fillna(m["prev_issuer"])
    m["class_name"] = m["cur_class"].fillna(m["prev_class"])

    # Infer forward split factors from both inverse implied-price movement and share-count jump.
    # This catches the large distortions that otherwise dominate an absolute-share ranking.
    prev_px = np.where(m.prev_shares > 0, m.prev_value / m.prev_shares, np.nan)
    cur_px  = np.where(m.cur_shares > 0, m.cur_value / m.cur_shares, np.nan)
    px_ratio = prev_px / cur_px
    sh_ratio = np.where(m.prev_shares > 0, m.cur_shares / m.prev_shares, np.nan)
    candidates = np.array([1.25,4/3,1.5,2,3,4,5,10,20,40], dtype=float)
    factors = np.ones(len(m))
    for i in range(len(m)):
        if not np.isfinite(px_ratio[i]) or not np.isfinite(sh_ratio[i]): continue
        errs = np.abs(np.log(px_ratio[i] / candidates)) + np.abs(np.log(sh_ratio[i] / candidates))
        f = candidates[np.argmin(errs)]
        if 0.60 <= px_ratio[i]/f <= 1.55 and 0.60 <= sh_ratio[i]/f <= 1.55:
            factors[i] = f
    m["split_factor"] = factors
    m["prev_shares_adj"] = m.prev_shares * m.split_factor
    m["abs_change"] = m.cur_shares - m.prev_shares_adj
    m["pct_change"] = np.where(m.prev_shares_adj > 0, m.abs_change / m.prev_shares_adj, np.nan)
    m["quarter"] = qlabel(q)
    m["period_end"] = q.isoformat()
    m["filing_cutoff"] = cutoff.isoformat()
    return m, {"matched_managers": int(len(managers)), "securities": int(len(m))}

def choose_winners(metrics):
    pos = metrics[metrics.abs_change > 0].copy()
    winners = {}
    if pos.empty: return winners
    winners["Absolute"] = pos.sort_values(["abs_change","pct_change"], ascending=False).iloc[0]
    pct = pos[np.isfinite(pd.to_numeric(pos["pct_change"], errors="coerce"))]
    if not pct.empty:
        winners["Percentage"] = pct.sort_values(["pct_change","abs_change"], ascending=False).iloc[0]
    for f in FLOORS:
        z = pct[pct.abs_change >= f]
        if not z.empty:
            winners[f"Percentage+floor:{int(f)}"] = z.sort_values(["pct_change","abs_change"], ascending=False).iloc[0]
    return winners

def next_price_date(idx, target, max_gap_days=7):
    target = pd.Timestamp(target)
    p = idx.searchsorted(target)
    if p >= len(idx):
        return None
    found = idx[p]
    if (found.tz_localize(None) if getattr(found, "tzinfo", None) else found) - target > pd.Timedelta(days=max_gap_days):
        return None
    return found

def download_prices(tickers):
    tickers = sorted(set(tickers) | {"SPY"})
    start = f"{START_YEAR}-01-01"
    end = (date.today() + timedelta(days=5)).isoformat()
    all_frames = {}
    # Chunk to reduce Yahoo throttling.
    for k in range(0, len(tickers), 40):
        chunk = tickers[k:k+40]
        for attempt in range(3):
            try:
                raw = yf.download(chunk, start=start, end=end, auto_adjust=True,
                                  progress=False, group_by="ticker", threads=True)
                break
            except Exception:
                if attempt == 2: raw = pd.DataFrame()
                time.sleep(3*(attempt+1))
        if raw.empty: continue
        if len(chunk) == 1:
            t = chunk[0]
            s = raw["Close"].dropna() if "Close" in raw else pd.Series(dtype=float)
            all_frames[t] = s
        else:
            for t in chunk:
                try:
                    s = raw[t]["Close"].dropna()
                    if not s.empty: all_frames[t] = s
                except Exception:
                    pass
    return all_frames

def add_returns(rows):
    tickers = [r["ticker"] for r in rows if r.get("ticker")]
    prices = download_prices(tickers)
    spy = prices.get("SPY", pd.Series(dtype=float))
    for r in rows:
        q = datetime.strptime(r["period_end"], "%Y-%m-%d").date()
        target_entry = cutoff_for(q) + timedelta(days=1)
        s = prices.get(r["ticker"], pd.Series(dtype=float))
        if s.empty or spy.empty:
            r.update(entry_date=None, exit_date=None, return_90d=np.nan, spy_return_90d=np.nan, spy_excess_return=np.nan)
            continue
        entry = next_price_date(s.index, target_entry)
        if entry is None:
            r.update(entry_date=None, exit_date=None, return_90d=np.nan, spy_return_90d=np.nan, spy_excess_return=np.nan)
            continue
        target_exit = entry.date() + timedelta(days=90)
        exitd = next_price_date(s.index, target_exit)
        spy_entry = next_price_date(spy.index, entry)
        spy_exit = next_price_date(spy.index, target_exit)
        if exitd is None or spy_entry is None or spy_exit is None:
            r.update(entry_date=entry.date().isoformat(), exit_date=None, return_90d=np.nan, spy_return_90d=np.nan, spy_excess_return=np.nan)
            continue
        ret = float(s.loc[exitd] / s.loc[entry] - 1)
        sret = float(spy.loc[spy_exit] / spy.loc[spy_entry] - 1)
        r.update(entry_date=entry.date().isoformat(), exit_date=exitd.date().isoformat(),
                 return_90d=ret, spy_return_90d=sret, spy_excess_return=ret-sret)

def summary(df):
    rows=[]
    for method,g in df.groupby("method"):
        z=g.dropna(subset=["return_90d"])
        if z.empty: continue
        rows.append({
          "method":method, "quarters":len(z),
          "avg_90d_return":z.return_90d.mean(),
          "median_90d_return":z.return_90d.median(),
          "positive_rate":(z.return_90d>0).mean(),
          "avg_spy_excess":z.spy_excess_return.mean(),
          "median_spy_excess":z.spy_excess_return.median(),
          "spy_outperformance_rate":(z.spy_excess_return>0).mean(),
          "compounded_return":float(np.prod(1+z.return_90d)-1),
          "compounded_spy":float(np.prod(1+z.spy_return_90d)-1),
        })
    return pd.DataFrame(rows)

def main():
    snap = load_snapshot()
    print("snapshot generated:", snap.get("generatedAt"))
    paths = cache_dataset_files(snap)
    db = build_db(paths)

    winner_rows=[]
    quality=[]
    metric_winners=[]
    for q in quarter_ends(START_YEAR, END_YEAR):
        print("processing", qlabel(q))
        m, meta = aggregate_quarter(db, q)
        quality.append({"quarter":qlabel(q),"period_end":q.isoformat(),**meta})
        if m.empty: continue
        wins = choose_winners(m)
        for method, row in wins.items():
            rec = {
              "quarter": qlabel(q), "period_end": q.isoformat(),
              "method": method, "ticker": row.ticker, "cusip": row.cusip, "issuer": row.issuer,
              "abs_share_change": float(row.abs_change),
              "pct_share_change": float(row["pct_change"]) if np.isfinite(float(row["pct_change"])) else np.nan,
              "split_factor": float(row.split_factor),
              "current_shares": float(row.cur_shares),
              "prior_shares_adjusted": float(row.prev_shares_adj),
              "matched_managers": meta["matched_managers"]
            }
            winner_rows.append(rec)

    add_returns(winner_rows)
    allw = pd.DataFrame(winner_rows)
    allw.to_csv(RESULTS/"all_method_winners.csv", index=False)

    # Requested headline table.
    primary_floor_label = f"Percentage+floor:{int(PRIMARY_FLOOR_SHARES)}"
    headline = allw[allw.method.isin(["Absolute","Percentage",primary_floor_label])].copy()
    headline["method"] = headline["method"].replace({primary_floor_label:"Percentage+floor"})
    headline = headline[[
      "quarter","method","ticker","cusip","issuer","entry_date","exit_date",
      "return_90d","spy_return_90d","spy_excess_return",
      "abs_share_change","pct_share_change","split_factor","matched_managers"
    ]]
    headline.to_csv(RESULTS/"quarterly_backtest.csv", index=False)

    sens = allw[allw.method.str.startswith("Percentage+floor:", na=False)].copy()
    sens["floor_shares"] = sens.method.str.split(":").str[1].astype(float)
    sens.to_csv(RESULTS/"floor_sensitivity.csv", index=False)

    summ = summary(headline)
    summ.to_csv(RESULTS/"method_summary.csv", index=False)
    floor_summ = summary(sens.rename(columns={"method":"method"}))
    floor_summ.to_csv(RESULTS/"floor_method_summary.csv", index=False)
    pd.DataFrame(quality).to_csv(RESULTS/"data_quality.csv", index=False)

    meta = {
      "dataset": DATASET,
      "dataset_snapshot_generated_at": snap.get("generatedAt"),
      "start_year": START_YEAR, "end_year": END_YEAR,
      "primary_floor_shares": PRIMARY_FLOOR_SHARES,
      "floor_grid": FLOORS,
      "generated_at_utc": datetime.utcnow().isoformat()+"Z",
      "note": "2026Q2 may not yet have a complete 90-calendar-day forward return."
    }
    (RESULTS/"run_metadata.json").write_text(json.dumps(meta, indent=2))
    print("\nHEADLINE")
    print(headline.to_string(index=False))
    print("\nSUMMARY")
    print(summ.to_string(index=False))

if __name__ == "__main__":
    main()
