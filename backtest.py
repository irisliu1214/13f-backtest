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
PRIMARY_FLOOR_MANAGERS = int(os.getenv("PRIMARY_FLOOR_MANAGERS", "5"))
LEADER_TOP_N = int(os.getenv("LEADER_TOP_N", "50"))
PORTFOLIO_SIZE = int(os.getenv("PORTFOLIO_SIZE", "5"))

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

    # Signals are defined ONLY on institution counts among matched managers.
    # Absolute = current holders - prior holders.
    # Percentage = Absolute / prior holders (new positions with prior_holders=0 are excluded from Percentage).
    m["holder_abs_change"] = m.cur_holders - m.prev_holders
    m["holder_pct_change"] = np.where(
        m.prev_holders > 0,
        m.holder_abs_change / m.prev_holders,
        np.nan
    )

    # Point-in-time leader definition used for the opportunistic universe.
    # Top N current-quarter common equities by aggregate 13F market value are "leaders".
    m["leader_rank"] = m.cur_value.where(m.cur_value > 0).rank(method="min", ascending=False)
    m["is_leader"] = m["leader_rank"].le(LEADER_TOP_N).fillna(False)

    m["quarter"] = qlabel(q)
    m["period_end"] = q.isoformat()
    m["filing_cutoff"] = cutoff.isoformat()
    return m, {"matched_managers": int(len(managers)), "securities": int(len(m))}

def rank_candidates(metrics, method, exclude_leaders=False):
    x = metrics[metrics.holder_abs_change > 0].copy()
    if exclude_leaders:
        x = x[~x.is_leader]
    if x.empty:
        return x

    if method == "Absolute":
        x = x.sort_values(
            ["holder_abs_change", "holder_pct_change", "cur_holders"],
            ascending=[False, False, False]
        )
    elif method == "Percentage":
        x = x[np.isfinite(pd.to_numeric(x["holder_pct_change"], errors="coerce"))]
        x = x.sort_values(
            ["holder_pct_change", "holder_abs_change", "cur_holders"],
            ascending=[False, False, False]
        )
    elif method == "Percentage+floor":
        x = x[
            np.isfinite(pd.to_numeric(x["holder_pct_change"], errors="coerce"))
            & (x.holder_abs_change >= PRIMARY_FLOOR_MANAGERS)
        ]
        x = x.sort_values(
            ["holder_pct_change", "holder_abs_change", "cur_holders"],
            ascending=[False, False, False]
        )
    else:
        raise ValueError(method)

    return x.drop_duplicates(subset=["ticker"]).head(PORTFOLIO_SIZE).copy()

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

def build_quarterly_portfolio(constituents):
    rows = []
    if constituents.empty:
        return pd.DataFrame()
    keys = ["quarter", "period_end", "universe", "method"]
    for key, grp in constituents.groupby(keys, sort=True):
        grp = grp.sort_values("portfolio_rank")
        valid = grp.dropna(subset=["return_90d"])
        complete = len(grp) == PORTFOLIO_SIZE and len(valid) == PORTFOLIO_SIZE
        rows.append({
            "quarter": key[0],
            "period_end": key[1],
            "universe": key[2],
            "method": key[3],
            "n_selected": int(len(grp)),
            "n_with_returns": int(len(valid)),
            "complete_top5": bool(complete),
            "tickers": "|".join(grp.ticker.astype(str).tolist()),
            "entry_date": valid.entry_date.mode().iloc[0] if not valid.empty and not valid.entry_date.mode().empty else None,
            "exit_date": valid.exit_date.mode().iloc[0] if not valid.empty and not valid.exit_date.mode().empty else None,
            "portfolio_return_90d": float(valid.return_90d.mean()) if complete else np.nan,
            "spy_return_90d": float(valid.spy_return_90d.mean()) if complete else np.nan,
            "spy_excess_return": float(valid.spy_excess_return.mean()) if complete else np.nan,
        })
    return pd.DataFrame(rows)

def portfolio_summary(portfolios):
    rows = []
    if portfolios.empty:
        return pd.DataFrame()
    for (universe, method), grp in portfolios.groupby(["universe", "method"], sort=True):
        z = grp.dropna(subset=["portfolio_return_90d"])
        if z.empty:
            continue
        rows.append({
            "universe": universe,
            "method": method,
            "complete_quarters": int(len(z)),
            "avg_90d_return": float(z.portfolio_return_90d.mean()),
            "median_90d_return": float(z.portfolio_return_90d.median()),
            "positive_rate": float((z.portfolio_return_90d > 0).mean()),
            "avg_spy_excess": float(z.spy_excess_return.mean()),
            "median_spy_excess": float(z.spy_excess_return.median()),
            "spy_outperformance_rate": float((z.spy_excess_return > 0).mean()),
            "compounded_return": float(np.prod(1 + z.portfolio_return_90d) - 1),
            "compounded_spy": float(np.prod(1 + z.spy_return_90d) - 1),
        })
    return pd.DataFrame(rows)

def main():
    snap = load_snapshot()
    print("snapshot generated:", snap.get("generatedAt"))
    paths = cache_dataset_files(snap)
    db = build_db(paths)

    constituent_rows = []
    quality = []
    methods = ["Absolute", "Percentage", "Percentage+floor"]
    universes = [
        ("AllStocks", False),
        ("OpportunisticExLeaders", True),
    ]

    for q in quarter_ends(START_YEAR, END_YEAR):
        print("processing", qlabel(q))
        m, meta = aggregate_quarter(db, q)
        quality.append({"quarter": qlabel(q), "period_end": q.isoformat(), **meta})
        if m.empty:
            continue

        for universe, exclude_leaders in universes:
            for method in methods:
                picks = rank_candidates(m, method, exclude_leaders=exclude_leaders)
                for rank, (_, row) in enumerate(picks.iterrows(), 1):
                    constituent_rows.append({
                        "quarter": qlabel(q),
                        "period_end": q.isoformat(),
                        "universe": universe,
                        "method": method,
                        "portfolio_rank": rank,
                        "weight": 1.0 / PORTFOLIO_SIZE,
                        "ticker": row.ticker,
                        "cusip": row.cusip,
                        "issuer": row.issuer,
                        "is_leader": bool(row.is_leader),
                        "leader_rank": float(row.leader_rank) if pd.notna(row.leader_rank) else np.nan,
                        "current_holders": float(row.cur_holders),
                        "prior_holders": float(row.prev_holders),
                        "holder_abs_change": float(row.holder_abs_change),
                        "holder_pct_change": float(row.holder_pct_change) if pd.notna(row.holder_pct_change) else np.nan,
                        "current_value_usd": float(row.cur_value),
                        "current_shares": float(row.cur_shares),
                        "matched_managers": meta["matched_managers"],
                    })

    add_returns(constituent_rows)
    constituents = pd.DataFrame(constituent_rows)
    constituents.to_csv(RESULTS / "top5_constituents.csv", index=False)

    portfolios = build_quarterly_portfolio(constituents)
    portfolios.to_csv(RESULTS / "top5_quarterly_portfolios.csv", index=False)

    summ = portfolio_summary(portfolios)
    summ.to_csv(RESULTS / "top5_method_summary.csv", index=False)
    pd.DataFrame(quality).to_csv(RESULTS / "data_quality.csv", index=False)

    meta = {
        "dataset": DATASET,
        "dataset_snapshot_generated_at": snap.get("generatedAt"),
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "portfolio_size": PORTFOLIO_SIZE,
        "weighting": "equal_weight_20pct_each_when_top5_complete",
        "absolute_definition": "current matched 13F holder count minus prior-quarter matched 13F holder count",
        "percentage_definition": "holder_abs_change divided by prior-quarter matched holder count; zero-base/new positions excluded",
        "percentage_floor_definition": f"Percentage ranking with holder_abs_change >= {PRIMARY_FLOOR_MANAGERS}",
        "leader_definition": f"top {LEADER_TOP_N} current-quarter common equities by aggregate 13F market value, point-in-time at filing cutoff",
        "universes": ["AllStocks", "OpportunisticExLeaders"],
        "entry_rule": "first trading day after the 45-day 13F filing cutoff",
        "holding_period": "90 calendar days, exit first trading day on/after day 90",
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "note": "2026Q2 may not yet have a complete 90-calendar-day forward return."
    }
    (RESULTS / "top5_run_metadata.json").write_text(json.dumps(meta, indent=2))

    print("\nTOP5 PORTFOLIOS")
    print(portfolios.to_string(index=False))
    print("\nSUMMARY")
    print(summ.to_string(index=False))

if __name__ == "__main__":
    main()
