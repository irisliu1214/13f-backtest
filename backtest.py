#!/usr/bin/env python3
import os, re, json, math, time, shutil, hashlib, io
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
CANDIDATE_DEPTH = int(os.getenv("CANDIDATE_DEPTH", "20"))
BUYER_EPS = float(os.getenv("BUYER_EPS", "0.01"))

EQUITY_BAD = re.compile(
    r"(NOTE|BOND|DEBT|PRFD|PREFERRED|PFD|WARRANT|UNIT|RIGHT|CVT|CONVERTIBLE|"
    r"ETF|ETN|INDEX|MUTUAL|ISHARES|VANGUARD|SPDR|PROSHARES|KRANESHARES|"
    r"GLOBAL X|FIDELITY COVINGTON|LISTED FD|INVESCO.*TRUST|WISDOMTREE|GRAYSCALE|SPROTT PHYSICAL|BITCOIN TRUST|GOLD TRUST|SILVER TRUST|URANIUM ETF)", re.I
)
TICKER_OK = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
SEC_FUND_TICKERS = set()
SEC_CEF_NAMES_BY_YEAR = {}
KNOWN_CEF_TICKERS = {"BWG","AOD","AWP","JRI","RMM","BSTZ","TYG","NFJ","BKT","CEF","PSLV","PHYS","SIVR","SLV","GLD","GBTC","FBTC","URNM"}

def load_sec_fund_tickers():
    """Load SEC's official mutual-fund/ETF class ticker file once per run."""
    url = "https://www.sec.gov/files/company_tickers_mf.json"
    headers = {
        "User-Agent": "13f-backtest research contact@example.com",
        "Accept-Encoding": "gzip, deflate",
    }
    try:
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        obj = r.json()
    except Exception as exc:
        print("WARNING: SEC fund ticker list unavailable:", exc)
        return set()

    symbols = set()

    # Common SEC table shape: {"fields":[...], "data":[[...], ...]}
    if isinstance(obj, dict) and isinstance(obj.get("fields"), list) and isinstance(obj.get("data"), list):
        fields = [str(x).lower() for x in obj["fields"]]
        idxs = [i for i, name in enumerate(fields) if "ticker" in name or "symbol" in name]
        for row in obj["data"]:
            if isinstance(row, list):
                for i in idxs:
                    if i < len(row) and isinstance(row[i], str) and row[i].strip():
                        symbols.add(row[i].strip().upper())

    # Also support dict/object shapes.
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                kl = str(k).lower()
                if ("ticker" in kl or "symbol" in kl) and isinstance(v, str) and v.strip():
                    symbols.add(v.strip().upper())
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    print("SEC registered fund tickers loaded:", len(symbols))
    return symbols


_CEF_ABBREV = {
    "FD":"FUND", "FDS":"FUNDS", "TR":"TRUST", "DIVD":"DIVIDEND",
    "INM":"INCOME", "OPRT":"OPPORTUNITIES", "OPP":"OPPORTUNITY",
    "GLBAL":"GLOBAL", "GBL":"GLOBAL", "PPTYS":"PROPERTIES", "PPTY":"PROPERTY",
    "INFRA":"INFRASTRUCTURE", "MUN":"MUNICIPAL", "DUR":"DURATION",
    "SCI":"SCIENCE", "TECH":"TECHNOLOGY", "GR":"GROWTH",
}
_CEF_DROP = {"INC","INCORPORATED","CORP","CORPORATION","CO","COMPANY","LTD","LIMITED",
             "PLC","LLC","LP","NEW","COM","COMMON","SH","SHS","CLASS","CL","ORD","ORDINARY"}

def _cef_tokens(name):
    x = str(name or "").upper().replace("&", " AND ")
    raw = re.findall(r"[A-Z0-9]+", x)
    out = []
    for t in raw:
        t = _CEF_ABBREV.get(t, t)
        if t in {"AND", "THE"} or t in _CEF_DROP:
            continue
        out.append(t)
    return set(out)

def _is_cef_name(issuer, year):
    """Conservative token match against SEC's annual active closed-end-fund registrant names."""
    if not issuer or not SEC_CEF_NAMES_BY_YEAR:
        return False
    itoks = _cef_tokens(issuer)
    if len(itoks) < 2:
        return False
    names = SEC_CEF_NAMES_BY_YEAR.get(int(year), set())
    for ntoks in names:
        if not ntoks:
            continue
        inter = len(itoks & ntoks)
        smaller = min(len(itoks), len(ntoks))
        # Exact/subset matches, or strong abbreviation-tolerant matches with >=3 informative tokens.
        if itoks == ntoks or (smaller >= 3 and inter == smaller) or (smaller >= 4 and inter / smaller >= 0.75 and inter >= 3):
            return True
    return False

def load_sec_cef_names_by_year():
    """Load annual SEC Closed-End Fund Information registrant names.

    The SEC publishes yearly active CEF registrant files. 2021 is XML-only in the
    catalog, so use the union of 2020 and 2022 as a conservative 2021 bridge.
    """
    base = "https://www.sec.gov/files/investment/data/other/closed-end-fund-information/"
    urls = {
        2016: base + "closed-end_investment_company.csv",
        2017: base + "closed-end_investment_company_2017.csv",
        2018: base + "closed-end_investment_company_2018.csv",
        2019: base + "closed-end_investment_company_2019.csv",
        2020: base + "closed-end_investment_company_2020.csv",
        2022: base + "closed-end_investment_company_2022.csv",
        2023: base + "closed-end_investment_company_2023.csv",
        2024: base + "closed-end-investment-company-2024.csv",
        2025: base + "closed-end-investment-company-2025.csv",
        2026: base + "closed-end-investment-company-2026.csv",
    }
    headers = {
        "User-Agent": "13f-backtest research contact@example.com",
        "Accept-Encoding": "gzip, deflate",
    }
    ans = {}
    for year, url in urls.items():
        names = set()
        try:
            r = requests.get(url, headers=headers, timeout=60)
            r.raise_for_status()
            try:
                df = pd.read_csv(io.BytesIO(r.content))
            except Exception:
                df = pd.read_csv(io.BytesIO(r.content), encoding="latin1")
            name_cols = [c for c in df.columns if "name" in str(c).lower() and "registrant" in str(c).lower()]
            if not name_cols:
                name_cols = [c for c in df.columns if "name" in str(c).lower()]
            if name_cols:
                for v in df[name_cols[0]].dropna().astype(str):
                    toks = frozenset(_cef_tokens(v))
                    if len(toks) >= 2:
                        names.add(toks)
        except Exception as exc:
            print(f"WARNING: SEC CEF list unavailable for {year}:", exc)
        ans[year] = names
        print(f"SEC CEF names loaded {year}: {len(names)}")

    # 2021 has XML but no CSV in the current SEC catalog. The adjacent-year union
    # is conservative for exclusion and avoids treating a CEF as an operating stock.
    ans[2021] = set(ans.get(2020, set())) | set(ans.get(2022, set()))
    print("SEC CEF names loaded 2021 (2020/2022 union):", len(ans[2021]))
    return ans


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
            # Do not evaluate a quarter before its full statutory 13F cutoff has arrived.
            if q <= date.today() and (q + timedelta(days=45)) <= date.today():
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

def clean_equity(df, year=None):
    if df.empty: return df
    text = (df["issuer"].fillna("") + " " + df["class_name"].fillna(""))
    good = ~text.str.contains(EQUITY_BAD)
    good &= ~text.str.contains(
        r"(FUND|PORTFOLIO|BETABUILDERS|ISHARES|VANGUARD|SPDR|PROSHARES|ETF|INDEX|ETN|"
        r"KRANESHARES|GLOBAL X|SCHWAB|FIRST TRUST|DIREXION|VANECK|VAN ECK|ALPS|PIMCO|"
        r"INVESCO|WISDOMTREE|SELECT SECTOR|EXCHANGE TRADED|NASDAQ[- ]?100|S&P 500|MSCI|"
        r"RUSSELL|UNIT SER|ETF SHS|ETF TR|CLOSED[- ]END|CLOSED END)",
        case=False, regex=True
    )
    good &= df["ticker"].fillna("").str.match(TICKER_OK)
    tickers_upper = df["ticker"].fillna("").str.upper()
    if SEC_FUND_TICKERS:
        good &= ~tickers_upper.isin(SEC_FUND_TICKERS)
    good &= ~tickers_upper.isin(KNOWN_CEF_TICKERS)
    if year is not None and SEC_CEF_NAMES_BY_YEAR:
        cef_mask = df["issuer"].fillna("").map(lambda x: _is_cef_name(x, int(year)))
        good &= ~cef_mask
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
        return clean_equity(x, year=period.year)

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

    # Breadth signal: net increase in matched 13F holder count.
    m["holder_abs_change"] = m.cur_holders - m.prev_holders
    m["holder_pct_change"] = np.where(
        m.prev_holders > 0,
        m.holder_abs_change / m.prev_holders,
        np.nan
    )

    # Split-adjust prior aggregate share counts so aggregate institutional-share growth
    # and manager-by-manager buy/sell comparisons are not distorted by common splits.
    prev_px = np.where(m.prev_shares > 0, m.prev_value / m.prev_shares, np.nan)
    cur_px = np.where(m.cur_shares > 0, m.cur_value / m.cur_shares, np.nan)
    px_ratio = prev_px / cur_px
    sh_ratio = np.where(m.prev_shares > 0, m.cur_shares / m.prev_shares, np.nan)
    candidates = np.array([0.025,0.05,0.1,0.2,0.25,1/3,0.5,1.0,1.25,4/3,1.5,2,3,4,5,10,20,40], dtype=float)
    factors = np.ones(len(m))
    for i in range(len(m)):
        if not np.isfinite(px_ratio[i]) or not np.isfinite(sh_ratio[i]):
            continue
        errs = np.abs(np.log(px_ratio[i] / candidates)) + np.abs(np.log(sh_ratio[i] / candidates))
        f = candidates[np.argmin(errs)]
        if f != 1.0 and 0.72 <= px_ratio[i]/f <= 1.38 and 0.72 <= sh_ratio[i]/f <= 1.38:
            factors[i] = f
    m["split_factor"] = factors
    m["prev_shares_adj"] = m.prev_shares * m.split_factor

    # Depth signal proxy. The 13F source has position shares but not a clean
    # point-in-time shares-outstanding denominator for every historical security.
    # So this first-pass institutional-ownership measure is split-adjusted growth
    # in aggregate matched-manager 13F shares.
    m["institutional_ownership_increase"] = np.where(
        m.prev_shares_adj > 0,
        (m.cur_shares - m.prev_shares_adj) / m.prev_shares_adj,
        np.nan
    )

    # Conviction signal: among continuing matched managers, what fraction of active
    # changers increased their split-adjusted share position?
    split_tbl = m[["cusip","split_factor"]].dropna().drop_duplicates("cusip")
    db.register("split_tbl", split_tbl)
    conviction = db.execute("""
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
        WHERE f.rn=1
        GROUP BY f.cik,h.cusip
      ), ppos AS (
        SELECT f.cik,h.cusip,SUM(h.shares) shares
        FROM pf f JOIN holdings h USING(accession)
        WHERE f.rn=1
        GROUP BY f.cik,h.cusip
      )
      SELECT c.cusip,
             SUM(CASE WHEN c.shares > p.shares*s.split_factor*(1+?) THEN 1 ELSE 0 END) buyers,
             SUM(CASE WHEN c.shares < p.shares*s.split_factor*(1-?) THEN 1 ELSE 0 END) sellers,
             COUNT(*) continuing_holders
      FROM cpos c
      JOIN ppos p ON c.cik=p.cik AND c.cusip=p.cusip
      JOIN split_tbl s ON c.cusip=s.cusip
      WHERE c.shares>0 AND p.shares>0
      GROUP BY c.cusip
    """, [q, cutoff, pq, pcutoff, BUYER_EPS, BUYER_EPS]).df()
    if not conviction.empty:
        conviction["active_changers"] = conviction.buyers + conviction.sellers
        conviction["buyer_ratio"] = np.where(
            conviction.active_changers > 0,
            conviction.buyers / conviction.active_changers,
            np.nan
        )
        m = m.merge(conviction, on="cusip", how="left")
    else:
        m["buyers"] = np.nan
        m["sellers"] = np.nan
        m["continuing_holders"] = np.nan
        m["active_changers"] = np.nan
        m["buyer_ratio"] = np.nan

    for col in ["buyers","sellers","continuing_holders","active_changers"]:
        if col not in m:
            m[col] = np.nan

    # Equal-weight composite of three cross-sectional standardized signals.
    # Winsorization reduces domination by tiny-base outliers.
    def zscore(s):
        s = pd.to_numeric(s, errors="coerce")
        finite = s[np.isfinite(s)]
        if len(finite) < 5:
            return pd.Series(np.nan, index=s.index)
        lo, hi = finite.quantile([0.01, 0.99])
        w = s.clip(lo, hi)
        sd = w.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(0.0, index=s.index)
        return (w - w.mean()) / sd

    m["z_holder_abs"] = zscore(m.holder_abs_change)
    m["z_inst_ownership"] = zscore(m.institutional_ownership_increase)
    m["z_buyer_ratio"] = zscore(m.buyer_ratio.where(m.active_changers >= 5))
    m["composite_accumulation_score"] = (
        m[["z_holder_abs","z_inst_ownership","z_buyer_ratio"]].mean(axis=1, skipna=False)
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
    x = metrics.copy()
    if exclude_leaders:
        x = x[~x.is_leader]
    if x.empty:
        return x

    if method == "AbsoluteHolderIncrease":
        x = x[x.holder_abs_change > 0]
        x = x.sort_values(
            ["holder_abs_change", "institutional_ownership_increase", "buyer_ratio", "cur_holders"],
            ascending=[False, False, False, False]
        )
    elif method == "InstitutionalOwnershipIncrease":
        x = x[np.isfinite(pd.to_numeric(x["institutional_ownership_increase"], errors="coerce"))]
        x = x[x.institutional_ownership_increase > 0]
        x = x.sort_values(
            ["institutional_ownership_increase", "holder_abs_change", "buyer_ratio"],
            ascending=[False, False, False]
        )
    elif method == "BuyerRatioConviction":
        x = x[
            np.isfinite(pd.to_numeric(x["buyer_ratio"], errors="coerce"))
            & (x.active_changers >= 5)
            & (x.buyer_ratio > 0.5)
        ]
        x = x.sort_values(
            ["buyer_ratio", "active_changers", "holder_abs_change"],
            ascending=[False, False, False]
        )
    elif method == "CompositeAccumulationScore":
        x = x[np.isfinite(pd.to_numeric(x["composite_accumulation_score"], errors="coerce"))]
        x = x.sort_values(
            ["composite_accumulation_score", "holder_abs_change", "institutional_ownership_increase"],
            ascending=[False, False, False]
        )
    else:
        raise ValueError(method)

    return x.drop_duplicates(subset=["ticker"]).head(CANDIDATE_DEPTH).copy()

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
    # Preserve caller priority rather than alphabetically sorting thousands of symbols.
    # This lets the highest-ranked candidates get priced first if Yahoo throttles later.
    ordered = []
    seen = set()
    for t in ["SPY"] + list(tickers):
        if t and t not in seen:
            ordered.append(t)
            seen.add(t)

    start = f"{START_YEAR}-01-01"
    end = (date.today() + timedelta(days=5)).isoformat()
    all_frames = {}

    def ingest(raw, chunk):
        if raw is None or raw.empty:
            return
        if len(chunk) == 1:
            t = chunk[0]
            try:
                s = raw["Close"].dropna() if "Close" in raw else pd.Series(dtype=float)
                if not s.empty:
                    all_frames[t] = s
            except Exception:
                pass
        else:
            for t in chunk:
                try:
                    s = raw[t]["Close"].dropna()
                    if not s.empty:
                        all_frames[t] = s
                except Exception:
                    pass

    # SPY is required by every row. Fetch it first and fail loudly rather than
    # silently producing empty portfolio files when Yahoo is rate-limited.
    for attempt in range(5):
        try:
            raw = yf.download(["SPY"], start=start, end=end, auto_adjust=True,
                              progress=False, group_by="ticker", threads=False)
            ingest(raw, ["SPY"])
            if "SPY" in all_frames:
                break
        except Exception:
            pass
        time.sleep(5 * (attempt + 1))
    if "SPY" not in all_frames:
        raise RuntimeError("SPY price download failed; refusing to write empty backtest outputs")

    # Download signal candidates in priority order, with no parallel hammering of Yahoo.
    # Smaller sequential chunks trade a little speed for much lower 429 risk.
    rest = [t for t in ordered if t != "SPY"]
    for k in range(0, len(rest), 20):
        chunk = rest[k:k+20]
        raw = pd.DataFrame()
        for attempt in range(3):
            try:
                raw = yf.download(chunk, start=start, end=end, auto_adjust=True,
                                  progress=False, group_by="ticker", threads=False)
                if raw is not None and not raw.empty:
                    break
            except Exception:
                raw = pd.DataFrame()
            time.sleep(4 * (attempt + 1))
        ingest(raw, chunk)
        time.sleep(0.75)
    return all_frames

def add_returns(rows):
    # Price the highest signal ranks first across every quarter/method/universe.
    # This is important because only the first five tradable names are needed.
    ranked = sorted(
        [r for r in rows if r.get("ticker")],
        key=lambda r: (int(r.get("signal_rank", 999999)), r.get("quarter", ""), r.get("method", ""), r.get("universe", ""))
    )
    tickers = [r["ticker"] for r in ranked]
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

def select_top5_from_candidates(candidates):
    """Select the first five signal-ranked names with usable historical price data.

    This is a data-availability fallback, not an additional alpha signal. For quarters
    whose 90-day window has fully elapsed, a usable return is required; for the still-
    open latest quarter, only an entry price is required.
    """
    if candidates.empty:
        return pd.DataFrame()

    selected = []
    keys = ["quarter", "period_end", "universe", "method"]
    for key, grp in candidates.groupby(keys, sort=True):
        grp = grp.sort_values("signal_rank").copy()
        q = datetime.strptime(key[1], "%Y-%m-%d").date()
        target_entry = cutoff_for(q) + timedelta(days=1)
        mature = (target_entry + timedelta(days=90)) <= date.today()

        eligible = grp[grp.entry_date.notna()].copy()
        if mature:
            eligible = eligible[eligible.return_90d.notna()].copy()

        picks = eligible.head(PORTFOLIO_SIZE).copy()
        if picks.empty:
            continue
        picks["portfolio_rank"] = np.arange(1, len(picks) + 1)
        picks["weight"] = 1.0 / PORTFOLIO_SIZE
        selected.append(picks)

    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


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
    global SEC_FUND_TICKERS, SEC_CEF_NAMES_BY_YEAR
    SEC_FUND_TICKERS = load_sec_fund_tickers()
    SEC_CEF_NAMES_BY_YEAR = load_sec_cef_names_by_year()
    snap = load_snapshot()
    print("snapshot generated:", snap.get("generatedAt"))
    paths = cache_dataset_files(snap)
    db = build_db(paths)

    candidate_rows = []
    quality = []
    methods = [
        "AbsoluteHolderIncrease",
        "InstitutionalOwnershipIncrease",
        "BuyerRatioConviction",
        "CompositeAccumulationScore",
    ]
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
                candidates = rank_candidates(m, method, exclude_leaders=exclude_leaders)
                for signal_rank, (_, row) in enumerate(candidates.iterrows(), 1):
                    candidate_rows.append({
                        "quarter": qlabel(q),
                        "period_end": q.isoformat(),
                        "universe": universe,
                        "method": method,
                        "signal_rank": signal_rank,
                        "ticker": row.ticker,
                        "cusip": row.cusip,
                        "issuer": row.issuer,
                        "is_leader": bool(row.is_leader),
                        "leader_rank": float(row.leader_rank) if pd.notna(row.leader_rank) else np.nan,
                        "current_holders": float(row.cur_holders),
                        "prior_holders": float(row.prev_holders),
                        "holder_abs_change": float(row.holder_abs_change),
                        "holder_pct_change": float(row.holder_pct_change) if pd.notna(row.holder_pct_change) else np.nan,
                        "institutional_ownership_increase": float(row.institutional_ownership_increase) if pd.notna(row.institutional_ownership_increase) else np.nan,
                        "split_factor": float(row.split_factor),
                        "buyers": float(row.buyers) if pd.notna(row.buyers) else np.nan,
                        "sellers": float(row.sellers) if pd.notna(row.sellers) else np.nan,
                        "active_changers": float(row.active_changers) if pd.notna(row.active_changers) else np.nan,
                        "buyer_ratio": float(row.buyer_ratio) if pd.notna(row.buyer_ratio) else np.nan,
                        "composite_accumulation_score": float(row.composite_accumulation_score) if pd.notna(row.composite_accumulation_score) else np.nan,
                        "current_value_usd": float(row.cur_value),
                        "current_shares": float(row.cur_shares),
                        "prior_shares_adjusted": float(row.prev_shares_adj),
                        "matched_managers": meta["matched_managers"],
                    })

    add_returns(candidate_rows)
    candidate_df = pd.DataFrame(candidate_rows)
    candidate_df.to_csv(RESULTS / "accumulation_candidate_pool.csv", index=False)

    constituents = select_top5_from_candidates(candidate_df)
    if constituents.empty:
        raise RuntimeError("No priced constituents were selected; refusing to commit empty backtest outputs")
    constituents.to_csv(RESULTS / "accumulation_constituents.csv", index=False)

    # Mature quarter/method/universe groups should have five priced names. Fail loudly
    # if widespread pricing failure would otherwise masquerade as a successful workflow.
    mature_cutoff = date.today() - timedelta(days=90)
    expected = candidate_df[
        pd.to_datetime(candidate_df["period_end"]).dt.date.map(lambda q: cutoff_for(q) + timedelta(days=1) <= mature_cutoff)
    ][["quarter","period_end","universe","method"]].drop_duplicates()
    actual = constituents.groupby(["quarter","period_end","universe","method"]).size().reset_index(name="n")
    check = expected.merge(actual, on=["quarter","period_end","universe","method"], how="left")
    incomplete = check[check["n"].fillna(0) < PORTFOLIO_SIZE]
    if len(incomplete) > 0:
        print("WARNING: incomplete mature groups:", len(incomplete))
        print(incomplete.to_string(index=False))
    if len(incomplete) > max(2, int(0.05 * max(1, len(expected)))):
        raise RuntimeError(f"Too many incomplete mature portfolio groups ({len(incomplete)}/{len(expected)}); likely price-source throttling")

    portfolios = build_quarterly_portfolio(constituents)
    portfolios.to_csv(RESULTS / "accumulation_quarterly_portfolios.csv", index=False)

    summ = portfolio_summary(portfolios)
    summ.to_csv(RESULTS / "accumulation_method_summary.csv", index=False)
    pd.DataFrame(quality).to_csv(RESULTS / "data_quality.csv", index=False)

    meta = {
        "dataset": DATASET,
        "dataset_snapshot_generated_at": snap.get("generatedAt"),
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "portfolio_size": PORTFOLIO_SIZE,
        "candidate_depth": CANDIDATE_DEPTH,
        "weighting": "equal_weight_20pct_each_when_top5_complete",
        "methods": [
            "AbsoluteHolderIncrease",
            "InstitutionalOwnershipIncrease",
            "BuyerRatioConviction",
            "CompositeAccumulationScore"
        ],
        "absolute_holder_increase_definition": "current matched 13F holder count minus prior-quarter matched 13F holder count",
        "institutional_ownership_increase_definition": "split-adjusted percentage growth in aggregate shares held by matched 13F managers; proxy used because the source does not contain point-in-time issuer shares outstanding for every security",
        "buyer_ratio_definition": f"among continuing matched managers that changed their split-adjusted position by more than {BUYER_EPS:.1%}, buyers / (buyers + sellers); require at least 5 active changers",
        "composite_definition": "equal-weight mean of cross-sectional z-scores for absolute holder increase, institutional ownership increase proxy, and buyer ratio; each input winsorized at 1st/99th percentile before z-scoring",
        "leader_definition": f"top {LEADER_TOP_N} current-quarter common equities by aggregate 13F market value, point-in-time at filing cutoff",
        "universes": ["AllStocks", "OpportunisticExLeaders"],
        "entry_rule": "first trading day after the 45-day 13F filing cutoff",
        "holding_period": "90 calendar days, exit first trading day on/after day 90",
        "fund_filter": "SEC company_tickers_mf.json plus issuer/class regex filters plus annual SEC Closed-End Fund Information registrant-name matching; applied before leader ranking and signals",
        "historical_price_fallback": "rank up to candidate_depth by signal, then take first five with usable historical Yahoo price data; no return-based ranking",
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "note": "2026Q2 may not yet have a complete 90-calendar-day forward return."
    }
    (RESULTS / "accumulation_run_metadata.json").write_text(json.dumps(meta, indent=2))

    print("\nTOP5 PORTFOLIOS")
    print(portfolios.to_string(index=False))
    print("\nSUMMARY")
    print(summ.to_string(index=False))

if __name__ == "__main__":
    main()
