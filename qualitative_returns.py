#!/usr/bin/env python3
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
import io, time, urllib.parse
import numpy as np
import pandas as pd
import requests

from pandas.tseries.holiday import USFederalHolidayCalendar

R = Path("results")
PICKS = R / "qualitative_picks_final.csv"
HORIZONS = [90, 180, 365]
UA = {"User-Agent": "Mozilla/5.0 qualitative-13f-backtest"}

_HOLIDAY_CAL = USFederalHolidayCalendar()
_HOLIDAYS = set(pd.Timestamp(x).date() for x in _HOLIDAY_CAL.holidays(start="2015-01-01", end="2028-12-31"))

def cutoff_for(q):
    d = q + timedelta(days=45)
    while d.weekday() >= 5 or d in _HOLIDAYS:
        d += timedelta(days=1)
    return d

def quarter_end(q):
    y = int(q[:4]); n = int(q[-1])
    return {1:date(y,3,31),2:date(y,6,30),3:date(y,9,30),4:date(y,12,31)}[n]

def yahoo_symbol(t):
    return str(t).replace(".","-")

def stooq_symbol(t):
    return str(t).lower().replace(".","-") + ".us"

def fetch_stooq(t,start,end):
    url = f"https://stooq.com/q/d/l/?s={urllib.parse.quote(stooq_symbol(t))}&d1={start:%Y%m%d}&d2={end:%Y%m%d}&i=d"
    try:
        r=requests.get(url,headers=UA,timeout=25)
        if r.status_code != 200 or "No data" in r.text or len(r.text) < 30:
            return pd.Series(dtype=float), None
        df=pd.read_csv(io.StringIO(r.text))
        if "Date" not in df or "Close" not in df:
            return pd.Series(dtype=float), None
        df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
        df["Close"]=pd.to_numeric(df["Close"],errors="coerce")
        s=df.dropna(subset=["Date","Close"]).set_index("Date")["Close"].sort_index()
        return s, "stooq"
    except Exception:
        return pd.Series(dtype=float), None

def fetch_yahoo(t,start,end):
    sym=yahoo_symbol(t)
    p1=int(datetime.combine(start,datetime.min.time(),tzinfo=timezone.utc).timestamp())
    p2=int(datetime.combine(end+timedelta(days=2),datetime.min.time(),tzinfo=timezone.utc).timestamp())
    url=f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(sym)}?period1={p1}&period2={p2}&interval=1d&events=history&includeAdjustedClose=true"
    try:
        r=requests.get(url,headers=UA,timeout=25)
        if r.status_code != 200:
            return pd.Series(dtype=float), None
        j=r.json(); res=(j.get("chart",{}).get("result") or [None])[0]
        if not res:
            return pd.Series(dtype=float), None
        ts=res.get("timestamp") or []
        adj=((res.get("indicators",{}).get("adjclose") or [{}])[0].get("adjclose") or [])
        close=((res.get("indicators",{}).get("quote") or [{}])[0].get("close") or [])
        px=adj if len(adj)==len(ts) else close
        rows=[(pd.to_datetime(x,unit="s",utc=True).tz_convert(None),p) for x,p in zip(ts,px) if p is not None]
        if not rows:
            return pd.Series(dtype=float), None
        return pd.Series({d:float(p) for d,p in rows}).sort_index(), "yahoo_chart"
    except Exception:
        return pd.Series(dtype=float), None

def fetch_prices(t,start,end):
    s,src=fetch_stooq(t,start,end)
    if len(s)>=2:
        return s,src
    time.sleep(0.15)
    return fetch_yahoo(t,start,end)

def on_or_after(s,target,max_gap=10):
    if s.empty: return None
    p=s.index.searchsorted(pd.Timestamp(target))
    if p >= len(s.index): return None
    d=s.index[p]
    if (d-pd.Timestamp(target)).days > max_gap: return None
    return d

def calc(t,q):
    qe=quarter_end(q)
    target_entry=cutoff_for(qe)+timedelta(days=1)
    end=target_entry+timedelta(days=max(HORIZONS)+20)
    s,src=fetch_prices(t,target_entry-timedelta(days=5),end)
    out={"quarter":q,"ticker":t,"source":src,"target_entry_date":target_entry.isoformat()}
    if s.empty:
        out["entry_date"]=None
        for h in HORIZONS:
            out[f"exit_{h}d"]=None; out[f"return_{h}d"]=np.nan
        return out
    ed=on_or_after(s,target_entry)
    out["entry_date"]=ed.date().isoformat() if ed is not None else None
    if ed is None:
        for h in HORIZONS:
            out[f"exit_{h}d"]=None; out[f"return_{h}d"]=np.nan
        return out
    for h in HORIZONS:
        xd=on_or_after(s,ed.date()+timedelta(days=h))
        out[f"exit_{h}d"]=xd.date().isoformat() if xd is not None else None
        out[f"return_{h}d"]=float(s.loc[xd]/s.loc[ed]-1) if xd is not None else np.nan
    return out

def main():
    picks=pd.read_csv(PICKS)
    invest=picks[picks["decision"].eq("INVEST")].copy()
    expanded=[]
    for _,r in invest.iterrows():
        tickers=[x.strip() for x in str(r.get("selected_tickers","")).split("|") if x.strip()]
        for rank,t in enumerate(tickers,1):
            expanded.append({
                "quarter":str(r["quarter"]),
                "theme":r["theme"],
                "confidence":r["confidence"],
                "portfolio_rank":rank,
                "ticker":t,
            })
    e=pd.DataFrame(expanded)
    if e.empty:
        raise RuntimeError("No INVEST selections found")

    pairs=set(zip(e.quarter,e.ticker))
    pairs |= {(q,"SPY") for q in e.quarter.unique()}
    rows=[]
    for q,t in sorted(pairs):
        print("price",q,t,flush=True)
        rows.append(calc(t,q))
        time.sleep(0.15)
    ret=pd.DataFrame(rows)
    ret.to_csv(R/"qualitative_forward_returns.csv",index=False)

    stocks=ret[~ret.ticker.eq("SPY")].copy()
    spy=ret[ret.ticker.eq("SPY")][["quarter"]+[f"return_{h}d" for h in HORIZONS]].rename(
        columns={f"return_{h}d":f"spy_{h}d" for h in HORIZONS}
    )
    c=e.merge(stocks,on=["quarter","ticker"],how="left").merge(spy,on="quarter",how="left")
    for h in HORIZONS:
        c[f"excess_{h}d"]=c[f"return_{h}d"]-c[f"spy_{h}d"]
    c.to_csv(R/"qualitative_constituents.csv",index=False)

    qrows=[]
    for (q,theme,conf),g in c.groupby(["quarter","theme","confidence"],sort=True):
        d={"quarter":q,"theme":theme,"confidence":conf,"n_selected":len(g),
           "tickers":"|".join(g.sort_values("portfolio_rank").ticker.astype(str))}
        for h in HORIZONS:
            z=g[f"return_{h}d"].dropna()
            d[f"n_with_{h}d"]=len(z)
            d[f"coverage_{h}d"]=float(len(z)/len(g)) if len(g) else np.nan
            d[f"portfolio_{h}d"]=float(z.mean()) if len(z)==len(g) and len(g)>0 else np.nan
            d[f"partial_portfolio_{h}d"]=float(z.mean()) if len(z)>0 else np.nan
            sp=g[f"spy_{h}d"].dropna()
            d[f"spy_{h}d"]=float(sp.iloc[0]) if len(sp) else np.nan
            d[f"excess_{h}d"]=d[f"portfolio_{h}d"]-d[f"spy_{h}d"] if pd.notna(d[f"portfolio_{h}d"]) and pd.notna(d[f"spy_{h}d"]) else np.nan
        qrows.append(d)
    qdf=pd.DataFrame(qrows)
    qdf.to_csv(R/"qualitative_quarterly.csv",index=False)

    srows=[]
    for h in HORIZONS:
        z=qdf.dropna(subset=[f"portfolio_{h}d"])
        srows.append({
            "horizon_days":h,
            "complete_invest_quarters":len(z),
            "avg_return":float(z[f"portfolio_{h}d"].mean()) if len(z) else np.nan,
            "median_return":float(z[f"portfolio_{h}d"].median()) if len(z) else np.nan,
            "positive_rate":float((z[f"portfolio_{h}d"]>0).mean()) if len(z) else np.nan,
            "avg_spy_return":float(z[f"spy_{h}d"].mean()) if len(z) else np.nan,
            "avg_excess":float(z[f"excess_{h}d"].mean()) if len(z) else np.nan,
            "spy_outperformance_rate":float((z[f"excess_{h}d"]>0).mean()) if len(z) else np.nan,
        })
    pd.DataFrame(srows).to_csv(R/"qualitative_summary.csv",index=False)

if __name__=="__main__":
    main()
