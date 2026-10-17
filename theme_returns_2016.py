#!/usr/bin/env python3
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
import io, math, time, urllib.parse
import requests
import pandas as pd
import numpy as np

R=Path("results")
HORIZONS=[90,180,365]
UA={"User-Agent":"Mozilla/5.0 13f-theme-research"}

def quarter_end(q):
    y=int(q[:4]); n=int(q[-1])
    return {1:date(y,3,31),2:date(y,6,30),3:date(y,9,30),4:date(y,12,31)}[n]

def stooq_symbol(t):
    return t.lower().replace(".","-")+".us"

def fetch_stooq(t,start,end):
    sym=stooq_symbol(t)
    url=f"https://stooq.com/q/d/l/?s={urllib.parse.quote(sym)}&d1={start:%Y%m%d}&d2={end:%Y%m%d}&i=d"
    try:
        r=requests.get(url,headers=UA,timeout=30)
        if r.status_code!=200 or "No data" in r.text or len(r.text)<30:
            return pd.Series(dtype=float),None
        df=pd.read_csv(io.StringIO(r.text))
        if "Date" not in df or "Close" not in df:
            return pd.Series(dtype=float),None
        df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
        df["Close"]=pd.to_numeric(df["Close"],errors="coerce")
        s=df.dropna(subset=["Date","Close"]).set_index("Date")["Close"].sort_index()
        return s,"stooq"
    except Exception:
        return pd.Series(dtype=float),None

def fetch_yahoo(t,start,end):
    # Direct chart endpoint, one ticker at a time to avoid yfinance bulk rate limits.
    sym=t.replace(".","-")
    p1=int(datetime.combine(start,datetime.min.time(),tzinfo=timezone.utc).timestamp())
    p2=int(datetime.combine(end+timedelta(days=2),datetime.min.time(),tzinfo=timezone.utc).timestamp())
    url=f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(sym)}?period1={p1}&period2={p2}&interval=1d&events=history&includeAdjustedClose=true"
    try:
        r=requests.get(url,headers=UA,timeout=30)
        if r.status_code!=200: return pd.Series(dtype=float),None
        j=r.json(); res=(j.get("chart",{}).get("result") or [None])[0]
        if not res: return pd.Series(dtype=float),None
        ts=res.get("timestamp") or []
        adj=((res.get("indicators",{}).get("adjclose") or [{}])[0].get("adjclose") or [])
        close=((res.get("indicators",{}).get("quote") or [{}])[0].get("close") or [])
        px=adj if len(adj)==len(ts) else close
        rows=[(pd.to_datetime(x,unit="s",utc=True).tz_convert(None),p) for x,p in zip(ts,px) if p is not None]
        if not rows: return pd.Series(dtype=float),None
        return pd.Series({d:float(p) for d,p in rows}).sort_index(),"yahoo_chart"
    except Exception:
        return pd.Series(dtype=float),None

def fetch_price(t,start,end):
    s,src=fetch_stooq(t,start,end)
    if len(s)>=2: return s,src
    time.sleep(.2)
    return fetch_yahoo(t,start,end)

def on_or_after(s,target,maxgap=10):
    if s.empty:return None
    idx=s.index
    p=idx.searchsorted(pd.Timestamp(target))
    if p>=len(idx):return None
    d=idx[p]
    if (d-pd.Timestamp(target)).days>maxgap:return None
    return d

def calc(t,q):
    qe=quarter_end(q)
    target_entry=qe+timedelta(days=46)
    end=target_entry+timedelta(days=max(HORIZONS)+20)
    s,src=fetch_price(t,target_entry-timedelta(days=5),end)
    if s.empty:
        return {"quarter":q,"ticker":t,"source":None,"entry_date":None,**{f"return_{h}d":np.nan for h in HORIZONS}}
    ed=on_or_after(s,target_entry)
    if ed is None:
        return {"quarter":q,"ticker":t,"source":src,"entry_date":None,**{f"return_{h}d":np.nan for h in HORIZONS}}
    out={"quarter":q,"ticker":t,"source":src,"entry_date":ed.date().isoformat()}
    for h in HORIZONS:
        xd=on_or_after(s,ed.date()+timedelta(days=h))
        out[f"exit_{h}d"]=xd.date().isoformat() if xd is not None else None
        out[f"return_{h}d"]=float(s.loc[xd]/s.loc[ed]-1) if xd is not None else np.nan
    return out

def main():
    quant=pd.read_csv(R/"quant_theme_constituents.csv")
    quant=quant[quant.quarter.str.startswith("2016")].copy()
    qual=pd.read_csv(R/"qualitative_picks_2016.csv")
    qual_rows=[]
    for _,r in qual.iterrows():
        if r["decision"]=="INVEST" and isinstance(r.get("selected_tickers"),str):
            for t in r["selected_tickers"].split("|"):
                if t: qual_rows.append((r["quarter"],t))
    pairs=set(zip(quant.quarter.astype(str),quant.ticker.astype(str)))|set(qual_rows)
    pairs|={(q,"SPY") for q,_ in pairs}
    rows=[]
    for q,t in sorted(pairs):
        print("price",q,t,flush=True)
        rows.append(calc(t,q))
        time.sleep(.25)
    ret=pd.DataFrame(rows)
    ret.to_csv(R/"theme_returns_2016_stocks.csv",index=False)

    def attach_and_summarize(name,picks):
        m=picks.merge(ret,on=["quarter","ticker"],how="left")
        spy=ret[ret.ticker.eq("SPY")][["quarter"]+[f"return_{h}d" for h in HORIZONS]].rename(
            columns={f"return_{h}d":f"spy_{h}d" for h in HORIZONS})
        m=m.merge(spy,on="quarter",how="left")
        for h in HORIZONS:
            m[f"excess_{h}d"]=m[f"return_{h}d"]-m[f"spy_{h}d"]
        m.to_csv(R/f"{name}_constituents_2016.csv",index=False)
        qrows=[]
        for q,g in m.groupby("quarter"):
            d={"quarter":q,"n":len(g),"tickers":"|".join(g.ticker.astype(str))}
            for h in HORIZONS:
                z=g[f"return_{h}d"].dropna()
                d[f"n_{h}d"]=len(z)
                d[f"portfolio_{h}d"]=float(z.mean()) if len(z)==len(g) and len(g)>0 else np.nan
                sp=g[f"spy_{h}d"].dropna()
                d[f"spy_{h}d"]=float(sp.iloc[0]) if len(sp) else np.nan
                d[f"excess_{h}d"]=d[f"portfolio_{h}d"]-d[f"spy_{h}d"] if pd.notna(d[f"portfolio_{h}d"]) and pd.notna(d[f"spy_{h}d"]) else np.nan
            qrows.append(d)
        pd.DataFrame(qrows).to_csv(R/f"{name}_quarterly_2016.csv",index=False)

    attach_and_summarize("quant_theme",quant[["quarter","ticker"]].drop_duplicates())
    qp=[]
    for q,t in qual_rows: qp.append({"quarter":q,"ticker":t})
    attach_and_summarize("qual_theme",pd.DataFrame(qp))

if __name__=="__main__":
    main()
