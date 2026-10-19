#!/usr/bin/env python3
from datetime import date
from pathlib import Path
import backtest as bt
from new_holder_top30_q2 import compute_new_holders

OUT=Path("results"); OUT.mkdir(exist_ok=True)

def main():
    bt.START_YEAR=2016
    bt.END_YEAR=2026
    bt.SEC_FUND_TICKERS=bt.load_sec_fund_tickers()
    bt.SEC_FUND_TICKERS_BY_YEAR=bt.load_sec_annual_fund_tickers()
    bt.SEC_CEF_NAMES_BY_YEAR=bt.load_sec_cef_names_by_year()
    snap=bt.load_snapshot()
    paths=bt.cache_dataset_files(snap)
    db=bt.build_db(paths)
    q=date(2026,6,30)
    top,n_mgr=compute_new_holders(db,q)
    top.to_csv(OUT/"new_holder_top30_2026Q2_only.csv",index=False)
    print("matched_managers",n_mgr)
    print(top[["rank","ticker","issuer","new_holders","current_holders","prior_holders","new_holder_share_of_current"]].to_string(index=False))

if __name__=="__main__":
    main()
