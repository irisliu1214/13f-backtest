#!/usr/bin/env python3
import os
import pandas as pd
s=pd.read_csv("results/b4v2_summary.csv")
q=pd.read_csv("results/b4v2_quarterly.csv")
c=pd.read_csv("results/b4v2_constituents.csv")
assert set(s.horizon_days)=={90,180,365}
assert len(q)>0 and len(c)>0
print("FREEZE_SHA_FROM_ENV",os.environ.get("FREEZE_SHA"))
print(s.to_string(index=False))
print(q.to_string(index=False))
