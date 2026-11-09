# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""W-P1 power check (protocol §12): target-availability counts only (no outcome statistics)."""
import json, os

import pandas as pd
from sentinel.core import paths
SPL = paths.REPO_ROOT / "docs" / "preregistration" / "sonde_split.json"
D = paths.processed("sonde_daily") / "sites"
s = json.load(open(SPL))
T = {"do": "do_mgl_mean", "turb": "turb_fnu_mean", "no3": "no3no2_mgl_mean", "chl": "chl_ugl_mean", "temp": "temp_c_mean", "sc": "spcond_uscm_mean"}
dev_end, tst0, tst1 = pd.Timestamp("2023-09-30"), pd.Timestamp("2023-11-12"), pd.Timestamp("2025-09-30")
rows = []
for site, role in s["site_role"].items():
    f = f"{D}/{site}.parquet"
    if not os.path.exists(f):
        continue
    import pyarrow.parquet as pq
    cols = set(pq.read_schema(f).names)
    use = [c for c in T.values() if c in cols]
    if not use:
        continue
    x = pd.read_parquet(f, columns=use)
    x.index = pd.to_datetime(x.index) if not isinstance(x.index, pd.DatetimeIndex) else x.index
    for k, c in T.items():
        if c not in x:
            continue
        v = x[c].notna()
        dev = int((v & (x.index >= "2008-10-01") & (x.index <= dev_end)).sum())
        tst = int((v & (x.index >= tst0) & (x.index <= tst1)).sum())
        rows.append((site, role, s["basin_group"].get(site), k, dev, tst))
r = pd.DataFrame(rows, columns=["site", "role", "bg", "var", "days_dev_window", "days_test_window"])
out = {}
for k in T:
    a = r[r["var"] == k]
    dv = a[a.role.isin(["train", "val"]) & (a.days_dev_window >= 365)]
    ts = a[(a.role == "test") & (a.days_test_window >= 180)]
    out[k] = {"dev_sites_ge1y": int(len(dv)), "dev_days": int(dv.days_dev_window.sum()), "dev_basin_groups": int(dv.bg.nunique()),
              "test_sites_ge180d": int(len(ts)), "test_days": int(ts.days_test_window.sum()), "test_basin_groups": int(ts.bg.nunique())}
print(json.dumps(out, indent=1))
json.dump(out, open(paths.REPO_ROOT / "results" / "fuse" / "wp1_power_check.json", "w"), indent=1)
