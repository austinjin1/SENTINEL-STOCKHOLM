#!/usr/bin/env python3
"""Independent re-verification of the FIB nowcast: does skill survive dropping
GEOGRAPHY (lat/lon/ghcn_dist) and same-day rain? Isolates pure antecedent
hydromet + season signal. Reuses the exact eval functions."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_fib_nowcast import temporal_holdout, site_grouped_holdout, DATA, PREDICTORS

df = pd.read_parquet(DATA); df["date"] = pd.to_datetime(df["date"])
for c in PREDICTORS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
rain = [c for c in PREDICTORS if c.startswith("rain")]
df[rain] = df[rain].fillna(0.0)
df["ghcn_dist_km"] = df["ghcn_dist_km"].fillna(df["ghcn_dist_km"].median())

VARIANTS = {
    "full(as-reported)": PREDICTORS,
    "no_geo(keep sameday)": ["rain_prior1_mm","rain_prior2_mm","rain_prior3_mm","rain_prior7_mm","rain_sameday_mm","doy_sin","doy_cos","month"],
    "no_geo_strict(no sameday)": ["rain_prior1_mm","rain_prior2_mm","rain_prior3_mm","rain_prior7_mm","doy_sin","doy_cos","month"],
    "rain_only_strict": ["rain_prior1_mm","rain_prior2_mm","rain_prior3_mm","rain_prior7_mm"],
}

out = {}
for name, preds in VARIANTS.items():
    print(f"\n########## {name}  ({len(preds)} predictors) ##########", flush=True)
    t = temporal_holdout(df, predictors=preds)
    s = site_grouped_holdout(df, predictors=preds)
    tm = t["model_hydromet"]["auroc"]
    tsub = t.get("model_hydromet_on_persistence_subset", {}).get("auroc")
    pers = t.get("baseline_persistence", {}).get("auroc")
    sm = s["model_hydromet_pooledOOF"]["auroc"]
    out[name] = {"temporal_auroc": tm, "temporal_ci": t["model_hydromet"]["auroc_ci95"],
                 "model_on_persist_subset": tsub, "persistence_auroc": pers,
                 "site_grouped_auroc": sm, "site_ci": s["model_hydromet_pooledOOF"]["auroc_ci95"],
                 "beats_persistence": (tsub > pers) if (tsub and pers) else None}
    print(f"  temporal={tm} (vs persistence {pers}, model-on-subset {tsub})  site-grouped={sm}", flush=True)

Path(__file__).resolve().parents[2].joinpath("results/benchmarks/disease_fib_nowcast_nogeo_verify.json").write_text(json.dumps(out, indent=2))
print("\n===== SUMMARY =====")
for k, v in out.items():
    print(f"{k:28s} temporal={v['temporal_auroc']}  site={v['site_grouped_auroc']}  beats_persist={v['beats_persistence']}")
