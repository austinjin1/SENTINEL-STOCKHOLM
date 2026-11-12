#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Validate the IV-derived daily rule of ``download_newsites.daily_from_iv`` against USGS DV (ew_v3 §2 / W-N5).

Population: TRAIN-ROLE sites of ``sonde_split.json`` that have raw legacy IV (``raw/usgs_iv/<site>/``) and a DV daily
tier; TRAIN BLOCK only (WY2008–2021). No new-site data, no val/test data. Variables: DO daily min/mean, SC mean.
Pre-specified acceptance (ew_v3_protocol.md §2): ≥ 99 % of common days within 0.1 mg/L for DO min; otherwise the
IV-only new sites form a separate secondary cell. H1 onset concordance (frozen ``hazards.daily_onsets``) reported.
Writes ``sentinel_runs/sonde/iv_daily_rule_validation/result.json``.

    python -B scripts/data/validate_iv_daily_rule.py --max-sites 60
"""

from __future__ import annotations

import argparse
import importlib.util
import json

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.data.sonde_hourly import clean_iv
from sentinel.eval.hazards import HAZARDS, daily_onsets

TRAIN_END = pd.Timestamp("2021-09-30")


def _dl():
    spec = importlib.util.spec_from_file_location("_dn", paths.REPO_ROOT / "scripts/data/download_newsites.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sites", type=int, default=60)
    a = ap.parse_args()
    dn = _dl()
    split = json.loads((paths.REPO_ROOT / "docs/preregistration/sonde_split.json").read_text())["site_role"]
    reg = pd.read_parquet(paths.PROCESSED / "sites" / "sites.parquet")
    tz = reg[reg["network"] == "USGS"].drop_duplicates("site_id").set_index("site_id")["tz_cd"] \
        if "tz_cd" in reg.columns else pd.Series(dtype=str)
    raw = paths.RAW / "usgs_iv"
    sites = sorted(s for s in split if split[s] == "train" and (raw / s).is_dir())[: a.max_sites]
    snap = (paths.PROCESSED / "sonde_daily").resolve() / "sites"
    rows, on = [], {"iv": 0, "dv": 0, "both": 0}
    for s in sites:
        files = sorted((raw / s).glob("*.parquet"))
        files = [f for f in files if int(f.stem) <= 2021]
        if not files or not (snap / f"{s}.parquet").exists():
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        d_iv = dn.daily_from_iv(clean_iv(df), tz.get(s))
        dv = pd.read_parquet(snap / f"{s}.parquet")
        dv.index = pd.to_datetime(dv.index)
        for v in ("do_mgl_min", "do_mgl_mean", "spcond_uscm_mean"):
            if v not in d_iv or v not in dv:
                continue
            a_ = d_iv[v][d_iv.index <= TRAIN_END]
            b_ = dv[v][dv.index <= TRAIN_END]
            j = pd.concat([a_.rename("iv"), b_.rename("dv")], axis=1).dropna()
            if len(j) == 0:
                continue
            tol = 0.1 if v.startswith("do") else max(1.0, 0.01 * float(j["dv"].median()))
            rows.append({"site": s, "var": v, "n": len(j), "frac_within_tol": float((np.abs(j.iv - j.dv) <= tol).mean()),
                         "only_iv": int(a_.notna().sum() - len(j)), "only_dv": int(b_.notna().sum() - len(j))})
            if v == "do_mgl_min":
                idx = pd.date_range(min(a_.index.min(), b_.index.min()), TRAIN_END)
                _, oa = daily_onsets(a_.reindex(idx).to_numpy(float), HAZARDS["H1_hypoxia"])
                _, ob = daily_onsets(b_.reindex(idx).to_numpy(float), HAZARDS["H1_hypoxia"])
                on["iv"] += int(oa.sum())
                on["dv"] += int(ob.sum())
                on["both"] += int((oa & ob).sum())
    r = pd.DataFrame(rows)
    res = {"n_sites": int(r["site"].nunique()) if len(r) else 0, "population": "train-role split sites, train block",
           "by_var": {v: {"n_days": int(g["n"].sum()), "frac_within_tol_pooled": float((g["frac_within_tol"] * g["n"]).sum() / g["n"].sum()),
                          "only_iv_days": int(g["only_iv"].sum()), "only_dv_days": int(g["only_dv"].sum())}
                      for v, g in r.groupby("var")} if len(r) else {},
           "h1_onsets": on}
    res["acceptance_do_min"] = bool(res["by_var"].get("do_mgl_min", {}).get("frac_within_tol_pooled", 0) >= 0.99)
    out = paths.RUNS_ROOT / "sonde" / "iv_daily_rule_validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(res, indent=1))
    r.to_csv(out / "per_site.csv", index=False)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
