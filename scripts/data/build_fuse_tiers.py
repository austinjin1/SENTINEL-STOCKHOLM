# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build the SENTINEL-Fuse aligned multimodal tables (Stage 0 audit).

    python -B scripts/data/build_fuse_tiers.py d      # processed('fuse_tier_d')  NEON
    python -B scripts/data/build_fuse_tiers.py c      # processed('fuse_tier_c')  cyanotoxins
    python -B scripts/data/build_fuse_tiers.py w      # processed('fuse_tier_w')  wide USGS/WQP
    python -B scripts/data/build_fuse_tiers.py wp1    # processed('fuse_tier_wp1') W-P1 harder track (protocol §12)

Each target writes Parquet tables, availability-pattern reports and a manifest
(``manifests/fuse_tier_<x>.json``). Datasheets: docs/datasheets/fuse_tier_<x>.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

import pandas as pd

from sentinel.core import paths
from sentinel.core.manifest import read_manifest, write_manifest
from sentinel.data import fuse_tiers as ft

TODAY = dt.date.today().isoformat()


def _inputs(names):
    out = {}
    for n in names:
        try:
            out[n] = read_manifest(n)["created"]
        except FileNotFoundError:
            out[n] = None
    return out


def t_d(args) -> None:
    res = ft.build_tier_d()
    out = paths.processed("fuse_tier_d")
    daily = res["daily"]
    files = []
    for name in ("daily", "chem_events", "micro_events"):
        f = out / f"{name}.parquet"
        res[name].to_parquet(f, compression="zstd", index=False)
        files.append(f)
    f = out / "micro_genus.parquet"
    res["micro_genus"].to_parquet(f, compression="zstd")
    files.append(f)
    (out / "columns.json").write_text(json.dumps(res["modalities"], indent=1))
    files.append(out / "columns.json")
    mods = ft.NON_HUB_D
    av = ft.availability_patterns(daily, mods)
    av.to_csv(out / "availability_all.csv", index=False)
    files.append(out / "availability_all.csv")
    by_t = {}
    for t in ["do_mgl", "turb_fnu", "no3n_mgl", "chl_ugl"]:
        m = daily[f"y__{t}__h7"].notna().to_numpy()
        a = ft.availability_patterns(daily, mods, extra_mask=m)
        a.insert(0, "target", f"y__{t}__h7")
        by_t[t] = a
    for t in ["tn_mgl", "tp_mgl", "doc_mgl"]:
        m = daily[f"ylab__{t}"].notna().to_numpy() | daily[f"ylab__{t}__cens"].fillna(False).to_numpy(bool)
        a = ft.availability_patterns(daily, mods, extra_mask=m)
        a.insert(0, "target", f"ylab__{t}")
        by_t[t] = a
    abt = pd.concat(by_t.values(), ignore_index=True)
    abt.to_csv(out / "availability_by_target.csv", index=False)
    files.append(out / "availability_by_target.csv")
    present = {m: int(daily[f"{m}__present"].sum()) for m in mods + ft.HUB_D if f"{m}__present" in daily}
    n_mod = sum(daily[f"{m}__present"].fillna(False).astype(int) for m in mods)
    stats = {
        "rows": int(len(daily)), "sites": int(daily.site.nunique()),
        "first": str(daily.date.min().date()), "last": str(daily.date.max().date()),
        "present_rows": present,
        "rows_by_n_nonhub_modalities": {int(k): int(v) for k, v in n_mod.value_counts().sort_index().items()},
        "chem_events": int(daily["is_chem_event"].sum()), "micro_events": int(daily["is_micro_event"].sum()),
        "target_rows_h7": {c: int(daily[c].notna().sum()) for c in daily.columns
                           if c.startswith("y__") and c.endswith("__h7")},
        "lab_target_rows": {c: int(daily[c].notna().sum()) for c in daily.columns
                            if c.startswith("ylab__") and not c.endswith(("__cens", "__dl"))},
        "hazards": {c: {"valid": int(daily[c].notna().sum()), "pos": int((daily[c] == 1).sum()),
                        "pos_sites": int(daily.loc[daily[c] == 1, "site"].nunique())}
                    for c in daily.columns if c.startswith("ev__")},
        "top_patterns": av.head(15).to_dict("records"),
        "rules": {"calendar": ft.CALENDAR, "chem_max_age_days": ft.CHEM_MAX_AGE, "chem_strictly_before": True,
                  "micro_max_age_days": ft.MICRO_MAX_AGE, "bio_max_age_days": ft.BIO_MAX_AGE,
                  "sensor_present_days": ft.SENSOR_PRESENT_DAYS, "hor_pref": ft.HOR_PREF,
                  "log_offsets": ft.LOG_OFFSET, "lab_log_offsets": ft.LAB_LOG_OFFSET},
        "p2_products_present": {k: (paths.PROCESSED / v[0]).exists() for k, v in ft.SENSOR_PRODUCTS.items()}
        | {"neon_macroinv": (paths.PROCESSED / "neon_macroinv").exists(),
           "neon_periphyton": (paths.PROCESSED / "neon_periphyton").exists()},
        "retrieved_inputs": _inputs(["neon_sites", "neon_sonde_daily", "neon_nitrate", "neon_tsw", "neon_tsd",
                                     "neon_chem", "microbe_wq", "forcings_daily", "streamcat", "sites",
                                     "neon_eos", "neon_discharge", "neon_macroinv", "neon_periphyton"]),
        "built": TODAY,
    }
    write_manifest("fuse_tier_d", files=files,
                   sources=["manifest:neon_sonde_daily", "manifest:neon_nitrate", "manifest:neon_tsw",
                            "manifest:neon_tsd", "manifest:neon_chem", "manifest:microbe_wq",
                            "manifest:forcings_daily", "manifest:streamcat", "manifest:sites", "manifest:neon_eos",
                            "manifest:neon_discharge", "manifest:neon_macroinv", "manifest:neon_periphyton"],
                   description="SENTINEL-Fuse Tier D: NEON site × issue-date aligned multimodal table "
                               "(sonde, nitrate, temperature, hydrology, lab chemistry, 16S, macroinvertebrates, periphyton, hub) "
                               "+ targets/hazards.",
                   license="CC BY 4.0 (NEON); public domain (gridMET, StreamCat)", synthetic=False, stats=stats)
    print(json.dumps({k: stats[k] for k in ("rows", "sites", "present_rows", "rows_by_n_nonhub_modalities")},
                     indent=1))
    print(av.head(20).to_string())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tier", choices=["d", "c", "w", "wp1"])
    args = ap.parse_args(argv)
    if args.tier == "d":
        t_d(args)
    elif args.tier == "c":
        from sentinel.data import fuse_tier_c
        fuse_tier_c.main_build()
    elif args.tier == "wp1":
        from sentinel.data import fuse_tier_wp1
        fuse_tier_wp1.main_build()
    else:
        from sentinel.data import fuse_tier_w
        fuse_tier_w.main_build()
    return 0


if __name__ == "__main__":
    sys.exit(main())
