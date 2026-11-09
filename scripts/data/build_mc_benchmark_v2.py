#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Re-extract Sentinel-2 features for the v1 microcystin benchmark (3,649 rows, 227 sites).

v1 inputs (read-only, SENTINEL_LEGACY): data/processed/biology/noaa_habs/mc_fusion_dataset.parquet
and mc_download_targets.parquet; coordinates via the positional ``idx`` join used by v1
(``targets.reset_index(drop=True).iloc[idx]``, verified row by row here).

Corrected pipeline (vs v1 ``download_microcystin_s2.py``): every S2 acquisition covering the site
within +-7 days (v1 window) is found (Earth Search C1, Planetary Computer fallback); SCL is screened
first; the 2 nearest usable acquisitions per row get full chips (BOA offset harmonised per item,
all bands on one 10 m grid, SCL + JRC layers, exact granule sensing time); features are recomputed
with the v1 definitions *and* as water-masked centre medians.

Outputs: processed('mc_benchmark_v2')/{mc_benchmark_v2.parquet, candidates.parquet, qc_report.json};
chips in processed('s2_chips')/mc_benchmark.

    python -B scripts/data/build_mc_benchmark_v2.py --read-workers 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data import chips as C  # noqa: E402
from sentinel.data import matchups as M  # noqa: E402
from sentinel.data.sources import stac as S  # noqa: E402

log = logging.getLogger("mc_benchmark_v2")
WINDOW = 7
V1_DIR = Path("data/processed/biology/noaa_habs")
OFFSET_SWITCH = pd.Timestamp("2022-01-25")


def load_v1() -> pd.DataFrame:
    base = paths.LEGACY_ROOT / V1_DIR
    fus = pd.read_parquet(base / "mc_fusion_dataset.parquet")
    tg = pd.read_parquet(base / "mc_download_targets.parquet").reset_index(drop=True)
    t = tg.iloc[fus["idx"].to_numpy()].reset_index(drop=True)
    chk = pd.DataFrame({
        "site_ok": t["Location_Identifier"].astype(str).to_numpy() == fus["site"].astype(str).to_numpy(),
        "date_ok": pd.to_datetime(t["date"]).dt.normalize().to_numpy() == pd.to_datetime(fus["date"]).dt.normalize().to_numpy(),
        "val_ok": np.isclose(t["val"].to_numpy(float), fus["mc"].to_numpy(float), equal_nan=True)})
    if not chk.all().all():
        raise ValueError(f"v1 idx join inconsistent: {(~chk).sum().to_dict()}")
    df = pd.DataFrame({
        "v1_idx": fus["idx"].astype(int), "site_id": fus["site"].astype(str),
        "date": pd.to_datetime(fus["date"]).dt.normalize(), "lat": t["Location_Latitude"].astype(float),
        "lon": t["Location_Longitude"].astype(float), "mc_ugL": fus["mc"].astype(float),
        "mc_unit": "ug/L", "label_gt8": fus["label"].astype(int),
        # v1 pigment columns mix lab ug/L, mg/m3 and probe RFU and contain sentinel codes; kept only
        # for traceability, never as features (docs/research/10 §0).
        "v1_CHLA_unharmonised": fus["CHLA"].astype(float), "v1_PC_unharmonised": fus["PC"].astype(float),
        "v1_cloud": fus["cloud"].astype(float)})
    for k in range(26):
        df[f"v1_sf{k}"] = fus[f"sf{k}"].astype(float)
    df["point_key"] = [M.point_key(lo, la) for lo, la in zip(df.lon, df.lat)]
    df["v1_era_post_offset"] = df.date >= OFFSET_SWITCH
    return df


def era_confound(res: pd.DataFrame, seed: int = 0) -> dict:
    """Data-QC diagnostic (not a model result): how well the processing era (pre/post 2022-01-25,
    the BOA-offset switch) can be predicted from the satellite features, with site-grouped 5-fold CV
    and a small gradient-boosting classifier. v1 features: AUROC 0.989 (docs/research/10 §0). Rows: QC-
    lenient rows with finite features in both sets."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    v1 = [f"v1_sf{k}" for k in range(26)]
    v2 = [c for c in res.columns if c.startswith("v2def_") and not c.startswith("v2def_water_")]
    sub = res[res.qc_lenient].dropna(subset=v1 + v2)
    y = sub.v1_era_post_offset.astype(int).to_numpy()
    if len(sub) < 100 or y.min() == y.max():
        return {"n": int(len(sub)), "note": "too few rows"}
    out = {"n": int(len(sub)), "n_post": int(y.sum()), "n_sites": int(sub.site_id.nunique())}
    for name, cols in (("v1_features", v1), ("v2_features", v2)):
        X = sub[cols].to_numpy(float)
        oof = np.full(len(y), np.nan)
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in cv.split(X, y, sub.site_id):
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_depth=3, random_state=seed)
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        out[f"auroc_era_{name}"] = float(roc_auc_score(y, oof))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--read-workers", type=int, default=2)
    ap.add_argument("--scene-workers", type=int, default=3)
    ap.add_argument("--per-row", type=int, default=2, help="usable acquisitions per row to extract")
    a = ap.parse_args()
    out = paths.processed("mc_benchmark_v2")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / "build.log"), logging.StreamHandler()])
    logging.getLogger("rasterio").setLevel(logging.WARNING)
    rows = load_v1()
    log.info("v1 rows %d, sites %d, dates %s..%s", len(rows), rows.site_id.nunique(), rows.date.min().date(),
             rows.date.max().date())
    # ---- planning: every S2 acquisition covering each site within +-7 d
    need: dict[str, list] = {}
    for r in rows.drop_duplicates(["point_key", "date"]).itertuples():
        for d in M.date_range_strings(r.date, WINDOW, WINDOW):
            need.setdefault(d, []).append((r.point_key, r.lon, r.lat))
    pbd = {d: pd.DataFrame(v, columns=["point_key", "lon", "lat"]).drop_duplicates("point_key")
           for d, v in need.items()}
    cand, cat = M.plan_s2_candidates(pbd, paths.interim("s2_scene_catalog") / "mc_benchmark", logger=log.info)
    log.info("planning: %d (point, scene) candidates over %d dates; %d scenes", len(cand), len(pbd), len(cat))
    scenes = M.scenes_from_catalog(cat)
    pts = rows.drop_duplicates("point_key").set_index("point_key")[["lon", "lat"]]
    cand = cand.drop_duplicates(["point_key", "item_id"])
    cand["request_key"] = cand.item_id + "|" + cand.point_key
    cand = cand.join(pts, on="point_key")
    # ---- SCL screening
    writer = C.ChipWriter(paths.processed("s2_chips") / "mc_benchmark", layers=C.S2_STORE_LAYERS,
                          dtype="int16", semantics=C.S2_SEMANTICS)
    ex = M.S2Extractor(writer, read_workers=a.read_workers, scene_workers=a.scene_workers, logger=log.info)
    scr = ex.screen_scl(cand[["request_key", "item_id", "lon", "lat"]], scenes,
                        paths.interim("s2_screen") / "mc_benchmark")
    scr = scr.drop_duplicates("request_key", keep="last")
    # ---- row x candidate table
    rc = rows[["v1_idx", "point_key", "date"]].merge(cand[["point_key", "item_id", "request_key"]], on="point_key")
    rc = rc.merge(scr, on=["request_key", "item_id"], how="left")
    st = pd.to_datetime(rc.sensing_time_utc, utc=True, errors="coerce")
    rc = pd.concat([rc, M.window_columns(rc.date, st)], axis=1)
    rc = rc[rc.abs_dt_days <= WINDOW]
    rc["usable"] = (rc.screen_status == "ok") & ~rc.nodata_centre.astype("boolean").fillna(True).astype(bool) & \
        (rc.scl_c7_n_valid >= 5)
    rc["water_hint"] = rc.scl_c3_n_water.fillna(0) >= 5
    rc = rc.sort_values(["v1_idx", "usable", "water_hint", "abs_dt_days", "scene_cloud_cover"],
                        ascending=[True, False, False, True, True])
    rc["rank"] = rc.groupby("v1_idx").cumcount()
    chosen = rc[rc.usable & (rc["rank"] < a.per_row)]
    req = chosen.drop_duplicates("request_key").merge(cand[["request_key", "lon", "lat"]], on="request_key")
    log.info("full extraction for %d (row, scene) choices -> %d unique chip requests", len(chosen), len(req))
    ex.run(req[["request_key", "item_id", "lon", "lat", "point_key"]], scenes)
    ex.close()
    idx = C.consolidate(writer.root)
    idx["offset_suspect"] = C.recompute_offset_suspect(idx)  # current flag definition
    # ---- features per chip (v1 definitions recomputed + water-only variant)
    store = C.ChipStore(writer.root)
    ok = idx[idx.status == "ok"]
    feats = []
    for r in ok.itertuples():
        chip = store.load(r.chip_id)
        f1 = {f"v2def_{k}": v for k, v in C.v1_definition_features(chip).items()}
        f2 = {f"v2def_water_{k}": v for k, v in C.v1_definition_features(chip, water_only=True).items()}
        feats.append({"request_key": r.request_key, **f1, **f2})
    store.close()
    feats = pd.DataFrame(feats)
    keep_idx = [c for c in idx.columns if c not in ("lon", "lat", "point_key", "item_id")]
    cands = rc.merge(idx[keep_idx], on="request_key", how="left", suffixes=("", "_chip"))
    cands = cands.merge(feats, on="request_key", how="left")
    cands["chip_ok"] = cands.status == "ok"
    cands["qc_strict"] = cands.chip_ok & (cands.scl_c3_n_water_chip >= 5) & \
        cands.jrc_water_ok.astype("boolean").fillna(False).astype(bool) & (cands.offset_suspect.fillna("") == "")
    cloud_c3 = cands[["scl_c3_cloud_medium_chip", "scl_c3_cloud_high_chip", "scl_c3_thin_cirrus_chip",
                      "scl_c3_cloud_shadow_chip"]].sum(axis=1)
    cands["qc_lenient"] = cands.chip_ok & (cands.scl_c7_n_water_chip >= 5) & (cloud_c3 == 0) & \
        (cands.offset_suspect.fillna("") == "")
    cands.to_parquet(out / "candidates.parquet", index=False, compression="zstd")
    # ---- one row per v1 row: best strict, else best lenient, else best chip
    cands["_q"] = np.select([cands.qc_strict, cands.qc_lenient, cands.chip_ok], [0, 1, 2], 3)
    best = cands.sort_values(["v1_idx", "_q", "abs_dt_days", "scene_cloud_cover"]).drop_duplicates("v1_idx")
    best = best.drop(columns=["_q", "date", "point_key"])
    res = rows.merge(best, on="v1_idx", how="left")
    agg = rc.groupby("v1_idx").agg(n_acq_7d=("request_key", "nunique"), n_usable_7d=("usable", "sum"),
                                  min_abs_dt_usable=("abs_dt_days", lambda s: np.nan))
    usable_dt = rc[rc.usable].groupby("v1_idx").abs_dt_days.min()
    agg["min_abs_dt_usable"] = usable_dt
    res = res.join(agg, on="v1_idx")
    res["n_acq_7d"] = res.n_acq_7d.fillna(0).astype(int)
    res["n_usable_7d"] = res.n_usable_7d.fillna(0).astype(int)
    for c in ("chip_ok", "qc_strict", "qc_lenient"):
        res[c] = res[c].astype("boolean").fillna(False).astype(bool)
    dest = out / "mc_benchmark_v2.parquet"
    res.to_parquet(dest, index=False, compression="zstd")
    # ---- QC survival report
    rep = {"n_rows": int(len(res)), "n_sites": int(res.site_id.nunique()),
           "any_acquisition_7d": int((res.n_acq_7d > 0).sum()),
           "usable_scl_7d": int((res.n_usable_7d > 0).sum()), "chip_ok": int(res.chip_ok.sum()),
           "qc_strict": int(res.qc_strict.sum()), "qc_lenient": int(res.qc_lenient.sum())}
    for w in (0, 1, 3, 7):
        m = res.abs_dt_days <= w
        rep[f"qc_strict_within_{w}d"] = int((res.qc_strict & m).sum())
        rep[f"qc_lenient_within_{w}d"] = int((res.qc_lenient & m).sum())
    for lab, m in (("pos_gt8", res.label_gt8 == 1), ("post_2022_01_25", res.v1_era_post_offset)):
        rep[f"{lab}_rows"] = int(m.sum())
        rep[f"{lab}_qc_strict"] = int((res.qc_strict & m).sum())
        rep[f"{lab}_qc_lenient"] = int((res.qc_lenient & m).sum())
    rep["sites_qc_strict"] = int(res[res.qc_strict].site_id.nunique())
    rep["stac_source_chosen"] = res[res.chip_ok].stac_source.value_counts().to_dict()
    rep["baseline_chosen"] = res[res.chip_ok].processing_baseline.value_counts().to_dict()
    rep["extraction_stats"] = ex.stats
    rep["offset_group_check"] = C.offset_group_check(idx).to_dict("records")
    try:
        rep["era_confound_diagnostic"] = era_confound(res)
    except Exception as e:  # noqa: BLE001 — diagnostic only
        rep["era_confound_diagnostic"] = {"error": repr(e)}
    (out / "qc_report.json").write_text(json.dumps(rep, indent=1, default=str))
    log.info("QC report: %s", rep)
    write_manifest("mc_benchmark_v2", files=[dest, out / "candidates.parquet", out / "qc_report.json",
                                            writer.root / "index.parquet"],
                   sources=[str(paths.LEGACY_ROOT / V1_DIR / "mc_fusion_dataset.parquet"),
                            str(paths.LEGACY_ROOT / V1_DIR / "mc_download_targets.parquet"),
                            S.EARTH_SEARCH_URL, S.PLANETARY_COMPUTER_URL],
                   description="v1 microcystin benchmark (3,649 rows) with Sentinel-2 L2A features re-extracted "
                               "by the corrected pipeline (+-7 d, SCL screening, per-item BOA offset, single "
                               "10 m grid, SCL/JRC masks, exact sensing time); strict/lenient QC flags.",
                   license="WQP (public); Copernicus Sentinel data (free, full and open); JRC GSW",
                   synthetic=False, stats=json.loads(json.dumps(rep, default=str)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
