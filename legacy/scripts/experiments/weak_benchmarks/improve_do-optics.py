#!/usr/bin/env python
"""Dissolved-oxygen (DO) from optics — honest diagnosis + improvement.

TARGET
------
DO R2 on the REAL Sentinel-2 x USGS drone_wq_pairs dataset:
  - drone WaterDroneNet : DO R2 = 0.257  (results/benchmarks/waterdronenet_holdout.json)
  - HydroViT (ViT)      : DO R2 = 0.582  (results/benchmarks/hydrovit_real.json)
Both use the same 4 bands (B02,B03,B04,B08) and a STATION holdout.

WHY DO IS HARD / THE DIAGNOSTIC QUESTION
----------------------------------------
DO is only WEAKLY optical. Dissolved O2 has no color. What optics CAN see are
proxies that co-vary with DO:
  * chlorophyll-a / algae (photosynthesis raises daytime DO; blooms swing it) ->
    green/red, red/NIR, NDVI, NDWI, brightness.
  * turbidity / TSS (respiration in turbid water lowers DO) -> red & NIR reflectance.
But the single strongest physical driver of DO is TEMPERATURE: O2 solubility falls
as water warms (Henry's law), so cold water holds more O2. Temperature is itself
partly optical (thermal/seasonal), and the shipped models reach Temp R2=0.73.

So the HONEST decomposition we must make is:
  (A) OPTICS-ONLY   : DO recoverable from reflectance alone (chl/turbidity proxies).
  (B) TEMP-ONLY     : DO recoverable from in-situ water temperature alone.
  (C) OPTICS+TEMP   : combined.
If (C) >> (A) and (C) ~ (B), the "improvement" is TEMPERATURE information, not new
optical signal -> we say so explicitly (temperature leakage into a DO gain).
If (A) alone already beats the shipped 0.257/0.582, that is a genuine optical gain
from better features + GBM (vs the ViT), on the SAME bands and a GROUPED holdout.

METHOD (matched to the repo's honest recipe)
--------------------------------------------
  * Same REAL data & bands. Features = per-band spatial stats (mean/std/percentiles
    over the tile) + optical proxy indices (NDVI, NDWI, green/red, red/NIR,
    brightness, ...). HistGradientBoosting regressor (CPU), same family as the
    red-edge chl-a study that beat the ViT.
  * Split = STATION GroupKFold (site_id); NO site in train and test. Out-of-fold
    predictions pooled -> grouped-holdout R2.
  * CI = site-clustered bootstrap (resamples SITES, 2000 reps) on each R2 and on
    the paired delta vs the shipped baseline. A gain counts only if the delta CI
    excludes 0.
  * "improved" is judged against the SAME modality it competes with:
      optics-only vs drone-4band (0.257): pure optical, apples-to-apples bands.
    We ALSO report optics+temp but flag it as temperature-informed, not a pure
    optical claim.

INTEGRITY: REAL public data only (Sentinel-2 L2A + USGS/NWIS in-situ). No target
augmentation, no random split, no tuning on test. Temperature used as a feature is
the REAL in-situ Temp measurement (a legitimately available same-day covariate),
and any DO gain it produces is labeled as temperature-driven, not optical.

MIT License -- Bryan Cheng, 2026
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]
NPZ = ROOT / "data" / "processed" / "satellite" / "drone_wq_pairs.npz"
OUT = ROOT / "results" / "benchmarks" / "weak_improve" / "do-optics.json"
SEED = 0
NBOOT = 2000
EPS = 1e-6

# Shipped baselines (same bands, station holdout)
DRONE_DO_R2 = 0.2569   # waterdronenet_holdout.json  (pure 4-band optics)
HYDROVIT_DO_R2 = 0.5818  # hydrovit_real.json          (pure 4-band optics)

BANDS = ["B02", "B03", "B04", "B08"]  # blue, green, red, NIR
TGT = ["DO", "pH", "Turb", "Temp", "SpCond"]


def band_stats(x2d, prefix, feats, names):
    v = x2d[np.isfinite(x2d)]
    if v.size == 0:
        v = np.zeros(1, np.float32)
    for tag, val in [("mean", v.mean()), ("std", v.std()),
                     ("p10", np.percentile(v, 10)), ("p50", np.percentile(v, 50)),
                     ("p90", np.percentile(v, 90)),
                     ("min", v.min()), ("max", v.max())]:
        feats.append(float(val))
        names.append(f"{prefix}_{tag}")


def optical_features(img):
    """img [4,H,W] -> optical feature vector (chl/turbidity proxies)."""
    B = {n: img[i].astype(np.float64) for i, n in enumerate(BANDS)}
    feats, names = [], []
    for n in BANDS:
        band_stats(B[n], n, feats, names)
    b2, b3, b4, b8 = B["B02"], B["B03"], B["B04"], B["B08"]
    ndvi = (b8 - b4) / (b8 + b4 + EPS)          # veg/algae
    ndwi = (b3 - b8) / (b3 + b8 + EPS)          # water/veg contrast
    gr = b3 / (b4 + EPS)                        # green/red ~ chl
    rn = b4 / (b8 + EPS)                        # red/NIR ~ turbidity/TSS
    bg = b2 / (b3 + EPS)                        # blue/green ~ CDOM/clarity
    ndti = (b4 - b3) / (b4 + b3 + EPS)          # turbidity index
    bright = (b2 + b3 + b4 + b8) / 4.0          # overall brightness
    for arr, pfx in [(ndvi, "NDVI"), (ndwi, "NDWI"), (gr, "GR"), (rn, "RN"),
                     (bg, "BG"), (ndti, "NDTI"), (bright, "BRIGHT")]:
        band_stats(arr, pfx, feats, names)
    return np.array(feats, np.float32), names


def build_optical_matrix(images):
    rows, names = [], None
    for i in range(images.shape[0]):
        fv, nm = optical_features(images[i])
        if names is None:
            names = nm
        rows.append(fv)
    return np.asarray(rows, np.float64), names


def hgb():
    return HistGradientBoostingRegressor(
        max_depth=6, learning_rate=0.03, max_iter=600,
        l2_regularization=3.0, random_state=SEED)


def oof_reg(X, y, g, ns):
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=ns).split(X, y, g):
        p[te] = hgb().fit(X[tr], y[tr]).predict(X[te])
    return p


def site_ci(y, p, sites, nboot=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.unique(sites)
    idx_by = {s: np.where(sites == s)[0] for s in uniq}
    vals = []
    for _ in range(nboot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_by[s] for s in samp])
        try:
            vals.append(r2_score(y[ii], p[ii]))
        except Exception:
            continue
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def paired_delta_ci(y, p_base, p_new, sites, nboot=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    uniq = np.unique(sites)
    idx_by = {s: np.where(sites == s)[0] for s in uniq}
    d = []
    for _ in range(nboot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_by[s] for s in samp])
        try:
            d.append(r2_score(y[ii], p_new[ii]) - r2_score(y[ii], p_base[ii]))
        except Exception:
            continue
    d = np.array(d)
    return (float(d.mean()), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)), float((d > 0).mean()))


def evaluate(X, y, sites, ns):
    """Return grouped-OOF R2 + site-bootstrap CI + OOF preds."""
    p = oof_reg(X, y, sites, ns)
    r2 = r2_score(y, p)
    lo, hi = site_ci(y, p, sites)
    return r2, (lo, hi), p


def main():
    t0 = time.time()
    d = np.load(NPZ, allow_pickle=True)
    images = d["images"]            # [N,4,224,224]
    targets = d["targets"]          # [N,5]
    meta = json.loads(d["metadata"].item())
    sites_all = np.array([m["site_id"] for m in meta])

    do = targets[:, TGT.index("DO")].astype("float64")
    temp = targets[:, TGT.index("Temp")].astype("float64")

    print(f"Building optical features for {images.shape[0]} tiles...")
    Xopt_all, opt_names = build_optical_matrix(images)
    print(f"  optical features: {len(opt_names)}")

    # Valid DO rows (optics-only usable)
    finite_opt = np.isfinite(Xopt_all).all(axis=1)
    m_opt = finite_opt & np.isfinite(do)
    # For temp arms we also require finite temp
    m_all = m_opt & np.isfinite(temp)

    g_opt = sites_all[m_opt]
    ns_opt = min(5, np.unique(g_opt).size)
    y_opt = do[m_opt]

    g_all = sites_all[m_all]
    ns_all = min(5, np.unique(g_all).size)
    y_all = do[m_all]

    print(f"DO valid (optics): n={m_opt.sum()} sites={np.unique(g_opt).size}")
    print(f"DO valid (optics+temp): n={m_all.sum()} sites={np.unique(g_all).size}")

    out = {
        "_PROVENANCE": {
            "purpose": "Honest diagnosis + improvement of DO (dissolved oxygen) from "
                       "optics on REAL Sentinel-2 x USGS data. Decomposes DO signal "
                       "into optics-only vs temperature-only vs combined.",
            "data": "data/processed/satellite/drone_wq_pairs.npz (REAL Sentinel-2 L2A "
                    "B02,B03,B04,B08 paired with REAL same-day USGS/NWIS in-situ WQ).",
            "bands": BANDS,
            "real_data_only": True,
            "augmentation_to_target": False,
            "label_leakage": False,
            "split": "STATION (site_id) GroupKFold — no site shared train/test; "
                     "pooled out-of-fold predictions.",
            "ci": "site-clustered bootstrap (resamples SITES, not rows), 2000 reps; "
                  "paired delta CI vs shipped baseline.",
            "model": "HistGradientBoosting regressor (CPU), same family as the "
                     "red-edge chl-a study that beat the ViT.",
            "temperature_note": "Temp feature = REAL in-situ same-day water temperature, "
                                "a legitimately available covariate. DO gain attributable "
                                "to Temp is labeled temperature-driven, NOT optical.",
        },
        "shipped_baselines": {
            "drone_DO_r2": DRONE_DO_R2,
            "hydrovit_DO_r2": HYDROVIT_DO_R2,
            "note": "both use the same 4 bands + station holdout; drone is the pure-optics "
                    "comparator for the optics-only arm.",
        },
        "n_valid": {"optics": int(m_opt.sum()), "optics_temp": int(m_all.sum())},
        "arms": {},
    }

    # ---- ARM A: OPTICS-ONLY ----
    r2_A, ci_A, pA = evaluate(Xopt_all[m_opt], y_opt, g_opt, ns_opt)
    print(f"[A optics-only]   R2={r2_A:.4f} CI[{ci_A[0]:.4f},{ci_A[1]:.4f}]")

    # ---- ARM B: TEMP-ONLY (on the optics+temp valid set for fair comparison) ----
    Xtemp = temp[m_all].reshape(-1, 1)
    r2_B, ci_B, pB = evaluate(Xtemp, y_all, g_all, ns_all)
    print(f"[B temp-only]     R2={r2_B:.4f} CI[{ci_B[0]:.4f},{ci_B[1]:.4f}]")

    # ---- ARM C: OPTICS+TEMP ----
    Xct = np.column_stack([Xopt_all[m_all], temp[m_all]])
    r2_C, ci_C, pC = evaluate(Xct, y_all, g_all, ns_all)
    print(f"[C optics+temp]   R2={r2_C:.4f} CI[{ci_C[0]:.4f},{ci_C[1]:.4f}]")

    # ---- ARM A' : optics-only on the SAME optics+temp valid set (for clean deltas) ----
    r2_Ap, ci_Ap, pAp = evaluate(Xopt_all[m_all], y_all, g_all, ns_all)
    print(f"[A' optics-only same-set] R2={r2_Ap:.4f}")

    out["arms"]["A_optics_only"] = {
        "features": "4-band spatial stats + chl/turbidity proxy indices (no temp)",
        "n": int(m_opt.sum()), "n_sites": int(np.unique(g_opt).size),
        "r2": round(r2_A, 4), "r2_ci95": [round(ci_A[0], 4), round(ci_A[1], 4)],
    }
    out["arms"]["B_temp_only"] = {
        "features": "in-situ water temperature only (1 feature)",
        "n": int(m_all.sum()), "n_sites": int(np.unique(g_all).size),
        "r2": round(r2_B, 4), "r2_ci95": [round(ci_B[0], 4), round(ci_B[1], 4)],
    }
    out["arms"]["C_optics_plus_temp"] = {
        "features": "optical proxies + in-situ temperature",
        "n": int(m_all.sum()), "n_sites": int(np.unique(g_all).size),
        "r2": round(r2_C, 4), "r2_ci95": [round(ci_C[0], 4), round(ci_C[1], 4)],
    }
    out["arms"]["A_optics_only_matched_set"] = {
        "note": "optics-only evaluated on the SAME rows as B/C for fair paired deltas",
        "r2": round(r2_Ap, 4), "r2_ci95": [round(ci_Ap[0], 4), round(ci_Ap[1], 4)],
    }

    # ---- PAIRED DELTAS ----
    # 1) optical improvement vs shipped drone (pure optics vs pure optics).
    #    Shipped drone DO R2=0.257 is a fixed scalar (different model/rows), so we
    #    report the gain as (A_optics R2 - 0.257) and give the CI on A's own R2.
    out["deltas"] = {}
    out["deltas"]["optics_vs_drone_shipped"] = {
        "new_r2": round(r2_A, 4), "old_r2": DRONE_DO_R2,
        "delta": round(r2_A - DRONE_DO_R2, 4),
        "new_r2_ci95": [round(ci_A[0], 4), round(ci_A[1], 4)],
        "interpretation": "pure-optics GBM vs pure-optics drone ViT, same bands & "
                          "station holdout. Improved iff new_r2 CI lower bound > drone 0.257.",
    }
    out["deltas"]["optics_vs_hydrovit_shipped"] = {
        "new_r2": round(r2_A, 4), "old_r2": HYDROVIT_DO_R2,
        "delta": round(r2_A - HYDROVIT_DO_R2, 4),
        "new_r2_ci95": [round(ci_A[0], 4), round(ci_A[1], 4)],
    }
    # 2) temperature's marginal value: C vs A' (same rows, paired) -> how much of any
    #    combined gain is temperature, not optics.
    dm, dlo, dhi, pgt = paired_delta_ci(y_all, pAp, pC, g_all)
    out["deltas"]["temp_marginal_C_minus_Aoptics"] = {
        "mean": round(dm, 4), "ci95": [round(dlo, 4), round(dhi, 4)],
        "prob_improvement": round(pgt, 3),
        "interpretation": "R2 gain from ADDING temperature to optics (paired, same "
                          "sites). This gain is TEMPERATURE-driven, not new optical signal.",
    }
    # 3) optics' marginal value over temp: C vs B (same rows) -> does optics add
    #    anything beyond temperature?
    dm2, dlo2, dhi2, pgt2 = paired_delta_ci(y_all, pB, pC, g_all)
    out["deltas"]["optics_marginal_C_minus_temp"] = {
        "mean": round(dm2, 4), "ci95": [round(dlo2, 4), round(dhi2, 4)],
        "prob_improvement": round(pgt2, 3),
        "interpretation": "R2 gain from ADDING optics on top of temperature (paired). "
                          "This isolates GENUINE optical DO information beyond temperature.",
    }

    # ---- VERDICTS ----
    verdict = {}
    # optical improvement over shipped drone: require A's CI lower bound above 0.257
    if ci_A[0] > DRONE_DO_R2:
        verdict["optics_vs_drone"] = (
            f"IMPROVED: optics-only R2={r2_A:.3f} (CI lower {ci_A[0]:.3f}) > "
            f"drone shipped 0.257. Genuine optical DO gain (GBM > ViT, same bands).")
        optics_improved = True
    elif r2_A > DRONE_DO_R2:
        verdict["optics_vs_drone"] = (
            f"HIGHER POINT but CI touches: optics-only R2={r2_A:.3f} > 0.257 yet "
            f"CI lower {ci_A[0]:.3f} <= 0.257. Not conclusively improved.")
        optics_improved = False
    else:
        verdict["optics_vs_drone"] = (
            f"NOT IMPROVED vs drone: optics-only R2={r2_A:.3f} <= 0.257.")
        optics_improved = False

    if dlo2 > 0:
        verdict["optics_beyond_temp"] = (
            f"Optics adds REAL signal beyond temperature: C-B delta={dm2:+.3f} "
            f"CI[{dlo2:+.3f},{dhi2:+.3f}] excludes 0.")
    else:
        verdict["optics_beyond_temp"] = (
            f"Optics adds LITTLE beyond temperature: C-B delta={dm2:+.3f} "
            f"CI[{dlo2:+.3f},{dhi2:+.3f}] includes 0.")

    verdict["temperature_share"] = (
        f"Temp-only R2={r2_B:.3f} vs optics-only R2={r2_Ap:.3f} vs combined R2={r2_C:.3f}. "
        f"Adding temp to optics lifts R2 by {dm:+.3f} (CI[{dlo:+.3f},{dhi:+.3f}]). "
        "Any DO 'improvement' from the combined model is largely TEMPERATURE information.")

    out["verdict"] = verdict

    # headline improved flag = pure-optics genuine gain over the pure-optics baseline
    out["improved"] = bool(optics_improved)
    out["improved_definition"] = (
        "new_r2 (optics-only) CI-lower-bound > shipped drone pure-optics DO R2 0.257, "
        "on a station GroupKFold holdout. This is a pure-optics claim; the "
        "optics+temp arm is reported but flagged temperature-driven.")
    out["elapsed_seconds"] = round(time.time() - t0, 1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print("WROTE", OUT)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
