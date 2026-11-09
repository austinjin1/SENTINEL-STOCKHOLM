#!/usr/bin/env python
"""HydroViT pH from optics — honest diagnosis + improvement (tier2).

TARGET
------
pH R2 on the REAL Sentinel-2 x USGS drone_wq_pairs dataset:
  - HydroViT (ViT-S/16): pH R2 = 0.3109  (results/benchmarks/hydrovit_real.json,
    STATION holdout, 4 bands B02/B03/B04/B08, MAE 0.36).
That 0.31 is the bar to beat with a PURE-OPTICS GBM on the same bands + station
GroupKFold + a site-clustered bootstrap CI.

WHY pH IS HARD / THE DIAGNOSTIC QUESTION
----------------------------------------
pH has NO intrinsic color; it is only WEAKLY optical. Reflectance can only see
proxies that co-vary with pH:
  * CDOM / dissolved organics: humic/tannic acids depress pH and absorb blue ->
    blue/green ratio, blue reflectance (CDOM darkens & yellows water).
  * Algal / photosynthetic activity: primary production consumes CO2 and RAISES
    daytime pH; blooms are visible via chl (green/red, red-edge-like NDCI proxy,
    NDVI-on-water) -> a genuine optical pH lever.
  * Turbidity / TSS: mineral-rich turbid water often tracks alkalinity/carbonate
    buffering -> red & NIR reflectance.
The strongest NON-optical covariates for pH are (a) TEMPERATURE (shifts carbonate
equilibria & photosynthesis) and (b) SPECIFIC CONDUCTANCE (ionic strength /
alkalinity / carbonate buffering — chemically tied to pH but NOT optical).

HONEST DECOMPOSITION
--------------------
  (A) OPTICS-ONLY   : pH recoverable from reflectance alone (CDOM/chl/turbidity).
  (B) COVAR-ONLY    : pH from in-situ Temp + SpCond alone (non-optical chemistry).
  (C) OPTICS+COVAR  : combined.
If (A) alone beats the shipped 0.31 with a CI lower bound above 0.31, that is a
GENUINE optical pH gain (better features + GBM vs the ViT), same bands & holdout.
If gains only appear in (C), we label them covariate-driven (Temp/SpCond chemistry),
NOT a new optical claim, and report the paired marginal deltas that isolate which
is which. A confirmed weak-optical ceiling is itself a valid, reportable result.

METHOD (matched to the repo's honest recipe)
--------------------------------------------
  * Same REAL data & 4 bands. Features = per-band spatial stats (mean/std/pctiles)
    + optical proxy indices (blue/green CDOM, green/red chl, red/NIR turbidity,
    NDVI/NDWI/NDTI/brightness). HistGradientBoosting regressor (CPU), same family
    as the red-edge chl-a study that beat the ViT.
  * Split = STATION GroupKFold (site_id); NO site in train and test. Pooled OOF
    predictions -> grouped-holdout R2.
  * CI = site-clustered bootstrap (resamples SITES, 2000 reps) on each R2 and on
    paired deltas. A gain counts only if the delta / new-R2 CI excludes the bar.
  * "improved" is a PURE-OPTICS claim: optics-only R2 CI-lower-bound > 0.31.

INTEGRITY: REAL public data only (Sentinel-2 L2A + USGS/NWIS in-situ). No target
augmentation, no random split, no tuning on test. Temp/SpCond used as features are
REAL same-day in-situ covariates; any pH gain they add is flagged covariate-driven,
NOT optical.

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

ROOT = Path(__file__).resolve().parents[4]
NPZ = ROOT / "data" / "processed" / "satellite" / "drone_wq_pairs.npz"
OUT = ROOT / "results" / "benchmarks" / "weak_improve" / "tier2" / "hydrovit-ph.json"
SEED = 0
NBOOT = 2000
EPS = 1e-6

# Shipped baseline (same 4 bands, station holdout)
HYDROVIT_PH_R2 = 0.3109   # results/benchmarks/hydrovit_real.json per_param.pH.r2

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
    """img [4,H,W] -> optical feature vector (CDOM/chl/turbidity proxies for pH)."""
    B = {n: img[i].astype(np.float64) for i, n in enumerate(BANDS)}
    feats, names = [], []
    for n in BANDS:
        band_stats(B[n], n, feats, names)
    b2, b3, b4, b8 = B["B02"], B["B03"], B["B04"], B["B08"]
    ndvi = (b8 - b4) / (b8 + b4 + EPS)          # veg/algae (bloom -> high pH)
    ndwi = (b3 - b8) / (b3 + b8 + EPS)          # water/veg contrast
    gr = b3 / (b4 + EPS)                        # green/red ~ chl (photosynthesis)
    rn = b4 / (b8 + EPS)                        # red/NIR ~ turbidity/TSS
    bg = b2 / (b3 + EPS)                        # blue/green ~ CDOM/clarity (acidity)
    br = b2 / (b4 + EPS)                        # blue/red ~ CDOM absorption
    ndti = (b4 - b3) / (b4 + b3 + EPS)          # turbidity index
    ndci_p = (b4 - b3) / (b4 + b3 + EPS)        # NDCI proxy (no red-edge; red-green)
    bright = (b2 + b3 + b4 + b8) / 4.0          # overall brightness
    for arr, pfx in [(ndvi, "NDVI"), (ndwi, "NDWI"), (gr, "GR"), (rn, "RN"),
                     (bg, "BG"), (br, "BR"), (ndti, "NDTI"), (ndci_p, "NDCIp"),
                     (bright, "BRIGHT")]:
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


def scalar_delta_ci(y, p_new, sites, bar, nboot=NBOOT, seed=SEED):
    """Bootstrap CI on (new_R2 - fixed shipped bar). The bar is a fixed scalar
    from a different model/rows, so we bootstrap only the new model's R2."""
    rng = np.random.default_rng(seed + 2)
    uniq = np.unique(sites)
    idx_by = {s: np.where(sites == s)[0] for s in uniq}
    d = []
    for _ in range(nboot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_by[s] for s in samp])
        try:
            d.append(r2_score(y[ii], p_new[ii]) - bar)
        except Exception:
            continue
    d = np.array(d)
    return (float(d.mean()), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)), float((d > 0).mean()))


def evaluate(X, y, sites, ns):
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

    ph = targets[:, TGT.index("pH")].astype("float64")
    temp = targets[:, TGT.index("Temp")].astype("float64")
    spc = targets[:, TGT.index("SpCond")].astype("float64")

    print(f"Building optical features for {images.shape[0]} tiles...")
    Xopt_all, opt_names = build_optical_matrix(images)
    print(f"  optical features: {len(opt_names)}")

    finite_opt = np.isfinite(Xopt_all).all(axis=1)
    m_opt = finite_opt & np.isfinite(ph)
    # covariate arms require finite temp & spcond too
    m_all = m_opt & np.isfinite(temp) & np.isfinite(spc)

    g_opt = sites_all[m_opt]
    ns_opt = min(5, np.unique(g_opt).size)
    y_opt = ph[m_opt]

    g_all = sites_all[m_all]
    ns_all = min(5, np.unique(g_all).size)
    y_all = ph[m_all]

    print(f"pH valid (optics): n={m_opt.sum()} sites={np.unique(g_opt).size} "
          f"range=[{y_opt.min():.2f},{y_opt.max():.2f}] std={y_opt.std():.3f}")
    print(f"pH valid (optics+covar): n={m_all.sum()} sites={np.unique(g_all).size}")

    out = {
        "_PROVENANCE": {
            "purpose": "Honest diagnosis + improvement of pH from optics on REAL "
                       "Sentinel-2 x USGS data. Decomposes pH signal into optics-only "
                       "vs covariate-only (Temp+SpCond) vs combined. Bar = shipped "
                       "HydroViT ViT pH R2 0.3109 (same bands, station holdout).",
            "data": "data/processed/satellite/drone_wq_pairs.npz (REAL Sentinel-2 L2A "
                    "B02,B03,B04,B08 paired with REAL same-day USGS/NWIS in-situ WQ).",
            "bands": BANDS,
            "real_data_only": True,
            "augmentation_to_target": False,
            "label_leakage": False,
            "split": "STATION (site_id) GroupKFold — no site shared train/test; "
                     "pooled out-of-fold predictions.",
            "ci": "site-clustered bootstrap (resamples SITES, not rows), 2000 reps; "
                  "paired delta CI vs matched-set optics; scalar delta CI vs shipped bar.",
            "model": "HistGradientBoosting regressor (CPU), same family as the "
                     "red-edge chl-a study that beat the ViT.",
            "covariate_note": "Temp & SpCond features = REAL in-situ same-day covariates. "
                              "SpCond (ionic strength/alkalinity) is chemically tied to pH "
                              "but is NOT optical; any pH gain from Temp/SpCond is labeled "
                              "covariate-driven, NOT an optical claim.",
        },
        "shipped_baseline": {
            "hydrovit_pH_r2": HYDROVIT_PH_R2,
            "hydrovit_pH_mae": 0.3636,
            "note": "ViT-S/16 on same 4 bands + station holdout; the pure-optics comparator.",
        },
        "n_valid": {"optics": int(m_opt.sum()), "optics_covar": int(m_all.sum())},
        "arms": {},
    }

    # ---- ARM A: OPTICS-ONLY (full valid optics set) ----
    r2_A, ci_A, pA = evaluate(Xopt_all[m_opt], y_opt, g_opt, ns_opt)
    print(f"[A optics-only]    R2={r2_A:.4f} CI[{ci_A[0]:.4f},{ci_A[1]:.4f}]")

    # ---- ARM B: COVAR-ONLY (Temp+SpCond) on optics+covar valid set ----
    Xcov = np.column_stack([temp[m_all], spc[m_all]])
    r2_B, ci_B, pB = evaluate(Xcov, y_all, g_all, ns_all)
    print(f"[B covar-only]     R2={r2_B:.4f} CI[{ci_B[0]:.4f},{ci_B[1]:.4f}]")

    # ---- ARM C: OPTICS+COVAR ----
    Xct = np.column_stack([Xopt_all[m_all], temp[m_all], spc[m_all]])
    r2_C, ci_C, pC = evaluate(Xct, y_all, g_all, ns_all)
    print(f"[C optics+covar]   R2={r2_C:.4f} CI[{ci_C[0]:.4f},{ci_C[1]:.4f}]")

    # ---- ARM A': optics-only on the SAME optics+covar valid set (clean paired deltas) ----
    r2_Ap, ci_Ap, pAp = evaluate(Xopt_all[m_all], y_all, g_all, ns_all)
    print(f"[A' optics-only matched-set] R2={r2_Ap:.4f}")

    out["arms"]["A_optics_only"] = {
        "features": "4-band spatial stats + CDOM/chl/turbidity proxy indices (no covar)",
        "n": int(m_opt.sum()), "n_sites": int(np.unique(g_opt).size),
        "n_features": len(opt_names),
        "r2": round(r2_A, 4), "r2_ci95": [round(ci_A[0], 4), round(ci_A[1], 4)],
    }
    out["arms"]["B_covar_only"] = {
        "features": "in-situ Temp + SpCond only (2 features, non-optical chemistry)",
        "n": int(m_all.sum()), "n_sites": int(np.unique(g_all).size),
        "r2": round(r2_B, 4), "r2_ci95": [round(ci_B[0], 4), round(ci_B[1], 4)],
    }
    out["arms"]["C_optics_plus_covar"] = {
        "features": "optical proxies + in-situ Temp + SpCond",
        "n": int(m_all.sum()), "n_sites": int(np.unique(g_all).size),
        "r2": round(r2_C, 4), "r2_ci95": [round(ci_C[0], 4), round(ci_C[1], 4)],
    }
    out["arms"]["A_optics_only_matched_set"] = {
        "note": "optics-only evaluated on the SAME rows as B/C for fair paired deltas",
        "r2": round(r2_Ap, 4), "r2_ci95": [round(ci_Ap[0], 4), round(ci_Ap[1], 4)],
    }

    # ---- DELTAS ----
    out["deltas"] = {}

    # 1) PURE-OPTICS improvement vs shipped ViT bar (0.31). Bootstrap new R2 - bar.
    dm0, dlo0, dhi0, pgt0 = scalar_delta_ci(y_opt, pA, g_opt, HYDROVIT_PH_R2)
    out["deltas"]["optics_vs_hydrovit_shipped"] = {
        "new_r2": round(r2_A, 4), "old_r2": HYDROVIT_PH_R2,
        "delta_mean": round(dm0, 4), "delta_ci95": [round(dlo0, 4), round(dhi0, 4)],
        "prob_improvement": round(pgt0, 3),
        "interpretation": "pure-optics GBM vs pure-optics ViT, same bands & station "
                          "holdout. IMPROVED iff delta CI lower bound > 0.",
    }

    # 2) covariate marginal value: C vs A' (paired, same rows) -> how much is Temp/SpCond
    dm, dlo, dhi, pgt = paired_delta_ci(y_all, pAp, pC, g_all)
    out["deltas"]["covar_marginal_C_minus_Aoptics"] = {
        "mean": round(dm, 4), "ci95": [round(dlo, 4), round(dhi, 4)],
        "prob_improvement": round(pgt, 3),
        "interpretation": "R2 gain from ADDING Temp+SpCond to optics (paired). This "
                          "gain is COVARIATE-driven chemistry, NOT new optical signal.",
    }

    # 3) optics' marginal value beyond covariates: C vs B (paired) -> genuine optical add
    dm2, dlo2, dhi2, pgt2 = paired_delta_ci(y_all, pB, pC, g_all)
    out["deltas"]["optics_marginal_C_minus_covar"] = {
        "mean": round(dm2, 4), "ci95": [round(dlo2, 4), round(dhi2, 4)],
        "prob_improvement": round(pgt2, 3),
        "interpretation": "R2 gain from ADDING optics on top of Temp+SpCond (paired). "
                          "Isolates GENUINE optical pH information beyond chemistry.",
    }

    # 4) combined vs shipped ViT bar (covariate-informed headline, flagged)
    dmc, dloc, dhic, pgtc = scalar_delta_ci(y_all, pC, g_all, HYDROVIT_PH_R2, seed=SEED + 3)
    out["deltas"]["combined_vs_hydrovit_shipped"] = {
        "new_r2": round(r2_C, 4), "old_r2": HYDROVIT_PH_R2,
        "delta_mean": round(dmc, 4), "delta_ci95": [round(dloc, 4), round(dhic, 4)],
        "prob_improvement": round(pgtc, 3),
        "interpretation": "optics+Temp+SpCond vs shipped ViT. NOT a pure-optics claim; "
                          "any lift over optics-only is covariate-driven chemistry.",
    }

    # ---- VERDICTS ----
    verdict = {}
    optics_improved = bool(dlo0 > 0)
    if optics_improved:
        verdict["optics_vs_hydrovit"] = (
            f"IMPROVED (pure optics): optics-only R2={r2_A:.3f} vs shipped 0.311, "
            f"delta CI[{dlo0:+.3f},{dhi0:+.3f}] excludes 0. Genuine optical pH gain "
            f"(GBM CDOM/chl/turbidity features > ViT, same bands).")
    elif r2_A > HYDROVIT_PH_R2:
        verdict["optics_vs_hydrovit"] = (
            f"HIGHER POINT, CI touches: optics-only R2={r2_A:.3f} > 0.311 but delta "
            f"CI[{dlo0:+.3f},{dhi0:+.3f}] includes 0. Not conclusively improved.")
    else:
        verdict["optics_vs_hydrovit"] = (
            f"NOT IMPROVED (pure optics): optics-only R2={r2_A:.3f} <= 0.311. "
            f"pH is weakly optical; this GBM does not beat the ViT on optics alone.")

    if dlo2 > 0:
        verdict["optics_beyond_covar"] = (
            f"Optics adds REAL signal beyond Temp+SpCond: C-B delta={dm2:+.3f} "
            f"CI[{dlo2:+.3f},{dhi2:+.3f}] excludes 0.")
    else:
        verdict["optics_beyond_covar"] = (
            f"Optics adds LITTLE beyond Temp+SpCond: C-B delta={dm2:+.3f} "
            f"CI[{dlo2:+.3f},{dhi2:+.3f}] includes 0.")

    verdict["covariate_share"] = (
        f"Covar-only R2={r2_B:.3f} vs optics-only R2={r2_Ap:.3f} vs combined R2={r2_C:.3f}. "
        f"Adding Temp+SpCond to optics lifts R2 by {dm:+.3f} (CI[{dlo:+.3f},{dhi:+.3f}]). "
        "Any combined 'improvement' over optics is COVARIATE chemistry, not optical.")

    verdict["weak_optical_limit"] = (
        f"pH has no intrinsic color; optical recovery is capped by CDOM/chl/turbidity "
        f"proxies. Optics-only R2={r2_A:.3f} quantifies that weak-optical ceiling.")

    out["verdict"] = verdict
    out["improved"] = optics_improved
    out["improved_definition"] = (
        "PURE-OPTICS claim: optics-only (new_r2 - shipped ViT 0.3109) site-bootstrap "
        "delta CI lower bound > 0, on a station GroupKFold holdout. The optics+covar "
        "arm is reported but flagged covariate-driven, not an optical claim.")
    out["elapsed_seconds"] = round(time.time() - t0, 1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print("WROTE", OUT)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
