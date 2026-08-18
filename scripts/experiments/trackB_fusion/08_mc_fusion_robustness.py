#!/usr/bin/env python
"""Track B 8 — ROBUSTNESS STRESS-TEST of the microcystin fusion win.

The verified win (07_microcystin_fusion.py): satellite spectral (sf*) + in-situ
pigment (CHLA,PC) fusion beats the best single modality for microcystin at WHO
thresholds >1.6 (drinking) and >0.3 (detection), with a site-clustered bootstrap
95% CI on the paired AUROC delta that EXCLUDES 0.

This script asks: is that win real under HARDER protocols, or fragile? It re-runs
the fusion>best-single test (thr>1.6, thr>0.3) under five stress protocols:

  (a) TEMPORAL holdout   — train on earlier dates, test on later (out-of-time).
  (b) LEAVE-ONE-REGION-OUT — cluster sites into geographic regions (by state code
      derived from WQX site prefix, the honest geographic label available), hold
      each out in turn (cross-region generalization).
  (c) CLOUD-FRACTION filter — drop high-cloud tiles (cloud>median, cloud>25),
      re-run the full station-grouped OOF test (does clean-tile data still win?).
  (d) MORE SEEDS (10) + LARGER site bootstrap (5000) — is CI-excludes-0 stable
      across model random states and bootstrap resampling?
  (e) NEGATIVE CONTROL — replace in-situ pigment with random noise; fusion MUST
      collapse to satellite-alone (delta ~0, CI includes 0). If it still "wins",
      the pipeline is leaking / the win is an artifact.

Ship-gate throughout: fusion AUROC must exceed the BEST single modality with a
site-clustered bootstrap 95% CI on the paired delta that excludes 0.
Everything uses REAL data, station-grouped CV, non-circular microcystin labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "processed" / "biology" / "noaa_habs" / "mc_fusion_dataset.parquet"
OUT = ROOT / "results" / "trackB_fusion" / "mc_fusion_robustness.json"

THRESHOLDS = {"mc_gt1.6_WHO_drinking": 1.6, "mc_gt0.3_detection": 0.3}
NBOOT = 5000
SEEDS = list(range(10))


def hgb(seed=0):
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                          l2_regularization=1.0, class_weight="balanced",
                                          random_state=seed)


def oof(X, y, g, ns, seed=0):
    """Station-grouped out-of-fold predictions (no site appears in train+test)."""
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=ns).split(X, y, g):
        p[te] = hgb(seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def fit_predict(Xtr, ytr, Xte, seed=0):
    """Fit on a train block, predict a disjoint test block (temporal / region)."""
    return hgb(seed).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def site_bootstrap_delta(y, p_fusion, p_best, sites, seed=0, B=NBOOT):
    """Paired site-clustered bootstrap on AUROC(fusion) - AUROC(best single)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(sites)
    by_site = {s: np.where(sites == s)[0] for s in uniq}
    deltas = []
    for _ in range(B):
        chosen = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([by_site[s] for s in chosen])
        yi = y[idx]
        if yi.min() == yi.max():
            continue
        deltas.append(roc_auc_score(yi, p_fusion[idx]) - roc_auc_score(yi, p_best[idx]))
    deltas = np.array(deltas)
    if len(deltas) == 0:
        return {"delta_median": None, "ci95": [None, None], "P_delta_gt0": None, "n_valid": 0}
    return {"delta_median": round(float(np.median(deltas)), 4),
            "ci95": [round(float(np.percentile(deltas, 2.5)), 4),
                     round(float(np.percentile(deltas, 97.5)), 4)],
            "P_delta_gt0": round(float((deltas > 0).mean()), 4),
            "n_valid": int(len(deltas))}


def region_of(site: str) -> str:
    """Geographic region from WQX site-code prefix (state / tribal-nation locale).

    These are genuinely distant water bodies: Florida lakes, California
    reservoirs/tribal rivers, Michigan/Great-Lakes, and a few AZ/IA/MN sites.
    Small states are merged into one 'other' cluster so every held-out region
    has enough test rows.
    """
    s = str(site)
    if s.startswith("21FL"):
        return "FL"
    if (s.startswith("CA_") or s.startswith("POMO") or s.startswith("YUROK")
            or s.startswith("KARUK") or s.startswith("HVTEP")):
        return "CA"
    if s.startswith("21MICH"):
        return "MI"
    return "other"  # 21IOWA, AZDEQ, MNPCA — small distinct states pooled


def get_arrays(df):
    sf = [c for c in df.columns if c.startswith("sf")]
    sat = df[sf].to_numpy("float64")
    insitu = df[["CHLA", "PC"]].to_numpy("float64")
    fusion = np.hstack([sat, insitu])
    sites = df["site"].astype(str).to_numpy()
    mc = df["mc"].to_numpy("float64")
    return sat, insitu, fusion, sites, mc


def eval_block(a_fus, a_sat, a_ins, boot):
    best_name, best_a = ("insitu", a_ins) if a_ins >= a_sat else ("satellite", a_sat)
    return {"satellite_auroc": round(a_sat, 4), "insitu_auroc": round(a_ins, 4),
            "fusion_auroc": round(a_fus, 4), "best_single": best_name,
            "fusion_minus_best": round(a_fus - best_a, 4),
            "site_bootstrap_delta": boot,
            "fusion_beats_best_ci_excludes_0": bool(boot["ci95"][0] is not None
                                                    and boot["ci95"][0] > 0)}


# ----------------------------------------------------------------------------
# (a) TEMPORAL holdout
# ----------------------------------------------------------------------------
def stress_temporal(df, seed=0):
    df = df.sort_values("date").reset_index(drop=True)
    cut = df["date"].quantile(0.7)  # earlier 70% train, later 30% test
    tr_mask = (df["date"] <= cut).to_numpy()
    te_mask = ~tr_mask
    sat, insitu, fusion, sites, mc = get_arrays(df)
    out = {"train_cutoff_date": str(pd.Timestamp(cut).date()),
           "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()), "labels": {}}
    for name, thr in THRESHOLDS.items():
        y = (mc > thr).astype(int)
        ytr, yte = y[tr_mask], y[te_mask]
        if yte.min() == yte.max() or ytr.sum() < 15:
            out["labels"][name] = {"skipped": "degenerate train/test labels"}
            continue
        p_sat = fit_predict(sat[tr_mask], ytr, sat[te_mask], seed)
        p_ins = fit_predict(insitu[tr_mask], ytr, insitu[te_mask], seed)
        p_fus = fit_predict(fusion[tr_mask], ytr, fusion[te_mask], seed)
        a_sat, a_ins, a_fus = (float(roc_auc_score(yte, p)) for p in (p_sat, p_ins, p_fus))
        p_best = p_ins if a_ins >= a_sat else p_sat
        boot = site_bootstrap_delta(yte, p_fus, p_best, sites[te_mask], seed=seed)
        out["labels"][name] = {"pos_test": int(yte.sum()), **eval_block(a_fus, a_sat, a_ins, boot)}
    return out


# ----------------------------------------------------------------------------
# (b) LEAVE-ONE-REGION-OUT
# ----------------------------------------------------------------------------
def stress_loro(df, seed=0):
    df = df.reset_index(drop=True)
    df["region"] = df["site"].map(region_of)
    sat, insitu, fusion, sites, mc = get_arrays(df)
    regions = sorted(df["region"].unique())
    out = {"regions": {r: {"rows": int((df.region == r).sum()),
                           "sites": int(df.loc[df.region == r, "site"].nunique())}
                       for r in regions}, "labels": {}}
    for name, thr in THRESHOLDS.items():
        y = (mc > thr).astype(int)
        held = {}
        for r in regions:
            te = (df["region"] == r).to_numpy()
            tr = ~te
            yte, ytr = y[te], y[tr]
            if yte.min() == yte.max() or ytr.sum() < 15 or int(yte.sum()) < 5:
                held[r] = {"skipped": "too few pos in held-out region", "pos_test": int(yte.sum())}
                continue
            p_sat = fit_predict(sat[tr], ytr, sat[te], seed)
            p_ins = fit_predict(insitu[tr], ytr, insitu[te], seed)
            p_fus = fit_predict(fusion[tr], ytr, fusion[te], seed)
            a_sat, a_ins, a_fus = (float(roc_auc_score(yte, p)) for p in (p_sat, p_ins, p_fus))
            p_best = p_ins if a_ins >= a_sat else p_sat
            # bootstrap over sites WITHIN the held-out region
            boot = site_bootstrap_delta(yte, p_fus, p_best, sites[te], seed=seed)
            held[r] = {"pos_test": int(yte.sum()), "n_test": int(te.sum()),
                       **eval_block(a_fus, a_sat, a_ins, boot)}
        # pooled across all held-out folds (each row appears once as OOR test)
        deltas_pos = sum(1 for r in regions if isinstance(held[r], dict)
                         and held[r].get("fusion_beats_best_ci_excludes_0"))
        evaluable = [r for r in regions if "skipped" not in held[r]]
        out["labels"][name] = {"per_region": held,
                               "n_regions_evaluated": len(evaluable),
                               "n_regions_fusion_wins_ci": deltas_pos}
    return out


# ----------------------------------------------------------------------------
# (c) CLOUD-FRACTION filter
# ----------------------------------------------------------------------------
def stress_cloud(df, seed=0):
    out = {"labels": {}}
    filters = {"cloud_le_median": df["cloud"].median(), "cloud_le_25": 25.0, "cloud_le_10": 10.0}
    for fname, thr_cloud in filters.items():
        sub = df[df["cloud"] <= thr_cloud].reset_index(drop=True)
        sat, insitu, fusion, sites, mc = get_arrays(sub)
        ns = min(5, sub["site"].nunique())
        block = {"cloud_max": round(float(thr_cloud), 2), "n_rows": int(len(sub)),
                 "n_sites": int(sub["site"].nunique()), "labels": {}}
        for name, thr in THRESHOLDS.items():
            y = (mc > thr).astype(int)
            if y.sum() < 15 or (len(y) - y.sum()) < 15:
                block["labels"][name] = {"skipped": "too few pos/neg", "pos": int(y.sum())}
                continue
            p_sat = oof(sat, y, sites, ns, seed)
            p_ins = oof(insitu, y, sites, ns, seed)
            p_fus = oof(fusion, y, sites, ns, seed)
            a_sat, a_ins, a_fus = (float(roc_auc_score(y, p)) for p in (p_sat, p_ins, p_fus))
            p_best = p_ins if a_ins >= a_sat else p_sat
            boot = site_bootstrap_delta(y, p_fus, p_best, sites, seed=seed)
            block["labels"][name] = {"pos": int(y.sum()), **eval_block(a_fus, a_sat, a_ins, boot)}
        out["labels"][fname] = block
    return out


# ----------------------------------------------------------------------------
# (d) MORE SEEDS (10) + LARGER site bootstrap (5000)
# ----------------------------------------------------------------------------
def stress_seeds(df, seeds=SEEDS):
    df = df.reset_index(drop=True)
    sat, insitu, fusion, sites, mc = get_arrays(df)
    ns = min(5, df["site"].nunique())
    out = {"n_seeds": len(seeds), "nboot": NBOOT, "labels": {}}
    for name, thr in THRESHOLDS.items():
        y = (mc > thr).astype(int)
        per_seed = []
        for seed in seeds:
            p_sat = oof(sat, y, sites, ns, seed)
            p_ins = oof(insitu, y, sites, ns, seed)
            p_fus = oof(fusion, y, sites, ns, seed)
            a_sat, a_ins, a_fus = (float(roc_auc_score(y, p)) for p in (p_sat, p_ins, p_fus))
            p_best = p_ins if a_ins >= a_sat else p_sat
            # bootstrap seed offset so resampling differs across model seeds
            boot = site_bootstrap_delta(y, p_fus, p_best, sites, seed=1000 + seed, B=NBOOT)
            per_seed.append({"seed": seed, "fusion_auroc": round(a_fus, 4),
                             "best_single": "insitu" if a_ins >= a_sat else "satellite",
                             "fusion_minus_best": round(a_fus - max(a_sat, a_ins), 4),
                             "ci95": boot["ci95"], "P_delta_gt0": boot["P_delta_gt0"],
                             "ci_excludes_0": bool(boot["ci95"][0] > 0)})
        deltas = [s["fusion_minus_best"] for s in per_seed]
        lows = [s["ci95"][0] for s in per_seed]
        n_win = sum(s["ci_excludes_0"] for s in per_seed)
        out["labels"][name] = {
            "per_seed": per_seed,
            "delta_mean": round(float(np.mean(deltas)), 4),
            "delta_min": round(float(np.min(deltas)), 4),
            "delta_max": round(float(np.max(deltas)), 4),
            "ci_low_min": round(float(np.min(lows)), 4),
            "ci_low_max": round(float(np.max(lows)), 4),
            "n_seeds_ci_excludes_0": int(n_win),
            "all_seeds_win": bool(n_win == len(seeds))}
    return out


# ----------------------------------------------------------------------------
# (e) NEGATIVE CONTROL — replace in-situ pigment with random noise
# ----------------------------------------------------------------------------
def stress_negctrl(df, seed=0):
    df = df.reset_index(drop=True)
    sat, insitu, _, sites, mc = get_arrays(df)
    rng = np.random.default_rng(seed)
    # random noise standardized to same shape as the pigment block (2 cols)
    noise = rng.standard_normal(insitu.shape)
    fusion_noise = np.hstack([sat, noise])
    ns = min(5, df["site"].nunique())
    out = {"note": "in-situ pigment replaced by Gaussian noise; fusion should "
                   "collapse to satellite-alone (delta ~0, CI includes 0).",
           "labels": {}}
    for name, thr in THRESHOLDS.items():
        y = (mc > thr).astype(int)
        p_sat = oof(sat, y, sites, ns, seed)
        p_fus = oof(fusion_noise, y, sites, ns, seed)
        a_sat = float(roc_auc_score(y, p_sat))
        a_fus = float(roc_auc_score(y, p_fus))
        # here "best single" is satellite (noise has no signal); delta = fusion - sat
        boot = site_bootstrap_delta(y, p_fus, p_sat, sites, seed=seed)
        out["labels"][name] = {
            "satellite_auroc": round(a_sat, 4),
            "fusion_with_noise_auroc": round(a_fus, 4),
            "fusion_minus_satellite": round(a_fus - a_sat, 4),
            "site_bootstrap_delta": boot,
            "collapses_as_expected": bool(boot["ci95"][0] is not None and boot["ci95"][0] <= 0)}
    return out


def main():
    df = pq.read_table(DATA).to_pandas()
    print(f"Loaded {len(df)} rows, {df['site'].nunique()} sites", flush=True)

    results = {"dataset": str(DATA.name), "n": int(len(df)),
               "n_sites": int(df["site"].nunique()),
               "note": "Robustness stress-test of the microcystin fusion win "
                       "(satellite spectral + in-situ pigment). Ship-gate: fusion "
                       "beats best single with site-clustered bootstrap 95% CI on "
                       "paired AUROC delta excluding 0. REAL data, station-grouped "
                       "CV, non-circular microcystin thresholds.",
               "baseline_reference": "results/trackB_fusion/mc_fusion.json"}

    print("(a) temporal holdout ...", flush=True)
    results["a_temporal_holdout"] = stress_temporal(df)
    print("(b) leave-one-region-out ...", flush=True)
    results["b_leave_one_region_out"] = stress_loro(df)
    print("(c) cloud-fraction filter ...", flush=True)
    results["c_cloud_filter"] = stress_cloud(df)
    print("(d) 10 seeds + 5000 bootstrap ...", flush=True)
    results["d_seeds_and_bootstrap"] = stress_seeds(df)
    print("(e) negative control (noise pigment) ...", flush=True)
    results["e_negative_control"] = stress_negctrl(df)

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWROTE {OUT}", flush=True)


if __name__ == "__main__":
    main()
