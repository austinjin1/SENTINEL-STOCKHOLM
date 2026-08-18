#!/usr/bin/env python
"""Diagnostic 3 (fusion cycle 2) — per-modality marginal contribution on the
microcystin 3-way task (satellite spectral + in-situ pigment + physicochemical
drivers).

For thr mc>1.6 and mc>0.3, with GroupKFold-BY-SITE OOF predictions and a
site-clustered bootstrap (resample sites w/ replacement) for 95% CIs, we compute
for EACH of the 3 modalities:
  - single-modality AUROC (+ site-clustered CI on the absolute AUROC);
  - LEAVE-ONE-MODALITY-OUT delta: AUROC(all 3) - AUROC(the other 2), with a
    site-clustered CI on the paired delta (does dropping this modality hurt?);
  - ADD-LAST / Shapley marginal: mean over all coalition orderings of the AUROC
    gain from adding this modality LAST to the coalition already present, with a
    site-clustered CI on the per-bootstrap Shapley value.

A modality GENUINELY contributes if its Shapley (or LOMO) CI EXCLUDES 0;
otherwise it is redundant given the others. Real data only; label = real
microcystin lab threshold (non-circular). All model fitting is per train-fold.

Reuses the WQP driver join + HGB + bootstrap protocol from 08_driver_arm.py.
"""
from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data" / "processed" / "biology" / "noaa_habs" / "mc_fusion_dataset.parquet"
RAW = ROOT / "data" / "raw" / "wqp_drivers"
OUT = ROOT / "results" / "trackB_fusion" / "fusion_cycle2" / "modality_contribution.json"
CHARS = {"temp": "Temperature, water", "phosphorus": "Phosphorus", "nitrogen": "Nitrogen"}
MODS = ["satellite", "pigment", "drivers"]
SEED = 0
B = 2000


# ---- driver join (identical protocol to 08_driver_arm.py) --------------------
def parse_driver(df):
    if len(df) == 0:
        return pd.DataFrame(columns=["site", "date", "val"])
    out = pd.DataFrame({
        "site": df["MonitoringLocationIdentifier"].astype(str),
        "date": pd.to_datetime(df["ActivityStartDate"], errors="coerce"),
        "val": pd.to_numeric(df["ResultMeasureValue"], errors="coerce"),
    }).dropna(subset=["date", "val"])
    return out.sort_values("date")


def nearest(driver, site, date, tol_days=14):
    d = driver[driver["site"] == site]
    if len(d) == 0:
        return np.nan
    dd = (d["date"] - date).abs().dt.days
    i = dd.values.argmin()
    return float(d["val"].values[i]) if dd.values[i] <= tol_days else np.nan


def hgb():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                          l2_regularization=1.0, class_weight="balanced",
                                          random_state=SEED)


def oof(X, y, g, ns):
    """GroupKFold-by-site OOF probabilities; model fit on train folds only."""
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=ns).split(X, y, g):
        p[te] = hgb().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def site_boot_indices(sites):
    """Precompute one set of B site-resampled index arrays (shared across metrics
    so paired deltas use matched resamples)."""
    rng = np.random.default_rng(SEED)
    uniq = np.unique(sites)
    by = {s: np.where(sites == s)[0] for s in uniq}
    out = []
    for _ in range(B):
        out.append(np.concatenate([by[s] for s in rng.choice(uniq, len(uniq), replace=True)]))
    return out


def ci_from(vals):
    vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)])
    if len(vals) == 0:
        return {"median": None, "ci95": [None, None], "P_gt0": None, "n_boot": 0}
    return {"median": round(float(np.median(vals)), 4),
            "ci95": [round(float(np.percentile(vals, 2.5)), 4),
                     round(float(np.percentile(vals, 97.5)), 4)],
            "P_gt0": round(float((vals > 0).mean()), 4),
            "n_boot": int(len(vals))}


def auc_safe(y, p, idx):
    yi = y[idx]
    if yi.min() == yi.max():
        return None
    return roc_auc_score(yi, p[idx])


def main():
    df = pq.read_table(DATA).to_pandas()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    sites_list = sorted(df["site"].astype(str).unique())

    # join drivers from cache (temp / phosphorus / nitrogen), nearest within +/-14d
    for tag in CHARS:
        cache = RAW / f"{tag}.csv"
        dv = parse_driver(pd.read_csv(cache, low_memory=False))
        df[f"drv_{tag}"] = [nearest(dv, str(s), d) for s, d in
                            zip(df["site"].astype(str), df["date"])]

    sf = [c for c in df.columns if c.startswith("sf")]
    feats = {
        "satellite": df[sf].to_numpy("float64"),
        "pigment": df[["CHLA", "PC"]].to_numpy("float64"),
        "drivers": df[[f"drv_{t}" for t in CHARS]].to_numpy("float64"),
    }
    mc = df["mc"].to_numpy("float64")
    site_arr = df["site"].astype(str).to_numpy()
    ns = min(5, len(sites_list))
    boot_idx = site_boot_indices(site_arr)

    results = {
        "n": int(len(df)), "n_sites": len(sites_list),
        "modalities": {"satellite": {"features": sf, "k": len(sf)},
                       "pigment": {"features": ["CHLA", "PC"], "k": 2},
                       "drivers": {"features": [f"drv_{t}" for t in CHARS], "k": len(CHARS)}},
        "driver_coverage": {t: round(float(np.isfinite(df[f"drv_{t}"]).mean()), 3) for t in CHARS},
        "protocol": {"split": "GroupKFold by site", "n_splits": ns,
                     "bootstrap": "site-clustered (resample sites w/ replacement)",
                     "B": B, "label": "real microcystin lab threshold (non-circular)"},
        "method": "single AUROC, leave-one-modality-out (LOMO) delta, and exact "
                  "Shapley add-last marginal over all 3! coalition orders; "
                  "site-clustered 95% CIs on all deltas.",
        "labels": {},
    }

    for label, thr in [("mc_gt1.6", 1.6), ("mc_gt0.3", 0.3)]:
        y = (mc > thr).astype(int)

        # --- OOF predictions for every non-empty subset of modalities ---
        subsets = {}  # frozenset(mods) -> oof prob
        from itertools import combinations
        all_subsets = []
        for r in range(1, len(MODS) + 1):
            for combo in combinations(MODS, r):
                all_subsets.append(frozenset(combo))
        for s in all_subsets:
            X = np.hstack([feats[m] for m in MODS if m in s])
            subsets[s] = oof(X, y, site_arr, ns)

        full = frozenset(MODS)
        res = {"pos": int(y.sum()), "prevalence": round(float(y.mean()), 4)}

        # point AUROCs
        res["single_auroc"] = {m: round(float(roc_auc_score(y, subsets[frozenset([m])])), 4) for m in MODS}
        res["full_3way_auroc"] = round(float(roc_auc_score(y, subsets[full])), 4)

        # per-bootstrap AUROC for each subset (matched resamples)
        subset_boot = {s: np.array([auc_safe(y, subsets[s], idx) for idx in boot_idx], dtype=object)
                       for s in all_subsets}

        # --- per-modality single absolute CI ---
        res["single_auroc_ci"] = {}
        for m in MODS:
            res["single_auroc_ci"][m] = ci_from(subset_boot[frozenset([m])])

        # --- LEAVE-ONE-MODALITY-OUT: full - (others) ---
        res["lomo_delta"] = {}
        for m in MODS:
            others = full - frozenset([m])
            d = [(a - b) if (a is not None and b is not None) else None
                 for a, b in zip(subset_boot[full], subset_boot[others])]
            point = res["full_3way_auroc"] - round(float(roc_auc_score(y, subsets[others])), 4)
            entry = ci_from(d)
            entry["point"] = round(float(point), 4)
            entry["auroc_without"] = round(float(roc_auc_score(y, subsets[others])), 4)
            res["lomo_delta"][m] = entry

        # --- SHAPLEY add-last marginal (exact over all orderings) ---
        # For each ordering pi, marginal(m) = AUROC(coalition before m + m) - AUROC(coalition before m).
        # Empty coalition AUROC := 0.5 (chance). Average over the 3! = 6 orders, per bootstrap.
        orders = list(permutations(MODS))

        def shapley_for_boot(auc_lookup):
            # auc_lookup: dict subset(frozenset)->auc (float or None); empty->0.5
            def A(s):
                if len(s) == 0:
                    return 0.5
                v = auc_lookup[s]
                return v
            phi = {m: [] for m in MODS}
            ok = True
            for order in orders:
                coalition = frozenset()
                for m in order:
                    prev = A(coalition)
                    nxt = A(coalition | {m})
                    if prev is None or nxt is None:
                        ok = False
                        break
                    phi[m].append(nxt - prev)
                    coalition = coalition | {m}
                if not ok:
                    break
            if not ok:
                return None
            return {m: float(np.mean(phi[m])) for m in MODS}

        # point Shapley (from point AUROCs)
        point_lookup = {s: float(roc_auc_score(y, subsets[s])) for s in all_subsets}
        point_shap = shapley_for_boot(point_lookup)

        # bootstrap Shapley
        shap_boot = {m: [] for m in MODS}
        for bi in range(B):
            lookup = {s: subset_boot[s][bi] for s in all_subsets}
            sv = shapley_for_boot(lookup)
            if sv is None:
                continue
            for m in MODS:
                shap_boot[m].append(sv[m])

        res["shapley"] = {}
        for m in MODS:
            entry = ci_from(shap_boot[m])
            entry["point"] = round(float(point_shap[m]), 4)
            res["shapley"][m] = entry

        # verdict per modality: genuine if Shapley CI excludes 0 (lower bound > 0)
        res["verdict"] = {}
        for m in MODS:
            sh = res["shapley"][m]["ci95"]
            lo = res["lomo_delta"][m]["ci95"]
            genuine_shap = sh[0] is not None and sh[0] > 0
            genuine_lomo = lo[0] is not None and lo[0] > 0
            res["verdict"][m] = {
                "shapley_ci_excludes_0": bool(genuine_shap),
                "lomo_ci_excludes_0": bool(genuine_lomo),
                "genuine_contributor": bool(genuine_shap),  # primary criterion
            }

        results["labels"][label] = res
        print(f"\n[{label}] pos={res['pos']} full3way={res['full_3way_auroc']}")
        for m in MODS:
            print(f"  {m:10s} single={res['single_auroc'][m]:.3f} "
                  f"shap={res['shapley'][m]['point']:+.3f} CI{res['shapley'][m]['ci95']} "
                  f"lomo={res['lomo_delta'][m]['point']:+.3f} CI{res['lomo_delta'][m]['ci95']} "
                  f"=> genuine={res['verdict'][m]['genuine_contributor']}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()
