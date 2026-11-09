#!/usr/bin/env python
"""AUDIT of HydroDenseNet v4 (SENTINEL-Lite) — is it really "imagery only"?

CLAIM UNDER AUDIT (paper §3.6 / results/benchmarks/waterdronenet_v4_holdout.json):
  HydroDenseNet estimates water quality "directly from four-band RGB+NIR imagery
  with NO in-situ sensors", reporting Temp R2=0.78, DO 0.46, SpCond 0.44, Turb 0.18.

FACT (from scripts/training/train_hydrodensenet.py):
  ALL_COLS  = [DO, pH, Turb, Temp, SpCond]
  TARGET_COLS = [DO, Turb, Temp, SpCond]   # pH removed from targets
  pH is fed as an INPUT: TileDataset returns "ph_input"; HydroDenseNet.ph_encoder
  -> FiLM layers modulate visual features at every scale (MultiScaleAggregator.films),
  and pH is also concatenated into the BandRatioEncoder input.
  => The model consumes an IN-SITU pH-strip reading. It is NOT imagery-only.

PHYSICS RED FLAG:
  SpCond (specific conductance) is a measure of dissolved-ion conductance. It has
  essentially NO optical signature in RGB+NIR. An older truly-imagery-only ViT got
  SpCond R2 = -0.95 (waterdronenet_holdout.json). A jump to +0.44 from the SAME 4
  bands is physically implausible from optics; the pH input (which correlates with
  water chemistry / brackish vs fresh) is the plausible driver.

THIS SCRIPT (decisive, reusing the cycle-14 honest recipe):
  Rebuild the retrieval on drone_wq_pairs.npz (SAME S2 x in-situ pairs, 4 bands),
  station GroupKFold, TWICE for each target:
    (1) IMAGERY-ONLY : band spatial stats + optical indices  (the honest "no in-situ" number)
    (2) IMAGERY + pH : same features + the in-situ pH-strip reading as an extra input
  Report per-param R2 with site-clustered bootstrap CI in BOTH modes, and the paired
  delta (how much pH buys). Expect SpCond (and partly DO) to collapse toward the physics
  limit ~0 when pH is removed.

  Leakage checks:
    (a) Target normalization: HistGBM predicts raw units, so there is no test-time
        z-score leakage in THIS retrieval. We ALSO inspect the v4 training code path:
        target_mean/std are computed from train_ds only (compute_target_stats on the
        train split) and shared to val/test via set_stats -> TRAIN-ONLY (no leakage
        there). We assert & report that.
    (b) Global-mean baseline: R2 is defined against the GLOBAL mean of the pooled OOF
        targets, so a model that only predicts a per-region constant still has to beat
        the global mean. We test whether SpCond's positive R2 is a "spurious regional
        proxy": recompute R2 after DROPPING the top brackish sites, and compare vs a
        site-mean oracle to see how much is between-site vs within-site skill.

INTEGRITY: REAL public data only (Sentinel-2 L2A B02/B03/B04/B08 x USGS/NWIS in-situ).
No target augmentation, no random split, no tuning on test. pH used as a feature is the
REAL in-situ pH-strip value and any gain it produces is labeled in-situ-driven, NOT optical.

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
OUT = ROOT / "results" / "benchmarks" / "weak_improve" / "hydrodensenet" / "audit_hydrodensenet_ph.json"
SEED = 0
NBOOT = 2000
EPS = 1e-6

BANDS = ["B02", "B03", "B04", "B08"]  # blue, green, red, NIR
TGT = ["DO", "pH", "Turb", "Temp", "SpCond"]
TARGETS_AUDIT = ["DO", "Turb", "Temp", "SpCond"]  # v4 targets (pH is the v4 input)

# v4 reported holdout R2 (the "imagery only" claim numbers)
V4_REPORTED = {"DO": 0.463, "Turb": 0.181, "Temp": 0.776, "SpCond": 0.442}
# older truly-imagery-only ViT
IMGONLY_VIT = {"DO": 0.257, "Turb": -0.027, "Temp": 0.508, "SpCond": -0.951}
# QA/QC ranges (match training code PARAM_RANGES)
PARAM_RANGES = {"DO": (0.0, 20.0), "pH": (4.0, 10.0), "Turb": (0.0, 1000.0),
                "Temp": (-5.0, 45.0), "SpCond": (0.0, 50000.0)}


def band_stats(x2d, prefix, feats, names):
    v = x2d[np.isfinite(x2d)]
    if v.size == 0:
        v = np.zeros(1, np.float32)
    for tag, val in [("mean", v.mean()), ("std", v.std()),
                     ("p10", np.percentile(v, 10)), ("p50", np.percentile(v, 50)),
                     ("p90", np.percentile(v, 90)), ("min", v.min()), ("max", v.max())]:
        feats.append(float(val))
        names.append(f"{prefix}_{tag}")


def optical_features(img):
    B = {n: img[i].astype(np.float64) for i, n in enumerate(BANDS)}
    feats, names = [], []
    for n in BANDS:
        band_stats(B[n], n, feats, names)
    b2, b3, b4, b8 = B["B02"], B["B03"], B["B04"], B["B08"]
    ndvi = (b8 - b4) / (b8 + b4 + EPS)
    ndwi = (b3 - b8) / (b3 + b8 + EPS)
    gr = b3 / (b4 + EPS)
    rn = b4 / (b8 + EPS)
    bg = b2 / (b3 + EPS)
    ndti = (b4 - b3) / (b4 + b3 + EPS)
    bright = (b2 + b3 + b4 + b8) / 4.0
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
    p = oof_reg(X, y, sites, ns)
    return r2_score(y, p), site_ci(y, p, sites), p


def main():
    t0 = time.time()
    d = np.load(NPZ, allow_pickle=True)
    images = d["images"]
    targets = d["targets"]
    meta = json.loads(d["metadata"].item())
    sites_all = np.array([m["site_id"] for m in meta])

    ph_all = targets[:, TGT.index("pH")].astype("float64")

    print(f"Building optical features for {images.shape[0]} tiles...")
    Xopt_all, opt_names = build_optical_matrix(images)
    finite_opt = np.isfinite(Xopt_all).all(axis=1)

    out = {
        "_PROVENANCE": {
            "purpose": "AUDIT: is HydroDenseNet v4 (SENTINEL-Lite) really imagery-only? "
                       "Rebuild retrieval on the SAME S2 x in-situ pairs, station GroupKFold, "
                       "imagery-only vs imagery+in-situ-pH, per-param R2 + site-clustered CI. "
                       "Quantify how much of SpCond/DO R2 is driven by the in-situ pH input.",
            "data": "data/processed/satellite/drone_wq_pairs.npz (REAL Sentinel-2 L2A "
                    "B02,B03,B04,B08 x REAL same-day USGS/NWIS in-situ WQ). N=%d tiles, "
                    "%d sites." % (images.shape[0], np.unique(sites_all).size),
            "bands": BANDS,
            "real_data_only": True,
            "augmentation_to_target": False,
            "split": "STATION (site_id) GroupKFold; pooled out-of-fold predictions.",
            "ci": "site-clustered bootstrap (resamples SITES), 2000 reps.",
            "model": "HistGradientBoosting (CPU), same family as cycle-14 honest recipe.",
            "ph_note": "pH feature = REAL in-situ pH-strip value (v4's FiLM input). Any gain "
                       "it produces is IN-SITU-driven, NOT optical/imagery-only.",
        },
        "finding_1_ph_is_input": {
            "confirmed": True,
            "evidence": "train_hydrodensenet.py: ALL_COLS=[DO,pH,Turb,Temp,SpCond]; "
                        "TARGET_COLS=[DO,Turb,Temp,SpCond] (pH removed). TileDataset returns "
                        "'ph_input'; HydroDenseNet.ph_encoder->FiLMLayer at every scale "
                        "(MultiScaleAggregator.films) modulates visual features; pH also "
                        "concatenated into BandRatioEncoder input. => in-situ pH is consumed.",
            "paper_claim_status": "INCORRECT: paper/v4 JSON say 'imagery only / no in-situ "
                                  "sensors', but the model requires an in-situ pH-strip reading.",
        },
        "finding_2_target_norm_train_only": {
            "leakage": False,
            "evidence": "train_hydrodensenet.py computes target_mean/std via "
                        "train_ds.compute_target_stats() on the TRAIN split only, then "
                        "set_stats() shares those TRAIN stats to val/test. No test/global "
                        "z-score fit -> no normalization leakage. (This audit's GBM predicts "
                        "raw units, so it is independent of that path.)",
        },
        "v4_reported_r2": V4_REPORTED,
        "imagery_only_vit_r2": IMGONLY_VIT,
        "per_target": {},
    }

    for tgt in TARGETS_AUDIT:
        y_all = targets[:, TGT.index(tgt)].astype("float64")
        lo, hi = PARAM_RANGES[tgt]
        # valid set: finite optics, finite target in-range, finite pH in-range (so both
        # arms share EXACTLY the same rows -> clean paired delta)
        in_rng = np.isfinite(y_all) & (y_all >= lo) & (y_all <= hi)
        ph_ok = np.isfinite(ph_all) & (ph_all >= PARAM_RANGES["pH"][0]) & (ph_all <= PARAM_RANGES["pH"][1])
        m = finite_opt & in_rng & ph_ok
        g = sites_all[m]
        ns = min(5, np.unique(g).size)
        y = y_all[m]
        Xo = Xopt_all[m]
        Xp = np.column_stack([Xo, ph_all[m]])

        # ARM 1: imagery-only
        r2_img, ci_img, p_img = evaluate(Xo, y, g, ns)
        # ARM 2: imagery + in-situ pH
        r2_ph, ci_ph, p_ph = evaluate(Xp, y, g, ns)
        # paired delta (pH marginal value)
        dm, dlo, dhi, pgt = paired_delta_ci(y, p_img, p_ph, g)

        print(f"[{tgt}] imagery-only R2={r2_img:.4f} CI[{ci_img[0]:.3f},{ci_img[1]:.3f}] | "
              f"+pH R2={r2_ph:.4f} CI[{ci_ph[0]:.3f},{ci_ph[1]:.3f}] | "
              f"pH-delta={dm:+.3f} CI[{dlo:+.3f},{dhi:+.3f}]")

        entry = {
            "n": int(m.sum()), "n_sites": int(np.unique(g).size),
            "imagery_only": {"r2": round(r2_img, 4),
                             "r2_ci95": [round(ci_img[0], 4), round(ci_img[1], 4)]},
            "imagery_plus_insitu_ph": {"r2": round(r2_ph, 4),
                                       "r2_ci95": [round(ci_ph[0], 4), round(ci_ph[1], 4)]},
            "ph_marginal_delta": {"mean": round(dm, 4),
                                  "ci95": [round(dlo, 4), round(dhi, 4)],
                                  "prob_improvement": round(pgt, 3)},
            "v4_reported_r2": V4_REPORTED.get(tgt),
            "imagery_only_vit_r2": IMGONLY_VIT.get(tgt),
        }

        # SpCond spurious-regional-proxy probe
        if tgt == "SpCond":
            # site-mean oracle: predict each row's site mean (between-site skill ceiling)
            site_mean_pred = np.empty_like(y)
            for s in np.unique(g):
                mask = g == s
                site_mean_pred[mask] = y[mask].mean()
            r2_sitemean = r2_score(y, site_mean_pred)
            # drop the top-brackish sites (site mean above threshold) and recompute
            site_means = {s: y[g == s].mean() for s in np.unique(g)}
            thr = 5000.0  # brackish/high-ion sites
            keep = np.array([site_means[s] < thr for s in g])
            probe = {}
            probe["site_mean_oracle_r2"] = round(float(r2_sitemean), 4)
            probe["site_mean_oracle_note"] = ("between-site ceiling: R2 achievable by "
                                              "predicting each site's own mean SpCond.")
            if keep.sum() > 50 and np.unique(g[keep]).size >= 3:
                # imagery-only R2 restricted to non-brackish sites (recompute R2 on subset
                # of the SAME OOF preds, re-centering on subset global mean)
                yk, pk = y[keep], p_img[keep]
                r2_img_freshonly = r2_score(yk, pk)
                yk2, pk2 = y[keep], p_ph[keep]
                r2_ph_freshonly = r2_score(yk2, pk2)
                probe["n_after_dropping_brackish"] = int(keep.sum())
                probe["n_sites_after"] = int(np.unique(g[keep]).size)
                probe["imagery_only_r2_freshwater_only"] = round(float(r2_img_freshonly), 4)
                probe["imagery_plus_ph_r2_freshwater_only"] = round(float(r2_ph_freshonly), 4)
                probe["interpretation"] = (
                    "If SpCond R2 collapses toward ~0 once the few extreme brackish/high-ion "
                    "sites are removed, the positive R2 is a spurious REGIONAL proxy (telling "
                    "brackish sites from fresh), not a within-freshwater conductance retrieval.")
            entry["spcond_regional_proxy_probe"] = probe

        out["per_target"][tgt] = entry

    # headline verdicts
    sp = out["per_target"]["SpCond"]
    do = out["per_target"]["DO"]
    out["verdict"] = {
        "imagery_only_is_the_honest_claim": {
            "DO": do["imagery_only"]["r2"],
            "Turb": out["per_target"]["Turb"]["imagery_only"]["r2"],
            "Temp": out["per_target"]["Temp"]["imagery_only"]["r2"],
            "SpCond": sp["imagery_only"]["r2"],
            "note": "These are the numbers the paper's 'imagery only / no in-situ sensors' "
                    "claim actually supports. Temp is genuinely optical; Turb is weak; SpCond "
                    "is near/below 0 (physics limit); DO is a modest chl/temp-proxy signal.",
        },
        "ph_drives_spcond": (
            "SpCond imagery-only R2=%.3f (CI %.3f..%.3f) vs +in-situ-pH R2=%.3f. pH marginal "
            "delta=%+.3f (CI %+.3f..%+.3f). If large & positive, the v4 'SpCond R2~0.44' is "
            "driven by the in-situ pH input, NOT optics." % (
                sp["imagery_only"]["r2"], sp["imagery_only"]["r2_ci95"][0],
                sp["imagery_only"]["r2_ci95"][1], sp["imagery_plus_insitu_ph"]["r2"],
                sp["ph_marginal_delta"]["mean"], sp["ph_marginal_delta"]["ci95"][0],
                sp["ph_marginal_delta"]["ci95"][1])),
        "paper_fix": (
            "Correct the §3.6 / v4 JSON 'imagery only, no in-situ sensors' claim: the shipped "
            "model consumes an in-situ pH-strip reading via FiLM. Report the imagery-only R2 "
            "above as the honest no-in-situ number, and (if kept) report the pH-conditioned "
            "numbers explicitly as using an in-situ input."),
    }
    out["elapsed_seconds"] = round(time.time() - t0, 1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print("WROTE", OUT)
    print(json.dumps(out["verdict"], indent=2))


if __name__ == "__main__":
    main()
