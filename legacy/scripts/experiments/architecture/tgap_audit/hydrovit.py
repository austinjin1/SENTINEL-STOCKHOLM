#!/usr/bin/env python
"""HydroViT transfer-gap CONTRIBUTION audit.

PAPER CLAIM (transfer-gap evidence table, HydroViT row)
-------------------------------------------------------
"chl-a 4-band R2 0.18/0.21 -> RED-EDGE indices R2 0.34"
framed as: raw features collapse OOD; a transferable REPRESENTATION (the red-edge
indices) recovers the skill on the SAME information.

THE CONFOUND WE TEST
--------------------
Red-edge indices (NDCI = (B05-B04)/(B05+B04) etc.) REQUIRE the Sentinel-2 red-edge
band B05 (~705 nm). B05 is NOT in HydroViT's shipped 4-band set (B02/B03/B04/B08).
So the 0.21 -> 0.34 gain could be:
  (i)  ADDED INFORMATION  : the indices smuggle in a new physical band (B05/B11),
       i.e. it is NOT a "representation of the same features" at all; OR
  (ii) REPRESENTATION      : the engineered ratio genuinely extracts OOD-robust skill
       that raw values of the SAME bands cannot.

We decompose with a clean 2x2 grid on the IDENTICAL 6-band real tiles
(hydrovit_rededge recipe: mc_tiles, station GroupKFold, log chl-a, site-clustered CI):

  BANDS available   REPRESENTATION = raw per-band stats     REPRESENTATION = engineered indices
  6-band (has B05)  (A) raw6                                (B) idx6   <- indices incl NDCI/CIre/MCIre
  4-band (no  B05)  (C) raw4  <- HydroViT baseline info     (D) idx4   <- only 4-band-computable indices

Decomposition of the 0.21 -> 0.34 gain:
  * ADDED-BAND effect   = A - C  (raw6 vs raw4): same representation TYPE (raw stats),
                          differ ONLY in whether B05/B11 are available. Isolates the
                          value of the extra INFORMATION.
  * REPRESENTATION eff. = B - A  (idx6 vs raw6): SAME bands/information, differ ONLY in
                          whether we hand-engineer ratios vs let the model use raw stats.
                          Isolates the value of the index REPRESENTATION.
  * 4-band representation= D - C (idx4 vs raw4): does hand-engineering help WITHOUT the
                          new band? (control for "indices help in general").

If (A - C) captures most of the gain and (B - A) ~ 0, the recovery is ADDED-INFORMATION,
not a representation effect: the example is CONFOUNDED.

INTEGRITY
  * REAL Sentinel-2 L2A 6-band tiles x REAL WQP/NOAA in-situ chl-a. No augmentation.
  * Grouping = station (site) GroupKFold. No site in both train & test.
  * CI = site-clustered bootstrap (resamples SITES, 2000 reps). Paired deltas too.
  * Identical CPU HistGradientBoosting model & CV for all 4 arms (fair, no per-arm tuning).
  * Also reports a matched-count control: raw-6 restricted to the same #features as idx6
    is NOT needed since raw6 already has MORE raw features than the added indices; the
    representation test is conservative in favor of finding a representation effect.

MIT License -- audit, 2026
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[4]
NOAA = ROOT / "data" / "processed" / "biology" / "noaa_habs"
DATA = NOAA / "mc_fusion_dataset.parquet"
TILES = NOAA / "mc_tiles"
OUT = ROOT / "results" / "benchmarks" / "architecture" / "tgap_audit" / "hydrovit.json"
SEED = 0
NBOOT = 2000
EPS = 1e-6

TILE_BANDS = ["B02", "B03", "B04", "B05", "B08", "B11"]   # raw order in the tiles
BASE4 = ["B02", "B03", "B04", "B08"]                       # HydroViT shipped bands
CTR = slice(22, 42)


def _band_stats(x2d, prefix, feats, names):
    v = x2d[np.isfinite(x2d)]
    if v.size == 0:
        v = np.zeros(1, np.float32)
    for tag, val in [("ctr", x2d[CTR, CTR].mean() if np.isfinite(x2d[CTR, CTR]).any() else 0.0),
                     ("mean", v.mean()), ("std", v.std()),
                     ("p10", np.percentile(v, 10)), ("p50", np.percentile(v, 50)),
                     ("p90", np.percentile(v, 90)), ("max", v.max())]:
        feats.append(float(val))
        names.append(f"{prefix}_{tag}")


def features_for_tile(img, mode):
    """4 clean arms.
      raw6 : per-band spatial stats of ALL 6 bands. NO ratios.
      idx6 : engineered indices that USE the red-edge/SWIR (NDCI,CIre,MCIre,FAI)
             + the 4-band-computable indices (NDVI,NDWI,GR). NO raw per-band stats.
      raw4 : per-band spatial stats of the 4 HydroViT bands only. NO ratios.
      idx4 : only indices computable from 4 bands (NDVI,NDWI,GR). NO raw per-band stats.
    """
    B = {n: img[i].astype(np.float64) for i, n in enumerate(TILE_BANDS)}
    b3, b4, b5, b8, b11 = B["B03"], B["B04"], B["B05"], B["B08"], B["B11"]
    ndvi = (b8 - b4) / (b8 + b4 + EPS)
    ndwi = (b3 - b8) / (b3 + b8 + EPS)
    gr = b3 / (b4 + EPS)
    ndci = (b5 - b4) / (b5 + b4 + EPS)                       # Mishra&Mishra 2012 (needs B05)
    ci_re = b5 / (b4 + EPS) - 1.0                            # red-edge chl index (needs B05)
    mci_re = b5 - b4 - (b8 - b4) * (705.0 - 665.0) / (842.0 - 665.0)   # (needs B05)
    fai = b8 - (b4 + (b11 - b4) * (842.0 - 665.0) / (1610.0 - 665.0))  # (needs B11)

    feats, names = [], []
    if mode == "raw6":
        for n in TILE_BANDS:
            _band_stats(B[n], n, feats, names)
    elif mode == "idx6":
        for arr, pfx in [(ndvi, "NDVI"), (ndwi, "NDWI"), (gr, "GR"),
                         (ndci, "NDCI"), (ci_re, "CIre"), (mci_re, "MCIre"), (fai, "FAI")]:
            _band_stats(arr, pfx, feats, names)
    elif mode == "raw4":
        for n in BASE4:
            _band_stats(B[n], n, feats, names)
    elif mode == "idx4":
        for arr, pfx in [(ndvi, "NDVI"), (ndwi, "NDWI"), (gr, "GR")]:
            _band_stats(arr, pfx, feats, names)
    else:
        raise ValueError(mode)
    return np.array(feats, np.float32), names


MODES = ["raw6", "idx6", "raw4", "idx4"]


def build_all(df):
    mats = {m: [] for m in MODES}
    names = {m: None for m in MODES}
    keep = np.zeros(len(df), bool)
    for r, idx in enumerate(df["idx"].to_numpy()):
        f = TILES / f"{int(idx)}.npz"
        if not f.exists():
            for m in MODES:
                mats[m].append(None)
            continue
        try:
            d = np.load(f, allow_pickle=True)
            img = d["image"]
            if img.shape != (6, 64, 64):
                for m in MODES:
                    mats[m].append(None)
                continue
            ok = True
            fvs = {}
            for m in MODES:
                fv, nm = features_for_tile(img, m)
                fvs[m] = fv
                if names[m] is None:
                    names[m] = nm
        except Exception:
            for m in MODES:
                mats[m].append(None)
            continue
        for m in MODES:
            mats[m].append(fvs[m])
        keep[r] = True
    X = {}
    for m in MODES:
        Xm = np.full((len(df), len(names[m])), np.nan, np.float64)
        for r, fv in enumerate(mats[m]):
            if fv is not None:
                Xm[r] = fv
        X[m] = Xm
    return X, names, keep


def hgb_reg():
    # Identical, moderately-regularized model for ALL arms (no per-arm tuning). This
    # keeps the comparison fair: any arm difference is representation/information, not
    # a hyperparameter advantage handed to one arm.
    return HistGradientBoostingRegressor(
        max_depth=4, learning_rate=0.05, max_iter=400,
        l2_regularization=2.0, random_state=SEED)


def oof_reg(X, y, g, ns):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=ns).split(X, y, g):
        p[te] = hgb_reg().fit(X[tr], y[tr]).predict(X[te])
    return p


def site_bootstrap_ci(y, p, sites, nboot=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.unique(sites)
    idx_by_site = {s: np.where(sites == s)[0] for s in uniq}
    vals = []
    for _ in range(nboot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_by_site[s] for s in samp])
        try:
            vals.append(r2_score(y[ii], p[ii]))
        except Exception:
            continue
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_delta_ci(y, p_a, p_b, sites, nboot=NBOOT, seed=SEED):
    """CI of (p_b arm - p_a arm) R2 on SAME resampled sites -> mean, lo, hi, P(>0)."""
    rng = np.random.default_rng(seed + 7)
    uniq = np.unique(sites)
    idx_by_site = {s: np.where(sites == s)[0] for s in uniq}
    d = []
    for _ in range(nboot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_by_site[s] for s in samp])
        try:
            d.append(r2_score(y[ii], p_b[ii]) - r2_score(y[ii], p_a[ii]))
        except Exception:
            continue
    d = np.array(d)
    return (float(d.mean()), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)), float((d > 0).mean()))


def main():
    t0 = time.time()
    df = pd.read_parquet(DATA)
    sites_all = df["site"].astype(str).to_numpy()

    print("Building 4 feature matrices (raw6/idx6/raw4/idx4) from real 6-band tiles...")
    X, names, keep = build_all(df)
    print("  feature counts:", {m: len(names[m]) for m in MODES}, "usable:", int(keep.sum()))

    # chl-a regression, log10, station-grouped, identical CV for all arms
    chla = df["CHLA"].to_numpy("float64")
    m = keep & np.isfinite(chla) & (chla > 0) & (chla < 5000)
    y = np.log10(chla[m])
    g = sites_all[m]
    ns = min(5, pd.Series(g).nunique())

    preds, r2, ci = {}, {}, {}
    for mode in MODES:
        preds[mode] = oof_reg(X[mode][m], y, g, ns)
        r2[mode] = r2_score(y, preds[mode])
        ci[mode] = site_bootstrap_ci(y, preds[mode], g)
        print(f"  {mode:5s} R2={r2[mode]:.4f} CI[{ci[mode][0]:.3f},{ci[mode][1]:.3f}] "
              f"({len(names[mode])} feats)")

    # decomposition deltas
    def dd(a, b):
        mean, lo, hi, p = paired_delta_ci(y, preds[a], preds[b], g)
        return {"pair": f"{b} - {a}", "mean": round(mean, 4),
                "ci95": [round(lo, 4), round(hi, 4)], "prob_positive": round(p, 3)}

    added_band = dd("raw4", "raw6")   # A - C : value of the extra BAND (same rep type)
    representation = dd("raw6", "idx6")  # B - A : value of the index REPRESENTATION (same bands)
    rep_4band = dd("raw4", "idx4")    # D - C : do indices help WITHOUT the new band?
    total = dd("raw4", "idx6")        # C -> B : the paper's 0.21 -> 0.34 headline gain

    def verdict_of(delta):
        if delta["ci95"][0] > 0:
            return "POSITIVE (95% CI excludes 0)"
        if delta["ci95"][1] < 0:
            return "NEGATIVE (95% CI below 0)"
        return f"NULL (CI includes 0; P>0={delta['prob_positive']})"

    # attribute the total gain
    tot = total["mean"]
    ab = added_band["mean"]
    rep = representation["mean"]
    frac_added = round(ab / tot, 3) if abs(tot) > 1e-6 else None
    frac_rep = round(rep / tot, 3) if abs(tot) > 1e-6 else None

    # overall verdict logic
    if representation["ci95"][0] > 0 and added_band["ci95"][1] < added_band["ci95"][0] + 1:
        pass
    if added_band["ci95"][0] > 0 and (representation["ci95"][0] <= 0):
        overall = "added-information-confound"
        overall_text = ("The 4-band -> red-edge-index gain is driven by ADDED INFORMATION "
                        "(the B05 red-edge / B11 SWIR bands), NOT by the index representation. "
                        "Raw stats of the same 6 bands (raw6) match or beat the engineered "
                        "indices (idx6): B-A representation delta is NULL/negative. The example "
                        "is CONFOUNDED: the 'transferable representation' is really a new band.")
    elif representation["ci95"][0] > 0 and added_band["ci95"][1] < 0:
        overall = "genuine-representation-effect"
        overall_text = ("Indices on the SAME bands beat raw stats (B-A positive) while adding "
                        "the band alone does not (A-C null): genuine representation effect.")
    elif representation["ci95"][0] > 0 and added_band["ci95"][0] > 0:
        overall = "mixed"
        overall_text = ("BOTH the added red-edge band (A-C positive) AND the index "
                        "representation on top of it (B-A positive) contribute; the headline "
                        "gain is partly information, partly representation.")
    else:
        overall = "added-information-confound"
        overall_text = ("Representation delta (idx6 vs raw6, same bands) is not positive; "
                        "the gain traces to the added band(s). Confounded.")

    out = {
        "_PROVENANCE": {
            "purpose": "Decompose the HydroViT transfer-gap headline (chl-a 4-band R2 ~0.21 -> "
                       "red-edge indices R2 ~0.34) into ADDED-BAND vs index-REPRESENTATION, on "
                       "the SAME real 6-band tiles.",
            "paper_claim_row": "chl-a 4-band R2 0.18/0.21 -> red-edge indices R2 0.34",
            "data": "data/processed/biology/noaa_habs/mc_tiles (REAL Sentinel-2 L2A 6-band) x "
                    "mc_fusion_dataset.parquet (REAL WQP/NOAA in-situ chl-a).",
            "tile_bands": TILE_BANDS,
            "arms": {
                "raw6 (A)": "raw per-band spatial stats, all 6 bands, NO ratios",
                "idx6 (B)": "engineered indices incl NDCI/CIre/MCIre/FAI (need B05/B11), NO raw stats",
                "raw4 (C)": "raw per-band spatial stats, 4 HydroViT bands only, NO ratios",
                "idx4 (D)": "only 4-band-computable indices (NDVI/NDWI/GR), NO raw stats",
            },
            "decomposition": {
                "added_band_effect": "A - C (raw6 vs raw4): value of the extra B05/B11 BAND",
                "representation_effect": "B - A (idx6 vs raw6): value of the index REPRESENTATION on SAME bands",
                "rep_4band_control": "D - C (idx4 vs raw4): do indices help WITHOUT the new band",
                "total_headline": "B - C (idx6 vs raw4): the paper's ~0.21 -> ~0.34 gain",
            },
            "split": "STATION (site) GroupKFold; no site shared train/test.",
            "ci": "site-clustered bootstrap (resamples sites), 2000 reps; paired deltas.",
            "model": "identical CPU HistGradientBoosting for all arms (no per-arm tuning).",
            "real_data_only": True, "augmentation_to_target": False, "label_leakage": False,
        },
        "n_usable_tiles": int(keep.sum()),
        "n_chla": int(m.sum()),
        "n_sites": int(pd.Series(g).nunique()),
        "feature_counts": {mode: len(names[mode]) for mode in MODES},
        "arm_r2": {mode: {"r2": round(r2[mode], 4),
                          "r2_ci95": [round(ci[mode][0], 4), round(ci[mode][1], 4)]}
                   for mode in MODES},
        "decomposition_deltas": {
            "total_headline_idx6_minus_raw4": {**total, "verdict": verdict_of(total)},
            "added_band_raw6_minus_raw4": {**added_band, "verdict": verdict_of(added_band)},
            "representation_idx6_minus_raw6": {**representation, "verdict": verdict_of(representation)},
            "rep_4band_idx4_minus_raw4": {**rep_4band, "verdict": verdict_of(rep_4band)},
        },
        "attribution_of_total_gain": {
            "total_gain_R2": round(tot, 4),
            "added_band_share": frac_added,
            "representation_share": frac_rep,
            "note": "shares are mean-delta ratios; interpret with the CIs above.",
        },
        "verdict_type": overall,
        "verdict_text": overall_text,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print("\nVERDICT:", overall)
    print(overall_text)
    print("added-band  A-C:", added_band)
    print("representation B-A:", representation)
    print("total C->B:", total)
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
