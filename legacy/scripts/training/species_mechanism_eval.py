#!/usr/bin/env python3
"""Step 3: does REAL environment carry occupancy MECHANISM beyond a geographic
range prior?

The honest leakage-free baseline (results/benchmarks/species_health_honest.json)
found that WITH geography the occupancy macro-AUROC is ~0.94, but env-only
(no lat/lon/HUC, using only watershed drainage + NON-target community richness +
sampling effort) is ~0.85 -- and the rare species' skill is mostly a lat/lon
RANGE PRIOR, not environmental mechanism. The one honest bioindicator is the
WIDESPREAD Unionid mussel (~16% prevalence): a range prior cannot cheat it,
and env-only it sat at AUROC 0.777 (GBM).

This script tests whether adding REAL, count-independent environmental predictors
-- StreamCat NHDPlus watershed/catchment attributes (land cover, climate, soils,
terrain, hydrology, anthropogenic stressors) -- raises env-only occupancy above
that ~0.85 floor, and specifically raises the WIDESPREAD mussel above its 0.777
env-only AUROC. If it does, there is genuine environmental mechanism; if not,
the earlier skill was essentially geography.

Feature sets (all leakage-free; per-species counts NEVER used as input):
  S0_proxy_env     : OLD env-only proxies -> reproduces Variant C floor.
                     [log_drainage, log_comm_richness_nt, log_comm_count_nt, log_n_events, elev]
  S1_streamcat     : REAL StreamCat env ONLY (no geo, no community). << KEY TEST
  S2_streamcat_comm: StreamCat + NON-target community context + effort (no geo).
  S3_geo_only      : lat/lon/abslat ONLY (pure range prior, for reference).
  S4_streamcat_geo : StreamCat + geography (ceiling / how much geo adds on top).

Probe: HistGradientBoostingClassifier (GBM) + LogisticRegression cross-check,
matching species_health_honest.json. Same real USGS BioData, same seed=42
spatial hash holdout. Rare always-absent species (Hellbender, Pearl mussel)
dropped; Mayfly (93% prevalence) reported separately; macro excludes Mayfly.

Bootstrap 95% CI (1000 resamples of the test set) on the WIDESPREAD mussel
AUROC for S1 and S2, and a paired bootstrap test of S1/S2 minus the S0 floor.

Output: results/benchmarks/species_mechanism.json

MIT License -- Bryan Cheng, 2026
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, f1_score

PROJECT = Path("/home/bcheng/SENTINEL")
MECH_DIR = PROJECT / "data" / "processed" / "biology" / "species_mechanism"
SITE_TABLE = MECH_DIR / "site_table.parquet"
SITE_COMID = MECH_DIR / "site_comid.parquet"
STREAMCAT = MECH_DIR / "streamcat.parquet"
OUT = PROJECT / "results" / "benchmarks" / "species_mechanism.json"

SEED = 42
SPECIES_NAMES = [
    "Freshwater mussels (Unionidae)",
    "Mayflies (Ephemeroptera)",
    "Brook trout (Salvelinus fontinalis)",
    "Hellbender (Cryptobranchus alleganiensis)",
    "Freshwater pearl mussel (Margaritifera margaritifera)",
    "American eel (Anguilla rostrata)",
]
ALWAYS_ABSENT_IDX = {3, 4}
MAYFLY_IDX = 1
MUSSEL_IDX = 0  # the widespread honest bioindicator


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load():
    tab = pd.read_parquet(SITE_TABLE)
    tab.columns = [str(c) for c in tab.columns]
    sc_map = pd.read_parquet(SITE_COMID)[["site_id", "comid"]]
    sc = pd.read_parquet(STREAMCAT)
    sc["comid"] = pd.to_numeric(sc["comid"], errors="coerce")
    sc = sc.dropna(subset=["comid"]).drop_duplicates("comid")
    sc["comid"] = sc["comid"].astype(int)
    # site -> comid -> streamcat metrics
    sc_map = sc_map.dropna(subset=["comid"]).copy()
    sc_map["comid"] = sc_map["comid"].astype(int)
    site_sc = sc_map.merge(sc, on="comid", how="left").set_index("site_id")
    metric_cols = [c for c in sc.columns if c != "comid"]
    # align to tab index
    site_sc = site_sc.reindex(tab.index)
    return tab, site_sc, metric_cols


def build_features(tab, site_sc, metric_cols):
    df = pd.DataFrame(index=tab.index)
    # OLD proxy env (reproduces Variant C floor)
    df["log_drainage"] = np.log1p(pd.to_numeric(tab["drainage"], errors="coerce").clip(lower=0).fillna(0))
    df["elev_site"] = pd.to_numeric(tab["elev"], errors="coerce")
    df["log_comm_richness_nt"] = np.log1p(pd.to_numeric(tab["comm_richness_nontarget"], errors="coerce"))
    df["log_comm_count_nt"] = np.log1p(pd.to_numeric(tab["comm_count_nontarget"], errors="coerce"))
    df["log_n_events"] = np.log1p(pd.to_numeric(tab["n_events"], errors="coerce"))
    # geography
    df["lat"] = pd.to_numeric(tab["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(tab["lon"], errors="coerce")
    df["abslat"] = df["lat"].abs()
    # REAL StreamCat env metrics
    for c in metric_cols:
        df["sc_" + c] = pd.to_numeric(site_sc[c], errors="coerce")
    return df


FEATURE_SETS = {
    "S0_proxy_env": lambda cols: ["log_drainage", "elev_site", "log_comm_richness_nt",
                                  "log_comm_count_nt", "log_n_events"],
    "S1_streamcat": lambda cols: cols,                       # StreamCat env only
    "S2_streamcat_comm": lambda cols: cols + ["log_comm_richness_nt",
                                              "log_comm_count_nt", "log_n_events", "log_drainage"],
    "S3_geo_only": lambda cols: ["lat", "lon", "abslat"],
    "S4_streamcat_geo": lambda cols: cols + ["lat", "lon", "abslat"],
}


def eval_set(df, occ, tr, te, feats):
    Xtr, Xte = df.loc[tr, feats].values, df.loc[te, feats].values
    res = {}
    for g in range(6):
        name = SPECIES_NAMES[g]
        if g in ALWAYS_ABSENT_IDX:
            res[name] = {"status": "dropped_always_absent"}
            continue
        ytr, yte = occ[tr.values, g].astype(int), occ[te.values, g].astype(int)
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            res[name] = {"status": "degenerate"}
            continue
        gbm = HistGradientBoostingClassifier(random_state=SEED, max_iter=300,
                                             learning_rate=0.05, l2_regularization=1.0)
        gbm.fit(Xtr, ytr)
        p_gbm = gbm.predict_proba(Xte)[:, 1]
        lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                           LogisticRegression(max_iter=2000, C=1.0))
        lr.fit(Xtr, ytr)
        p_lr = lr.predict_proba(Xte)[:, 1]
        res[name] = {
            "status": "ok",
            "test_prevalence": float(yte.mean()),
            "n_test": int(len(yte)),
            "n_present_test": int(yte.sum()),
            "auroc_gbm": float(roc_auc_score(yte, p_gbm)),
            "f1_gbm": float(f1_score(yte, (p_gbm >= 0.5).astype(int), zero_division=0)),
            "auroc_logistic": float(roc_auc_score(yte, p_lr)),
            "_p_gbm": p_gbm, "_y": yte,   # for bootstrap (stripped before JSON)
        }
    return res


def macro(res, incl_mayfly=False):
    idxs = [g for g in range(6) if g not in ALWAYS_ABSENT_IDX
            and (incl_mayfly or g != MAYFLY_IDX)]
    vals = [res[SPECIES_NAMES[g]]["auroc_gbm"] for g in idxs
            if res[SPECIES_NAMES[g]].get("status") == "ok"]
    f1s = [res[SPECIES_NAMES[g]]["f1_gbm"] for g in idxs
           if res[SPECIES_NAMES[g]].get("status") == "ok"]
    return (float(np.mean(vals)) if vals else None,
            float(np.mean(f1s)) if f1s else None)


def bootstrap_auroc(y, p, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    N = len(y); out = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], p[idx]))
    a = np.array(out)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def paired_delta_ci(y, p_a, p_b, n=1000, seed=SEED):
    """Bootstrap 95% CI on AUROC(a) - AUROC(b), same resample indices (paired)."""
    rng = np.random.default_rng(seed)
    N = len(y); d = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    d = np.array(d)
    return {"delta_median": float(np.median(d)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_gt_0_frac": float((d > 0).mean())}


def main():
    tab, site_sc, metric_cols = load()
    sc_feat_cols = ["sc_" + c for c in metric_cols]
    df = build_features(tab, site_sc, metric_cols)
    occ = (tab[[str(g) for g in range(6)]].values > 0).astype(np.float32)

    tr = tab["split"].isin(["train", "val"])   # train probe on train+val
    te = tab["split"] == "test"
    n_sc = int(site_sc[metric_cols[0]].notna().sum()) if metric_cols else 0
    log(f"Sites: {len(tab)} | train+val={int(tr.sum())} test={int(te.sum())} "
        f"| StreamCat coverage: {n_sc}/{len(tab)} sites "
        f"({100*n_sc/len(tab):.1f}%) | {len(metric_cols)} metric cols")

    all_res, macro_summary = {}, {}
    for sname, selector in FEATURE_SETS.items():
        feats = selector(sc_feat_cols)
        feats = [f for f in feats if f in df.columns]
        res = eval_set(df, occ, tr, te, feats)  # tr/te are boolean Series aligned to df/occ
        m_excl = macro(res, incl_mayfly=False)
        m_incl = macro(res, incl_mayfly=True)
        macro_summary[sname] = {
            "n_features": len(feats),
            "macro_auroc_excl_mayfly": m_excl[0],
            "macro_f1_excl_mayfly": m_excl[1],
            "macro_auroc_incl_mayfly": m_incl[0],
            "mussel_auroc": res[SPECIES_NAMES[MUSSEL_IDX]].get("auroc_gbm"),
        }
        all_res[sname] = res
        log(f"  {sname:20s} nf={len(feats):3d} | mussel AUROC="
            f"{res[SPECIES_NAMES[MUSSEL_IDX]].get('auroc_gbm')} | "
            f"macro(excl mayfly)={m_excl[0]}")

    # ---- Bootstrap CIs on the WIDESPREAD mussel ----
    boot = {}
    m_name = SPECIES_NAMES[MUSSEL_IDX]
    y = all_res["S0_proxy_env"][m_name]["_y"]
    for sname in ["S0_proxy_env", "S1_streamcat", "S2_streamcat_comm", "S3_geo_only", "S4_streamcat_geo"]:
        r = all_res[sname][m_name]
        lo, hi = bootstrap_auroc(r["_y"], r["_p_gbm"])
        boot[sname] = {"mussel_auroc": r["auroc_gbm"], "ci95": [lo, hi]}
    # paired deltas vs the S0 proxy-env floor (does REAL env beat the floor?)
    p0 = all_res["S0_proxy_env"][m_name]["_p_gbm"]
    paired = {
        "S1_streamcat_minus_S0": paired_delta_ci(y, all_res["S1_streamcat"][m_name]["_p_gbm"], p0),
        "S2_streamcat_comm_minus_S0": paired_delta_ci(y, all_res["S2_streamcat_comm"][m_name]["_p_gbm"], p0),
    }

    # ---- strip arrays before JSON ----
    def clean(res):
        out = {}
        for k, v in res.items():
            if isinstance(v, dict):
                out[k] = {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            else:
                out[k] = v
        return out
    per_set_clean = {s: clean(r) for s, r in all_res.items()}

    floor_mussel = 0.7772797698096503   # Variant C GBM mussel AUROC (species_health_honest.json)
    floor_macro = 0.8528712389833438    # Variant C GBM macro AUROC
    s1_mussel = macro_summary["S1_streamcat"]["mussel_auroc"]
    s2_mussel = macro_summary["S2_streamcat_comm"]["mussel_auroc"]

    # Verdict logic: mechanism = env-only (S1/S2) beats the geography-free floor
    # for the WIDESPREAD mussel, with the gain CI excluding 0.
    beats_floor = paired["S2_streamcat_comm_minus_S0"]["ci95"][0] > 0 or \
                  paired["S1_streamcat_minus_S0"]["ci95"][0] > 0
    verdict = (
        "REAL ENVIRONMENTAL MECHANISM CONFIRMED for the widespread mussel: adding "
        "StreamCat watershed environment lifts env-only occupancy AUROC above the "
        "geography-free proxy floor, with the paired bootstrap gain CI excluding 0."
        if beats_floor else
        "NO clear environmental mechanism beyond the proxy floor: StreamCat env does "
        "not significantly raise the widespread mussel's env-only AUROC above the "
        "geography-free baseline (gain CI includes 0). The earlier occupancy skill is "
        "essentially geography / community context, not fine-grained watershed environment."
    )

    out = {
        "_PROVENANCE": (
            "Species-mechanism test: does REAL environment carry occupancy mechanism "
            "beyond a geographic range prior? Same real USGS BioData and same seed=42 "
            "spatial hash holdout as species_health_leakagefree. Each survey site's "
            "coordinates were resolved to an NHDPlusV2 COMID via the EPA WATERS "
            "PointIndexing service, then REAL StreamCat catchment+watershed landscape "
            "metrics (NLCD land cover, PRISM climate, STATSGO soils, terrain, hydrology, "
            "anthropogenic stressors) were downloaded per COMID. Per-species counts are "
            "NEVER used as input (leakage-free). ENV-ONLY feature sets exclude lat/lon/HUC "
            "so no species range prior can cheat. GBM probe + logistic cross-check, "
            "matching species_health_honest.json. Bootstrap 95% CIs on the WIDESPREAD "
            "Unionid mussel (the one honest bioindicator; a range prior cannot cheat it)."
        ),
        "data_source": "USGS BioData (real surveys) + EPA StreamCat (NHDPlusV2) + EPA WATERS COMID",
        "synthetic_data": False,
        "seed": SEED,
        "n_sites_total": int(len(tab)),
        "n_train_val": int(tr.sum()),
        "n_test": int(te.sum()),
        "streamcat_coverage_sites": n_sc,
        "streamcat_coverage_frac": float(n_sc / len(tab)),
        "n_streamcat_metric_cols": len(metric_cols),
        "streamcat_metric_cols": metric_cols,
        "occupancy_prevalence": {SPECIES_NAMES[g]: float((tab[str(g)] > 0).mean()) for g in range(6)},
        "baseline_floor_from_species_health_honest": {
            "variant_C_env_only_no_geography_macro_auroc_gbm": floor_macro,
            "variant_C_env_only_mussel_auroc_gbm": floor_mussel,
            "variant_B_with_geography_macro_auroc_gbm": 0.943669227579283,
            "note": "These are the pre-existing honest leakage-free numbers this test builds on.",
        },
        "feature_sets": {
            "S0_proxy_env": "OLD env-only proxies (reproduces Variant C floor): drainage, elev, non-target community richness/count, effort.",
            "S1_streamcat": "REAL StreamCat cat+ws env ONLY (land cover/climate/soils/terrain/hydrology/stressors); NO geography, NO community.",
            "S2_streamcat_comm": "StreamCat env + non-target community context + effort + drainage; NO geography.",
            "S3_geo_only": "lat/lon/|lat| only (pure geographic range prior).",
            "S4_streamcat_geo": "StreamCat env + geography (ceiling; how much geo adds on top of real env).",
        },
        "macro_summary": macro_summary,
        "widespread_mussel_bootstrap": boot,
        "widespread_mussel_paired_gain_vs_proxy_floor": paired,
        "per_feature_set_per_species": per_set_clean,
        "VERDICT": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    log(f"Wrote {OUT}")
    log("VERDICT: " + verdict)


if __name__ == "__main__":
    main()
