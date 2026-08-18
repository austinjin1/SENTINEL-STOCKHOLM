#!/usr/bin/env python
"""TASK B — task complementarity: is ANY on-disk label genuinely bimodal?

For each candidate label we decompose:
  sensor-only  vs  satellite-only  vs  fusion (concat)
and the CONDITIONAL gain of each modality over the other via out-of-fold
stacking: fit modality-A OOF probability p_A, then fit a HistGBM on
[modality-B features + p_A] and ask whether adding B raises AUROC beyond A
alone. Symmetric for B over A. A task is "bimodal" iff BOTH conditional gains
are positive AND fusion CI-beats each single modality.

All models: HistGradientBoostingClassifier, station-grouped 5-fold CV,
AUROC + Hanley-McNeil 95% CI. REAL data only.

Feature blocks per tile (from embeddings_v2.npz + sensor_noleak.npz):
  sensor          : AquaSSM 256-d embedding (NON-CIRCULAR window for Turb/DO/multi)
  sat_cls         : WaterDroneNet 384-d cls feature
  sat_spec        : 16 chl/turbidity-sensitive band ratios+indices (spectral[16:32])
  satellite       : sat_cls (+ sat_spec)  [the satellite modality]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]
TB = ROOT / "data" / "processed" / "trackB"
OUT = ROOT / "results" / "trackB_fusion"
OUT.mkdir(parents=True, exist_ok=True)
RNG = 0


def hgb():
    return HistGradientBoostingClassifier(
        max_depth=3, max_iter=300, learning_rate=0.05,
        l2_regularization=1.0, min_samples_leaf=20, random_state=RNG)


def hm_ci(a, y):
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return [None, None]
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a**2) + (n0 - 1) * (q2 - a**2)) / (n1 * n0))
    return [round(max(0, a - 1.96 * se), 3), round(min(1, a + 1.96 * se), 3)]


def oof_proba(X, y, g, ns):
    """Out-of-fold predicted probabilities under station-grouped CV."""
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=ns).split(X, y, g):
        m = hgb().fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def cv_auroc(X, y, g, ns):
    return roc_auc_score(y, oof_proba(np.nan_to_num(X), y, g, ns))


def conditional_gain(Xa, Xb, y, g, ns):
    """AUROC of A alone, and of B stacked on A's OOF prob (B's marginal value)."""
    pa = oof_proba(np.nan_to_num(Xa), y, g, ns)
    auc_a = roc_auc_score(y, pa)
    Xstack = np.hstack([np.nan_to_num(Xb), pa.reshape(-1, 1)])
    p_stack = oof_proba(Xstack, y, g, ns)
    auc_stack = roc_auc_score(y, p_stack)
    return auc_a, auc_stack, auc_stack - auc_a


def perm_importance_grouped(X, y, g, ns, names, blocks):
    """Permutation importance aggregated by feature block, OOF-averaged."""
    # single train/test split by group to keep it cheap but honest
    gk = GroupKFold(n_splits=ns)
    tr, te = next(iter(gk.split(X, y, g)))
    m = hgb().fit(np.nan_to_num(X[tr]), y[tr])
    r = permutation_importance(m, np.nan_to_num(X[te]), y[te],
                               n_repeats=10, random_state=RNG,
                               scoring="roc_auc")
    imp = r.importances_mean
    out = {}
    for bname, (lo, hi) in blocks.items():
        out[bname] = round(float(imp[lo:hi].sum()), 4)
    return out


def run_task(name, sensor, sat, spec, y, g, note=""):
    ns = min(5, len(set(g)))
    y = y.astype(int)
    satellite = np.hstack([sat, spec])
    res = {"note": note, "n": int(len(y)), "pos": int(y.sum()),
           "prev": round(float(y.mean()), 3), "n_sites": int(len(set(g))),
           "cv_folds": ns}
    # unimodal + fusion
    auc = {}
    auc["sensor"] = cv_auroc(sensor, y, g, ns)
    auc["satellite"] = cv_auroc(satellite, y, g, ns)
    auc["sat_spectral_only"] = cv_auroc(spec, y, g, ns)
    fusion = np.hstack([sensor, satellite])
    auc["fusion"] = cv_auroc(fusion, y, g, ns)
    res["auroc"] = {k: round(float(v), 3) for k, v in auc.items()}
    res["ci95"] = {k: hm_ci(v, y) for k, v in auc.items()}

    best_single = max(auc["sensor"], auc["satellite"])
    best_single_name = "sensor" if auc["sensor"] >= auc["satellite"] else "satellite"
    # fusion CI-beats each single?
    fus_lo = res["ci95"]["fusion"][0]
    res["fusion_ci_beats_sensor"] = bool(fus_lo is not None and fus_lo > auc["sensor"])
    res["fusion_ci_beats_satellite"] = bool(fus_lo is not None and fus_lo > auc["satellite"])
    res["fusion_gain_over_best_single"] = round(float(auc["fusion"] - best_single), 3)
    res["best_single"] = best_single_name

    # conditional gains (both directions)
    a_alone, sat_on_sen, gain_sat_over_sen = conditional_gain(sensor, satellite, y, g, ns)
    b_alone, sen_on_sat, gain_sen_over_sat = conditional_gain(satellite, sensor, y, g, ns)
    res["conditional"] = {
        "sensor_alone": round(float(a_alone), 3),
        "add_satellite_over_sensor": round(float(sat_on_sen), 3),
        "gain_satellite_over_sensor": round(float(gain_sat_over_sen), 3),
        "satellite_alone": round(float(b_alone), 3),
        "add_sensor_over_satellite": round(float(sen_on_sat), 3),
        "gain_sensor_over_satellite": round(float(gain_sen_over_sat), 3),
    }
    res["both_conditional_gains_positive"] = bool(
        gain_sat_over_sen > 0.005 and gain_sen_over_sat > 0.005)
    res["bimodal"] = bool(
        res["both_conditional_gains_positive"] and
        res["fusion_ci_beats_sensor"] and res["fusion_ci_beats_satellite"])

    # feature-block permutation importance on the fusion model
    X = fusion
    d_sen, d_sat, d_spec = sensor.shape[1], sat.shape[1], spec.shape[1]
    blocks = {"sensor": (0, d_sen),
              "sat_cls": (d_sen, d_sen + d_sat),
              "sat_spectral": (d_sen + d_sat, d_sen + d_sat + d_spec)}
    res["perm_importance_blocks"] = perm_importance_grouped(
        X, y, g, ns, None, blocks)

    print(f"\n=== {name} ===  {note}")
    print(f"  n={res['n']} pos={res['pos']} prev={res['prev']} sites={res['n_sites']} folds={ns}")
    for k in ["sensor", "satellite", "sat_spectral_only", "fusion"]:
        print(f"  {k:20s} AUROC {res['auroc'][k]:.3f} CI{res['ci95'][k]}")
    c = res["conditional"]
    print(f"  cond: +sat over sensor {c['gain_satellite_over_sensor']:+.3f} | "
          f"+sensor over sat {c['gain_sensor_over_satellite']:+.3f}")
    print(f"  fusion gain over best single ({res['best_single']}): "
          f"{res['fusion_gain_over_best_single']:+.3f}  BIMODAL={res['bimodal']}")
    print(f"  perm importance: {res['perm_importance_blocks']}")
    return res


def load_blocks():
    emb = np.load(TB / "embeddings_v2.npz", allow_pickle=True)
    row = {int(t): i for i, t in enumerate(emb["tile_idx"])}
    sen_nl = np.load(TB / "sensor_noleak.npz", allow_pickle=True)
    nl_row = {int(t): i for i, t in enumerate(sen_nl["tile_idx"])}
    return emb, row, sen_nl, nl_row


def gather(tiles, emb, row, sen_nl, nl_row, use_noleak):
    """Return sensor, sat_cls, spec arrays for a list of tile indices."""
    ridx = [row[int(t)] for t in tiles]
    sat = emb["sat"][ridx]
    spec = emb["spectral"][ridx][:, 16:32]  # chl/turb ratio+index half
    if use_noleak:
        sensor = np.stack([sen_nl["sensor"][nl_row[int(t)]] for t in tiles])
    else:
        sensor = emb["sensor"][ridx]
    return sensor, sat, spec


def main():
    import json as _json
    z = np.load(ROOT / "data" / "processed" / "satellite" / "drone_wq_pairs.npz",
                allow_pickle=True)
    md = _json.loads(str(z["metadata"]))
    targets = z["targets"]  # [DO,pH,Turb,Temp,SpCond]
    emb, row, sen_nl, nl_row = load_blocks()
    have = set(row) & set(nl_row)  # tiles with BOTH a v2 emb and a no-leak sensor

    results = {}

    # ---- Task 1: chl-a > 8 ug/L (NON-circular; leaky same-day window is fine
    # because chl-a is NOT a sensor channel). Use ORIGINAL sensor emb. ----
    pc = np.load(TB / "pairs_chl.npz", allow_pickle=True)
    tiles = [int(t) for t in pc["tile_idx"] if int(t) in row]
    yc = np.array([int(l) for t, l in zip(pc["tile_idx"], pc["label"]) if int(t) in row])
    gc = np.array([md[t]["site_id"] for t in tiles])
    sen_c = emb["sensor"][[row[t] for t in tiles]]
    sat_c = emb["sat"][[row[t] for t in tiles]]
    spec_c = emb["spectral"][[row[t] for t in tiles]][:, 16:32]
    results["chl_gt8"] = run_task(
        "chl-a > 8 ug/L", sen_c, sat_c, spec_c, yc, gc,
        note="non-circular (chl not a sensor channel); original sensor window")

    # ---- Tasks with same-day sensor channels -> use NON-CIRCULAR sensor ----
    tiles_all = sorted(t for t in have if np.isfinite(targets[t, 2]))  # valid Turb
    g_all = np.array([md[t]["site_id"] for t in tiles_all])
    sensor_nl, sat_nl, spec_nl = gather(tiles_all, emb, row, sen_nl, nl_row, True)
    turb = targets[tiles_all, 2]
    do = targets[tiles_all, 0]

    # Task 3: turbidity exceedance (> 10 NTU). satellite-visible AND sensor-measured
    y_turb = (turb > 10).astype(int)
    results["turb_gt10"] = run_task(
        "turbidity > 10 NTU", sensor_nl, sat_nl, spec_nl, y_turb, g_all,
        note="Turb is a sensor channel; sensor window is STRICTLY BEFORE label day")

    # Task 4a: multi-parameter any-of {DO<5, Turb>10} (both same-day targets)
    valid = np.isfinite(do) & np.isfinite(turb)
    tiles_mp = [t for t, v in zip(tiles_all, valid) if v]
    g_mp = g_all[valid]
    sensor_mp, sat_mp, spec_mp = sensor_nl[valid], sat_nl[valid], spec_nl[valid]
    y_mp = ((do[valid] < 5) | (turb[valid] > 10)).astype(int)
    results["multi_DOorTurb"] = run_task(
        "any-of {DO<5, Turb>10}", sensor_mp, sat_mp, spec_mp, y_mp, g_mp,
        note="multi-parameter, non-circular sensor window (strictly before label day)")

    # Task 4b: multi-parameter any-of {DO<5, chl>8, Turb>10} on the chl subset
    chl_map = {int(t): (float(v), int(l)) for t, v, l in
               zip(pc["tile_idx"], pc["value"], pc["label"])}
    tiles_c3 = [t for t in tiles_all if t in chl_map and np.isfinite(do[tiles_all.index(t)])]
    if len(tiles_c3) >= 40:
        idxmap = {t: i for i, t in enumerate(tiles_all)}
        ii = [idxmap[t] for t in tiles_c3]
        g_c3 = g_all[ii]
        sensor_c3, sat_c3, spec_c3 = sensor_nl[ii], sat_nl[ii], spec_nl[ii]
        y_c3 = np.array([int((do[i] < 5) or (turb[i] > 10) or (chl_map[t][1] == 1))
                         for i, t in zip(ii, tiles_c3)])
        results["multi_DOchlTurb"] = run_task(
            "any-of {DO<5, chl>8, Turb>10}", sensor_c3, sat_c3, spec_c3, y_c3, g_c3,
            note="3-parameter multi on chl-labeled subset; non-circular sensor")

    # ---- summary: most bimodal task ----
    bimodal = {k: v for k, v in results.items() if v.get("bimodal")}
    summary = {
        "tasks": list(results.keys()),
        "bimodal_tasks": list(bimodal.keys()),
        "phycocyanin": "NOT TESTABLE: phyco WQP data is 2019-2021 but all "
                       "Sentinel-2 tiles are 2022-2025; min site-date gap >=233 "
                       "days, so zero co-located pairs exist within any usable "
                       "tolerance. Reported as a data-coverage null.",
    }
    results["_summary"] = summary
    (OUT / "taskB_complementarity.json").write_text(json.dumps(results, indent=2))
    print("\n\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("WROTE", OUT / "taskB_complementarity.json")


if __name__ == "__main__":
    main()
