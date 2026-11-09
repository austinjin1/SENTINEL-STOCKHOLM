#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Track S (protocol §11): a-priori difficulty strata for saved M6 predictions — descriptive.

Per stratum × endpoint × model: median over sites (>= 10 stratum days) of per-site RMSE and bias on the stratum
days of the common scored cells (same cells as ``hydromet_score.py``), basin-group bootstrap CI; paired
Δ median RMSE vs the comparator selected for that endpoint (``comparators.json`` of a primary score dir); and
exceedance CSI over all scored cells (temperature: pooled train P90; ``do_mgl_min``: < 5 mg/L). Thresholds use
FINAL train-role sites × train block only.

    python -B scripts/eval/hydromet_strata.py --config configs/hydromet/common.yaml --block val --roles val \\
        --comparators <score_dir>/comparators.json --predictions <dirs...> --out <dir> [--register --experiment M6S/val_v1]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.models.hydromet import runner

MIN_STRATUM_DAYS = 10
PRIMARY = ["temp_c_mean", "temp_c_7dadm", "do_mgl_mean", "do_mgl_min"]


def main() -> int:
    ap = runner.base_parser(__doc__)
    ap.add_argument("--predictions", nargs="+", required=True)
    ap.add_argument("--block", default="val")
    ap.add_argument("--roles", default="val")
    ap.add_argument("--comparators", required=True)
    ap.add_argument("--models", default="", help="comma list of models to report (default: all loaded + ensembles)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--experiment", default=None)
    a = ap.parse_args()
    a.eval_blocks = a.block
    runner.env_threads(6)
    cfg = runner.load_config(a.config, smoke=a.smoke)
    if a.track_r:
        cfg["track_r"] = a.track_r
    roles = tuple(r.strip() for r in a.roles.split(",") if r.strip())
    out = Path(a.out)
    log = runner.make_logger(out / "log.txt")
    from sentinel.eval import metrics as M
    from sentinel.eval.bootstrap import bootstrap_ci, paired_bootstrap
    from sentinel.models.hydromet import data as D
    from sentinel.models.hydromet import evaluate as E
    from sentinel.models.hydromet import pipeline
    from sentinel.models.hydromet import tracks as TR

    # train-role sites are loaded only for the thresholds (their val-block values are never used here)
    data, _, _ = runner.prepare(a, cfg, out, training=False, log=log, roles=tuple(dict.fromkeys(("train",) + roles)))
    si, ti = runner.eval_cells(data, a.block, roles)
    preds, sources = pipeline.load_prediction_set(a.predictions, data, si, ti, a.block, smoke=cfg["_smoke"],
                                                  access=data.access)
    preds = {**preds, **pipeline.seed_blends(preds)}
    ens, _ = pipeline.seed_ensembles(preds)
    preds = {**preds, **ens}
    keep = [m.strip() for m in a.models.split(",") if m.strip()]
    comps = json.loads(Path(a.comparators).read_text())
    comps = comps.get("comparators", comps)
    tr_sites = data.site_idx(["train"]) if not cfg.get("track_r") else np.flatnonzero(
        np.isin(data.sites, data.split.sites("train")))
    train_days = data.block_mask("train")
    thr = TR.train_thresholds(data.y, tr_sites, train_days, list(D.TARGETS))
    tmax = data.forc[:, :, D.FORCING_VARS.index("tmax_c")].astype("float64")
    masks = TR.strata_masks(data.y, data.dates, tmax, thr, train_days, list(D.TARGETS))
    log(f"[strata] thresholds (train-role x train block): { {k: round(v, 3) for k, v in thr.items()} }")
    rows, csi_rows = [], []
    for v in PRIMARY:
        models = [k for k, p in preds.items() if v in p and (not keep or k in keep or k in comps.get(v, {}).values())]
        cm, _ = E.common_mask(data, v, a.block, si, ti, {k: preds[k] for k in models}, True)
        n_site = cm.sum(axis=1)
        scored = n_site >= int(cfg.get("min_days_per_site", 60))
        obs = data.y[si][:, ti, data.vi(v)].astype("float64")
        bg = data.basin_group[si]
        for (stratum, sv), sm in masks.items():
            if sv != v:
                continue
            m = cm & sm[si][:, ti] & scored[:, None]
            ok = m.sum(axis=1) >= MIN_STRATUM_DAYS
            per = {}
            for k in models:
                P = preds[k][v].astype("float64")
                per[k] = np.array([M.rmse(obs[i, m[i]], P[i, m[i]]) if ok[i] else np.nan for i in range(len(si))])
                b = np.array([M.bias(obs[i, m[i]], P[i, m[i]]) if ok[i] else np.nan for i in range(len(si))])
                good = np.isfinite(per[k])
                if good.sum() < 3:
                    continue
                ci = bootstrap_ci(lambda x: np.nanmedian(x), per[k][good], groups=bg[good], n_boot=2000)
                rows.append({"stratum": stratum, "var": v, "model": k, "metric": "median_site_rmse",
                             "value": float(np.nanmedian(per[k][good])), "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                             "median_bias": float(np.nanmedian(b[good])), "n_sites": int(good.sum()),
                             "n_groups": int(np.unique(bg[good]).size), "n_cells": int(m[ok].sum())})
            for fam, c in comps.get(v, {}).items():
                if c not in per:
                    continue
                for k in models:
                    if k == c or k.startswith(("forcing_gbm", "doy_", "air2", "dosat")):
                        continue
                    good = np.isfinite(per[k]) & np.isfinite(per[c])
                    if good.sum() < 3:
                        continue
                    res = paired_bootstrap(lambda _y, p: np.nanmedian(p), np.zeros(good.sum()), per[k][good],
                                           per[c][good], groups=bg[good], n_boot=2000, higher_is_better=False)
                    rows.append({"stratum": stratum, "var": v, "model": k, "metric": f"diff_median_site_rmse_vs_best_{fam}",
                                 "comparator": c, "value": float(res["diff"]), "ci_low": res["ci_low"],
                                 "ci_high": res["ci_high"], "p": res.get("p_value"), "n_sites": int(good.sum()),
                                 "n_groups": int(np.unique(bg[good]).size)})
        # exceedance CSI over all scored cells (pooled), group bootstrap
        thr_x = thr[v] if v.startswith("temp") else (TR.DO_CRITERION_MGL if v == "do_mgl_min" else None)
        if thr_x is None:
            continue
        mm = cm & scored[:, None]
        for k in models:
            P = preds[k][v].astype("float64")
            oe = (obs > thr_x) if v.startswith("temp") else (obs < thr_x)
            pe = (P > thr_x) if v.startswith("temp") else (P < thr_x)
            cnt = np.stack([(oe & pe & mm).sum(1), (oe & ~pe & mm).sum(1), (~oe & pe & mm).sum(1)], axis=1)
            has = mm.any(axis=1)
            cnt_s, bg_s = cnt[has], bg[has]
            f = lambda c: c[:, 0].sum() / max(c.sum(), 1)   # noqa: E731
            ci = bootstrap_ci(f, cnt_s, groups=bg_s, n_boot=2000)
            csi_rows.append({"stratum": "exceedance", "var": v, "model": k, "metric": "pooled_csi", "threshold": thr_x,
                             "value": float(f(cnt_s)), "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                             "n_sites": int(has.sum()), "n_groups": int(np.unique(bg_s).size),
                             "n_obs_exceed": int(cnt_s[:, :2].sum())})
    res = pd.DataFrame(rows + csi_rows)
    res.to_csv(out / f"strata_{a.block}.csv", index=False)
    runner.write_json(out / "thresholds.json", thr)
    runner.write_json(out / "sources.json", sources)
    log(res[res["metric"].isin(["median_site_rmse", "pooled_csi"])].round(3).to_string(index=False)[:6000])
    if a.register:
        from sentinel.core.registry import log_result

        exp = a.experiment or f"M6S/{a.block}"
        split = runner._track_label(cfg) + f"{'/'.join(roles)}-role sites x {a.block} block, track S strata"
        n = 0
        for r in res.itertuples(index=False):
            if not np.isfinite(r.value):
                continue
            log_result(workstream=runner.WORKSTREAM, experiment=exp, model=runner.MODEL_PREFIX + r.model,
                       metric=f"{r.stratum}:{r.metric}", value=float(r.value), ci_low=float(r.ci_low),
                       ci_high=float(r.ci_high), split=split, n=int(r.n_sites), target=r.var,
                       is_baseline=runner._is_baseline(r.model),
                       ci_method="percentile bootstrap resampling basin groups (B=2000)",
                       config=runner.public_config(cfg), data_manifests=["sonde_daily", "forcings_daily", "streamcat"],
                       artifacts=list(sources.values()),
                       extra={"n_groups": int(r.n_groups), "thresholds": thr, "split_sha256": data.split.sha256(),
                              "descriptive": True})
            n += 1
        log(f"[registry] {n} entries ({exp})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
