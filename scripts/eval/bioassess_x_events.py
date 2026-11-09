#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M5-X secondary EVENT TRACK on the development folds (protocol amendment v8; event_layer_plan.md).

Run after ``bioassess_x_select.py``. For each (block, event): event labels of the pseudo-test rows; derived route
(Gaussian on the target with cross-fitted out-of-fold residual SD → ``events.exceedance_probability`` →
cross-fitted ``events.recalibrate``) for the selected C-X and the threshold-on-baseline comparators RF[geography] and
TabPFNv2; prevalence and climatology from each fold's training rows; the direct LightGBM challenger for powered
events only; ``events.state_metrics`` and ``events.compare_routes`` (location-cluster bootstrap B = 2,000).
Writes ``x_track/dev/events/events_dev_v1.json`` and the frozen development calibration data
``events_calibration_v1.parquet`` (all-dev residual SD and raw out-of-fold probabilities; used by the test look),
plus the test-block capacity projection. Every fold's training ids pass ``xtrack.assert_fit_ids``.

    python -B scripts/eval/bioassess_x_events.py [--log]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from sentinel.core.registry import file_sha256, log_result
from sentinel.events import bioassess_x as XE
from sentinel.models.bioassess import xdev as XD
from sentinel.models.bioassess import xtrack as X

OUT = XD.DEV_DIR / "events"
FOLDS = XD.DEV_DIR / "folds_dev_v1.json"
SEL = XD.DEV_DIR / "selection_x_v1.json"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jsonable(d):
    return json.loads(json.dumps(d, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="store_true")
    a = ap.parse_args()
    XE.events_module()                                   # fail early if the shared module is not there yet
    S, frozen, D = XD.load_context()
    folds = XD.read_folds(FOLDS, S, frozen)
    sel = json.loads(SEL.read_text())
    TM = XD.TaxonMatrix(D)
    split = S[S["role"] == "test"].groupby("block").size().to_dict()
    res, calib = {"selection_sha256": file_sha256(SEL), "folds_sha256": file_sha256(FOLDS), "events": {}}, []
    for (b, code), (t, _, _) in XE.TRACK.items():
        for fo in folds[b]:
            X.assert_fit_ids(XD.keys_of(fo.train), S, frozen)
        ids = pd.Index(np.concatenate([fo.test for fo in folds[b]]))
        fold = np.concatenate([[fo.g] * len(fo.test) for fo in folds[b]])
        cl = D.samples.loc[ids, "location_id"].astype(str).to_numpy()
        yt = XD.labels(D, b, t, ids, "same")
        ye = np.full(len(yt), np.nan)
        ye[np.isfinite(yt)] = XE.labels(yt[np.isfinite(yt)], b, code)
        cx = sel["blocks"][b]["selected"][0]
        choice = sel["blocks"][b]["variant_choice"]
        models = {"C-X:" + cx: XD.DEV_DIR / "preds" / b / (f"{cx}__chosen__s0.parquet" if cx.startswith("ens_") else
                                                           next((XD.DEV_DIR / "preds" / b).glob(f"{cx}__*__s0.parquet")).name)}
        for m in XE.COMPARATORS:
            v = choice.get(m, {}).get(t, "same")
            models[m] = XD.DEV_DIR / "preds" / b / f"{m}__{v}__s0.parquet"
        probs = {}
        for name, f in models.items():
            mu = pd.read_parquet(f).reindex(ids)[t].to_numpy(float)
            raw = np.full(len(mu), np.nan)
            for g in np.unique(fold):                      # cross-fitted SD for the development numbers
                sel_g = fold == g
                raw[sel_g] = XE.derived_probability(mu[sel_g], XE.residual_sd(yt, mu, fold, leave_out=g), b, code)
            probs[name] = XE.crossfit_calibrated(raw, ye, fold)
            sd_all = XE.residual_sd(yt, mu, fold)
            raw_all = XE.derived_probability(mu, sd_all, b, code)
            calib.append(pd.DataFrame({"block": b, "event": code, "model": name, "id": ids, "fold": fold,
                                       "p_raw": raw_all, "y": ye, "sd_all": sd_all}))
        prev, clim = np.full(len(ids), np.nan), np.full(len(ids), np.nan)
        for fo in folds[b]:
            m_ = ids.isin(fo.test)
            pv, cm = XE.fold_prevalence_climatology(D, b, code, fo)
            prev[m_], clim[m_] = pv, cm
        probs["prevalence"], probs["climatology"] = prev, clim
        if (b, code) in XE.POWERED:
            raw = np.concatenate([XE.fold_direct(D, TM, b, code, fo, 0) for fo in folds[b]])
            probs["direct"] = XE.crossfit_calibrated(raw, ye, fold)
            calib.append(pd.DataFrame({"block": b, "event": code, "model": "direct", "id": ids, "fold": fold,
                                       "p_raw": raw, "y": ye, "sd_all": np.nan}))
        ok = np.isfinite(ye)
        pw = XE.power_flag(ye[ok], cl[ok])
        avail = ok.mean()
        proj = {"n_test_split": int(split.get(b, 0)), "dev_label_availability": float(avail),
                "dev_prevalence": float(np.nanmean(ye)),
                "projected_test_positives": float(split.get(b, 0) * avail * np.nanmean(ye))}
        mets = {n: XE.metrics(ye[ok], p[ok], cl[ok], p_ref=clim[ok]) for n, p in probs.items()}
        cxn = "C-X:" + cx
        comps = {f"{cxn} - {n}": XE.compare(ye[ok], probs[cxn][ok], probs[n][ok], cl[ok]) for n in probs if n != cxn}
        status = "descriptive only" if (b, code) in XE.DESCRIPTIVE_ONLY else "secondary (powered)"
        res["events"][f"{b}|{code}"] = jsonable({"target": t, "power_dev": pw, "status": status, "projection": proj,
                                                 "metrics": mets, "contrasts": comps})
        log(b, code, pw, status, {n: round(XE.metric_value(v, "brier")[0], 4) for n, v in mets.items()})
        if a.log:
            for n, v in mets.items():
                for k in ("auprc", "brier", "bss"):
                    val, lo, hi = XE.metric_value(v, k)
                    if np.isfinite(val):
                        log_result(workstream="bioassess", experiment="bioassess_x_events_dev", model=n, metric=k,
                                   value=val, split=f"{b}_dev", target=code, n=int(ok.sum()), ci_low=lo, ci_high=hi,
                                   ci_method="location-cluster bootstrap B=2000 (events.state_metrics)",
                                   is_baseline=not n.startswith("C-X"),
                                   notes=f"M5-X event track (secondary; {status}); amendment v8",
                                   extra={"power_dev": pw, "projection": proj})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(calib, ignore_index=True).to_parquet(OUT / "events_calibration_v1.parquet")
    (OUT / "events_dev_v1.json").write_text(json.dumps(res, indent=1, default=float))
    log("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
