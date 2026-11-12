#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Microbe single-pass test scorer (protocol §8–§9; events protocol §7). DO NOT RUN without a
lead-written ``docs/preregistration/FREEZE_microbe.json`` (status FROZEN, authorised_by).

Look ``M3/test`` contains both H-M3 routes (16 hypotheses = 8 targets × 2 routes, Holm per resampling unit):
* ``timeout``: for each NEON domain g, fit the frozen configuration on development events outside g
  (plus .001 2017–18 rows outside g if the frozen track is ``lodo_x001``); predict holdout events
  (collect date ≥ 2024-01-01) in g.
* ``reserve``: for each domain g, fit on .002 development events outside g (descriptors disabled); predict
  the reserved DP1.20141.001 events (< 2017-01-01) in g.
Every model and comparator comes from the freeze file, never from the command line. The guard sequence is
in ``sentinel/models/microbe/testlook.py``. The event track (``timeout`` route, frozen σ/Platt maps) never fails
the primary look: errors are recorded as ``event_track_not_run``.

    python -B scripts/eval/microbe_final.py                      # claim -> predict -> seal -> score
    python -B scripts/eval/microbe_final.py --rescore-from-saved # only if sealed and no scoring rows exist
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.models.microbe import testlook as TL

N_JOBS = 6


def _folds(S: pd.DataFrame, route: str, track: str):
    from sentinel.models.microbe.splits import Fold

    src = S["source_product"].astype(str)
    dom = S["domain_id"].astype(str)
    date = pd.to_datetime(S["collect_date"])
    v2, v1 = src == "DP1.20141.002", src == "DP1.20141.001"
    dev2 = v2 & (date < pd.Timestamp("2024-01-01"))
    hold = v2 & (date >= pd.Timestamp("2024-01-01"))
    dev1 = v1 & (date >= pd.Timestamp("2017-01-01"))
    resv = v1 & (date < pd.Timestamp("2017-01-01"))
    folds = []
    for g in sorted(dom.unique()):
        if route == "timeout":
            tr = (dev2 | (dev1 if track == "lodo_x001" else False)) & (dom != g)
            te = hold & (dom == g)
        else:
            tr = dev2 & (dom != g)
            te = resv & (dom == g)
        if te.sum() and tr.sum():
            f = Fold(f"{route}|domain={g}", S.index[tr].to_numpy(), S.index[te].to_numpy())
            assert not set(dom[f.train]) & {g} and not (date[f.train] >= pd.Timestamp("2024-01-01")).any()
            assert not ((src[f.train] == "DP1.20141.001") & (date[f.train] < pd.Timestamp("2017-01-01"))).any()
            folds.append(f)
    return folds


def _predict(name: str, design, folds, targets, route: str) -> pd.DataFrame:
    from sentinel.models.microbe import baselines as B
    from sentinel.models.microbe import cv as CV

    if name.startswith("ens_"):
        parts = [_predict(m, design, folds, targets, route) for m in name[4:].split("+")]
        return sum(parts) / len(parts)
    params = {"descriptors": False} if route == "reserve" else {}
    if name in B.BASELINES:
        if name in B.SEQEMB:
            params["seq_table"] = CV.load_seq_table(B.SEQEMB[name], design.tokens["seq_id"].unique())
        seeds = (0, 1, 2) if name in B.STOCHASTIC else (0,)
        preds = [CV.oof_baseline(name, design, folds, targets=targets, seed=s, n_jobs=N_JOBS, params=params)
                 for s in seeds]
        return sum(p[list(targets)] for p in preds) / len(preds)
    cfg = CV.load_config(TL.paths.REPO_ROOT / "configs" / "microbe" / "seqset.yaml", name)   # frozen SeqSet config
    cfg["inner_val_unit"] = "domain"
    if route == "reserve":
        cfg["descriptors"] = False
    table, seq_row = (CV.load_seq_table(cfg["encoder"], design.tokens["seq_id"].unique())
                      if cfg.get("seq_mode", "frozen") == "frozen" else (None, None))
    oof, _, _ = CV.oof_seqset(cfg, design, folds, targets=targets, seeds=(0, 1, 2), device="cpu", table=table,
                              seq_row=seq_row)[:3]
    return oof.loc[np.concatenate([f.test for f in folds])]


def predict_and_seal(fr: dict, look: str) -> None:
    from sentinel.models.microbe.design import TARGET_NAMES, load_design

    design = load_design(allow_holdout=True, with_release_v1=True)       # requires the open claim
    m = fr["model"]
    names = [m["name"]] + list(m["comparators"])
    out = TL.pred_dir(look)
    out.mkdir(parents=True, exist_ok=True)
    for route in ("timeout", "reserve"):
        folds = _folds(design.samples, route, m["track"])
        for nm in names:
            p = _predict(nm, design, folds, list(TARGET_NAMES), route)
            p.to_parquet(out / f"{route}__{nm}.parquet")
    ids = {r: sorted(map(str, np.concatenate([f.test for f in _folds(design.samples, r, m["track"])])))
           for r in ("timeout", "reserve")}
    (out / "test_ids.json").write_text(json.dumps(ids))
    TL.seal_predictions(look)        # holdout access closes here


def score(fr: dict, look: str) -> dict:
    from sentinel.core.registry import log_result
    from sentinel.eval.bootstrap import holm_bonferroni
    from sentinel.models.microbe import evaluate as E
    from sentinel.models.microbe.design import TARGET_NAMES, add_targets
    from sentinel.core import paths

    d = TL.check_sealed(look)
    TL.assert_no_scoring_rows(look)
    m = fr["model"]
    S = pd.concat([pd.read_parquet(paths.PROCESSED / n / "samples.parquet") for n in ("microbe_wq", "microbe_wq_v1")])
    S = add_targets(S).set_index("sample_id", drop=False)      # labels only; predictions come from the sealed files
    n_hyp = 2 * len(TARGET_NAMES)
    res, pv = {}, {"site": {}, "domain": {}}
    for route in ("timeout", "reserve"):
        P = {nm: pd.read_parquet(d / f"{route}__{nm}.parquet") for nm in [m["name"]] + list(m["comparators"])}
        idx = P[m["name"]].index
        Sx = S.loc[idx]
        dom, site = Sx["domain_id"].astype(str), Sx["site_id"].astype(str)
        for t in TARGET_NAMES:
            y = Sx[t].to_numpy(float)
            if np.isfinite(y).sum() < 10:
                continue
            sm = E.summary(y, P[m["name"]][t].to_numpy(float), dom.to_numpy())
            ci = E.cis(y, P[m["name"]][t].to_numpy(float), dom.to_numpy(), site.to_numpy(), n_boot=2000)
            rec = {"r2": sm["r2"], "ci": ci["r2"], "n": sm["n"], "vs": {}}
            for ref in m["pass_refs"][t] if isinstance(m["pass_refs"], dict) else m["pass_refs"]:
                pr = P[ref][t] if ref in P else None
                if pr is None:
                    continue
                a = E.paired_vs(y, P[m["name"]][t], pr, dom, site, n_boot=2000)
                b = E.paired_vs(y, P[m["name"]][t], pr, dom, dom, n_boot=2000)
                rec["vs"][ref] = {"diff": a["diff"], "ci_site": [a["ci_low"], a["ci_high"]], "p_site": a["p_value"],
                                  "ci_domain": [b["ci_low"], b["ci_high"]], "p_domain": b["p_value"]}
                pv["site"][(route, t, ref)] = a["p_value"]
                pv["domain"][(route, t, ref)] = b["p_value"]
            res[(route, t)] = rec
    # Holm over the 16 (route, target) hypotheses, per comparator and resampling unit
    refs = sorted({k[2] for k in pv["site"]})
    passed = {}
    for ref in refs:
        rej = {u: holm_bonferroni({f"{k[0]}|{k[1]}": v for k, v in pv[u].items() if k[2] == ref and np.isfinite(v)})
               for u in ("site", "domain")}
        for (route, t), rec in res.items():
            if ref in rec["vs"]:
                key = f"{route}|{t}"
                rec["vs"][ref]["holm_both"] = bool(rej["site"].get(key) and rej["domain"].get(key)
                                                   and rec["vs"][ref]["diff"] > 0)
    for (route, t), rec in res.items():
        passed[f"{route}|{t}"] = bool(rec["vs"]) and all(v.get("holm_both") for v in rec["vs"].values())
    hm3 = {r: sum(v for k, v in passed.items() if k.startswith(r)) for r in ("timeout", "reserve")}
    lab = m["name"]
    for (route, t), rec in res.items():
        log_result(workstream=TL.WORKSTREAM, experiment=look, model=lab, metric="r2", value=rec["r2"], target=t,
                   ci_low=rec["ci"]["ci_low"], ci_high=rec["ci"]["ci_high"], ci_method="cluster bootstrap B=2000, unit=site",
                   split=f"M3/test/{route}", n=rec["n"], data_manifests=["microbe_wq", "microbe_wq_v1"],
                   extra={"vs": rec["vs"], "passes_all_refs": passed[f"{route}|{t}"], "n_hypotheses": n_hyp,
                          "freeze_sha256": fr["_sha256"]})
    log_result(workstream=TL.WORKSTREAM, experiment=look, model=lab, metric="hm3_targets_passing",
               value=float(max(hm3.values())), split="M3/test", data_manifests=["microbe_wq", "microbe_wq_v1"],
               extra={"by_route": hm3, "pass": any(v >= 2 for v in hm3.values()), "passed": passed})
    summary = {"hm3_by_route": hm3, "hm3_pass": any(v >= 2 for v in hm3.values())}
    try:
        summary["events"] = score_events(fr, look, d, S)
    except Exception:      # never fails the primary look
        err = traceback.format_exc()
        TL.record_event_track_not_run(look, err)
        summary["events"] = {"status": "not run", "error": err[-1000:]}
    return summary


def score_events(fr: dict, look: str, d: Path, S: pd.DataFrame) -> dict:
    from sentinel.core.registry import log_result
    from sentinel.eval import events as EV
    from sentinel.events import microbe as ME

    maps = json.loads(Path(TL.paths.REPO_ROOT / fr["event_maps"]).read_text())
    m = fr["model"]
    P = {nm: pd.read_parquet(d / f"timeout__{nm}.parquet") for nm in [m["name"]] + list(m["comparators"])}
    idx = P[m["name"]].index
    Sx = S.loc[idx]
    out = {}
    for code, mp_all in maps.items():
        y = ME.labels(Sx, code)
        probs = {}
        for nm, mp in mp_all.items():
            mu = P[nm].loc[idx, mp["target"]].to_numpy(float)
            p_raw = EV.exceedance_probability(EV.Gaussian(mu, np.full(mu.shape, mp["sigma"]), space=mp["space"]),
                                              ME.event_def(code))
            a, b = mp["platt"].get("a", np.nan), mp["platt"].get("b", np.nan)
            ok = np.isfinite(a) and abs(a) <= ME.PLATT_MAX_SLOPE and np.isfinite(b)
            probs[nm] = EV.Calibrator("platt", mp["platt"]).transform(p_raw) if ok else p_raw
        dom = Sx["domain_id"].astype(str).to_numpy()
        site = Sx["site_id"].astype(str).to_numpy()
        rec = {"model": {u: EV.state_metrics(y, probs[m["name"]], groups=g) for u, g in (("domain", dom), ("site", site))},
               "paired": {c: {u: EV.compare_routes(y, probs[m["name"]], probs[c], groups=g)
                              for u, g in (("domain", dom), ("site", site))} for c in probs if c != m["name"]}}
        out[code] = rec
        br = rec["model"]["domain"]["brier"]
        log_result(workstream=TL.WORKSTREAM, experiment="microbe_events_test", model=m["name"], metric=f"{code}_brier",
                   value=br["point"], ci_low=br["ci_low"], ci_high=br["ci_high"], split="M3/test/timeout",
                   ci_method="cluster bootstrap B=2000, unit=domain", data_manifests=["microbe_wq"],
                   extra={"auprc": rec["model"]["domain"]["auprc"], "site": rec["model"]["site"]["brier"],
                          "paired": {c: v["domain"]["delta_brier"] for c, v in rec["paired"].items()}})
    return {"status": "scored", "events": list(out)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--look", default="M3/test", choices=TL.LOOKS)
    ap.add_argument("--rescore-from-saved", action="store_true")
    a = ap.parse_args(argv)
    fr = TL.load_freeze()                 # lead-written freeze only; code hash checked
    TL.verify_data_pins(fr)               # before the claim: a mismatch never burns the look
    TL.verify_artifacts(fr)
    if a.look not in fr["looks"]:
        raise TL.ProtectedDataError(f"the freeze has no block for {a.look}")
    if a.rescore_from_saved:
        TL.assert_no_scoring_rows(a.look)
    else:
        TL.claim(a.look, fr)
        try:
            predict_and_seal(fr, a.look)
        except BaseException:          # incl. KeyboardInterrupt / SystemExit: the claim is closed, never left open
            TL.abort(a.look, traceback.format_exc())
            raise
    summary = score(fr, a.look)
    TL.mark_scored(a.look, summary)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
