#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M3 Microbe event track on DEVELOPMENT LODO OOF predictions (docs/preregistration/events_microbe_protocol.md).

For the model under evaluation (``--model``; M* once frozen) and every comparator: derived event
probabilities (route i, cross-fitted sigma + Platt), state metrics with domain- and site-cluster CIs, BSS vs
the season+location event climatology, paired ΔBrier/ΔAUPRC vs each comparator (Holm across the inferential
events, both units), and the direct head (route ii) for powered events. Never touches holdout rows.
Registry: workstream 'M3-microbe', experiment 'microbe_events_dev'.

    python -B scripts/eval/events_microbe.py --model xgb_v4clr --scheme lodo
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.core.registry import log_result
from sentinel.core.runlog import TrainingRun

WS, EXP = "M3-microbe", "microbe_events_dev"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--scheme", default="lodo")
    ap.add_argument("--comparators", nargs="+", default=["tier0_best", "rf_clr", "xgb_clr"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-direct", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args(argv)
    from sentinel.eval import events as EV
    from sentinel.eval.bootstrap import holm_bonferroni
    from sentinel.events import microbe as ME
    from sentinel.models.microbe import splits as SP
    from sentinel.models.microbe.baselines import LABELS
    from sentinel.models.microbe.design import load_design

    design = load_design(with_release_v1=a.scheme == "lodo_x001")
    folds = SP.make_folds(design.samples, a.scheme)
    S = design.samples
    d = paths.RUNS_ROOT / "microbe" / "cv" / a.scheme
    names = [a.model] + list(a.comparators)
    oof = {m: pd.read_parquet(d / f"{m}.parquet") for m in names}
    idx = oof[a.model].index
    for m in names:
        idx = idx.intersection(oof[m].index)
    out = {"model": a.model, "scheme": a.scheme, "comparators": a.comparators, "events": {}}
    cfg = {"model": a.model, "scheme": a.scheme, "comparators": a.comparators, "n_boot": a.n_boot,
           "protocol": "events_microbe_protocol.md", "events_py_sha256": EV.__dict__.get("__sha256__", "see file")}
    run_ctx = TrainingRun(WS, f"{EXP}:{a.model}:{a.scheme}", config=cfg, data=["microbe_wq"],
                          plan_ref="docs/plans/event_layer_plan.md; events_microbe_protocol.md",
                          iteration="events", artifacts_dir=str(d))
    with run_ctx as run:
        clim_cache = {}
        pvals = {}          # (comparator, unit) -> {event: p}
        for code in ME.INFERENTIAL + ME.BORDERLINE + ME.DESCRIPTIVE:
            der = {m: ME.derived_probability(S, oof[m].loc[idx], code) for m in names}
            y = der[a.model]["y"]
            dom = S.loc[idx, "domain_id"].astype(str).to_numpy()
            site = S.loc[idx, "site_id"].astype(str).to_numpy()
            cl = clim_cache.setdefault(code, ME.climatology_probability(design, folds, code))
            clim = pd.Series(cl["climatology"], index=cl["index"]).reindex(idx).to_numpy(float)
            prev = pd.Series(cl["prevalence"], index=cl["index"]).reindex(idx).to_numpy(float)
            common = np.isfinite(y) & np.all([np.isfinite(der[m]["p"]) for m in names], axis=0) & np.isfinite(clim)
            n_pos = int(np.nansum(y[common]))
            n_dom_pos = len(np.unique(dom[common & (y == 1)]))
            flag = EV.power_flag(n_pos, n_dom_pos)
            status = ("inferential" if code in ME.INFERENTIAL else
                      ("inferential" if flag == "feasible" else "descriptive") if code in ME.BORDERLINE else "descriptive")
            rec = {"status": status, "power_flag": flag, "n": int(common.sum()), "n_pos": n_pos,
                   "domains_with_pos": n_dom_pos, "models": {}, "paired": {}}
            yc, dc, sc = y[common], dom[common], site[common]
            probs = {m: der[m]["p"][common] for m in names}
            probs["event_climatology"], probs["train_prevalence"] = clim[common], prev[common]
            for m, pm in probs.items():
                rec["models"][m] = {"domain": EV.state_metrics(yc, pm, groups=dc, p_ref=clim[common], n_boot=a.n_boot),
                                    "site": EV.state_metrics(yc, pm, groups=sc, p_ref=clim[common], n_boot=a.n_boot)}
            if status == "inferential" and not a.no_direct:
                dr = ME.direct_probability(design, folds, code, a.model)
                pdir = pd.Series(dr["p"], index=dr["index"]).reindex(idx).to_numpy(float)[common]
                rec["direct"] = {"skipped_folds": dr["skipped_folds"],
                                 "domain": EV.compare_routes(yc, probs[a.model], pdir, groups=dc, n_boot=a.n_boot),
                                 "site": EV.compare_routes(yc, probs[a.model], pdir, groups=sc, n_boot=a.n_boot)}
            for c in list(a.comparators) + ["event_climatology", "train_prevalence"]:
                rec["paired"][c] = {u: EV.compare_routes(yc, probs[a.model], probs[c], groups=g, n_boot=a.n_boot)
                                    for u, g in (("domain", dc), ("site", sc))}
                if status == "inferential":
                    for u in ("domain", "site"):
                        pvals.setdefault((c, u), {})[code] = rec["paired"][c][u]["delta_brier"]["p_value"]
            out["events"][code] = rec
        # Holm across inferential events, per comparator and unit; claim needs both units + ΔBrier < 0
        for (c, u), pv in pvals.items():
            rej = holm_bonferroni({k: v for k, v in pv.items() if np.isfinite(v)})
            for code in pv:
                out["events"][code]["paired"][c][u]["holm_reject"] = bool(rej.get(code, False))
        for code, rec in out["events"].items():
            for c, pr in rec["paired"].items():
                pr["claim"] = bool(rec["status"] == "inferential" and pr["domain"].get("holm_reject")
                                   and pr["site"].get("holm_reject") and pr["domain"]["delta_brier"]["diff"] < 0)
        # frozen sigma/Platt maps of the evaluated model (fitted on all development rows; for the test pass)
        out["frozen_maps"] = {code: {m: ME.frozen_maps(S, oof[m].loc[idx], code) for m in names}
                              for code in out["events"]}
        dest = paths.ensure(paths.RUNS_ROOT / "microbe" / "events") / f"events_dev_{a.model}_{a.scheme}.json"
        dest.write_text(json.dumps(out, indent=1, default=float))
        if not a.no_log:
            lab = LABELS.get(a.model, a.model)
            for code, rec in out["events"].items():
                for m, mm in rec["models"].items():
                    for met in ("brier", "auprc", "bss"):
                        v = mm["domain"].get(met)
                        if isinstance(v, dict) and np.isfinite(v["point"]):
                            log_result(workstream=WS, experiment=EXP, model=LABELS.get(m, m), metric=f"{code}_{met}",
                                       value=v["point"], ci_low=v["ci_low"], ci_high=v["ci_high"],
                                       ci_method=f"cluster bootstrap B={a.n_boot}, unit=domain", split=f"neon_dev/{a.scheme}_cv",
                                       n=rec["n"], data_manifests=["microbe_wq"], is_baseline=(m != a.model),
                                       extra={"status": rec["status"], "n_pos": rec["n_pos"], "site_ci": [
                                           mm["site"][met]["ci_low"], mm["site"][met]["ci_high"]],
                                           "prevalence": mm["domain"]["prevalence"], "evaluated_model": lab},
                                       artifacts=[str(dest)])
                for c, pr in rec["paired"].items():
                    db = pr["domain"]["delta_brier"]
                    log_result(workstream=WS, experiment=EXP, model=lab, metric=f"{code}_delta_brier_vs_{c}",
                               value=db["diff"], ci_low=db["ci_low"], ci_high=db["ci_high"],
                               ci_method=f"paired cluster bootstrap B={a.n_boot}, unit=domain",
                               split=f"neon_dev/{a.scheme}_cv", n=rec["n"], data_manifests=["microbe_wq"],
                               extra={"p_value": db["p_value"], "site": pr["site"]["delta_brier"],
                                      "claim": pr["claim"], "status": rec["status"],
                                      "delta_auprc_domain": pr["domain"]["delta_auprc"]}, artifacts=[str(dest)])
        for code, rec in out["events"].items():
            run.metric(f"{code}_brier", rec["models"][a.model]["domain"]["brier"]["point"])
    for code, rec in out["events"].items():
        mm = rec["models"][a.model]["domain"]
        print(f"{code:9s} {rec['status']:11s} n={rec['n']} pos={rec['n_pos']} dom+={rec['domains_with_pos']} "
              f"AUPRC={mm['auprc']['point']:.3f} prev={mm['prevalence']:.3f} BSS={mm.get('bss', {}).get('point', np.nan):.3f} "
              + " ".join(f"dB[{c}]={pr['domain']['delta_brier']['diff']:+.4f}{'*' if pr['claim'] else ''}"
                         for c, pr in rec["paired"].items()))


if __name__ == "__main__":
    main()
