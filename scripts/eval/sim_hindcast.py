# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Score SENTINEL-Sim hindcasts (protocol §4) on the validation cell (val-role sites × val block).

    python -B scripts/eval/sim_hindcast.py --runs base_s0 base_s1 base_s2 --tag base [--register]

The simulation of the listed runs is averaged (ensemble); every model of §4.3 is scored on common rows.
Writes $SENTINEL_RUNS/sim/eval/<tag>/{scores.csv, hindcast_summary.json}; with --register logs the
primary-endpoint CRPSS values (+ CIs) and the comparator CRPS values to the registry (split "val").
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.core.registry import log_result
from sentinel.core.runlog import TrainingRun
from sentinel.data.sonde_splits import SondeSplit
from sentinel.models.sim import data as D
from sentinel.models.sim import guard, hindcast, runner


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--block", default="val", choices=["val"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--register", action="store_true")
    a = ap.parse_args(argv)
    d = runner.load_data("validation")
    split = SondeSplit.load(guard.SPLIT_FILE)
    files = [paths.RUNS_ROOT / "sim" / r / "sim_eval.npz" for r in a.runs]
    sim = runner.ensemble_sims(files, d)
    ev = d.idx("val")
    out = paths.ensure(paths.RUNS_ROOT / "sim" / "eval" / a.tag)
    cfg = {"runs": a.runs, "block": a.block, "n_boot": a.n_boot, "horizons": list(hindcast.HORIZONS)}
    with TrainingRun("sim", "sim_hindcast_eval", config=cfg, data=list(D.DATA_MANIFESTS),
                     plan_ref="sim_protocol §4", iteration=a.tag, artifacts_dir=str(out)) as run:
        hc = hindcast.hindcast_rows(d, sim, ev, a.block, split)
        sc = hindcast.score(hc, n_boot=a.n_boot)
        sc.to_csv(out / "scores.csv", index=False)
        prim = sc[sc["model"].eq("sim_ec_site") & sc["var"].isin(hindcast.PRIMARY_VARS)
                  & sc["h"].isin(hindcast.PRIMARY_H)]
        verdict = {}
        for _, r in prim.iterrows():
            for ref in ("persistence", "climatology", "physics_only", "damped_anomaly"):
                verdict[f"{r['var']}_h{r['h']}_vs_{ref}"] = {
                    "crpss": r[f"crpss_vs_{ref}"], "lo": r[f"crpss_vs_{ref}_lo"], "hi": r[f"crpss_vs_{ref}_hi"],
                    "pass": bool(r[f"crpss_vs_{ref}_lo"] > 0)}
        sh1 = all(v["pass"] for k, v in verdict.items() if k.endswith(("persistence", "climatology")))
        gate = sh1 and all(v["pass"] for k, v in verdict.items() if k.endswith("physics_only"))
        summ = {"S_H1_pass_val": sh1, "baseline_gate_pass_val": gate, "endpoints": verdict,
                "n_eval_sites": int(len(ev)), "provenance": d.provenance(), "runs": a.runs}
        (out / "hindcast_summary.json").write_text(json.dumps(summ, indent=1, default=str))
        run.metric("S_H1_pass_val", float(sh1))
        run.metric("baseline_gate_pass_val", float(gate))
        print(prim[["var", "h", "n_rows", "n_sites", "crps", "crpss_vs_persistence", "crpss_vs_persistence_lo",
                    "crpss_vs_climatology", "crpss_vs_climatology_lo", "crpss_vs_physics_only",
                    "crpss_vs_physics_only_lo", "crpss_vs_damped_anomaly"]].to_string())
        if a.register:
            model_name = f"SENTINEL-Sim v1 ({a.tag}, {len(a.runs)}-seed ensemble)"
            for _, r in sc.iterrows():
                is_sim = r["model"] in hindcast.SIM_MODELS
                name = f"{model_name}:{r['model']}" if is_sim else r["model"]
                log_result(workstream="sim", experiment="sim_hindcast", model=name, metric=f"crps_h{r['h']}",
                           value=r["crps"], split="val", n=int(r["n_rows"]), target=r["var"],
                           is_baseline=not is_sim, config=cfg, data_manifests=D.DATA_MANIFESTS,
                           artifacts=[str(out / "scores.csv")],
                           notes="hindcast with observed forcings; val-role sites x WY2022-2023; model space",
                           extra={"n_sites": int(r["n_sites"]), "n_groups": int(r["n_groups"]),
                                  "rmse_median": r["rmse_median"], "nse_pooled": r["nse_pooled"]})
                if is_sim:
                    for ref in hindcast.REFS:
                        log_result(workstream="sim", experiment="sim_hindcast", model=name,
                                   metric=f"crpss_vs_{ref}_h{r['h']}", value=r[f"crpss_vs_{ref}"],
                                   ci_low=r[f"crpss_vs_{ref}_lo"], ci_high=r[f"crpss_vs_{ref}_hi"],
                                   ci_method=f"basin-group cluster bootstrap B={a.n_boot}, paired ratio",
                                   split="val", n=int(r["n_rows"]), target=r["var"], config=cfg,
                                   data_manifests=D.DATA_MANIFESTS, artifacts=[str(out / "scores.csv")],
                                   notes="hindcast with observed forcings (not a forecast)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
