# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Score the train-cool/test-hot experiment (protocol §5.4) on cal + val-role sites × hot train-block years.

    python -B scripts/eval/sim_dsst.py --dsst-runs dsst_s0 [...] --control-runs dsstctl_s0 [...] --tag dsst [--register]
"""

from __future__ import annotations

import argparse
import json

from sentinel.core import paths
from sentinel.core.registry import log_result
from sentinel.core.runlog import TrainingRun
from sentinel.models.sim import data as D
from sentinel.models.sim import dsst, runner


def _years(run: str) -> dict:
    return json.loads((paths.RUNS_ROOT / "sim" / run / "dsst_years.json").read_text())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsst-runs", nargs="+", required=True)
    ap.add_argument("--control-runs", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--register", action="store_true")
    a = ap.parse_args(argv)
    d = runner.load_data("validation")
    out = paths.ensure(paths.RUNS_ROOT / "sim" / "eval" / a.tag)
    import numpy as np

    # evaluation sites: cal + val-role sites (never trained on), hot train-block years (review C3)
    cal = np.concatenate([d.idx("cal"), d.idx("val")])
    n_groups = len(set(d.basin_group[cal].tolist()))
    if n_groups < 10:
        raise SystemExit(f"DSST needs >= 10 basin groups, got {n_groups}")
    cfg = {"dsst_runs": a.dsst_runs, "control_runs": a.control_runs, "n_boot": a.n_boot}
    with TrainingRun("sim", "sim_dsst_eval", config=cfg, data=list(D.DATA_MANIFESTS), plan_ref="sim_protocol §5.4",
                     iteration=a.tag, artifacts_dir=str(out)) as run:
        res = {}
        for kind, runs in (("hot", a.dsst_runs), ("control", a.control_runs)):
            yrs = _years(runs[0])
            assert all(_years(r)["fit_wy"] == yrs["fit_wy"] for r in runs)
            sim = runner.ensemble_sims([paths.RUNS_ROOT / "sim" / r / "sim_eval.npz" for r in runs], d)
            rows = dsst.dsst_rows(d, sim, cal, yrs["fit_wy"], yrs["eval_wy"])
            sc = dsst.score_dsst(rows, n_boot=a.n_boot)
            sc["experiment"] = kind
            sc.to_csv(out / f"dsst_{kind}.csv", index=False)
            res[kind] = sc
            print(kind, yrs["eval_wy"])
            print(sc[sc.model.eq("sim_ec_site")].drop(columns=["experiment"]).to_string())
        h = res["hot"]
        prim = h[h.model.eq("sim_ec_site")]
        passed = bool(((prim["crpss_vs_persistence_lo"] > 0) & (prim["crpss_vs_climatology_lo"] > 0)).all()
                      and len(prim) == 4)
        c = res["control"].set_index(["var", "h", "model"])
        loss = {f"{r['var']}_h{r['h']}": float(c.loc[(r["var"], r["h"], "sim_ec_site"), "crpss_vs_climatology"]
                                               - r["crpss_vs_climatology"]) for _, r in prim.iterrows()}
        summ = {"S_H4_pass": passed, "skill_loss_vs_control_crpss_clim": loss,
                "hot_years": _years(a.dsst_runs[0])["eval_wy"], "control_eval_years": _years(a.control_runs[0])["eval_wy"]}
        (out / "dsst_summary.json").write_text(json.dumps(summ, indent=1))
        print(json.dumps(summ, indent=1))
        run.metric("S_H4_pass", float(passed))
        if a.register:
            for _, r in prim.iterrows():
                for ref in ("persistence", "climatology", "physics_only"):
                    log_result(workstream="sim", experiment="sim_dsst", model=f"SENTINEL-Sim v1 DSST ({a.tag}):sim_ec_site",
                               metric=f"crpss_vs_{ref}_h{r['h']}", value=r[f"crpss_vs_{ref}"],
                               ci_low=r[f"crpss_vs_{ref}_lo"], ci_high=r[f"crpss_vs_{ref}_hi"],
                               ci_method=f"basin-group cluster bootstrap B={a.n_boot}", split="cal_hot_years",
                               n=int(r["n_rows"]), target=r["var"], config=cfg, data_manifests=D.DATA_MANIFESTS,
                               artifacts=[str(out / "dsst_hot.csv")],
                               notes="trained on cool WYs at train-role sites; scored at cal+val-role sites x hot WYs; "
                                     "all comparators fitted on cool WYs only",
                               extra={"skill_loss_vs_control": loss.get(f"{r['var']}_h{r['h']}")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
