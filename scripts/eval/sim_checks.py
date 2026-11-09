# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Physics/metamorphic checks (protocol §5.2) and latent validation (§6) for trained Sim members.

    python -B scripts/eval/sim_checks.py --runs base_s0 base_s1 base_s2 --tag base [--register]

Checks run per member on val-role sites × val block (share of sites passing per member and the
member-wise minimum); latent validation uses the ensemble-mean simulation vs Powell Center metabolism
at val-role Powell sites.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from sentinel.core import paths
from sentinel.core.registry import log_result
from sentinel.core.runlog import TrainingRun
from sentinel.models.sim import checks, latent, runner
from sentinel.models.sim import data as D


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    d = runner.load_data("validation")
    out = paths.ensure(paths.RUNS_ROOT / "sim" / "eval" / a.tag)
    cfg = {"runs": a.runs, "block": "val"}
    with TrainingRun("sim", "sim_checks_latent", config=cfg, data=list(D.DATA_MANIFESTS),
                     plan_ref="sim_protocol §5.2, §6", iteration=a.tag, artifacts_dir=str(out)) as run:
        val = d.idx("val")
        member = {}
        for r in a.runs:
            model, _ = runner.load_model(paths.RUNS_ROOT / "sim" / r / "model.pt", device=device, d=d)
            res = checks.run_checks(model, d, val, "val", device=device)
            res["site_rows"].to_csv(out / f"checks_sites_{r}.csv", index=False)
            member[r] = res["summary"]
            print(r, json.dumps({k: (v.get("share", v.get("value")), v["pass"]) for k, v in res["summary"].items()}))
        keys = list(next(iter(member.values())).keys())
        combined = {}
        for k in keys:
            vals = [m[k].get("share", m[k].get("value")) for m in member.values()]
            combined[k] = {"per_member": vals, "all_members_pass": all(m[k]["pass"] for m in member.values()),
                           "required": member[a.runs[0]][k].get("required_share", member[a.runs[0]][k].get("required"))}
        s_h2 = all(v["all_members_pass"] for v in combined.values())
        sim = runner.ensemble_sims([paths.RUNS_ROOT / "sim" / r / "sim_eval.npz" for r in a.runs], d)
        lat = latent.compare(d, sim, roles=("val",))
        lat["per_site"].to_csv(out / "latent_sites.csv", index=False)
        sp = latent.compare(d, sim, roles=("val",), labels=latent.load_streampulse())
        if len(sp["per_site"]):
            sp["per_site"].to_csv(out / "latent_sites_streampulse.csv", index=False)
        summ = {"S_H2_metamorphic_pass_val": s_h2, "checks": combined,
                "latent": {k: v for k, v in lat.items() if k != "per_site"},
                "latent_streampulse_secondary": {k: v for k, v in sp.items() if k not in ("per_site", "S_H3_pass")}}
        (out / "checks_latent_summary.json").write_text(json.dumps(summ, indent=1, default=str))
        print(json.dumps(summ["latent"], indent=1, default=str))
        run.metric("S_H2_metamorphic_pass_val", float(s_h2))
        run.metric("S_H3_pass_val", float(lat["S_H3_pass"]))
        if a.register:
            name = f"SENTINEL-Sim v1 ({a.tag})"
            for k, v in combined.items():
                for r, val_ in zip(a.runs, v["per_member"]):
                    if val_ is None or not np.isfinite(val_):
                        continue
                    log_result(workstream="sim", experiment="sim_physics_checks", model=f"{name}:{r}",
                               metric=f"{k}_share" if k != "M2_identity" else "M2_identity_max_abs_err_mgl",
                               value=float(val_), split="val", config=cfg, data_manifests=D.DATA_MANIFESTS,
                               artifacts=[str(out / f"checks_sites_{r}.csv")],
                               notes=f"required {v['required']}; protocol §5.2")
            for v in ("gpp", "er", "k600"):
                if v in lat:
                    log_result(workstream="sim", experiment="sim_latent_validation", model=name,
                               metric=f"median_site_spearman_{v}_vs_powell", value=lat[v]["median_rho"],
                               ci_low=lat[v]["ci_low"], ci_high=lat[v]["ci_high"],
                               ci_method="basin-group bootstrap of the site median, B=2000", split="val",
                               n=lat[v]["n_sites"], target=v, config=cfg,
                               data_manifests=list(D.DATA_MANIFESTS) + ["labels_powell_metabolism"],
                               artifacts=[str(out / "latent_sites.csv")],
                               notes="Powell Center H/M confidence, plausible days; val-role sites; spatial hold-out only "
                                     "(Powell period WY2008-2016 lies in the train block)",
                               extra={"between_site_rho": lat[v].get("between_site_rho")})
                if v in sp:
                    log_result(workstream="sim", experiment="sim_latent_validation", model=name,
                               metric=f"median_site_spearman_{v}_vs_streampulse", value=sp[v]["median_rho"],
                               ci_low=sp[v]["ci_low"], ci_high=sp[v]["ci_high"],
                               ci_method="basin-group bootstrap of the site median, B=2000", split="val",
                               n=sp[v]["n_sites"], target=v, config=cfg,
                               data_manifests=list(D.DATA_MANIFESTS) + ["labels_streampulse"],
                               artifacts=[str(out / "latent_sites_streampulse.csv")],
                               notes="secondary comparator (protocol §6); StreamPULSE valid, unflagged days; val-role sites")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
