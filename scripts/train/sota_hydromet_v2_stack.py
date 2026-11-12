#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Fit one stack of the M6 architecture-v2 round on inner rows (amendment §6.3; lead ruling L11.5;
``configs/hydromet_sota/v2/stack.yaml``): D3 SENTINEL-Stack, the mandatory references-only control, or the second
references-only stack with R-TD-W.

    python -B scripts/train/sota_hydromet_v2_stack.py --stack ref_stack
    python -B scripts/train/sota_hydromet_v2_stack.py --stack d3 --smoke --synthetic

* Inputs: the early-stopped inner predictions (``inner_pred.npz``, seeds 0–2 averaged) of the members, in the label
  form that gate G-D1a determined (``sota_v2/gates/g_d1a.json``; there is no command-line choice), and the inner
  labels of the pinned pool. Train-role inner groups × train block only; no validation row; no validation metric.
* Output: weights per endpoint (``weights.json``), the cross-fitted inner predictions (leave one inner basin group
  out) and their J through the same criterion code as every model; registry rows ``M6V2/inner_v1``.
* A member without complete inner predictions (D2 before its release, or after its kill rule) is left out and
  reported; no other member is exchanged.
* Non-smoke runs pass :func:`sentinel.hydromet_sota.gso_mt.runner.assert_released` first (lead rulings L10).
  ``--smoke --synthetic`` runs the same code on synthetic members; nothing is registered. CPU only.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="smoke only: synthetic pool and synthetic members")
    a = ap.parse_args()
    from sentinel.hydromet_sota.gso_mt import runner as VR
    from sentinel.models.hydromet import runner as HR

    if a.synthetic != a.smoke:
        raise SystemExit("a stack smoke run is synthetic (members come from counted runs); use --smoke --synthetic")
    cfg = VR.load_config(a.config or VR.CFG_DIR / "stack.yaml", "fixed", smoke=a.smoke)
    if a.stack not in cfg["stacks"]:
        raise SystemExit(f"unknown stack {a.stack!r}; one of {sorted(cfg['stacks'])}")
    spec = cfg["stacks"][a.stack]
    cfg = {**cfg, "arm": a.stack, "stack_members": list(spec["members"]), "role": spec["role"]}
    HR.env_threads(int(cfg["n_jobs"]))
    stamp = None if a.smoke else VR.assert_released(f"M6V2 stack {a.stack}")
    cfg = VR.with_release(cfg, stamp)                           # release record of the run (entry-1 rulings F1, F5)
    name = f"{a.stack}_fixed"
    rdir = VR.run_dir(a.stack, name, smoke=a.smoke)
    if (rdir / ".done").exists() and not a.smoke:
        print(f"{name}: already done")
        return 0
    log = HR.make_logger(rdir / "log.txt")
    from sentinel.hydromet_sota import runner as SR
    from sentinel.hydromet_sota.gso_mt import data as VD
    from sentinel.hydromet_sota.stack import build as B

    jT, jD = VD.DAILY.index("T_mean"), VD.DAILY.index("DO_mean")
    if a.synthetic:
        from sentinel.hydromet_sota.gso_mt.synthetic import synth_v2

        form, v2 = "grab", synth_v2(20, 6, 20, 4, n_forc=22)
    else:
        gate = VR.ROOT / "gates" / "g_d1a.json"
        if not gate.is_file():
            raise SystemExit(f"{gate} is missing: gate G-D1a decides the label form of every stack member")
        form = json.loads(gate.read_text())["label_form"]
        v2 = VD.load_v2_train(grab=form == "grab", require_pins=True, log=log)
        SR.assert_training_sites_clean(v2.data.sites, where=f"M6V2 {name}")
        VR.assert_pool_read(cfg, v2)                            # load_v2_train(require_pins=True) only
    VR.assert_pool_clean(v2.data.sites, where=f"M6V2 {name}")
    scales = VD.fit_scales(v2)
    Yd, inst, _ = VD.standardise_labels(v2, scales)
    ii, gi = v2.inner_idx, v2.grab_inner_idx
    y_d, y_i = Yd[ii][:, :, :7].astype("float64"), (inst[gi].astype("float64") if len(gi) else None)
    lm, ls = scales.label_mean, scales.label_std
    members, absent, selected, combined = {}, [], {}, {}
    if a.synthetic:
        rs = np.random.default_rng(0)
        for k, m in enumerate(cfg["stack_members"]):
            fill = np.where(np.isfinite(y_d), y_d, 0.0)
            members[m] = {"daily": fill + rs.normal(0, 0.2 + 0.15 * k, y_d.shape) + 0.05 * k,
                          "inst": np.where(np.isfinite(y_i), y_i, 0.0) + rs.normal(0, 0.3 + 0.1 * k, y_i.shape)}
    else:
        for arm in ("d1", "d2"):
            f = VR.ROOT / arm / "selection.json"
            selected[arm] = json.loads(f.read_text())["selected"] if f.is_file() else None
        for m in cfg["stack_members"]:
            dirs = B.member_run_dirs(VR.ROOT, m, form, selected)
            z = B.load_member(dirs, n_inner=len(ii), n_grab_inner=len(gi), date_start=str(v2.data.dates[0].date()),
                              release=cfg["_release"], what=f"stack {a.stack}, member {m}") if dirs else None
            if z is None:
                if m != "d2":
                    raise SystemExit(f"member {m} has no complete inner predictions in the '{form}' form; only D2 "
                                     "may be absent (stack.yaml)")
                absent.append(m)
                continue
            members[m] = {"daily": (z["daily"][:, :, :7] - lm[:7]) / ls[:7],
                          "inst": None if z["inst"] is None else (z["inst"] - lm[[jT, jD]]) / ls[[jT, jD]],
                          "n_seeds": z["n_seeds"]}
            combined[m] = z["combined"]                         # same pin state and code-pin sha256 as this process
    VR.write_json(rdir / "config.json", VR.run_config(cfg))
    VR.log_config_tried(cfg, rdir)
    log(f"[{name}] {'smoke (synthetic)' if a.smoke else 'stamped ' + stamp['amendment_sha256'][:12]}; label form "
        f"{form}; members {list(members)}; absent {absent}")
    g = np.asarray(v2.data.basin_group)
    with VR.training_run(cfg, rdir, seed=0, refit=False, sites=v2.data.sites) as run:
        res = B.fit_stack(members, y_d, y_i, g[ii], g[gi], scales.crit_factor(), form=form, ridge=float(cfg["ridge"]),
                          min_rows=int(cfg["min_rows"]) if not a.smoke else 20)
        if res["n_rows_dropped"] and not a.smoke:
            raise RuntimeError(f"{res['n_rows_dropped']} labelled inner rows lack a member prediction: the same rows "
                               "must score every model")
        info = {k: res[k] for k in ("weights", "folds", "rows", "j_sonde", "by_target", "j_grab", "J", "form",
                                    "n_rows_dropped")}
        info.update({"members": list(members), "absent": absent, "selected_points": selected, "smoke": a.smoke,
                     "combined_runs": combined,
                     "n_inner": len(ii), "n_grab_inner": len(gi), "n_inner_groups": int(len(set(g[ii].tolist())))})
        VR.write_json(rdir / "stack_info.json", info)
        VR.write_json(rdir / "weights.json", {"stack": a.stack, "form": form, "weights": res["weights"],
                                              "absent": absent, "selected_points": selected,
                                              "release": VR.run_release(cfg), "combined_runs": combined})
        if not a.smoke:
            for k in ("J", "j_sonde", "j_grab"):
                if info.get(k) is not None:
                    run.metric(f"best_{k}", info[k])
        run.note(f"{len(members)} members, {info['n_inner_groups']} inner groups, form {form}")
    log(f"[{name}] weights {json.dumps(res['weights'])}")
    if a.smoke:
        log(f"[{name}] SMOKE: nothing registered")
        return 0
    out = {"daily": res["daily"] * ls[:7] + lm[:7]}
    if res["inst"] is not None:
        out["inst"] = res["inst"] * ls[[jT, jD]] + lm[[jT, jD]]
    np.savez_compressed(rdir / "inner_pred.npz", **{k: v.astype("float32") for k, v in out.items()},
                        date_start=str(v2.data.dates[0].date()), n_inner=len(ii), n_grab_inner=len(gi))
    VR.log_inner(cfg, {"best_J": info["J"], "best_j_sonde": info["j_sonde"], "best_j_grab": info["j_grab"],
                       "best_by_target": info["by_target"], "best_epoch": None}, seed=0, rdir=rdir)
    (rdir / ".done").touch()
    log(f"[{name}] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
