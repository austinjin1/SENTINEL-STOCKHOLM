#!/usr/bin/env python
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Train one network of a seed unit of an architecture-round-v2 arm (orbit_arch_v2_amendment.md §3).

    python scripts/train/orbitv2_arm.py --config configs/orbit_v2/a1_c2.yaml --unit 0 --init 1

One process = one network (network seed = 100 x unit + init), <= 2 h. When the time budget runs out the process
writes a checkpoint and exits with status 75; the same command continues the run.

A real run is refused before the lead's stamp of the amendment (``stamp.require_stamp``) and before the arm file
names the selected dev_v5 configuration. It writes the network, its mixtures on the development rows outside the fit
cell and the training log entry. It computes no validation number: unit and arm scores are computed and registered
by the unit script.

``--smoke`` is a train-only test of the code path: <= 500 fit rows, 2 epochs, mixtures written for inner
(train-role) rows only, nothing registered, no row of a validation cell touched. It needs ``--smoke-base`` (a v1
configuration) and ``--data-dir`` (a dev_v5m-format cache; never the default).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.core.runlog import TrainingRun  # noqa: E402
from sentinel.models.orbit import data as D  # noqa: E402
from sentinel.models.orbit import pipeline as PL  # noqa: E402
from sentinel.models.orbit_v2 import arm as ARM  # noqa: E402
from sentinel.models.orbit_v2 import devm as DM  # noqa: E402
from sentinel.models.orbit_v2 import stamp as ST  # noqa: E402

EXIT_RESUME = 75
PLAN_REF = "docs/preregistration/orbit_arch_v2_amendment.md §3; " + PL.PLAN_REF


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/orbit_v2/<arm>.yaml")
    ap.add_argument("--unit", type=int, required=True)
    ap.add_argument("--init", type=int, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-base", default=None, help="smoke test: a v1 configuration file (repo-relative)")
    ap.add_argument("--data-dir", default=None, help="smoke test: a dev_v5m-format cache folder")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="cpu: smoke tests only")
    a = ap.parse_args(argv)
    cfg = ARM.resolve_config(Path(a.config), smoke=a.smoke, smoke_base=a.smoke_base)
    arm = cfg["arm"]
    if a.smoke:
        if not a.data_dir:
            raise SystemExit("a smoke test needs --data-dir")
        if Path(a.data_dir).resolve() == DM.cache_dir().resolve() and not ST.stamp_status()["stamped"]:
            raise SystemExit("before the stamp a smoke test reads a smoke cache, not the dev_v5m cache")
    else:
        if a.data_dir or a.smoke_base or a.device != "cuda":
            raise SystemExit("--data-dir, --smoke-base and --device cpu are for smoke tests")
        ST.require_stamp(f"orbitv2_arm: a network of arm {arm}")
    # the same for the 9 networks of the arm, however the path of the arm file was spelled
    arm_hash = ARM.config_hash({k: v for k, v in cfg.items() if k != "arm_file"})
    cfg.update({"unit": a.unit, "init": a.init})
    name = f"{arm}_u{a.unit}i{a.init}"
    run_dir = paths.run_dir("orbit", f"smoke/v2/{name}" if a.smoke else f"v2/{arm}/u{a.unit}i{a.init}")
    if (run_dir / "DONE").exists():
        print(f"{run_dir}: already complete")
        return 0
    if a.device == "cuda":
        PL.setup_gpu(f"orbit_v2_{name}")
    import torch

    from sentinel.models.orbit import satwq as V1

    torch.set_num_threads(2)      # shared-host rule: GPU jobs use 2 CPU threads
    device = torch.device("cuda" if a.device == "cuda" and torch.cuda.is_available() else "cpu")
    t0 = time.time()
    part = len(list(run_dir.glob("part_*.json")))
    if not a.smoke:
        PL.log_config_tried(arm, cfg, run_dir, "started" if part == 0 else "resumed")
    with TrainingRun(PL.WORKSTREAM, f"{'SMOKE/' if a.smoke else ''}orbit_v2_{arm}/{name}_part{part}", config=cfg,
                     data=[f"orbit_{DM.VERSION}", "gloria"], plan_ref=PLAN_REF, iteration="arch_v2",
                     artifacts_dir=str(run_dir)) as run:
        base = None
        if not a.smoke:
            base, _, _ = D.load_dev(DM.BASE_VERSION)
        df, pix, feats, ctx_cache, info = DM.load(Path(a.data_dir) if a.data_dir else None, base=base)
        run.note(f"data {info.get('version')} table sha256 {info['sha256'][DM.TABLE][:12]}; config hash "
                 f"{ARM.config_hash(cfg)[:12]}; arm file sha256 {cfg['arm_file_sha256'][:12]}")
        if a.smoke:
            run.note("SMOKE TEST (train-only): <=500 fit rows, 2 epochs; NOT a result; nothing registered")
        gloria = D.gloria_rows() if cfg["v1"]["pretrain"].get("enabled") else None
        smoke_n = int((cfg["v1"].get("smoke_n_fit") or 500)) if a.smoke else None
        res = ARM.train_network(cfg, df, pix, feats, ctx_cache, unit=a.unit, init=a.init, device=device,
                                gloria=gloria, run_dir=run_dir, smoke_n_fit=smoke_n,
                                pretrain_cache=paths.run_dir("orbit", "smoke/v2/pretrain" if a.smoke else "v2/pretrain"),
                                max_seconds=cfg["budget_minutes"] * 60 - (time.time() - t0))
        model, norm, targets, cl = res["model"], res["norm"], res["targets"], res["cells"]
        run.metric("n_params", V1.n_params(model))
        run.metric("n_fit", int(cl["fit_used"].sum()))
        run.metric("n_inner", int(cl["inner"].sum()))
        run.metric("n_heads", len(targets))
        run.metric("minutes", (time.time() - t0) / 60)
        run.note(f"heads {targets}; fit-cell values of the new targets {res['head_counts']} "
                 f"(a head below {ARM.MIN_HEAD_FIT} is dropped); pretrained network {res['pretrain'].get('key')} "
                 f"(reused {res['pretrain'].get('reused')}); {res['n_transferred']} tensors transferred")
        PL.write_json(run_dir / f"part_{part}.json", {"history": res["history"], "pretrain": res["pretrain"],
                                                      "complete": res["complete"], "seconds": time.time() - t0})
        if not res["complete"]:
            run.note("time budget reached: checkpoint written; the same command continues the run")
            print(f"NOT COMPLETE: rerun the same command to continue ({run_dir})")
            return EXIT_RESUME
        run.metric("best_epoch", res["history"]["best_epoch"])
        run.metric("inner_mdsa_select", res["history"]["best_inner"])       # inner cell = train role (not validation)
        ARM.save_network(run_dir / "model.pt", cfg, res)
        out_mask = cl["inner"] if a.smoke else ~cl["fit_cell_all"]
        ds = ARM.make_set(arm, df, pix, feats, ctx_cache, norm, out_mask, targets)
        ctx_off = bool(res["aug"].get("ctx_off", False))
        w, mu, sg = V1.predict_mixture(model, ds, device, norm.y_mean, norm.y_std, ctx_off=ctx_off)
        PL.save_predictions(run_dir / "preds.npz", df, out_mask, {"w": w, "mu": mu, "sg": sg})
        if ARM.ARMS[arm]["use_set"]:
            w, mu, sg = V1.predict_mixture(model, ARM.SetOff(ds), device, norm.y_mean, norm.y_std, ctx_off=ctx_off)
            PL.save_predictions(run_dir / "preds_set_off.npz", df, out_mask, {"w": w, "mu": mu, "sg": sg})
        PL.write_json(run_dir / "network.json", {
            "arm": arm, "unit": a.unit, "init": a.init, "seed": res["seed"], "targets": targets,
            "head_counts": res["head_counts"], "config_hash": ARM.config_hash(cfg), "arm_config_hash": arm_hash,
            "arm_file_sha256": cfg["arm_file_sha256"], "smoke": bool(a.smoke),
            "data": {"version": info.get("version"), "sha256": info["sha256"]},
            "prediction_rows": "inner cell only (smoke)" if a.smoke else "development rows outside the fit cell",
            "n_prediction_rows": int(np.asarray(out_mask).sum()), "best_epoch": res["history"]["best_epoch"]})
        if not a.smoke:
            PL.log_config_tried(arm, cfg, run_dir, "completed", {"unit": a.unit, "init": a.init})
        (run_dir / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    print(f"DONE {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
