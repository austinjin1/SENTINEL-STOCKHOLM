#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Train one M6 architecture-v2 neural configuration × seed (``docs/preregistration/hydromet_arch_v2_amendment.md``).

    python -B scripts/train/sota_hydromet_v2_nn.py --config configs/hydromet_sota/v2/d1.yaml --point g1 --seed 0
    python -B scripts/train/sota_hydromet_v2_nn.py --config configs/hydromet_sota/v2/d1.yaml --point g1 --seed 0 --refit
    python -B scripts/train/sota_hydromet_v2_nn.py --config configs/hydromet_sota/v2/d1.yaml --point g1 --seed 0 --smoke

* Data: the pinned arch-v2 pool (:func:`sentinel.hydromet_sota.gso_mt.data.load_v2_train`): train-role locations ×
  train block only. Early stopping on J (inner sonde sites + inner grab set). No validation- or test-block row can
  be read by this script; it writes no validation metric.
* ``--refit`` trains on fit + inner rows for E = best epoch + 1 of the early-stopped run of the same point and seed.
* Non-smoke runs pass :func:`sentinel.hydromet_sota.gso_mt.runner.assert_released` first (lead rulings L10): the
  stamped amendment, the stamped deviations file as it is on disk, and every pin of ``pins.json``; they assert the
  US input set v2 (30 dynamic / 191 static inputs) and, for R-TD, the transcribed published recipe.
* ``--smoke``: ≤ 50 train-role locations, a few steps, finite-gradient assertions; measures s/step and VRAM; logged
  with ``smoke: true``; never registered.
* D2 kill rule (amendment §8): ``--smoke`` on ``d2.yaml`` point g1 (``--d2-fallback`` for step 2), then
  ``--smoke --d2-kill-rule step1|step2`` writes ``sota_v2/d2/kill_rule.json`` (times, memory, counts, hashes) and
  one registry row. It releases nothing: the D2 grid needs its own stamped deviations entry (rulings L4).
* One GPU, always through the lock helper (the script refuses to pick a GPU itself):
  ``python -m sentinel.core.gpu --name <job> --min-free-mb 10000 -- python -B scripts/train/sota_hydromet_v2_nn.py …``.
  A job stops at an epoch boundary after ``job_minutes`` with exit code 75; rerunning the same command resumes
  exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--point", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--base", default=None, help="selected grid point whose hyperparameters an ablation/variant reuses")
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-steps", type=int, default=30)
    ap.add_argument("--synthetic", action="store_true", help="smoke only: synthetic arrays instead of the pool")
    ap.add_argument("--smoke-new-usgs", action="store_true",
                    help="smoke only: include the lever pool's new USGS fit sites (needs the lever engineer's "
                         "upstream_forcing_sota.parquet in its final state; off by default)")
    ap.add_argument("--cpu", action="store_true", help="debug only: no GPU lock (never for a result)")
    ap.add_argument("--d2-fallback", action="store_true",
                    help="smoke only, D2: diel sub-model on the fallback months (kill rule, step 2)")
    ap.add_argument("--d2-kill-rule", choices=("step1", "step2", "dry"), default=None,
                    help="no training: write sota_v2/d2/kill_rule.json from the train-only smoke run(s) of this "
                         "point (step2 also reads the --d2-fallback run; dry prints without writing)")
    a = ap.parse_args()
    from sentinel.hydromet_sota.gso_mt import runner as VR
    from sentinel.models.hydromet import runner as HR

    HR.env_threads(6)
    cfg = VR.load_config(a.config, a.point, base=a.base, smoke=a.smoke)
    if (a.synthetic or a.smoke_new_usgs or a.cpu or a.d2_fallback) and not a.smoke:
        raise SystemExit("--synthetic, --smoke-new-usgs, --cpu and --d2-fallback are smoke options")
    if a.d2_kill_rule:                                          # amendment §8: record of the measured step time
        if not a.smoke or cfg["model"] != "thermox" or a.seed != 0:
            raise SystemExit("--d2-kill-rule reads the smoke run of a D2 point at seed 0 (add --smoke)")
        d1 = VR.ROOT / "smoke" / f"{cfg['arm']}_{a.point}_s0"
        d2 = VR.ROOT / "smoke" / f"{cfg['arm']}_{a.point}_s0_fallback" if a.d2_kill_rule == "step2" else None
        if a.d2_kill_rule == "dry":
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                out = VR.record_d2_kill_rule(cfg, d1, None, record=f"{tmp}/kill_rule.json", register=False)
        else:
            out = VR.record_d2_kill_rule(cfg, d1, d2)
        print(json.dumps(out, indent=1))
        return 0
    VR.assert_recipe(cfg)
    stamp = None if a.smoke else VR.assert_released(f"M6V2 {cfg['arm']}.{a.point} seed {a.seed}")
    cfg = VR.with_release(cfg, stamp)                           # release record of the run (entry-1 rulings F1, F5)
    VR.assert_arm_released(cfg)                                 # D2 needs its own stamped entry (rulings L4)
    if cfg["model"] == "thermox":
        cfg = VR.resolve_d2_diel(cfg, fallback=a.d2_fallback)   # kill rule (amendment §8): days of the diel sub-model
    name = f"{cfg['arm']}_{a.point}_s{a.seed}{'_refit' if a.refit else ''}{'_fallback' if a.d2_fallback else ''}"
    rdir = VR.run_dir(cfg["arm"], name, smoke=a.smoke)
    if (rdir / ".done").exists() and not a.smoke:
        print(f"{name}: already done")
        return 0
    log = HR.make_logger(rdir / "log.txt")
    gpu = None if a.cpu else VR.locked_gpu()
    from sentinel.hydromet_sota import runner as SR
    from sentinel.hydromet_sota.gso_mt import data as VD
    from sentinel.hydromet_sota.gso_mt.trainer import ResumeNeeded, Trainer

    make = None
    if cfg["model"] == "thermox":
        from sentinel.hydromet_sota.thermox.nets import make_net as make
    resume = rdir / "resume.pt"
    if a.smoke and resume.exists():
        resume.unlink()
    fresh = not resume.exists()
    VR.write_json(rdir / "config.json", VR.run_config(cfg, seed=a.seed, refit=a.refit))
    if fresh and not a.refit:
        VR.log_config_tried(cfg, rdir)
    log(f"[{name}] GPU {gpu}; {'smoke' if a.smoke else 'stamped ' + stamp['amendment_sha256'][:12]}; "
        f"{'fresh start' if fresh else 'resume'}")
    t0 = time.time()
    if a.synthetic:
        from sentinel.hydromet_sota.gso_mt.synthetic import synth_v2

        v2 = synth_v2(20, 6, 20, 4, n_forc=22, n_static_extra=180)
    else:
        from sentinel.hydromet_sota import data as LD

        new_usgs = not a.smoke or a.smoke_new_usgs
        if not new_usgs:
            log(f"[{name}] smoke without the new USGS fit sites ({LD.UPSTREAM_SOTA.name} is the lever build)")
        v2 = VD.load_v2_train(grab=bool(cfg["data"].get("grab", True)), require_pins=not a.smoke, new_usgs=new_usgs,
                              log=log)
        SR.assert_training_sites_clean(v2.data.sites, where=f"M6V2 {name} training set")
        if a.smoke:
            v2 = VD.smoke_subset(v2, np.random.default_rng(0))
    if a.smoke and len(v2.data.sites) > 50:
        raise RuntimeError("a smoke run uses at most 50 train-role locations")
    VR.assert_pool_read(cfg, v2)                                # non-smoke: load_v2_train(require_pins=True) only
    VR.write_json(rdir / "data_info.json", {**{k: v for k, v in v2.info.items()}, "blocks": ["train"],
                                            "n_locations_used": len(v2.data.sites), "smoke": a.smoke})
    log(f"[{name}] data ready in {(time.time() - t0) / 60:.1f} min ({len(v2.data.sites)} locations)")
    tcfg = VR.train_cfg(cfg, a.seed)
    if a.smoke:
        tcfg.max_epochs, tcfg.steps_per_epoch = 2, a.smoke_steps
    refit_epochs = None
    if a.refit:
        src = VR.run_dir(cfg["arm"], f"{cfg['arm']}_{a.point}_s{a.seed}", smoke=a.smoke) / "train_info.json"
        refit_epochs = int(json.loads(src.read_text())["best_epoch"]) + 1
        if not a.smoke:                                         # the early-stopped run must be of the same state
            VR.assert_combinable([src.parent], cfg["_release"], f"{name}: number of epochs")
        if a.smoke:
            refit_epochs = 1
    tr = Trainer(tcfg) if make is None else Trainer(tcfg, make=make)
    if not a.smoke:
        tr.expect_inputs = (VR.N_DYNAMIC, VR.N_STATIC)          # US input set v2 (rulings L5), checked before step 1
    ckpt = rdir / f"{name}.pt"
    need_resume = False
    with VR.training_run(cfg, rdir, seed=a.seed, refit=a.refit, sites=v2.data.sites) as run:
        try:
            info = tr.fit(v2, refit_epochs=refit_epochs, device="cpu" if a.cpu else "cuda", log=log,
                          checkpoint_path=ckpt, resume_path=resume, job_minutes=float(cfg.get("job_minutes", 100)),
                          assert_finite=a.smoke)
        except ResumeNeeded:
            need_resume = True
            run.note(f"job chunk ended after {len(tr.history)} epochs; resumed by the next job")
        if not need_resume:
            tr.save(ckpt)
            info = {**info, "gpu_memory": VR.gpu_memory_mb(), "seed": a.seed, "refit": a.refit, "smoke": a.smoke}
            VR.write_json(rdir / "train_info.json", {**info, "history": tr.history})
            if not a.smoke and not a.refit:
                for k in ("best_J", "best_j_sonde", "best_j_grab"):
                    if info.get(k) is not None:
                        run.metric(k, info[k])
            run.note(f"epochs {info['epochs']}, best epoch {info['best_epoch']}, params {info['n_params']}, "
                     f"{info['s_per_step_median']} s/step, {info['gpu_memory']}")
    if need_resume:
        log(f"[{name}] exiting with code {VR.EXIT_RESUME} (resume)")
        return VR.EXIT_RESUME
    if a.smoke:
        log(f"[{name}] SMOKE: {info['s_per_step_median']} s/step at batch {tcfg.batch_size}; {info['gpu_memory']}; "
            f"params {info['n_params']}; inputs {tr.n_dyn} dynamic / {tr.n_static} static")
        return 0
    if not a.refit:
        # inner predictions of the early-stopped model (train-role inner rows; members of the stack, gates G-D1a, G-DO)
        dyn, st, _, _ = VD.make_inputs(v2.data, v2.solar, v2.vaa, tr.scales)
        Yd, inst, _ = VD.standardise_labels(v2, tr.scales)
        ii, gi = v2.inner_idx, v2.grab_inner_idx
        ps = tr.predict(dyn, st, ii, np.isfinite(Yd[ii][:, :, :7]).any(-1))
        out = {"daily": ps["daily"] * tr.scales.label_std[:7] + tr.scales.label_mean[:7]}
        if len(gi):
            pg = tr.predict(dyn, st, gi, np.isfinite(inst[gi]).any(-1), hours=v2.inst_hour[gi])
            j = [VD.DAILY.index("T_mean"), VD.DAILY.index("DO_mean")]
            out["inst"] = pg["inst"] * tr.scales.label_std[j] + tr.scales.label_mean[j]
        np.savez_compressed(rdir / "inner_pred.npz", **{k: v.astype("float32") for k, v in out.items()},
                            date_start=str(v2.data.dates[0].date()), n_inner=len(ii), n_grab_inner=len(gi))
        VR.log_inner(cfg, info, seed=a.seed, rdir=rdir)
    (rdir / ".done").touch()
    log(f"[{name}] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
