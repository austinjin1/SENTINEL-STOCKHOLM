#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Train one seed of a GBM reference of the M6 architecture-v2 round: G1 or G2 (PF-GBM), amendment §5 and §7.

    python -B scripts/train/sota_hydromet_v2_gbm.py --variant g1 --seed 0            # grid of 3 + selection
    python -B scripts/train/sota_hydromet_v2_gbm.py --variant g1 --seed 1            # selected members only
    python -B scripts/train/sota_hydromet_v2_gbm.py --variant g1 --seed 0 --nograb   # no-grab variant (gate G-D1a)
    python -B scripts/train/sota_hydromet_v2_gbm.py --variant g2 --seed 0 --smoke [--synthetic]

* Data: the pinned arch-v2 pool (:func:`sentinel.hydromet_sota.gso_mt.data.load_v2_train`): train-role locations ×
  train block only; the same rows as the neural arms. No validation- or test-block row can be read by this script; it
  writes no validation metric.
* **G1** = frozen causal features (``FeatureSet.build``, lag 0) + the 4 solar channels + the 3 VAA statics (the
  additions every reference gets, amendment §3). **G2** = G1 + prior-parameter equilibrium temperature (trailing
  means), DO saturation at it, and out-of-fold stacks of log Q, T min, T max and DO max.
* Seed 0 fits every grid member, selects (per primary target: form × grid point with the lowest inner MSE; hour-
  resolved output: lowest capped inner grab MSE) and writes ``sota_v2/<variant>/selection.json`` once. Seeds 1–2 and
  ``--nograb`` fit only the selected members (no-grab: same forms and row caps, grab share 0); nothing is selected
  again. Selected members are refitted on fit + inner rows with their iteration count.
* Non-smoke runs need the lead's stamp and the pool pins. ``--smoke``: ≤ 50 train-role locations, grid point 1
  only; on real data the log and the record hold timing and row counts only (no inner score); never registered.
* CPU only, ≤ 6 threads. A job stops before the next member after ``job_minutes`` with exit code 75; rerunning the
  same command continues from the member cache.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time

import numpy as np


def point_cfg(base: dict, variant: str, point: str, **over) -> dict:
    """The public configuration of one counted grid point (or variant) of a reference: what the shared counter
    hashes. G1 does not carry the G2-only blocks."""
    keep = ("model", "lgbm", "n_jobs") + (("stack", "te_prior") if variant == "g2" else ())
    return {"arm": variant, "point": point, **{k: base[k] for k in keep}, "inputs": base["arms"][variant]["inputs"],
            "variant_of": None, "_smoke": base["_smoke"], "_config_file": base["_config_file"], **over}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=("g1", "g2"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--nograb", action="store_true", help="no-grab variant of the selected members")
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="smoke only: synthetic arrays instead of the pool")
    ap.add_argument("--smoke-new-usgs", action="store_true",
                    help="smoke only: include the lever pool's new USGS fit sites (off by default)")
    a = ap.parse_args()
    import yaml

    from sentinel.hydromet_sota.gso_mt import runner as VR
    from sentinel.models.hydromet import runner as HR

    if (a.synthetic or a.smoke_new_usgs) and not a.smoke:
        raise SystemExit("--synthetic and --smoke-new-usgs are smoke options")
    cpath = a.config or str(VR.CFG_DIR / "gbm.yaml")
    if not a.smoke:
        cpath = str(VR.pinned_config_path(cpath))               # a pinned file of configs/hydromet_sota/v2/ only
    base = yaml.safe_load(open(cpath).read())
    base.update({"_smoke": bool(a.smoke), "_config_file": cpath})
    HR.env_threads(int(base["n_jobs"]))
    stamp = None if a.smoke else VR.assert_released(f"M6V2 {a.variant} seed {a.seed}")
    point = "nograb" if a.nograb else "sel"
    name = f"{a.variant}_{point}_s{a.seed}"
    rdir = VR.run_dir(a.variant, name, smoke=a.smoke)
    if (rdir / ".done").exists() and not a.smoke:
        print(f"{name}: already done")
        return 0
    log = HR.make_logger(rdir / "log.txt")
    from sentinel.hydromet_sota import runner as SR
    from sentinel.hydromet_sota.gso_mt import data as VD
    from sentinel.hydromet_sota.gso_mt import gbm_ref as G
    from sentinel.hydromet_sota.gso_mt.solar import SOLAR_CHANNELS
    from sentinel.hydromet_sota.thermox import physics as TP
    from sentinel.models.hydromet.features import FeatureSet

    grid = {k: (g["max_rows_per_site"], float(g["grab_share"])) for k, g in base["grid"].items()}
    if tuple(grid.values()) != tuple((c, float(s)) for c, s in G.GRID):
        raise SystemExit("the grid in the config differs from the amendment's grid (gbm_ref.GRID)")
    if a.variant == "g2" and {k: float(v) for k, v in base["te_prior"].items()} != TP.PRIOR:
        raise SystemExit("te_prior in the config differs from thermox.physics.PRIOR")
    if a.variant == "g2" and list(base["stack"]["targets"]) != list(G.STACKED):
        raise SystemExit("stack targets in the config differ from gbm_ref.STACKED")
    lcfg = G.lgbm_config(base["lgbm"], n_jobs=int(base["n_jobs"]))
    quiet = a.smoke and not a.synthetic          # real-data smoke: timing and row counts only
    flog = (lambda m: None) if quiet else log
    # ---- which members: the grid (seed 0) or the written selection (seeds 1-2, no-grab)
    sel_p = VR.ROOT / a.variant / "selection.json"
    use_grid, fixed, sel = list(grid.values()), None, None
    if a.smoke:
        use_grid = use_grid[:1]
        if a.nograb or a.seed != 0:
            c, s = use_grid[0]
            fixed = {"selected": {v: ["derived", c, s] for v in VD.PRIMARY}, "selected_inst": {v: [c, s] for v in VD.INST}}
    elif a.nograb or a.seed != 0:
        if not sel_p.is_file():
            raise SystemExit(f"{sel_p} is missing: run seed 0 of {a.variant} first (the grid and its selection)")
        sel = json.loads(sel_p.read_text())
        fixed = {k: sel[k] for k in ("selected", "selected_inst")}
    if fixed is not None and a.nograb:
        fixed = G.no_grab_selection(fixed)
    if fixed is None:
        tried = [point_cfg(base, a.variant, k, max_rows_per_site=c, grab_share=s) for k, (c, s) in grid.items()
                 if (c, s) in use_grid]
    elif a.nograb:
        tried = [point_cfg(base, a.variant, "nograb", grab_share=0.0, variant_of="sel", selection=fixed)]
    else:
        used = {(s[1], s[2]) for s in fixed["selected"].values() if s[0] == "derived"} | \
               {(c, s) for c, s in map(tuple, fixed["selected_inst"].values())}
        caps = {s[1] for s in fixed["selected"].values() if s[0] == "direct"}
        tried = [point_cfg(base, a.variant, k, max_rows_per_site=c, grab_share=s) for k, (c, s) in grid.items()
                 if (c, s) in used or c in caps]
    cfg = point_cfg(base, a.variant, point, variant_of="sel" if a.nograb else None, grid=base["grid"],
                    fixed_selection=fixed, seed_role="grid + selection" if fixed is None else "selected members only")
    cfg = VR.with_release(cfg, stamp)                           # release record of the run (entry-1 rulings F1, F5)
    sel_from = None
    if sel is not None:                                         # the seed-0 selection must be of the same state
        sel_from = VR.assert_combinable([sel["source_run"]], cfg["_release"], f"{name}: seed-0 selection")
    cache_dir = rdir / "members"
    if a.smoke and cache_dir.exists():
        shutil.rmtree(cache_dir)
    fresh = not cache_dir.exists()
    VR.write_json(rdir / "config.json", VR.run_config(cfg, seed=a.seed))
    if fresh:
        for c in tried:
            VR.log_config_tried(c, rdir)
    log(f"[{name}] {'smoke' if a.smoke else 'stamped ' + stamp['amendment_sha256'][:12]}; "
        f"{'fresh start' if fresh else 'resume'}; {len(tried)} point(s) logged; threads {lcfg.n_jobs}")
    t0 = time.time()
    cache = G.MemberCache(cache_dir, deadline=t0 + 60.0 * float(base.get("job_minutes", 100)))
    # ---- data (train-role locations x train block)
    if a.synthetic:
        from sentinel.hydromet_sota.gso_mt.synthetic import synth_v2

        v2 = synth_v2(20, 6, 20, 4, n_forc=22)
    else:
        if a.smoke and not a.smoke_new_usgs:
            log(f"[{name}] smoke without the new USGS fit sites (their forcing table is the lever build)")
        v2 = VD.load_v2_train(grab=True, require_pins=not a.smoke, new_usgs=not a.smoke or a.smoke_new_usgs, log=flog)
        SR.assert_training_sites_clean(v2.data.sites, where=f"M6V2 {name} training set")
        if a.smoke:
            v2 = VD.smoke_subset(v2, np.random.default_rng(0))
    if a.smoke and len(v2.data.sites) > 50:
        raise RuntimeError("a smoke run uses at most 50 train-role locations")
    VR.assert_pool_clean(v2.data.sites, where=f"M6V2 {name} training set")
    VR.assert_pool_read(cfg, v2)                                # non-smoke: load_v2_train(require_pins=True) only
    VR.write_json(rdir / "data_info.json", {**{k: v for k, v in v2.info.items()}, "blocks": ["train"],
                                            "n_locations_used": len(v2.data.sites), "smoke": a.smoke})
    scales = VD.fit_scales(v2)
    elev = v2.data.static["elev_m"].to_numpy()
    fs = G.v2_inputs(FeatureSet.build(v2.data, lag=0), v2.solar, v2.vaa)
    stacks, sinfo = {}, {}
    need_resume = False
    with VR.training_run(cfg, rdir, seed=a.seed, refit=False, sites=v2.data.sites) as run:
        try:
            if a.variant == "g2":
                te = TP.te_prior(v2.data.forc, elev, v2.solar[:, :, SOLAR_CHANNELS.index("cloud_ratio")],
                                 prior=base["te_prior"])
                fs = G.physics_features(fs, te=te, elev_m=elev)
                fs, stacks, sinfo = G.add_stacks(v2, fs, lcfg, a.seed, log=flog, n_folds=int(base["stack"]["n_folds"]),
                                                 max_rows_per_site=base["stack"]["max_rows_per_site"], cache=cache)
            cols = fs.columns()
            csha = G.columns_sha256(cols + G.extra_names())
            if sel is not None and sel["columns_sha256"] != csha:
                raise RuntimeError(f"{name}: the input columns differ from those of the seed-0 selection")
            log(f"[{name}] features ready in {(time.time() - t0) / 60:.1f} min: {len(v2.data.sites)} locations, "
                f"{len(cols)} + {G.N_EXTRA} columns, cube {fs.dyn.nbytes / 2 ** 30:.1f} GiB")
            ref = G.GBMRef(a.variant, lcfg, grid=use_grid, seed=a.seed, fixed=fixed)
            out = ref.fit(v2, fs, scales, log=flog, cache=cache)
        except G.ResumeNeeded as e:
            need_resume = True
            run.note(f"job chunk ended before member {e}; {cache.n_built} built, {cache.n_loaded} loaded; resumed "
                     "by the next job")
        if not need_resume:
            info = {**ref.info, "stacks": sinfo, "columns_sha256": csha, "n_locations": len(v2.data.sites),
                    "selection_from": sel_from,
                    "minutes_this_job": round((time.time() - t0) / 60, 2), "members_built_this_job": cache.n_built,
                    "members_loaded_this_job": cache.n_loaded, "smoke": a.smoke, "variant_of": cfg["variant_of"],
                    "cube_gib": round(fs.dyn.nbytes / 2 ** 30, 2)}
            VR.write_json(rdir / "gbm_info.json", G.strip_scores(info) if quiet else info)
            if not a.smoke:
                for k in ("J", "j_sonde", "j_grab"):
                    if info.get(k) is not None:
                        run.metric(f"best_{k}", info[k])
            run.note(f"{len(ref.inst)} + {len(ref.amp)} + {len(ref.direct)} refitted members; "
                     f"{info['n_columns']} columns; fixed selection {fixed is not None}")
    if need_resume:
        log(f"[{name}] exiting with code {VR.EXIT_RESUME} (resume)")
        return VR.EXIT_RESUME
    mem = {grp: {k: {q: m[q] for q in ("n_train", "n_inner", "best_iteration", "n_columns", "fit_s", "rows_by_src",
                                       "refit_rows") if q in m} for k, m in d.items()}
           for grp, d in info["members"].items()}
    log(f"[{name}] members (rows, trees, seconds): {json.dumps(mem)}")
    if a.smoke:
        log(f"[{name}] SMOKE: {info['minutes_this_job']} min; {info['n_columns']} columns; {len(v2.data.sites)} "
            "locations; nothing registered")
        return 0
    import joblib

    joblib.dump({"ref": ref, "stacks": stacks, "columns": cols, "scales": scales.state(), "variant": a.variant,
                 "te_prior": base.get("te_prior") if a.variant == "g2" else None, "seed": a.seed}, rdir / "model.joblib",
                compress=3)
    ii, gi = v2.inner_idx, v2.grab_inner_idx
    np.savez_compressed(rdir / "inner_pred.npz", **{k: v.astype("float32") for k, v in out.items()},
                        date_start=str(v2.data.dates[0].date()), n_inner=len(ii), n_grab_inner=len(gi))
    if fixed is None:
        rec = {"variant": a.variant, "seed": a.seed, "selected": info["selected"],
               "selected_inst": info["selected_inst"], "columns_sha256": csha, "n_columns": info["n_columns"],
               "source_run": str(rdir), "rule": "per primary target: lowest inner MSE over {derived, direct} x grid; "
               "hour-resolved output: lowest capped inner grab MSE (amendment §5); written once by seed 0",
               "release": VR.run_release(cfg)}
        if sel_p.is_file():
            old = json.loads(sel_p.read_text())
            if any(old[k] != rec[k] for k in ("selected", "selected_inst", "columns_sha256")):
                raise RuntimeError(f"{sel_p} exists with a different selection; it is written once")
        else:
            VR.write_json(sel_p, rec)
            VR.log_decision("selection", f"{a.variant}.sel", {k: rec[k] for k in ("selected", "selected_inst", "rule",
                                                                              "columns_sha256")}, rdir=rdir,
                            release=VR.run_release(cfg), value=float(len(rec["selected"]) + len(rec["selected_inst"])))
    VR.log_inner(cfg, {"best_J": info["J"], "best_j_sonde": info["j_sonde"], "best_j_grab": info["j_grab"],
                       "best_by_target": info["by_target"], "best_epoch": None}, seed=a.seed, rdir=rdir)
    (rdir / ".done").touch()
    shutil.rmtree(cache_dir, ignore_errors=True)
    log(f"[{name}] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
