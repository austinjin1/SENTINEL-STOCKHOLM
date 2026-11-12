#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Fit the physics-family reference P of the M6 architecture-v2 round (amendment §5; ``configs/hydromet_sota/v2/
physics.yaml``): the frozen ``air2stream_regional`` / ``dosat_ratio[...]`` classes, site-cell and basin-mean (``_ws``)
members, refitted on the daily rows of the pinned arch-v2 pool.

    python -B scripts/train/sota_hydromet_v2_physics.py                 # step 1 (inner) + step 2 (refit)
    python -B scripts/train/sota_hydromet_v2_physics.py --smoke [--synthetic]

* Data: the pinned pool (:func:`sentinel.hydromet_sota.gso_mt.data.load_v2_train`), train-role locations × train
  block only. P fits daily sonde rows only; the inner grab stations are loaded to be *scored* (J_grab with the daily
  mean as the hour-resolved value). No validation- or test-block row can be read; no validation metric is written.
* Step 1: donors and DO ratios from the fit sites; the inner sites and inner grab stations are predicted. Per primary
  endpoint the member with the lowest standardised inner-sonde MSE is chosen (ties: the earlier member), per
  hour-resolved variable the member with the lowest term of J_grab. The choice goes to ``sota_v2/p/selection.json``
  and to the registry (``M6V2/decisions_v1``). Step 2: the members are refitted on fit + inner sites.
* The classes, their settings (``configs/hydromet/baselines_ws_physics.yaml``) and the DO saturation routine are the
  frozen ones; this file holds no formula and no physical constant.
* Non-smoke runs pass :func:`sentinel.hydromet_sota.gso_mt.runner.assert_released` first (lead rulings L10).
  ``--smoke``: ≤ 50 train-role locations, 20 calibration iterations; on real data the log and the record hold timing
  and counts only (no inner score); never registered.
* CPU only, ≤ 6 threads. A job stops before the next calibration after ``job_minutes`` with exit code 75; rerunning
  the same command continues from the fitted members on disk.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time

import numpy as np

MEMBERS = ("air2stream_regional", "air2stream_regional_ws")          # order = tie order (earlier member wins)


def member_view(data, kind: str):
    """The frozen feature builder sees 11 forcing channels: the site-cell block or the basin-mean block of the pool's
    22 (the lever driver's ``_replace``)."""
    from sentinel.models.hydromet.data import FORCING_VARS

    n = len(FORCING_VARS)
    if data.forc.shape[-1] != 2 * n:
        raise RuntimeError(f"the pool has {data.forc.shape[-1]} forcing channels; expected {2 * n}")
    d = copy.copy(data)
    d.forc = data.forc[:, :, :n] if kind == "site_cell" else data.forc[:, :, n:]
    return d


def choose(scores: dict, order) -> str:
    """Lowest score; ties to the earlier name in ``order``."""
    best = None
    for m in order:
        if scores.get(m) is not None and (best is None or scores[m] < scores[best]):
            best = m
    if best is None:
        raise RuntimeError("no member has an inner score")
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="smoke only: synthetic arrays instead of the pool")
    ap.add_argument("--smoke-new-usgs", action="store_true", help="smoke only: include the lever pool's new USGS sites")
    a = ap.parse_args()
    from sentinel.hydromet_sota.gso_mt import runner as VR
    from sentinel.models.hydromet import runner as HR

    if (a.synthetic or a.smoke_new_usgs) and not a.smoke:
        raise SystemExit("--synthetic and --smoke-new-usgs are smoke options")
    cfg = VR.load_config(a.config or VR.CFG_DIR / "physics.yaml", "fixed", smoke=a.smoke)
    HR.env_threads(int(cfg["n_jobs"]))
    stamp = None if a.smoke else VR.assert_released("M6V2 p")
    cfg = VR.with_release(cfg, stamp)                           # release record of the run (entry-1 rulings F1, F5)
    name = "p_fixed"
    rdir = VR.run_dir("p", name, smoke=a.smoke)
    if (rdir / ".done").exists() and not a.smoke:
        print(f"{name}: already done")
        return 0
    log = HR.make_logger(rdir / "log.txt")
    import joblib

    from sentinel.hydromet_sota import runner as SR
    from sentinel.hydromet_sota.gso_mt import criterion as C
    from sentinel.hydromet_sota.gso_mt import data as VD
    from sentinel.models.hydromet.baselines import Air2StreamRegional, DOSatRatio
    from sentinel.models.hydromet.features import FeatureSet

    if list(cfg["forcing"]) != list(MEMBERS) or any(list(v)[0].endswith("_ws") for v in cfg["members"].values()):
        raise SystemExit("the member list of the config differs from this driver's (site-cell first, then _ws)")
    quiet = a.smoke and not a.synthetic          # real-data smoke: timing and counts only
    flog = (lambda m: None) if quiet else log
    fresh = not any(rdir.glob("member_*.joblib"))
    if a.smoke:
        for f in rdir.glob("member_*.joblib"):
            f.unlink()
    VR.write_json(rdir / "config.json", VR.run_config(cfg))
    if fresh:
        VR.log_config_tried(cfg, rdir)
    log(f"[{name}] {'smoke' if a.smoke else 'stamped ' + stamp['amendment_sha256'][:12]}; "
        f"{'fresh start' if fresh else 'resume'}")
    t0 = time.time()
    deadline = t0 + 60.0 * float(cfg.get("job_minutes", 100))
    if a.synthetic:
        from sentinel.hydromet_sota.gso_mt.synthetic import synth_v2

        v2 = synth_v2(20, 6, 20, 4, n_forc=22)
    else:
        v2 = VD.load_v2_train(grab=True, require_pins=not a.smoke, new_usgs=not a.smoke or a.smoke_new_usgs, log=flog)
        SR.assert_training_sites_clean(v2.data.sites, where=f"M6V2 {name} training set")
        if a.smoke:
            v2 = VD.smoke_subset(v2, np.random.default_rng(0))
    if a.smoke and len(v2.data.sites) > 50:
        raise RuntimeError("a smoke run uses at most 50 train-role locations")
    VR.assert_pool_read(cfg, v2)                                # non-smoke: load_v2_train(require_pins=True) only
    scales = VD.fit_scales(v2)                   # the same label scales as every other model (full pool)
    Yd, inst, _ = VD.standardise_labels(v2, scales)
    n_all = len(v2.data.sites)
    # P reads daily sonde rows only: keep the fit sites, the inner sites and the inner grab stations (scored)
    keep = np.sort(np.r_[v2.fit_idx, v2.inner_idx, v2.grab_inner_idx].astype(int))
    pos = {int(o): i for i, o in enumerate(keep)}
    y_in, y_gi = Yd[v2.inner_idx][:, :, :7], inst[v2.grab_inner_idx]
    v2 = v2.subset(keep)
    VR.assert_pool_clean(v2.data.sites, where=f"M6V2 {name} training set")
    VD.assert_no_forcing_only_site(v2.data.sites)
    d, fi, ii, gi = v2.data, v2.fit_idx, v2.inner_idx, v2.grab_inner_idx
    VR.write_json(rdir / "data_info.json", {**{k: v for k, v in v2.info.items()}, "blocks": ["train"],
                                            "n_locations_pool": n_all, "n_locations_used": len(d.sites),
                                            "n_fit": len(fi), "n_inner": len(ii), "n_grab_inner": len(gi),
                                            "smoke": a.smoke})
    a2s_kw = dict(cfg["air2stream"])
    if a.smoke:
        a2s_kw["iters"] = 20
    t_all = np.arange(len(d.dates))
    jT, jD = VD.DAILY.index("T_mean"), VD.DAILY.index("DO_mean")
    col = {v: VD.DAILY.index(k) for v, k in VD.PRIMARY.items()}
    cf = scales.crit_factor()

    def fitted(tag: str, kind: str, fs, dv, inner):
        """One member (temperature + DO models) with donors / ratios from ``fi`` (+ ``inner``); cached on disk."""
        f = rdir / f"member_{tag}_{kind}.joblib"
        if f.is_file():
            return joblib.load(f)
        if time.time() > deadline:
            raise TimeoutError(f"{tag}_{kind}")
        ts = time.time()
        temp = Air2StreamRegional(**a2s_kw)
        temp.name = MEMBERS[0] if kind == "site_cell" else MEMBERS[1]
        temp.fit(dv, fs, fi, inner, log=flog)
        do = DOSatRatio(temp, MEMBERS[0]).fit(dv, fs, fi, inner, log=flog)
        m = {"temp": temp, "do": do, "kind": kind, "fit_s": round(time.time() - ts, 1),
             "n_donors": int(temp.info["n_donors"]), "n_candidates": int(temp.info["n_candidates"])}
        joblib.dump(m, f, compress=3)
        log(f"[{name}] {tag} {kind}: {m['n_donors']} donors of {m['n_candidates']} candidates, {m['fit_s']} s")
        return m

    def predict(m, fs, dv, idx):
        """(n, T, 7) daily array in raw units (primary columns filled) for the locations ``idx``."""
        out = np.full((len(idx), len(d.dates), 7), np.nan, dtype="float32")
        if len(idx):
            for src in (m["temp"].predict(dv, fs, idx, t_all), m["do"].predict(dv, fs, idx, t_all)):
                for v, x in src.items():
                    out[:, :, col[v]] = x
        return out

    def to_model(x):
        return (x - scales.label_mean[:7]) / scales.label_std[:7]

    info = {"members": {}, "smoke": a.smoke}
    preds, refits = {}, {}
    need_resume = False
    with VR.training_run(cfg, rdir, seed=0, refit=False, sites=d.sites) as run:
        try:
            for kind in cfg["forcing"].values():
                dv = member_view(d, kind)
                fs = FeatureSet.build(dv, lag=0)
                m1 = fitted("inner", kind, fs, dv, np.array([], int))
                preds[kind] = {"daily": predict(m1, fs, dv, ii), "grab": predict(m1, fs, dv, gi)}
                refits[kind] = fitted("refit", kind, fs, dv, ii)
                info["members"][kind] = {k: {q: m[q] for q in ("fit_s", "n_donors", "n_candidates")}
                                         for k, m in (("inner", m1), ("refit", refits[kind]))}
                del fs
        except TimeoutError as e:
            need_resume = True
            run.note(f"job chunk ended before member {e}; resumed by the next job")
        if not need_resume:
            kinds = list(cfg["forcing"].values())
            names = dict(zip(kinds, MEMBERS))
            sc_d = {k: C.j_sonde(to_model(preds[k]["daily"]), y_in, cf)["by_target"] for k in kinds}
            g_pred = {k: to_model(preds[k]["grab"])[:, :, [jT, jD]] for k in kinds}
            sc_g = {k: C.j_grab(g_pred[k], y_gi)["by_variable"] if len(gi) else {} for k in kinds}
            sel = {v: choose({k: sc_d[k].get(v) for k in kinds}, kinds) for v in VD.PRIMARY}
            sel_i = {v: (choose({k: sc_g[k].get(v) for k in kinds}, kinds) if len(gi) and
                         any(sc_g[k].get(v) is not None for k in kinds) else sel[cfg["inst_members"][v]])
                     for v in VD.INST}
            daily = np.full_like(preds[kinds[0]]["daily"], np.nan)
            for v, k in sel.items():
                daily[:, :, col[v]] = preds[k]["daily"][:, :, col[v]]
            ginst = np.stack([preds[sel_i[v]]["grab"][:, :, j] for v, j in zip(VD.INST, (jT, jD))], axis=-1)
            n_lab = int(np.isfinite(y_in[:, :, list(col.values())]).sum())
            n_nopred = int((np.isfinite(y_in[:, :, list(col.values())]) &
                            ~np.isfinite(daily[:, :, list(col.values())])).sum())
            n_nopred_g = int((np.isfinite(y_gi) & ~np.isfinite(ginst)).sum()) if len(gi) else 0
            if (n_nopred or n_nopred_g) and not a.smoke:
                raise RuntimeError(f"P has no prediction at {n_nopred} labelled inner cells and {n_nopred_g} inner "
                                   "grab rows: the same rows must score every model")
            js = C.j_sonde(to_model(daily), y_in, cf)
            jg = C.j_grab((ginst - scales.label_mean[[jT, jD]]) / scales.label_std[[jT, jD]], y_gi) if len(gi) else None
            mname = lambda v, k: cfg["members"][v][kinds.index(k)]                  # noqa: E731
            info.update({"selected": {v: mname(v, k) for v, k in sel.items()},
                         "selected_inst": {v: mname(cfg["inst_members"][v], k) for v, k in sel_i.items()},
                         "selected_forcing": {**sel, **sel_i},
                         "J": C.combine(js, jg), "j_sonde": js["j_sonde"], "j_grab": None if jg is None else jg["j_grab"],
                         "by_target": js["by_target"], "by_member": {names[k]: sc_d[k] for k in kinds},
                         "by_member_inst": {names[k]: sc_g[k] for k in kinds}, "n_labelled_inner_cells": n_lab,
                         "n_inner_cells_without_prediction": n_nopred, "n_grab_rows_without_prediction": n_nopred_g,
                         "minutes_this_job": round((time.time() - t0) / 60, 2), "n_locations": len(d.sites)})
            keep_keys = ("members", "smoke", "minutes_this_job", "n_locations", "n_labelled_inner_cells",
                         "n_inner_cells_without_prediction", "n_grab_rows_without_prediction")
            VR.write_json(rdir / "physics_info.json", {k: info[k] for k in keep_keys} if quiet else info)
            if not a.smoke:
                for k in ("J", "j_sonde", "j_grab"):
                    if info.get(k) is not None:
                        run.metric(f"best_{k}", info[k])
            run.note(f"{len(kinds)} members x (inner, refit); deterministic; one label form")
    if need_resume:
        log(f"[{name}] exiting with code {VR.EXIT_RESUME} (resume)")
        return VR.EXIT_RESUME
    if a.smoke:
        log(f"[{name}] SMOKE: {info['minutes_this_job']} min; {len(d.sites)} locations; members "
            f"{json.dumps(info['members'])}; cells without a prediction {n_nopred} + {n_nopred_g}; nothing registered")
        return 0
    rec = {"selected": info["selected"], "selected_inst": info["selected_inst"],
           "selected_forcing": info["selected_forcing"], "source_run": str(rdir), "release": VR.run_release(cfg),
           "rule": "per primary endpoint: lowest standardised inner-sonde MSE; per hour-resolved variable: lowest "
                   "term of J_grab; ties: the earlier member (amendment §5); written once"}
    sel_p = VR.ROOT / "p" / "selection.json"
    if sel_p.is_file():
        old = json.loads(sel_p.read_text())
        if any(old[k] != rec[k] for k in ("selected", "selected_inst")):
            raise RuntimeError(f"{sel_p} exists with a different selection; it is written once")
    else:
        VR.write_json(sel_p, rec)
        VR.log_decision("selection", "p.fixed", {k: rec[k] for k in ("selected", "selected_inst", "selected_forcing", "rule")}, rdir=rdir,
                        release=VR.run_release(cfg), value=float(len(rec["selected"]) + len(rec["selected_inst"])))
    joblib.dump({"members": {names[k]: refits[k] for k in kinds}, "selected": info["selected"],
                 "selected_inst": info["selected_inst"], "selected_forcing": info["selected_forcing"],
                 "scales": scales.state(), "forcing": dict(cfg["forcing"])},
                rdir / "model.joblib", compress=3)
    out = {"daily": daily, "inst": ginst} if len(gi) else {"daily": daily}
    np.savez_compressed(rdir / "inner_pred.npz", **{k: v.astype("float32") for k, v in out.items()},
                        date_start=str(d.dates[0].date()), n_inner=len(ii), n_grab_inner=len(gi))
    VR.log_inner(cfg, {"best_J": info["J"], "best_j_sonde": info["j_sonde"], "best_j_grab": info["j_grab"],
                       "best_by_target": info["by_target"], "best_epoch": None}, seed=0, rdir=rdir)
    (rdir / ".done").touch()
    log(f"[{name}] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
