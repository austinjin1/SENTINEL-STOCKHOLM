#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Orbit event track, development (orbit_protocol.md §12): fit the event objects on the fit cell (tuning on
inner), compute raw and cross-fitted-recalibrated P(E4a) etc. on the val cells, fit the frozen full-val Platt
calibrators, register validation metrics, and write the event artifact ``events_orbit_<version>.pkl`` (pinned in the
freeze). Validation only: protected rows are never loaded (``PL.load_dev``).

    python -B scripts/eval/events_orbit.py --version dev_v4 --satwq satwq_v4_noaug_s0 satwq_v4_noaug_s1 \
        satwq_v4_noaug_s2 --baselines baselines_dev_v4 \
        --comparators-json $SENTINEL_RUNS/orbit/compare_satwq_v4_noaug/comparators_val_selected.json \
        --label satwq_v4_noaug --register
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.events import orbit as EO  # noqa: E402
from sentinel.eval import events as EV  # noqa: E402
from sentinel.eval import metrics as M  # noqa: E402
from sentinel.models.orbit import baselines as B  # noqa: E402
from sentinel.models.orbit import data as D  # noqa: E402
from sentinel.models.orbit import features as F  # noqa: E402
from sentinel.models.orbit import pipeline as PL  # noqa: E402

CELLS = ("valxval_lake", "valxany_lake")
CAL_CELL = "valxval_lake"
N_FOLDS = 5


def recall_at_cut(y, p, cut):
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    return float((p[y == 1] >= cut).mean()) if (y == 1).any() else np.nan, float((p[y == 0] >= cut).mean())


def fpr_cut(y, p, fpr=0.10):
    ok = np.isfinite(y) & np.isfinite(p)
    neg = np.sort(p[ok & (y == 0)])
    return float(np.quantile(neg, 1 - fpr)) if neg.size else np.nan


def crossfit_calibrate(p, y, groups, n_folds=N_FOLDS):
    """Platt map per fold fitted on the other water-body folds (val metrics are out-of-fold)."""
    from sklearn.model_selection import GroupKFold

    out = np.full(len(p), np.nan)
    ok = np.isfinite(p) & np.isfinite(y)
    idx = np.where(ok)[0]
    for tr, te in GroupKFold(n_folds).split(idx, groups=groups[idx]):
        out[idx[te]] = EV.recalibrate(p[idx[tr]], y[idx[tr]])(p[idx[te]])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--satwq", nargs="+", required=True)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--comparators-json", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--fit-scheme", default="base", choices=["base", "track_u"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    import cloudpickle

    from sentinel.core.runlog import TrainingRun

    R = paths.RUNS_ROOT / "orbit"
    out_dir = paths.ensure(R / (f"events_{a.label}" + ("" if a.fit_scheme == "base" else f"_{a.fit_scheme}")))
    exp = f"orbit_events_val_{a.version}"
    cfg = {"version": a.version, "satwq": a.satwq, "baselines": a.baselines, "fit_scheme": a.fit_scheme,
           "head_Cs": list(EO.HEAD_CS), "gbm_cls_grid_n": len(EO.GBM_CLS_GRID), "n_folds": N_FOLDS}
    EO.EV()                                  # refuses anything but the released events.py v1.0
    with TrainingRun(PL.WORKSTREAM, f"events_{a.label}", config=cfg, data=[f"orbit_{a.version}"],
                     plan_ref="orbit_protocol.md §12; docs/plans/event_layer_plan.md", iteration="events-v1",
                     artifacts_dir=str(out_dir)) as run:
        if a.gpu:
            PL.setup_gpu(f"orbit_events_{a.label}")
        import torch

        torch.set_num_threads(2)
        device = "cuda" if a.gpu else "cpu"
        df, pix, info, masks = PL.load_dev(a.version, a.fit_scheme)
        use = masks["fit"] | masks["inner"] | masks["valxany_lake"]
        sub = df[use].reset_index(drop=True)
        psub = {k: v[use] for k, v in pix.items()}
        mk = {k: v[use] for k, v in masks.items() if k in ("fit", "inner") + CELLS}
        print(f"[events] rows: fit {mk['fit'].sum()} inner {mk['inner'].sum()} "
              + " ".join(f"{c} {mk[c].sum()}" for c in CELLS), flush=True)
        ckpts = [R / r / "model.pt" for r in a.satwq]
        (w, mu, sg), emb = EO.satwq_mixture_embedding(ckpts, sub, psub, device)
        jt = {t: j for j, t in enumerate(D.TARGETS)}
        mix_chl = (w[:, jt["chla"]], mu[:, jt["chla"]], sg[:, jt["chla"]])
        fitted = cloudpickle.loads((R / a.baselines / "fitted_baselines.pkl").read_bytes())["fitted"]
        comparators = json.loads(Path(a.comparators_json).read_text())
        Ximg, Xctx = F.image_features(sub, psub), B.location_season_matrix(sub)
        Xgbm = pd.concat([Ximg, Xctx], axis=1)
        spec = EO.spectral_indices(sub)
        chl_pt, mdn_mix, names = EO.chl_point_inputs(fitted, comparators, sub, Ximg, Xctx)
        ylog = {t: np.log10(sub[t].to_numpy(float)) for t in D.TARGETS}
        fit, inner = mk["fit"], mk["inner"]
        art = {"version": a.version, "fit_scheme": a.fit_scheme, "satwq_runs": a.satwq, "baselines": a.baselines,
               "comparator_models": names, "events_module_version": EV.__version__,
               "events_module_sha256": EO.EVENTS_V1_SHA256,
               "resid_sd": {k: EO.residual_sd(v[inner], ylog["chla"][inner]) for k, v in chl_pt.items()},
               "by_event": {}}
        run.note(f"resid_sd inner: {art['resid_sd']}")
        rows, pairs, per_row = [], [], []
        for code in EO.E4A_CODES:
            y = EO.labels(sub["chla"].to_numpy(float), code)
            ev = {"direct_head": EO.fit_direct_head(emb[:, fit], y[fit], emb[:, inner], y[inner]),
                  "gbm_direct": EO.fit_gbm_classifier(Xgbm[fit], y[fit], Xgbm[inner], y[inner]),
                  "climatology": EO.fit_gbm_classifier(Xctx[fit], y[fit], Xctx[inner], y[inner]),
                  "spectral": {s: EO.fit_univariate_logistic(spec[s].to_numpy(float)[fit], y[fit]) for s in EO.SPECTRAL},
                  "prevalence_fit": float(np.nanmean(y[fit]))}
            art["by_event"][code] = ev
            print(f"[events] {code}: head C={ev['direct_head']['C']} gbm_direct cfg={ev['gbm_direct']['config_index']} "
                  f"clim cfg={ev['climatology']['config_index']} prev_fit={ev['prevalence_fit']:.3f}", flush=True)
            raw = EO.event_scores(art, mix=mix_chl, emb=emb, chl_preds=chl_pt, Xgbm=Xgbm, Xclim=Xctx, spec=spec,
                                  mdn_mix=mdn_mix, code=code)
            cal_m = mk[CAL_CELL] & np.isfinite(y)
            ev["calibrators"] = {c: EV.recalibrate(raw[c].to_numpy()[cal_m], y[cal_m]) for c in raw.columns}
            calp = EO.calibrate_all(art, raw, code)
            ev["fpr10_cut"] = {c: fpr_cut(y[cal_m], calp[c].to_numpy()[cal_m]) for c in raw.columns}
            for cell in CELLS:
                m = mk[cell] & np.isfinite(y)
                yy, wb, bl = y[m], sub.waterbody.to_numpy()[m], sub.block.to_numpy()[m]
                versions = {"raw": raw[m].reset_index(drop=True),
                            "crossfit": pd.DataFrame({c: crossfit_calibrate(raw[c].to_numpy()[m], yy, wb)
                                                      for c in raw.columns})}
                for vname, P in versions.items():
                    ref = P["climatology"].to_numpy()
                    for c in P.columns:
                        p = P[c].to_numpy()
                        for unit, g in (("waterbody", wb), ("role_block", bl)):
                            if unit == "role_block" and c not in ("derived", "direct", "climatology"):
                                continue
                            sm = EV.state_metrics(yy, p, groups=g, p_ref=ref, n_boot=a.n_boot)
                            rec, fpr_r = recall_at_cut(yy, p, fpr_cut(yy, p))
                            rows.append({"event": code, "cell": cell, "calib": vname, "score": c, "unit": unit,
                                         "n": sm["n"], "n_pos": sm["n_pos"], "prevalence": sm["prevalence"],
                                         "n_groups": sm["n_groups"], "n_groups_pos": sm["n_groups_with_pos"],
                                         "power_flag": sm["power_flag"],
                                         **{f"{k}{s}": sm[k][s2] for k in ("auprc", "brier", "bss")
                                            for s, s2 in (("", "point"), ("_lo", "ci_low"), ("_hi", "ci_high"))},
                                         "bss_prev": M.brier_skill_score(yy, p, np.full(len(yy), ev["prevalence_fit"])),
                                         "ece": sm["ece"],
                                         "auroc": M.auroc(yy, p), "recall_fpr10": rec})
                        if c == "derived":
                            continue
                        for unit, g in (("waterbody", wb), ("role_block", bl)):
                            cr = EV.compare_routes(yy, P["derived"].to_numpy(), p, groups=g, n_boot=a.n_boot)
                            pairs.append({"event": code, "cell": cell, "calib": vname, "comparator": c, "unit": unit,
                                          "n": cr["n"], "n_pos": cr["n_pos"],
                                          **{f"{k}{s}": cr[k][s2] for k in ("delta_brier", "delta_auprc")
                                             for s, s2 in (("", "diff"), ("_lo", "ci_low"), ("_hi", "ci_high"),
                                                           ("_p", "p_value"))}})
                    if cell == CAL_CELL:
                        per_row.append(P.assign(event=code, calib=vname, row_id=sub.row_id.to_numpy()[m], y=yy,
                                                waterbody=wb, block=bl))
        # E3 (descriptive, derived route only): counts on every val row, metrics only where positives exist
        e3 = []
        for code, t in EO.E3_DEFS:
            y = EO.labels(sub[t].to_numpy(float), code)
            p = EO.mixture_p(w[:, jt[t]], mu[:, jt[t]], sg[:, jt[t]], code)
            for cell in CELLS:
                m = mk[cell] & np.isfinite(y)
                rec = {"event": code, "cell": cell, "n": int(m.sum()), "n_pos": int(np.nansum(y[m])),
                       "n_waterbodies_pos": int(pd.Series(sub.waterbody.to_numpy()[m][y[m] == 1]).nunique()),
                       "mean_p": float(np.nanmean(p[m])) if m.any() else np.nan}
                rec["power_flag"] = EV.power_flag(rec["n_pos"], rec["n_waterbodies_pos"])
                if rec["n_pos"] > 0 and rec["n_pos"] < rec["n"]:
                    rec["auprc_raw"] = M.auprc(y[m], p[m])
                e3.append(rec)
        res, pr, e3 = pd.DataFrame(rows), pd.DataFrame(pairs), pd.DataFrame(e3)
        res.to_csv(out_dir / "event_metrics.csv", index=False)
        pr.to_csv(out_dir / "event_paired.csv", index=False)
        e3.to_csv(out_dir / "event_e3_descriptive.csv", index=False)
        pd.concat(per_row, ignore_index=True).to_parquet(out_dir / "val_event_scores.parquet", index=False)
        art_path = out_dir / f"events_orbit_{a.version}.pkl"
        art_path.write_bytes(cloudpickle.dumps(art))
        run.note(f"artifact {art_path}")
        prim = res[(res.event == "E4a") & (res.cell == CAL_CELL) & (res.unit == "waterbody")]
        print(prim[["calib", "score", "n", "n_pos", "auprc", "auprc_lo", "auprc_hi", "brier", "bss", "bss_lo",
                    "bss_hi", "ece", "auroc"]].round(3).to_string(index=False))
        pp = pr[(pr.event == "E4a") & (pr.cell == CAL_CELL) & (pr.unit == "waterbody") & (pr.calib == "crossfit")]
        print(pp[["comparator", "delta_brier", "delta_brier_lo", "delta_brier_hi", "delta_auprc", "delta_auprc_lo",
                  "delta_auprc_hi"]].round(4).to_string(index=False))
        print(e3.to_string(index=False))
        d = prim[(prim.calib == "crossfit") & (prim.score == "derived")].iloc[0]
        run.metric("E4a_auprc_derived_crossfit", d.auprc)
        run.metric("E4a_bss_clim_derived_crossfit", d.bss)
        if a.register:
            from sentinel.core.registry import log_result

            common = dict(workstream=PL.WORKSTREAM, experiment=exp, data_manifests=[f"orbit_{a.version}"],
                          artifacts=[str(art_path)], config=cfg)
            for r in res.itertuples():
                split = f"{r.cell}@{a.version}:{r.calib}:{r.unit}"
                for met in ("auprc", "brier", "bss"):
                    v = getattr(r, met)
                    if np.isfinite(v):
                        log_result(model=f"{a.label}:{r.score}", metric=f"{r.event}_{met}" + ("_vs_clim" if met == "bss" else ""),
                                   value=float(v), split=split, n=int(r.n), ci_low=float(getattr(r, f"{met}_lo")),
                                   ci_high=float(getattr(r, f"{met}_hi")), target="chla",
                                   ci_method=f"{r.unit} cluster bootstrap (B={a.n_boot})",
                                   is_baseline=r.score not in EO.ROUTES,
                                   notes="validation (development), event track §12",
                                   extra={"n_pos": int(r.n_pos), "prevalence": r.prevalence, "power_flag": r.power_flag},
                                   **common)
                if r.unit == "waterbody":
                    for met in ("ece", "auroc", "recall_fpr10", "bss_prev"):
                        v = getattr(r, met)
                        if np.isfinite(v):
                            log_result(model=f"{a.label}:{r.score}", metric=f"{r.event}_{met}", value=float(v),
                                       split=split, n=int(r.n), target="chla", is_baseline=r.score not in EO.ROUTES,
                                       notes="validation (development), event track §12", **common)
            for p in pr.itertuples():
                split = f"{p.cell}@{a.version}:{p.calib}:{p.unit}"
                for met in ("delta_brier", "delta_auprc"):
                    log_result(model=a.label, metric=f"{p.event}_{met}_derived_vs_{p.comparator}",
                               value=float(getattr(p, met)), split=split, n=int(p.n), target="chla",
                               ci_low=float(getattr(p, f"{met}_lo")), ci_high=float(getattr(p, f"{met}_hi")),
                               ci_method=f"paired {p.unit} cluster bootstrap (B={a.n_boot})",
                               extra={"p_value": float(getattr(p, f"{met}_p")), "n_pos": int(p.n_pos)},
                               notes="validation (development), event track §12", **common)
            for r in e3.itertuples():
                log_result(model=f"{a.label}:derived", metric=f"{r.event}_n_pos", value=float(r.n_pos),
                           split=f"{r.cell}@{a.version}", n=int(r.n), target=r.event,
                           notes=f"descriptive (power flag {r.power_flag}), event track §12", **common)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
