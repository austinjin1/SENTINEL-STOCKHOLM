#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Orbit frozen test path (orbit_protocol.md §9, §9a, §9b, §11, §12) with code-level single-look
enforcement (``sentinel/models/orbit/testlook.py``).

Looks:
* ``M2/test``  — headline cell (in-lake rows, test role x 2024-2025), Test-B (test role x 2026-01..09), pooled;
  strata: >10-km isolation, turbidity unit classes, track-E bloom strata; event track §12 (E4a + tiers, E3 descriptive);
* ``M2U/test`` — track U (held-out HUC2 regions x 2024-2025; U-fit artifacts from FREEZE ``track_u``).

Order (non-dry-run): ``guard.load_freeze`` (ONLY docs/preregistration/FREEZE_orbit.json, status FROZEN, lead
authorisation, candidate sha, code hash, split, dev manifest) -> ``testlook.verify_data_pins`` -> registry refusal +
O_EXCL claim marker + ``look_claimed`` registry row -> protected rows (guard requires the open claim) -> all
predictions written once and sealed (0444, SHA256SUMS, PREDICTIONS_SAVED) -> scoring from the sealed files only ->
registry rows -> O_EXCL SCORED marker. Every artifact is verified against FREEZE ``weights_sha256``; models,
comparators and the blend come only from the freeze, never from the command line.

``--dry-run-dev`` runs the identical predict -> seal -> score path on the development selection cell (val role x
val years, in-lake) under ``$SENTINEL_RUNS/orbit/test_looks_dryrun/<label>/`` WITHOUT a freeze or protected rows,
for checking against scripts/train/orbit_compare.py and scripts/eval/events_orbit.py.

    python -B scripts/eval/orbit_test.py --dry-run-dev --version dev_v5 --satwq satwq_v5_X_s0 ... \
        --baselines baselines_dev_v5 --comparators-json .../comparators_val_selected.json \
        --events-artifact $SENTINEL_RUNS/orbit/events_X/events_orbit_dev_v5.pkl [--blend-with gbm_imagery_context] \
        --label X --register
    python -B scripts/eval/orbit_test.py --allow-test --look M2/test     # only after the lead writes FREEZE_orbit.json
    python -B scripts/eval/orbit_test.py --allow-test --look M2/test --rescore-from-saved   # crash recovery only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.models.orbit import baselines as B  # noqa: E402
from sentinel.models.orbit import data as D  # noqa: E402
from sentinel.models.orbit import evaluate as E  # noqa: E402
from sentinel.models.orbit import features as F  # noqa: E402
from sentinel.models.orbit import guard  # noqa: E402
from sentinel.models.orbit import pipeline as PL  # noqa: E402
from sentinel.models.orbit import testlook as TL  # noqa: E402
from sentinel.models.orbit.split import nearest_distance_km  # noqa: E402

MDN_B7 = ("B01", "B02", "B03", "B04", "B05", "B06", "B07")
SATWQ = "SENTINEL"           # column name of the evaluated model in the sealed predictions


def baseline_predict(name: str, t: str, fitted: dict, df, Ximg, Xctx):
    """Predictions of a (possibly variant) baseline for target t from the persisted fitted objects."""
    if name.startswith("gbm_imagery_context") or name.startswith("gbm_imagery"):
        base, _, v = name.partition("#v")
        X = pd.concat([Ximg, Xctx], axis=1) if base == "gbm_imagery_context" else Ximg
        return fitted[base](df, X, **({"variant": int(v)} if v else {}))[t]
    if name.startswith("location_knn"):
        base, _, k = name.partition("#k")
        return fitted[base](df, None, **({"k": int(k)} if k else {}))[t]
    if name == "gbm_season_location":
        return fitted[name](df, Xctx)[t]
    if name in ("global_median",):
        return fitted[name](df)[t]
    if name == "index_opt":
        return fitted[name](df, Ximg)[t]
    if re.match(r"^mdn_(b(7|12)(_\d+k_(lin|log))?|gloria)$", name):
        bands = D.BANDS if name.startswith("mdn_b12") else MDN_B7
        div = np.pi if name == "mdn_gloria" else 1.0
        Xl = np.stack([df[f"rw3_{b}"].to_numpy(float) / div for b in bands], 1)
        return fitted[(name, t)].predict(Xl)
    if name == "mdn_official":
        from sentinel.models.orbit.mdn import OfficialMDN

        off = OfficialMDN(paths.RAW / "mdn_stream_weights" / "MSI" / "x2025")
        po = off.predict_log10(np.stack([df[f"rw3_{b}"].to_numpy(float) / np.pi for b in OfficialMDN.MSI_BANDS], 1))
        return po.get({"chla": "chl", "tss": "tss"}.get(t, ""), np.full(len(df), np.nan))
    raise KeyError(name)


# =========================================================================== stage 1: predictions (written once)
def predict_cells(cells: dict, df, pix, *, ckpts, fitted, comparators, blend_with, events_art, dev_train, device,
                  pred_dir: Path, log=print) -> dict:
    """Write one parquet per cell: labels (log10), strata inputs, every model's prediction per target and every
    event score (raw and frozen-calibrated). Nothing is scored here."""
    from sentinel.events import orbit as EO
    from sentinel.models.orbit.satwq import mixture_quantiles

    pred_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for cell_name, sel in cells.items():
        if sel.sum() == 0:
            continue
        sub = df[sel].reset_index(drop=True)
        psub = {k: v[sel] for k, v in pix.items()}
        (w, mu, sg), emb = EO.satwq_mixture_embedding(ckpts, sub, psub, device)
        Ximg, Xctx = F.image_features(sub, psub), B.location_season_matrix(sub)
        out = {"row_id": sub.row_id.to_numpy(), "block": sub.block.to_numpy(), "waterbody": sub.waterbody.to_numpy(),
               "dist_train_km": nearest_distance_km(sub.lat, sub.lon, dev_train.lat, dev_train.lon),
               "turbidity_unit_class": sub.turbidity_unit_class.astype(str).to_numpy()}
        for col in list(D.TARGETS) + ["turbidity_fnu", "turbidity_all_units"]:
            if col in sub:
                out[f"y|{col}"] = np.log10(sub[col].to_numpy(float))
                out[f"ylin|{col}"] = sub[col].to_numpy(float)          # event labels use linear values
        for j, t in enumerate(D.TARGETS):
            med = mixture_quantiles(w[:, j], mu[:, j], sg[:, j], (0.5,))[:, 0]
            out[f"pred|{t}|{SATWQ}_mixture_median"] = med
            out[f"pred|{t}|{SATWQ}"] = (0.5 * med + 0.5 * baseline_predict(blend_with, t, fitted, sub, Ximg, Xctx)
                                        if blend_with else med)
            for fam, c in comparators.get(t, {}).items():
                out[f"pred|{t}|{fam}[{c['model']}]"] = baseline_predict(c["model"], t, fitted, sub, Ximg, Xctx)
            if t in ("chla", "tss"):
                out[f"pred|{t}|mdn_official"] = baseline_predict("mdn_official", t, fitted, sub, Ximg, Xctx)
        if events_art is not None:
            jt = {t: j for j, t in enumerate(D.TARGETS)}
            mix = (w[:, jt["chla"]], mu[:, jt["chla"]], sg[:, jt["chla"]])
            chl_pt, mdn_mix, _ = EO.chl_point_inputs(fitted, comparators, sub, Ximg, Xctx)
            Xgbm, spec = pd.concat([Ximg, Xctx], axis=1), EO.spectral_indices(sub)
            for code in EO.E4A_CODES:
                raw = EO.event_scores(events_art, mix=mix, emb=emb, chl_preds=chl_pt, Xgbm=Xgbm, Xclim=Xctx,
                                      spec=spec, mdn_mix=mdn_mix, code=code)
                cal = EO.calibrate_all(events_art, raw, code)
                for c in raw.columns:
                    out[f"event|{code}|{c}|raw"] = raw[c].to_numpy(float)
                    out[f"event|{code}|{c}|cal"] = cal[c].to_numpy(float)
            for code, t in EO.E3_DEFS:
                out[f"event|{code}|derived|raw"] = EO.mixture_p(w[:, jt[t]], mu[:, jt[t]], sg[:, jt[t]], code)
        pd.DataFrame(out).to_parquet(pred_dir / f"cell_{cell_name}.parquet", index=False)
        summary[cell_name] = int(len(sub))
        log(f"[predict] {cell_name}: {len(sub)} rows")
    (pred_dir / "cells.json").write_text(json.dumps(summary, indent=1))
    return summary


# =========================================================================== stage 2: scoring (sealed files only)
def score_sealed(pred_dir: Path, *, n_boot: int, events_art, label: str):
    from sentinel.eval import events as EV
    from sentinel.eval import metrics as M
    from sentinel.events import orbit as EO

    trk = json.loads((paths.REPO_ROOT / "docs" / "preregistration" / "orbit_track_u.json").read_text())["track_E"]
    cells = json.loads((pred_dir / "cells.json").read_text())
    rows, pairs, evrows, evpairs = [], [], [], []
    for cell_name in cells:
        P = pd.read_parquet(pred_dir / f"cell_{cell_name}.parquet")
        blocks, wb = P.block.to_numpy(), P.waterbody.to_numpy()
        for t in D.TARGETS:
            y = P[f"y|{t}"].to_numpy(float)
            if np.isfinite(y).sum() < 20:
                continue
            preds = {(label if m == SATWQ else m): P[c].to_numpy(float) for c in P.columns
                     if c.startswith(f"pred|{t}|") for m in [c.split("|", 2)[2]]}
            strata = {"all": (np.ones(len(P), bool), y), "isolated_gt10km": (P.dist_train_km.to_numpy() > 10, y)}
            if t == "chla":
                for thr in list(trk["thresholds_ugL_fixed"]) + [trk["threshold_fit_p95_ugL"]]:
                    strata[f"trackE_chla_ge{thr:g}"] = (y >= np.log10(thr), y)
            if t == "turbidity":
                cls = P.turbidity_unit_class.to_numpy()
                for u in ("NTU", "unspecified"):
                    strata[f"unit={u}"] = (cls == u, y)
                for col, sname in (("turbidity_fnu", "unit=FNU"), ("turbidity_all_units", "pooled_all_units")):
                    if f"y|{col}" in P:
                        strata[sname] = (np.ones(len(P), bool), P[f"y|{col}"].to_numpy(float))
            for sname, (sm, ys) in strata.items():
                yy = np.where(sm, ys, np.nan)
                if np.isfinite(yy).sum() < 20:
                    continue
                tab, ok = E.score_models(yy, preds, blocks, n_boot=n_boot)
                for k, v in (("stratum", sname), ("cell", cell_name), ("target", t)):
                    tab.insert(0, k, v)
                rows.append(tab)
                for comp in [k for k in preds if k not in (label, f"{SATWQ}_mixture_median")]:
                    r = E.paired_mdsa(yy[ok], preds[label][ok], preds[comp][ok], blocks[ok], n_boot=n_boot)
                    pairs.append({"cell": cell_name, "target": t, "stratum": sname, "comparator": comp, **r})
        # ---- event track (§12)
        if events_art is None:
            continue
        ychl = P["ylin|chla"].to_numpy(float)
        for code in EO.E4A_CODES:
            y = EO.labels(ychl, code)
            m = np.isfinite(y)
            if m.sum() < 20:
                continue
            ref = P[f"event|{code}|climatology|cal"].to_numpy(float)[m]
            prev = events_art["by_event"][code]["prevalence_fit"]
            cuts = events_art["by_event"][code]["fpr10_cut"]
            scs = [c.split("|")[2] for c in P.columns if c.startswith(f"event|{code}|") and c.endswith("|cal")]
            for calib in ("cal", "raw"):
                pd_ = P[f"event|{code}|derived|{calib}"].to_numpy(float)[m]
                for sc in scs:
                    p = P[f"event|{code}|{sc}|{calib}"].to_numpy(float)[m]
                    for unit, g in (("waterbody", wb[m]), ("role_block", blocks[m])):
                        if unit == "role_block" and sc not in ("derived", "direct", "climatology"):
                            continue
                        smx = EV.state_metrics(y[m], p, groups=g, p_ref=ref, n_boot=n_boot)
                        rec = {"cell": cell_name, "event": code, "calib": calib, "score": sc, "unit": unit,
                               "n": smx["n"], "n_pos": smx["n_pos"], "prevalence": smx["prevalence"],
                               "n_groups_pos": smx["n_groups_with_pos"], "power_flag": smx["power_flag"],
                               "ece": smx["ece"], "auroc": M.auroc(y[m], p),
                               "bss_prev": M.brier_skill_score(y[m], p, np.full(m.sum(), prev))}
                        for k in ("auprc", "brier", "bss"):
                            rec.update({k: smx[k]["point"], f"{k}_lo": smx[k]["ci_low"], f"{k}_hi": smx[k]["ci_high"]})
                        if calib == "cal":
                            yy, pp = y[m], p
                            rec["recall_at_frozen_fpr10_cut"] = float((pp[yy == 1] >= cuts[sc]).mean()) if (yy == 1).any() else np.nan
                            rec["fpr_at_frozen_cut"] = float((pp[yy == 0] >= cuts[sc]).mean()) if (yy == 0).any() else np.nan
                        evrows.append(rec)
                        if sc != "derived":
                            cr = EV.compare_routes(y[m], pd_, p, groups=g, n_boot=n_boot)
                            evpairs.append({"cell": cell_name, "event": code, "calib": calib, "comparator": sc,
                                            "unit": unit, "n": cr["n"], "n_pos": cr["n_pos"],
                                            **{f"{k}{s}": cr[k][s2] for k in ("delta_brier", "delta_auprc")
                                               for s, s2 in (("", "diff"), ("_lo", "ci_low"), ("_hi", "ci_high"),
                                                             ("_p", "p_value"))}})
        for code, t in EO.E3_DEFS:
            y = EO.labels(P[f"ylin|{t}"].to_numpy(float), code)
            m = np.isfinite(y)
            p = P[f"event|{code}|derived|raw"].to_numpy(float)
            n_pos = int(np.nansum(y[m]))
            ngp = int(pd.Series(wb[m][y[m] == 1]).nunique())
            rec = {"cell": cell_name, "event": code, "calib": "raw", "score": "derived", "unit": "descriptive",
                   "n": int(m.sum()), "n_pos": n_pos, "n_groups_pos": ngp, "power_flag": EV.power_flag(n_pos, ngp)}
            if 0 < n_pos < m.sum():
                rec.update(auprc=M.auprc(y[m], p[m]), auroc=M.auroc(y[m], p[m]), brier=M.brier(y[m], p[m]),
                           prevalence=float(np.mean(y[m])))
            evrows.append(rec)
    res, pr = pd.concat(rows, ignore_index=True), pd.DataFrame(pairs)
    return res, pr, pd.DataFrame(evrows), pd.DataFrame(evpairs)


def register(res, pr, ev, evp, *, experiment: str, label: str, note: str, version: str, extra: dict):
    from sentinel.core.registry import log_result

    common = dict(workstream=PL.WORKSTREAM, experiment=experiment, data_manifests=[f"orbit_{version}"], extra=extra)
    for r in res.itertuples():
        log_result(model=r.model, metric="mdsa", value=r.mdsa, split=f"{r.cell}:{r.stratum}", n=r.n,
                   ci_low=r.mdsa_ci_low, ci_high=r.mdsa_ci_high, ci_method="role-block cluster bootstrap (B=2000)",
                   target=r.target, is_baseline=r.model != label, notes=note, **common)
    for p in pr.itertuples():
        log_result(model=label, metric=f"mdsa_diff_vs_{p.comparator}", value=p.diff, split=f"{p.cell}:{p.stratum}",
                   n=p.n, ci_low=p.ci_low, ci_high=p.ci_high, target=p.target,
                   ci_method="paired role-block cluster bootstrap (B=2000)",
                   **{**common, "extra": {**extra, "p_value": p.p_value}}, notes=note)
    for r in ev.itertuples():
        for met in ("auprc", "brier", "bss", "ece", "auroc", "bss_prev", "recall_at_frozen_fpr10_cut", "n_pos"):
            v = getattr(r, met, np.nan)
            if v is None or not np.isfinite(v):
                continue
            ci = {}
            if met in ("auprc", "brier", "bss") and np.isfinite(getattr(r, f"{met}_lo", np.nan)):
                ci = dict(ci_low=float(getattr(r, f"{met}_lo")), ci_high=float(getattr(r, f"{met}_hi")),
                          ci_method=f"{r.unit} cluster bootstrap (B=2000)")
            log_result(model=f"{label}:{r.score}", metric=f"{r.event}_{met}" + ("_vs_clim" if met == "bss" else ""),
                       value=float(v), split=f"{r.cell}:{r.calib}:{r.unit}", n=int(r.n), target=r.event,
                       is_baseline=r.score not in ("derived", "direct"), notes=f"{note}; event track §12", **ci, **common)
    for p in evp.itertuples():
        for met in ("delta_brier", "delta_auprc"):
            log_result(model=label, metric=f"{p.event}_{met}_derived_vs_{p.comparator}", value=float(getattr(p, met)),
                       split=f"{p.cell}:{p.calib}:{p.unit}", n=int(p.n), target=p.event,
                       ci_low=float(getattr(p, f"{met}_lo")), ci_high=float(getattr(p, f"{met}_hi")),
                       ci_method=f"paired {p.unit} cluster bootstrap (B=2000)",
                       **{**common, "extra": {**extra, "p_value": float(getattr(p, f"{met}_p"))}},
                       notes=f"{note}; event track §12")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--look", default="M2/test", choices=list(TL.LOOKS))
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--rescore-from-saved", action="store_true", help="crash recovery: score the sealed predictions")
    ap.add_argument("--dry-run-dev", action="store_true")
    ap.add_argument("--satwq", nargs="*", default=None, help="dry-run only (test: from FREEZE)")
    ap.add_argument("--baselines", default=None, help="dry-run only (test: from FREEZE)")
    ap.add_argument("--comparators-json", default=None, help="dry-run only (test: FREEZE comparators)")
    ap.add_argument("--events-artifact", default=None, help="dry-run only (test: FREEZE events_artifact)")
    ap.add_argument("--blend-with", default=None, help="dry-run only (test: FREEZE blend_with)")
    ap.add_argument("--version", default=D.CACHE_VERSION, help="dry-run only (test: FREEZE data_version)")
    ap.add_argument("--label", default="sentinel_orbit", help="dry-run only: run label")
    ap.add_argument("--register", action="store_true", help="dry-run only: register under orbit_dryrun_<version>")
    ap.add_argument("--n-boot", type=int, default=E.N_BOOT)
    ap.add_argument("--gpu", action="store_true", help="lock a GPU via the shared helper (cap 0,1) for inference")
    a = ap.parse_args()
    import cloudpickle

    label = "SENTINEL"
    if a.dry_run_dev:
        if a.allow_test or a.rescore_from_saved:
            raise SystemExit("--dry-run-dev never takes --allow-test / --rescore-from-saved")
        if a.look != "M2/test":
            raise SystemExit("the dry run covers the M2/test code path (track U has no val cell, §11)")
        root = f"{TL.DRYRUN_ROOT}/{a.label}"
        freeze = {"_sha256": f"dryrun:{a.label}", "_path": None, "data_version": a.version,
                  "scoring_code_sha256": guard.scoring_code_sha256()}
        df, pix, _, masks = PL.load_dev(a.version)
        cells = {"valxval_lake": masks[PL.PRIMARY_CELL]}
        dev_train = df[df.role == "train"]
        ckpts = [paths.RUNS_ROOT / "orbit" / r / "model.pt" for r in a.satwq]
        fitted_path = paths.RUNS_ROOT / "orbit" / a.baselines / "fitted_baselines.pkl"
        comparators = json.loads(Path(a.comparators_json).read_text())
        blend_with = a.blend_with
        ev_path = Path(a.events_artifact) if a.events_artifact else None
        experiment, version = f"orbit_dryrun_{a.version}", a.version
        TL.claim(a.look, freeze, root=root, register=False)
    else:
        freeze = guard.load_freeze()       # ONLY FREEZE_orbit.json (status FROZEN, lead-authorised); never a candidate
        if not a.allow_test:
            raise guard.ProtectedDataError("pass --allow-test to score the frozen test look")
        TL.verify_data_pins(freeze)        # before the claim: a data mismatch never burns the look
        root = TL.ROOT
        version = freeze["data_version"]
        experiment = TL.experiment(a.look, version)
        art = freeze if a.look == "M2/test" else freeze["track_u"]
        ckpts = art["satwq_checkpoints"]
        for c in ckpts:
            guard.verify_weights(freeze, Path(c).parent.name, c)
        fitted_path = art["fitted_baselines"]
        guard.verify_weights(freeze, "fitted_baselines" if a.look == "M2/test" else "fitted_baselines_track_u",
                             fitted_path)
        ev_path = Path(art["events_artifact"]) if art.get("events_artifact") else None
        if ev_path is not None:
            guard.verify_weights(freeze, "events_orbit" if a.look == "M2/test" else "events_orbit_track_u", ev_path)
        blend_with = freeze.get("blend_with")
        comparators = freeze["comparators"]
        if not a.rescore_from_saved:
            TL.claim(a.look, freeze)       # registry refusal + O_EXCL marker + look_claimed row
    look_dir = TL.look_dir(a.look, root)
    pred_dir = look_dir / "predictions"
    events_art = cloudpickle.loads(Path(ev_path).read_bytes()) if ev_path else None
    if events_art is not None:
        from sentinel.events import orbit as EO

        EO.EV()                                              # only the released events.py v1.0
        if events_art.get("events_module_sha256") != EO.EVENTS_V1_SHA256:
            raise guard.ProtectedDataError("event artifact was not fitted with the released events.py v1.0")
    if not a.rescore_from_saved:
        if a.gpu:
            PL.setup_gpu(f"orbit_test_{a.label}")
        import torch

        torch.set_num_threads(2)
        device = "cuda" if a.gpu else "cpu"
        fitted = cloudpickle.loads(Path(fitted_path).read_bytes())["fitted"]
        if not a.dry_run_dev:
            df_all, _ = D.build_table(allow_test=True)            # guard: FREEZE + open claim of this freeze
            df = df_all[df_all.protected.to_numpy()].reset_index(drop=True)
            pix = D.extract_pixels(df)
            dev_train = D.load_dev(version, training=False)[0]
            dev_train = dev_train[dev_train.role == "train"]
            if a.look == "M2U/test":
                from sentinel.models.orbit import track_u as U

                U.verify_track_file()
                cells = {"trackU_test": U.test_mask(df, U.assign(df.anchor_lon, df.anchor_lat))}
            else:
                base = (df.role == "test") & df.in_lake & ~df.buffer_excluded
                sel = (base & (df.tblock == "test")).to_numpy()
                # Test-B (protocol v0.3 §9a, declared 2026-11-08 before any test look)
                sel_b = (base & (df.tblock == "test_b") & (pd.to_datetime(df.date) <= pd.Timestamp("2026-09-30"))).to_numpy()
                cells = {"headline": sel, "testB_2026": sel_b, "pooled_2024_2026": sel | sel_b}
        predict_cells(cells, df, pix, ckpts=ckpts, fitted=fitted, comparators=comparators, blend_with=blend_with,
                      events_art=events_art, dev_train=dev_train, device=device, pred_dir=pred_dir)
        TL.seal_predictions(a.look, pred_dir, root=root)
    sealed = TL.check_sealed(a.look, root=root)
    res, pr, ev, evp = score_sealed(sealed, n_boot=a.n_boot, events_art=events_art, label=label)
    out = paths.ensure(look_dir / "scores")
    for name, tab in (("metrics", res), ("paired", pr), ("event_metrics", ev), ("event_paired", evp)):
        tab.to_csv(out / f"{name}.csv", index=False)
    pd.set_option("display.width", 250)
    print(res[res.stratum == "all"].round(3).to_string(index=False))
    print(pr[pr.stratum == "all"].round(3).to_string(index=False))
    if len(ev):
        print(ev[(ev.event == "E4a") & (ev.unit == "waterbody")].round(3).to_string(index=False))
    note = ("TEST (single look after freeze)" if not a.dry_run_dev else
            "dry run of orbit_test.py on the val cell (development, not a claim)")
    extra = {"look": a.look, "freeze_sha256": freeze["_sha256"], "dry_run": bool(a.dry_run_dev)}
    if not a.dry_run_dev or a.register:
        register(res, pr, ev, evp, experiment=experiment, label=label, note=note, version=version, extra=extra)
    TL.mark_scored(a.look, {"experiment": experiment, "n_metric_rows": int(len(res)), "n_event_rows": int(len(ev)),
                            "freeze_sha256": freeze["_sha256"]}, root=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
