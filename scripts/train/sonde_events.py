#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""EW-v2 event heads for SENTINEL-Sonde: P(onset within 1/3/7 d) for H1 (+tiers), H4, H5b.

Two phases (pre-training review C2: never fit on rows that are scored):

``--phase calibrate`` (train + VAL blocks only; refuses test data)
  * fits every method on train-role sites × TRAIN block (event GBM: fit on issue dates with
    d + L <= 2019-09-30, early-stopped on the unscored inner slice WY2020–21 issued >= 2019-11-12;
    climatology: pooled fit + per-site deviations from each site's OWN training-year rows);
  * on the VAL block fits Platt recalibration for EVERY probabilistic method INCLUDING the
    climatology reference (protocol decision 2026-10-24: BSS reference = equally recalibrated
    climatology; paired ΔBS vs climatology/persistence/logistic reported) and the alarm
    thresholds at the PRIMARY alarm-day budget (pooled alarm days on valid non-onset days
    <= 1 / 30, sensitivity 2 / 30; ``early_warning.calibrate_threshold``);
  * writes model artifacts + ``event_calibration.json`` (thresholds, Platt a/b, artifact SHA-256s)
    and its SHA-256 (the value for FREEZE_sonde.json ``thresholds_sha256``);
  * descriptive val scores are OUT-OF-FOLD by water year (Platt and thresholds fitted on the other
    WY), so no reported val number uses its own rows.

``--phase score --calibration <event_calibration.json>``
  * loads the frozen artifacts (SHA-256 checked) and applies them unchanged; the only per-site
    computation is the climatology deviation of sites unseen at calibration, from their own
    training-year labels. Protected blocks need --allow-test, a valid freeze file and
    ``thresholds_sha256`` == SHA-256 of the calibration file. Scoring the val block with the frozen
    file is labelled in-sample.

Metrics: Brier score and BSS vs the site × DOY climatology (site-month block-bootstrap CI),
AUROC/AUPRC, realised alarm-day rates at the frozen thresholds (pooled, per site, time in alarm),
placebo comparison (clean, season-matched pseudo-onsets) incl. POD with >= 24 h warning; episode
counts only as labelled secondary descriptives. Quantile-model forecasts for the scored block must
come from scripts/eval/sonde_predict.py (frozen checkpoints).

    python -B scripts/train/sonde_events.py --phase calibrate --config configs/sonde/events.yaml \
        --quantile-model gbm=<predict_run>/forecasts --lstm-checkpoint <lstm_run>/lstm_best.pt
    python -B scripts/train/sonde_events.py --phase score --calibration <cal_run>/event_calibration.json \
        --config configs/sonde/events.yaml --eval-blocks test --allow-test --include-test-sites \
        --quantile-model gbm=<predict_test_run>/forecasts --lstm-checkpoint <lstm_run>/lstm_best.pt
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.data.sonde_splits import TIME_BLOCKS
from sentinel.eval import ew_controls as EWC
from sentinel.eval import ew_scoring as EWS
from sentinel.eval.hazards import HAZARDS
from sentinel.models.sonde import common, guard, runner
from sentinel.models.sonde import events as E
from sentinel.models.sonde.features import FeatureBuilder

NOT_PROB = {"threshold"}          # scores that are not probabilities (no Brier/BSS, no Platt)
NO_PLATT = {"stack"}              # already calibrated on the val block (logistic stack); no Platt on top
STACK_DEFAULT = ("event_gbm", "event_gbm_mono", "logistic", "persistence", "climatology")
BASELINES = ("climatology", "persistence", "logistic", "event_gbm")   # paired ΔBS references (protocol §7, H-b)
PRIMARY_FAMILY = ("H1_hypoxia", "H2_chloride", "H4_nitrate")         # Holm family (fixed, m = 3)
PRIMARY_LEAD = 3
MIN_ONSETS = 50
ROLE_CELLS = {"test_role": "test", "train_role": "train", "val_role": "val"}
FIT_END = "2019-09-30"            # event-GBM fit rows: d + L <= FIT_END
ES_START = pd.Timestamp("2019-10-01") + pd.Timedelta(days=42)   # inner early-stopping slice


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- shared pieces

def site_frames(data, lab, cand_mask, rows: E.EventRows, score: np.ndarray, block: str, L: int):
    """Per-site daily frames over the block's issue dates + onset events, ALL event runs and the
    clean placebo-candidate mask (for the placebo control)."""
    blocks_ = (block,) if isinstance(block, str) else tuple(block)
    t_idx = np.unique(np.concatenate([data.issue_idx(b, L) for b in blocks_]))
    dates = data.cube.dates[t_idx]
    pos_of_t = {int(t): k for k, t in enumerate(t_idx)}
    lo, hi = int(t_idx[0]), min(int(t_idx[-1]) + L, len(data.cube.dates) - 1)
    frames, onsets, events, excl, cands = {}, {}, {}, {}, {}
    x = data.cube.y[:, :, data.cube.var_index(HAZARDS[lab.hazard].variable)]
    for s in np.unique(rows.si):
        m = rows.si == s
        sc = np.full(t_idx.size, np.nan)
        use = np.zeros(t_idx.size, bool)
        k = np.array([pos_of_t[int(t)] for t in rows.ti[m]])
        sc[k], use[k] = score[m], True
        site = data.sites[s]
        frames[site] = pd.DataFrame({"score": sc, "usable": use}, index=dates)
        on_idx = np.flatnonzero(lab.onset[s, lo:hi + 1]) + lo
        onsets[site] = pd.DatetimeIndex(data.cube.dates[on_idx])
        evs = EWS.daily_events(site, data.cube.dates, lab.event[s], lab.onset[s], x[s])
        events[site] = [e for e in evs if data.cube.dates[lo] <= e.onset <= data.cube.dates[hi]]
        excl[site] = E.event_runs(site, data.cube.dates, lab.event[s])
        cands[site] = cand_mask[s, t_idx]
    return frames, onsets, events, excl, cands


def evaluate_scores(data, lab, cand, ev, scores: dict, p_ref, thresholds: dict, block, L, ew, seed,
                    tag: str | None = None, cell: str = "all", keep_frames: dict | None = None) -> list[dict]:
    """thresholds[name] = {'b1': thr, 'b2': thr} or per-row arrays (out-of-fold). ``block`` is the
    data block; ``tag`` the label written to the output (e.g. 'val_oof_by_wy')."""
    out = []
    for name, p in scores.items():
        p = np.asarray(p, float)
        rec = {"hazard": lab.hazard, "lead": L, "block": tag or block, "cell": cell, "method": name,
               "n_nonfinite": int((~np.isfinite(p)).sum())}
        if name not in NOT_PROB:
            rec.update(EWS.brier_skill(ev.y, p, p_ref, site=ev.site_no, dates=ev.issue_date,
                                       n_boot=int(ew["n_boot"]), seed=seed,
                                       basin={str(k): str(v) for k, v in data.split.basin_group.items()}))
        rec.update(EWS.discrimination(ev.y, p))
        if name not in NOT_PROB:   # paired ΔBS vs each baseline on the same rows (negative = better)
            for base in BASELINES:
                if base in scores and base != name:
                    d = EWS.delta_brier(ev.y, p, scores[base], site=ev.site_no, dates=ev.issue_date, n_boot=int(ew["n_boot"]),
                                        seed=seed, basin={str(k): str(v) for k, v in data.split.basin_group.items()})
                    rec.update({f"vs_{base}_{k}": v for k, v in d.items()})
        valid = np.ones(len(ev), bool)
        for bt in ("b1", "b2"):
            thr = thresholds[name][bt]
            margin = p - thr                     # alarm iff margin >= 0 (thr may vary by fold)
            st = EWS.alarm_day_stats(margin, ev.y, valid, ev.site_no, 0.0,
                                     float(ew["budget_per_30d" if bt == "b1" else "budget_sensitivity"]))
            rec.update({f"{bt}_{k}": v for k, v in st.items() if k not in ("threshold",)})
            if np.ndim(thr) == 0:
                rec[f"{bt}_threshold"] = float(thr)
            if bt == "b1":
                frames, onsets, events, excl, cands = site_frames(data, lab, cand, ev, margin, block, L)
                plc = EWS.placebo_comparison(frames, events, 0.0, L, k_per_onset=int(ew["k_placebo"]), seed=seed,
                                             exclude_events=excl, candidates=cands, return_frames=True)
                cl = EWC.placebo_cluster(plc.pop("_det"), plc.pop("_plc"),
                                         {str(k): str(v) for k, v in data.split.basin_group.items()},
                                         n_boot=int(ew["n_boot"]), seed=seed)
                rec.update({f"b1_{k}": v for k, v in plc.items()})
                rec.update({f"b1_plc_cluster_{k}": v for k, v in cl.items()})
                if keep_frames is not None:
                    keep_frames[name] = (frames, events)
                epi = EWS.StackedFrames(frames, onsets, L).stats(0.0)          # SECONDARY descriptive only
                rec.update({f"b1_desc_episode_{k}": v for k, v in epi.items()
                            if k in ("n_episodes", "n_false_episodes", "fa_per_site_30d")})
        out.append(rec)
    return out


def restrict_to_sc_rows(data, ev, block):
    """H2: keep only event rows that also have SC forecast rows (identical rows for every method)."""
    rows_sc = data.rows("spcond_uscm_mean", block)
    key = pd.MultiIndex.from_arrays([rows_sc.si, rows_sc.ti])
    ok = key.get_indexer(pd.MultiIndex.from_arrays([ev.si, ev.ti])) >= 0
    return E.EventRows(ev.hazard, ev.lead, ev.block, ev.si[ok], ev.ti[ok], ev.y[ok], ev.site_no[ok],
                       ev.issue_date[ok], ev.role[ok])


def quantile_inputs(data, var, block, ev, qmodels: dict, naive: list, log) -> dict:
    if var in E.CHLORIDE_HAZARD_VARS:   # forecast SC, map through each site's fixed Cl = a + b·SC
        reg = common.chloride_regressions()
        a_ = reg["a"].reindex(ev.site_no).to_numpy(float)
        b_ = reg["b"].reindex(ev.site_no).to_numpy(float)
        q_sc = quantile_inputs(data, "spcond_uscm_mean", block, ev, qmodels, naive, log)
        return {m: E.sc_quantiles_to_chloride(q, a_, b_) for m, q in q_sc.items()}
    rows_v = data.rows(var, block)
    key = pd.MultiIndex.from_arrays([rows_v.si, rows_v.ti])
    pos = key.get_indexer(pd.MultiIndex.from_arrays([ev.si, ev.ti]))
    if (pos < 0).any():
        raise ValueError("event rows not contained in the common forecast rows")
    out = {m: common.baseline_forecast(data, var, rows_v, m)[pos] for m in naive}
    for name, d in qmodels.items():
        f = Path(d) / f"{name}__{var}__{block}.npz"
        if f.is_file():
            out[name] = common.align_forecast(common.load_forecast(f), rows_v, data.horizons)[pos]
        else:
            log(f"[events] missing forecast file {f} (method skipped)")
    return out


def raw_scores(data, hz, L, ev, X, models: dict, qsets: dict, combine, lstm) -> dict:
    sc = {"climatology": models["climatology"].predict(ev.site_no, E.doy_of(data, ev.ti)),
          "persistence": models["persistence"].predict(X["margin_last"].to_numpy()),
          "threshold": -X["margin_last"].to_numpy(dtype=float)}
    if "logistic" in models:
        sc["logistic"] = models["logistic"].predict(X)
    for name in sorted(k for k in models if k.startswith("event_gbm")):
        sc[name] = models[name].predict(X)
    if "lstm_head" in models and lstm is not None:
        sc["lstm_head"] = models["lstm_head"].predict(lstm.embed(ev.si, ev.ti))
    for m, q in qsets.items():
        for comb in combine:
            sc[f"q_{m}_{comb}"] = E.onset_prob_from_quantiles(q, data.horizons, L, hz, combine=comb)
    return sc


def extra_feature_blocks(cfg, data, var, log):
    """Optional feature blocks shared by ALL feature-based methods (parity): lagged observed weather
    (use_weather) and hourly-tier sub-daily DO features (use_subdaily, DO hazards only)."""
    from sentinel.models.sonde.subdaily import load_subdaily, subdaily_blocks

    out = {}
    wvars = cfg.get("weather_vars")   # optional: restrict the weather block to these hazard variables
    if cfg.get("use_weather") and data.weather is not None and (not wvars or var in wvars):
        fbw = FeatureBuilder(data.cube, data.clim_mean, data.clim_std, data.static)
        out.update(fbw.weather_block(data.weather))
        if var in E.CHLORIDE_HAZARD_VARS:
            out.update(fbw.snow_block(data.weather))
    if cfg.get("use_subdaily") and var.startswith("do_mgl"):
        if "_subdaily" not in data.__dict__:
            reg = pd.read_parquet(common.SITE_REGISTRY)
            reg = reg[reg["network"] == "USGS"].set_index("site_id")["utc_offset_h"].to_dict()
            data.__dict__["_subdaily"] = load_subdaily(data.sites, data.cube.dates, reg)
            n = int(np.isfinite(data.__dict__["_subdaily"][..., 0]).any(axis=1).sum())
            log(f"[events] sub-daily features: {n} sites with hourly DO")
        out.update(subdaily_blocks(data.__dict__["_subdaily"]))
    return out


def site_training_rows(data, lab, L, roles):
    tr_all = E.event_rows(data, lab, L, "train", roles=roles)
    assert (data.cube.dates[tr_all.ti] + pd.Timedelta(days=L)).max() <= pd.Timestamp(TIME_BLOCKS["train"][1])
    return {s: (E.doy_of(data, tr_all.ti[tr_all.site_no == s]), tr_all.y[tr_all.site_no == s])
            for s in np.unique(tr_all.site_no)}


# ----------------------------------------------------------------------------- phases

def check_chloride_consistency(data, rtol: float = 1e-5, atol: float = 1e-3) -> dict:
    """Review C1 (2026-11-08): the attached derived chloride must equal clip(a + b·SC, 0) of the SC
    in the loaded cube (daily mean -> cl_mgl_mean, daily max -> cl_mgl_max) on every cube day, with
    the identical NaN pattern, so a series built from a superseded SC snapshot cannot be scored."""
    regs = common.chloride_regressions()
    cube = data.cube
    jm, jx = cube.var_index("spcond_uscm_mean"), cube.var_index("spcond_uscm_max") \
        if "spcond_uscm_max" in cube.variables else None
    km, kx = cube.var_index("cl_mgl_mean"), cube.var_index("cl_mgl_max")
    bad, n_sites, max_abs = [], 0, 0.0
    for i, s in enumerate(cube.sites):
        if str(s) not in regs.index:
            if np.isfinite(cube.y[i, :, km]).any():
                bad.append((str(s), "series without an eligible regression"))
            continue
        a_, b_ = float(regs.loc[str(s), "a"]), float(regs.loc[str(s), "b"])
        for j, k in ((jm, km), (jx, kx)):
            if j is None:
                continue
            want = np.clip(a_ + b_ * cube.y[i, :, j].astype(float), 0, None)
            have = cube.y[i, :, k].astype(float)
            fin = np.isfinite(have)
            if not np.array_equal(np.isfinite(want), fin):
                bad.append((str(s), f"{cube.variables[k]} NaN pattern"))
                break
            if fin.any():
                d = np.abs(want[fin] - have[fin])
                max_abs = max(max_abs, float(d.max()))
                if not np.all(d <= atol + rtol * np.abs(want[fin])):
                    bad.append((str(s), cube.variables[k]))
                    break
        n_sites += 1
    if bad:
        raise guard.ProtectedDataError(f"derived chloride inconsistent with the loaded SC cube "
                                       f"({len(bad)} sites, e.g. {bad[:5]}): rebuild chloride_derived")
    return {"chloride_sites_checked": n_sites, "chloride_max_abs_diff": max_abs}


def maybe_attach_chloride(cfg, data, log):
    if any(h.startswith("H2_") for h in cfg["hazards"]):
        common.attach_chloride(data)
        n = int(np.isfinite(data.cube.y[:, :, data.cube.var_index("cl_mgl_4dmean")]).any(axis=1).sum())
        chk = check_chloride_consistency(data)
        log(f"[events] derived chloride attached: {n} sites with an eligible SC->Cl regression; "
            f"consistency with the loaded SC cube {chk}")
    return data


def load_lstm(lstm_ck, data, log):
    if not lstm_ck:
        return None
    from sentinel.models.sonde.lstm import SondeLSTM

    m = SondeLSTM()
    m.load_state(lstm_ck, data)
    log(f"[events] LSTM encoder loaded from {lstm_ck}")
    return m


def calibrate(a, cfg, run_dir, log, qmodels, lstm_ck, lstm_sha):
    if tuple(runner.eval_blocks(a)) != ("val",):
        raise SystemExit("calibrate phase uses the val block only (--eval-blocks val)")
    data = runner.prepare(a, cfg, run_dir, training=True, log=log)
    data = maybe_attach_chloride(cfg, data, log)
    lstm = load_lstm(lstm_ck, data, log)
    fb = FeatureBuilder(data.cube, data.clim_mean, data.clim_std, data.static)
    ew, seed, leads = cfg["ew"], int(cfg.get("seed", 0)), tuple(cfg.get("leads", E.LEADS))
    combine = cfg.get("quantile_combine", ["max"])
    roles_all = tuple(sorted(set(data.roles.tolist())))
    entries, results, counts = {}, [], []
    for hz in cfg["hazards"]:
        var = HAZARDS[hz].variable
        if var not in data.cube.variables:
            continue
        t0 = time.time()
        lab = E.hazard_labels(data, hz, leads)
        cand = E.placebo_candidates(data, lab)
        fblocks = {**fb.target_block(var), **fb.context_block(exclude=var), **extra_feature_blocks(cfg, data, var, log)}
        for L in leads:
            tr = E.event_rows(data, lab, L, "train", roles=("train",))
            ev = E.event_rows(data, lab, L, "val", roles=roles_all)
            if var in E.CHLORIDE_HAZARD_VARS:
                ev = restrict_to_sc_rows(data, ev, "val")
            c = {"hazard": hz, "lead": L, "n_train_rows": len(tr), "n_train_onset_rows": int(tr.y.sum()),
                 "n_val_rows": len(ev), "n_val_pos_rows": int(ev.y.sum()),
                 "n_val_sites": int(np.unique(ev.si).size), "n_val_sites_with_onset": int(np.unique(ev.si[ev.y == 1]).size),
                 **{f"n_val_pos_rows_{r}": int(ev.y[ev.role == r].sum()) for r in roles_all}}
            counts.append(c)
            log(f"[events] {hz} L={L}: {c}")
            if len(ev) == 0 or ev.y.sum() == 0 or tr.y.sum() == 0:
                continue
            mdir = run_dir / "event_models" / f"{hz}_L{L}"
            mdir.mkdir(parents=True, exist_ok=True)
            X_tr = E.event_features(fb, data, tr, fblocks)
            X_ev = E.event_features(fb, data, ev, fblocks)
            models = {"climatology": E.HarmonicClimatology(**cfg["climatology"]).fit(
                E.doy_of(data, tr.ti), tr.y, site_training_rows(data, lab, L, roles_all)),
                "persistence": E.StatePersistence(**cfg["persistence"]).fit(X_tr["margin_last"].to_numpy(), tr.y)}
            if tr.y.sum() >= int(cfg.get("min_train_onsets", 5)):
                models["logistic"] = E.LagLogistic(**cfg["logistic"]).fit(X_tr, tr.y)
                d_tr = data.cube.dates[tr.ti]
                fit = np.asarray(d_tr + pd.Timedelta(days=L) <= pd.Timestamp(FIT_END))
                es = np.asarray(d_tr >= ES_START)
                if tr.y[fit].sum() >= 3:
                    models["event_gbm"] = E.EventGBM(E.EventGBMConfig(**cfg["event_gbm"], seed=seed)).fit(
                        X_tr[fit], tr.y[fit], X_tr[es], tr.y[es])
                    if cfg.get("event_gbm_mono"):   # same GBM + monotone constraints on the margin features
                        models["event_gbm_mono"] = E.EventGBM(E.EventGBMConfig(
                            **{**cfg["event_gbm"], **cfg["event_gbm_mono"]}, seed=seed)).fit(
                            X_tr[fit], tr.y[fit], X_tr[es], tr.y[es])
                if lstm is not None:
                    rs = np.random.default_rng(seed)
                    keep = min(1.0, cfg["lstm_head"]["max_train_rows"] / max(int((tr.y == 0).sum()), 1))
                    sel = (tr.y == 1) | (rs.random(len(tr)) < keep)
                    w = np.where(tr.y[sel] == 1, 1.0, 1.0 / keep)
                    models["lstm_head"] = E.EmbeddingLogistic(C=cfg["lstm_head"]["C"]).fit(
                        lstm.embed(tr.si[sel], tr.ti[sel]), tr.y[sel], sample_weight=w)
            artifacts = {}
            ser = {"climatology": E.climatology_to_dict, "persistence": E.persistence_to_dict,
                   "logistic": E.logistic_to_dict, "lstm_head": E.embedding_to_dict}
            for name, mdl in models.items():
                if name.startswith("event_gbm"):
                    f = mdir / f"{name}.json"
                    f.write_text(mdl.to_json())
                else:
                    f = mdir / f"{name}.json"
                    f.write_text(json.dumps(ser[name](mdl)))
                artifacts[name] = {"file": str(f.relative_to(run_dir)), "sha256": sha256_file(f)}
            qsets = quantile_inputs(data, var, "val", ev, qmodels, cfg.get("naive_quantile_models", []), log)
            raw = raw_scores(data, hz, L, ev, X_ev, models, qsets, combine, lstm)
            stack_members = [m for m in (cfg.get("stack") or {}).get("members", STACK_DEFAULT) if m in raw]
            use_stack = bool(cfg.get("stack")) and len(stack_members) >= 2
            if use_stack:   # fitted on ALL val rows (frozen); OOF version below for the descriptives
                C_st = float(cfg["stack"].get("C", 1.0))
                stack = E.StackLogistic(stack_members, C_st).fit(raw, ev.y)
                raw["stack"] = stack.predict(raw)
            # ---- frozen calibration on ALL val rows
            methods, final = {}, {}
            for name, p in raw.items():
                ab = E.platt_fit(p, ev.y) if name not in NOT_PROB | NO_PLATT else None
                pf = E.platt_apply(p, ab) if ab else p
                final[name] = pf
                methods[name] = {"platt": list(ab) if ab else None,
                                 "threshold_b1": EWS.alarm_day_threshold(pf, ev.y, np.ones(len(ev), bool),
                                                                         float(ew["budget_per_30d"])),
                                 "threshold_b2": EWS.alarm_day_threshold(pf, ev.y, np.ones(len(ev), bool),
                                                                         float(ew["budget_sensitivity"]))}
            if use_stack:
                methods["stack"]["stack"] = stack.to_dict()
            entries[f"{hz}|{L}"] = {"hazard": hz, "lead": L, "artifacts": artifacts, "methods": methods,
                                    "quantile_models": sorted(qsets)}
            # ---- out-of-fold (water-year) descriptives on val: never scored on rows they were fitted on
            wy = E.water_year(ev.issue_date)
            oof, thr_oof = {}, {}
            for name, p in raw.items():
                po = np.full(len(ev), np.nan)
                t1, t2 = np.full(len(ev), np.nan), np.full(len(ev), np.nan)
                for f in np.unique(wy):
                    te, trn = wy == f, wy != f
                    if name == "stack":   # stack refitted on the other water year only
                        st_f = E.StackLogistic(stack_members, C_st).fit({m: raw[m][trn] for m in stack_members}, ev.y[trn])
                        po[te] = st_f.predict({m: raw[m][te] for m in stack_members})
                        p_tr = st_f.predict({m: raw[m][trn] for m in stack_members})
                        t1[te] = EWS.alarm_day_threshold(p_tr, ev.y[trn], np.ones(trn.sum(), bool), float(ew["budget_per_30d"]))
                        t2[te] = EWS.alarm_day_threshold(p_tr, ev.y[trn], np.ones(trn.sum(), bool),
                                                         float(ew["budget_sensitivity"]))
                        continue
                    ab = (E.platt_fit(p[trn], ev.y[trn]) if (name not in NOT_PROB and trn.any()) else None)
                    tf = (lambda z, ab=ab: E.platt_apply(z, ab)) if ab else (lambda z: z)
                    po[te] = tf(p[te])
                    if trn.any():
                        t1[te] = EWS.alarm_day_threshold(tf(p[trn]), ev.y[trn], np.ones(trn.sum(), bool),
                                                         float(ew["budget_per_30d"]))
                        t2[te] = EWS.alarm_day_threshold(tf(p[trn]), ev.y[trn], np.ones(trn.sum(), bool),
                                                         float(ew["budget_sensitivity"]))
                    else:
                        t1[te] = t2[te] = np.inf
                oof[name] = po
                thr_oof[name] = {"b1": t1, "b2": t2}
            recs = evaluate_scores(data, lab, cand, ev, oof, oof["climatology"], thr_oof, "val", L, ew, seed,
                                   tag="val_oof_by_wy")
            results += recs
            pd.DataFrame({"site_no": ev.site_no, "issue_date": ev.issue_date, "role": ev.role, "y": ev.y,
                          **{f"raw_{k}": v for k, v in raw.items()}, **{f"oof_{k}": v for k, v in oof.items()}}
                         ).to_parquet(run_dir / "event_probs" / f"{hz}_L{L}_val.parquet", index=False)
        log(f"[events] {hz} calibrated in {(time.time() - t0) / 60:.1f} min")
    cal = {"created": _dt.datetime.now().isoformat(timespec="seconds"), "phase": "calibrate",
           "split_sha256": data.split_info["sha256"], "split_path": data.split_info["path"],
           "config_hash": runner.config_hash(runner.public_config(cfg)),
           "lstm_checkpoint_sha256": lstm_sha, "quantile_model_dirs": qmodels,
           "budget_per_30d": ew["budget_per_30d"], "budget_sensitivity": ew["budget_sensitivity"],
           "budget_definition": "alarm days on valid non-onset issue days per site per 30 days, pooled POFD",
           "smoke": cfg["_smoke"], "entries": entries}
    cf = run_dir / "event_calibration.json"
    cf.write_text(json.dumps(cal, indent=1, default=float))
    sha = sha256_file(cf)
    (run_dir / "event_calibration.sha256").write_text(sha + "\n")
    log(f"[events] calibration file {cf} sha256 {sha} (FREEZE_sonde.json thresholds_sha256 if frozen)")
    return results, counts


# ----------------------------------------------------------------------------- frozen test path (C2, C3)

PASSES = {("test",): "test", ("test", "test_b"): "test_ab"}   # Test-A primary; pre-declared pooled secondary
EVENTS_CONFIG = Path("configs/sonde/events.yaml")
DEFAULT_SPLIT = "docs/preregistration/sonde_split.json"
MIN_ONSET_BASIN_GROUPS = 2        # power rule (review C3): < 2 groups = degenerate outer bootstrap stage
POOLED_DATA_FILE = Path("docs/preregistration/FREEZE_ew_test_ab_data.json")   # W2: 2027 data pin of test_ab


def frozen_config_hash(cfg: dict) -> str:
    """Config hash exactly as at calibration (include_test_sites forced False)."""
    c = dict(cfg)
    c["include_test_sites"] = False
    return runner.config_hash(runner.public_config(c))


def check_frozen_invocation(a, cfg: dict, cal: dict, freeze: dict, qmodels: dict, lstm_ck) -> str:
    """Review C2: protected scoring runs only the invocation recorded in the freeze record."""
    from sentinel.core import paths

    inv = freeze.get("invocation") or {}
    bad = []
    if a.set:
        bad.append("--set is not allowed")
    if a.seed is not None:
        bad.append("--seed is not allowed")
    if a.smoke or a.temp_split:
        bad.append("--smoke/--temp-split are not allowed")
    if Path(a.config).resolve() != (paths.REPO_ROOT / EVENTS_CONFIG).resolve():
        bad.append(f"--config must be {EVENTS_CONFIG}")
    h = frozen_config_hash(cfg)
    if h != cal.get("config_hash"):
        bad.append(f"config hash {h} != calibration config_hash {cal.get('config_hash')}")
    if inv.get("config_hash") and h != inv["config_hash"]:
        bad.append("config hash != freeze invocation config_hash")
    if dict(qmodels) != dict(inv.get("quantile_models") or {}):
        bad.append(f"--quantile-model set {sorted(qmodels)} != frozen {sorted(inv.get('quantile_models') or {})}")
    if (lstm_ck or None) != (inv.get("lstm_checkpoint") or None):
        bad.append("--lstm-checkpoint differs from the frozen invocation")
    if not a.include_test_sites:
        bad.append("--include-test-sites is required (primary population = test-role sites)")
    sp = getattr(a, "split", None)   # W5: refused BEFORE the marker (a wrong split must not burn the pass)
    if sp is not None and Path(sp).resolve() != (paths.REPO_ROOT / cfg.get("split", DEFAULT_SPLIT)).resolve():
        bad.append(f"--split must be omitted (frozen split {cfg.get('split', DEFAULT_SPLIT)})")
    sf = paths.REPO_ROOT / cfg.get("split", DEFAULT_SPLIT)
    if freeze.get("split_sha256") and (not sf.is_file() or sha256_file(sf) != freeze["split_sha256"]):
        bad.append(f"split file {sf} does not match the frozen split_sha256")
    blocks = tuple(runner.eval_blocks(a))
    if blocks not in PASSES:
        bad.append(f"--eval-blocks must be one of {[','.join(k) for k in PASSES]}")
    if bad:
        raise guard.ProtectedDataError("frozen invocation violated: " + "; ".join(bad))
    return PASSES[blocks]


MARKER_RE = r"^\.ew_scored_{p}(_[0-9a-f]{{12}})?\.json$"   # fixed name; legacy freeze-sha-suffixed names too


def registry_has_pass(pass_name: str, registry_path: Path | None = None) -> bool:
    """True if the registry holds any row of experiment ``ew_v2_<pass>_score`` (parsed, not grepped)."""
    from sentinel.core import paths

    reg = Path(registry_path) if registry_path else paths.REPO_ROOT / "results" / "registry.jsonl"
    if not reg.is_file():
        return False
    want = f"ew_v2_{pass_name}_score"
    with open(reg, encoding="utf-8") as fh:
        for line in fh:
            if want in line:
                try:
                    if json.loads(line).get("experiment") == want:
                        return True
                except ValueError:
                    continue
    return False


def claim_single_pass(pass_name: str, freeze_sha: str, argv: list, marker_dir: Path | None = None,
                      registry_path: Path | None = None) -> Path:
    """Review C3/C4: atomically claim the one allowed pass BEFORE any protected value is loaded.
    The marker name is FIXED per pass (``.ew_scored_<pass>.json``), independent of the freeze record,
    so re-writing FREEZE_ew.json can never re-open a pass; refused if any marker of the pass exists
    (incl. legacy ``.ew_scored_<pass>_<sha12>.json``) or the registry holds ``ew_v2_<pass>_score`` rows
    (the score phase logs a ``pass_claimed`` row right after claiming, see :func:`log_pass_rows`)."""
    import os
    import re
    import sys

    from sentinel.core import paths

    d = Path(marker_dir) if marker_dir else paths.RUNS_ROOT / "sonde"
    d.mkdir(parents=True, exist_ok=True)
    rx = re.compile(MARKER_RE.format(p=re.escape(pass_name)))
    prior = sorted(f.name for f in d.iterdir() if rx.match(f.name))
    if prior:
        raise guard.ProtectedDataError(f"single-pass marker(s) {prior} exist in {d}: pass {pass_name} was already run")
    if (marker_dir is None or registry_path is not None) and registry_has_pass(pass_name, registry_path):
        raise guard.ProtectedDataError(f"registry already holds ew_v2_{pass_name}_score rows: pass {pass_name} was run")
    m = d / f".ew_scored_{pass_name}.json"
    try:
        fd = os.open(m, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError as e:
        raise guard.ProtectedDataError(f"single-pass marker {m} exists: pass {pass_name} was already run") from e
    with os.fdopen(fd, "w") as f:
        json.dump({"pass": pass_name, "freeze_sha256": freeze_sha, "claimed": _dt.datetime.now().isoformat(),
                   "argv": list(argv or sys.argv)}, f)
    return m


def log_pass_rows(pass_name: str, split_label: str, run_dir: Path, *, stage: str, primary: dict | None = None,
                  controls: list | None = None, freeze: dict | None = None, registry_path: Path | None = None) -> int:
    """Registry rows of a scoring pass (experiment ``ew_v2_<pass>_score``). ``stage='claimed'`` writes one
    ``pass_claimed`` row immediately after the marker (makes the registry fallback of
    :func:`claim_single_pass` effective); ``stage='scored'`` writes the primary endpoint per hazard and
    the numeric control columns of the selected method."""
    from sentinel.core.registry import log_result

    exp = f"ew_v2_{pass_name}_score"
    common_kw = dict(workstream="M1-sonde", experiment=exp, split=split_label, artifacts=[str(run_dir)],
                     data_manifests=sorted((freeze or {}).get("data_snapshots", {}) or []),
                     registry_path=registry_path)
    fsha = (freeze or {}).get("_sha256")
    if stage == "claimed":
        log_result(model="ew_v2", metric="pass_claimed", value=1.0, notes="single-pass marker claimed; "
                   "scoring starts", extra={"freeze_sha256": fsha, "run": run_dir.name}, **common_kw)
        return 1
    n = 0
    for hz, r in (primary or {}).items():
        for met in ("bss", "p_one_sided_descriptive", "p_holm_input", "p_holm", "n_evaluable_onsets",
                    "n_basin_groups_with_onsets", "reject_holm", "claim_allowed", "underpowered"):
            v = r.get(met)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                continue
            log_result(model=str(r.get("method", "none")), metric=f"primary_{met}_L{PRIMARY_LEAD}", value=float(v),
                       target=hz, ci_low=r.get("bss_ci_low") if met == "bss" else None,
                       ci_high=r.get("bss_ci_high") if met == "bss" else None,
                       ci_method="basin-group > site-month two-stage block bootstrap" if met == "bss" else None,
                       n=r.get("n_evaluable_onsets"), notes=f"EW-v2 primary endpoint ({r.get('cell')})",
                       extra={"freeze_sha256": fsha, "run": run_dir.name}, **common_kw)
            n += 1
    for c in controls or []:
        for k, v in c.items():
            if k in ("hazard", "lead", "block", "cell", "method") or v is None or not np.isscalar(v) \
                    or isinstance(v, (str, bool, np.bool_)):
                continue
            if not np.isfinite(float(v)):
                continue
            log_result(model=str(c["method"]), metric=f"control_{k}_L{int(c['lead'])}", value=float(v),
                       target=c["hazard"], notes=f"EW-v2 §8 control ({c['cell']})",
                       extra={"freeze_sha256": fsha, "run": run_dir.name}, **common_kw)
            n += 1
    return n


def subset_rows(ev: E.EventRows, m: np.ndarray) -> E.EventRows:
    return E.EventRows(ev.hazard, ev.lead, ev.block, ev.si[m], ev.ti[m], ev.y[m], ev.site_no[m], ev.issue_date[m],
                       ev.role[m])


def concat_rows(evs: list) -> E.EventRows:
    return E.EventRows(evs[0].hazard, evs[0].lead, "+".join(e.block for e in evs),
                       *(np.concatenate([getattr(e, k) for e in evs]) for k in
                         ("si", "ti", "y", "site_no", "issue_date", "role")))


def clean_sites_do(data, min_do: float = 6.0) -> set:
    """Negative-control sites (§8): DO daily minimum never below ``min_do`` in the TRAIN block."""
    j = data.cube.var_index("do_mgl_min")
    tr = np.asarray(data.cube.dates <= pd.Timestamp(TIME_BLOCKS["train"][1]))
    x = data.cube.y[:, tr, j]
    has = np.isfinite(x).sum(axis=1) >= 365
    with np.errstate(all="ignore"):
        mn = np.nanmin(np.where(np.isfinite(x), x, np.inf), axis=1)
    return {data.sites[i] for i in np.flatnonzero(has & (mn >= min_do))}


COVARIATE_PREFIXES = ("precip_", "tmax_", "tmin_", "srad_", "wind_", "vpd_", "snow", "sd_", "anom0_", "logq", "dlogq")


def shuffled_covariates(X: pd.DataFrame, ev, seed: int) -> pd.DataFrame:
    """Covariate-shuffle control: non-target covariates permuted across sites within each issue date."""
    cols = [c for c in X.columns if c.startswith(COVARIATE_PREFIXES)]
    Xs = X.copy()
    rs = np.random.default_rng(seed)
    d = pd.Series(np.arange(len(X))).groupby(pd.to_datetime(ev.issue_date)).apply(lambda g: g.to_numpy())
    for idx in d:
        if idx.size > 1:
            Xs.iloc[idx, [X.columns.get_loc(c) for c in cols]] = X.iloc[rs.permutation(idx)][cols].to_numpy()
    return Xs


def truncation_audit(cfg, data, fb, ev, var, log, n_dates: int = 3, max_rows: int = 20000) -> dict:
    """Input-truncation audit (§8): features at issue dates <= D are recomputed from a cube with every
    value after D removed (incl. hourly features and lagged weather); any difference = look-ahead."""
    import copy

    dmax = 0.0
    qs = np.quantile(ev.ti, np.linspace(0.25, 0.75, n_dates)).astype(int)
    base_blocks = {**fb.target_block(var), **fb.context_block(exclude=var), **extra_feature_blocks(cfg, data, var, log)}
    for D in qs:
        dc = copy.copy(data)
        cube = copy.copy(data.cube)
        cube.y = data.cube.y.copy()
        cube.y[:, D + 1:, :] = np.nan
        dc.cube = cube
        if data.weather is not None:
            dc.weather = data.weather.copy()
            dc.weather[:, D + 1:, :] = np.nan
        if "_subdaily" in data.__dict__:
            sd = data.__dict__["_subdaily"].copy()
            sd[:, D + 1:, ...] = np.nan
            dc.__dict__["_subdaily"] = sd
        fbt = FeatureBuilder(cube, data.clim_mean, data.clim_std, data.static)
        tb = {**fbt.target_block(var), **fbt.context_block(exclude=var), **extra_feature_blocks(cfg, dc, var, log)}
        m = np.flatnonzero(ev.ti <= D)[-max_rows:]
        for k, arr in base_blocks.items():
            a0, a1 = arr[ev.si[m], ev.ti[m]].astype(float), tb[k][ev.si[m], ev.ti[m]].astype(float)
            both = np.isfinite(a0) & np.isfinite(a1)
            if (np.isfinite(a0) != np.isfinite(a1)).any():
                dmax = np.inf
            elif both.any():
                dmax = max(dmax, float(np.max(np.abs(a0[both] - a1[both]))))
    return {"truncation_max_abs_feature_diff": dmax, "truncation_dates": [str(data.cube.dates[q].date()) for q in qs],
            "truncation_pass": bool(dmax <= 1e-6)}


# directories the loaders read (processed/<dir>) and the files in them that the loaders read (review W1)
LOADER_DIRS = {"sonde_daily": "sonde_daily", "sonde_hourly": "sonde_hourly", "forcings_daily": "forcings_daily",
               "chloride_derived": "chloride_derived"}
SNAPSHOT_FILES = {"sonde_daily": ("index.parquet", "sites/*.parquet"), "sonde_hourly": ("sites/*.parquet",),
                  "forcings_daily": ("forcings_daily.parquet",),
                  "chloride_derived": ("regressions.parquet", "sites/*.parquet")}


def _sha256_stream(f: Path) -> str:
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_content_sha256(role: str, directory: Path | None = None) -> dict:
    """Content pin of one loader directory (review W1): SHA-256 over the sorted lines
    '<relative path>\t<file sha256>' of every file the loader reads (:data:`SNAPSHOT_FILES`)."""
    from sentinel.core import paths

    d = Path(directory) if directory else paths.PROCESSED / LOADER_DIRS[role]
    files = sorted({f for pat in SNAPSHOT_FILES[role] for f in d.glob(pat) if f.is_file()})
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.relative_to(d).as_posix()}\t{_sha256_stream(f)}\n".encode())
    return {"content_sha256": h.hexdigest(), "n_content_files": len(files), "resolved_dir": str(d.resolve())}


def verify_snapshots(freeze: dict, strict: bool, log, pins: dict | None = None) -> dict:
    """Reviews W3/W1/C1: every loader directory (daily, hourly, forcings, chloride_derived) must hold
    exactly the pinned snapshot: the pinned manifest is unchanged AND the content hash over the files
    the loader reads equals the pinned ``content_sha256``. A missing pin is itself a failure."""
    from sentinel.core import paths

    pins = pins if pins is not None else (freeze.get("data_snapshots") or {})
    out = {}
    for role in LOADER_DIRS:
        pin = pins.get(role)
        problems = []
        if not pin:
            problems.append("no pin in the freeze record")
        else:
            mf = paths.MANIFESTS / f"{pin['manifest']}.json"
            if not mf.is_file() or sha256_file(mf) != pin.get("manifest_sha256"):
                problems.append(f"pinned manifest {pin['manifest']} changed or missing")
            if not pin.get("content_sha256"):
                problems.append("pin lacks content_sha256")
            else:
                got = snapshot_content_sha256(role)
                if got["content_sha256"] != pin["content_sha256"]:
                    problems.append(f"content sha256 {got['content_sha256'][:12]} != pinned {pin['content_sha256'][:12]}")
                out[role] = {"pinned": pin["manifest"], **got}
        out.setdefault(role, {})["match"] = not problems
        if problems:
            msg = f"{role}: loader directory does not hold the pinned snapshot ({'; '.join(problems)})"
            if strict:
                raise guard.ProtectedDataError(msg)
            log(f"[events] WARNING (rehearsal) {msg}")
    return out


def pooled_data_pins(freeze: dict) -> tuple[dict, dict]:
    """Review W2: the pooled Test-A ∪ Test-B pass (``test_ab``, after 2027-06-30) reads a NEW snapshot.
    Its pins come from the dated record :data:`POOLED_DATA_FILE` (written after the approval cut-off,
    never by the freeze), which must hold ``created`` >= cut-off, ``data_snapshots`` (same format as the
    freeze record, all four roles) and ``identity_check`` = {'until': '2023-09-30', 'identical': true}
    (train/val rows identical to the Test-A snapshot; Test-A rows are re-derived from the new snapshot,
    and the number of WY2024-25 label days that changed vs the Test-A snapshot is reported)."""
    from sentinel.core import paths

    f = paths.REPO_ROOT / POOLED_DATA_FILE
    if not f.is_file():
        raise guard.ProtectedDataError(f"pooled pass needs the dated data pin {POOLED_DATA_FILE}")
    rec = json.loads(f.read_text())
    cut = pd.Timestamp((freeze.get("pooled_secondary") or {}).get("approval_cutoff", "2027-06-30"))
    if pd.Timestamp(str(rec.get("created", "1900-01-01"))[:10]) < cut:
        raise guard.ProtectedDataError(f"{POOLED_DATA_FILE} created before the approval cut-off {cut.date()}")
    ic = rec.get("identity_check") or {}
    if not (ic.get("identical") is True and str(ic.get("until")) == "2023-09-30"):
        raise guard.ProtectedDataError("pooled data pin lacks a passed identity check of train/val rows (<= 2023-09-30)")
    return rec.get("data_snapshots") or {}, {"pooled_data_file": str(f), "pooled_data_file_sha256": sha256_file(f)}


def score(a, cfg, run_dir, log, qmodels, lstm_ck, lstm_sha):
    cf = Path(a.calibration)
    cal = json.loads(cf.read_text())
    cal_sha = sha256_file(cf)
    blocks = tuple(runner.eval_blocks(a))
    protected = bool(set(blocks) & set(guard.PROTECTED_BLOCKS)) or bool(a.include_test_sites)
    rehearsal = getattr(a, "rehearsal_freeze", None)
    if protected and rehearsal:
        raise SystemExit("--rehearsal-freeze is for the val block only")
    freeze, pass_name, marker = None, "val", None
    snap_check, split_label, reg_path = None, None, None
    if protected or rehearsal:
        freeze = guard.load_freeze(rehearsal or a.freeze_file or guard.FREEZE_FILE_EW)
        if protected and Path(a.freeze_file or "").resolve() != guard.FREEZE_FILE_EW.resolve():
            raise guard.ProtectedDataError(f"protected EW scoring requires --freeze-file {guard.FREEZE_FILE_EW}")
        # every check BEFORE the single-pass marker is claimed (a failed check never burns the pass)
        if rehearsal:   # exercise every protected check on val; the only difference is the block/roles
            a_chk = type("A", (), {**vars(a), "include_test_sites": True, "eval_blocks": "test"})()
            check_frozen_invocation(a_chk, cfg, cal, freeze, qmodels, lstm_ck)
            pass_name = "rehearsal_val"
        else:
            pass_name = check_frozen_invocation(a, cfg, cal, freeze, qmodels, lstm_ck)
            if pass_name == "test_ab":
                cut = pd.Timestamp((freeze.get("pooled_secondary") or {}).get("approval_cutoff", "2027-06-30"))
                if pd.Timestamp.now() < cut:
                    raise guard.ProtectedDataError(f"pooled Test-A+Test-B pass not before the approval cut-off {cut.date()}")
        guard.verify_thresholds(freeze, cf)
        if cal["split_sha256"] != freeze["split_sha256"]:
            raise guard.ProtectedDataError("calibration split differs from the frozen split")
        pins, pooled_info = (pooled_data_pins(freeze) if pass_name == "test_ab" else (None, {}))
        snap_check = verify_snapshots(freeze, strict=protected, log=log, pins=pins)
        snap_check.update(pooled_info)
        reg_path = run_dir / "registry_rehearsal.jsonl" if rehearsal else None
        marker = claim_single_pass(pass_name, freeze["_sha256"], getattr(a, "_argv", None),
                                   marker_dir=run_dir if rehearsal else None, registry_path=reg_path)
        split_label = f"{pass_name}/sonde_split_{freeze['split_sha256'][:12]}"
        log_pass_rows(pass_name, split_label, run_dir, stage="claimed", freeze=freeze, registry_path=reg_path)
        log(f"[events] pass {pass_name}: marker {marker}; calibration sha256 verified: {cal_sha}; snapshots {snap_check}")
    data = runner.prepare(a, cfg, run_dir, need_blocks=("train",), log=log)
    data = maybe_attach_chloride(cfg, data, log)
    lstm = load_lstm(lstm_ck, data, log)
    if data.access["protected"]:
        for name, d in qmodels.items():   # quantile forecasts must come from frozen, verified checkpoints
            man = Path(d).parent / "predict_manifest.json"
            if not man.is_file():
                raise guard.ProtectedDataError(f"{d}: no predict_manifest.json (use scripts/eval/sonde_predict.py)")
            mm = json.loads(man.read_text())
            if mm.get("checkpoint_sha256") and mm["checkpoint_sha256"] != data.access["freeze"]["weights_sha256"].get(name):
                raise guard.ProtectedDataError(f"{name}: forecasts not produced by the frozen checkpoint")
    if cal["split_sha256"] != data.split_info["sha256"]:
        raise guard.ProtectedDataError("calibration file was produced on a different split")
    if cal.get("lstm_checkpoint_sha256") and cal["lstm_checkpoint_sha256"] != lstm_sha:
        raise guard.ProtectedDataError("LSTM checkpoint differs from the one used at calibration")
    fb = FeatureBuilder(data.cube, data.clim_mean, data.clim_std, data.static)
    ew, seed = cfg["ew"], int(cfg.get("seed", 0))
    combine = cfg.get("quantile_combine", ["max"])
    roles_all = tuple(sorted(set(data.roles.tolist())))
    basin = {str(k): str(v) for k, v in data.split.basin_group.items()}
    selected = (freeze or {}).get("selected_method") or {}
    primary_role = "val" if rehearsal or not protected else "test"
    base = cf.parent
    load = {"climatology": E.climatology_from_dict, "persistence": E.persistence_from_dict,
            "logistic": E.logistic_from_dict, "lstm_head": E.embedding_from_dict}
    results, controls, primary = [], [], {}
    clean = clean_sites_do(data)
    (run_dir / "reliability").mkdir(exist_ok=True)
    for key, ent in cal["entries"].items():
        hz, L = ent["hazard"], int(ent["lead"])
        var = HAZARDS[hz].variable
        lab = E.hazard_labels(data, hz, (L,))
        cand = E.placebo_candidates(data, lab)
        fblocks = {**fb.target_block(var), **fb.context_block(exclude=var), **extra_feature_blocks(cfg, data, var, log)}
        models = {}
        for name, art in ent["artifacts"].items():
            f = base / art["file"]
            if sha256_file(f) != art["sha256"]:
                raise guard.ProtectedDataError(f"artifact {f} changed since calibration")
            if name.startswith("event_gbm"):
                models[name] = E.EventGBM.from_file(f)
            else:
                models[name] = load[name](json.loads(f.read_text()))
        if rehearsal:   # W5: treat val-role sites as unseen (drop their calibration-time deviations)
            for s_ in [x for x in models["climatology"].delta if x in set(np.asarray(data.sites)[data.roles == "val"])]:
                models["climatology"].delta.pop(s_)
        # climatology deviations for sites unseen at calibration: own TRAINING-year labels only
        for s, (d, y) in site_training_rows(data, lab, L, roles_all).items():
            if s not in models["climatology"].delta:
                E.fit_site_deviation(models["climatology"], s, d, y)
        if pass_name == "test_ab":
            evsets = {"test_b": [E.event_rows(data, lab, L, "test_b", roles=roles_all)],
                      "test_ab_pooled": [E.event_rows(data, lab, L, b, roles=roles_all) for b in ("test", "test_b")]}
        else:
            evsets = {b: [E.event_rows(data, lab, L, b, roles=roles_all)] for b in blocks}
        for tag0, evl in evsets.items():
            blk = tuple(e.block for e in evl)
            if var in E.CHLORIDE_HAZARD_VARS:
                evl = [restrict_to_sc_rows(data, e, e.block) for e in evl]
            ev = concat_rows(evl) if len(evl) > 1 else evl[0]
            if len(ev) == 0:
                continue
            X = E.event_features(fb, data, ev, fblocks)
            if tag0 == "test_ab_pooled" and L == PRIMARY_LEAD:   # W6: approval status of WY2026 label days
                jv = data.cube.var_index(var)
                tb = data.issue_idx("test_b", L)
                pv_ = data.cube.prov[:, tb, jv] & np.isfinite(data.cube.y[:, tb, jv])
                ob_ = np.isfinite(data.cube.y[:, tb, jv])
                with np.errstate(invalid="ignore"):
                    fr_ = pv_.sum(axis=1) / np.maximum(ob_.sum(axis=1), 1)
                pd.DataFrame({"site_no": data.sites, "role": data.roles, "frac_provisional_wy2026": fr_,
                              "n_obs_days_wy2026": ob_.sum(axis=1)}).to_csv(
                    run_dir / f"provisional_wy2026_{hz}.csv", index=False)

            def finals(Xm, ev_=ev):
                qsets = quantile_inputs_multi(data, var, evl, qmodels, cfg.get("naive_quantile_models", []), log) \
                    if Xm is X else {}
                raw = raw_scores(data, hz, L, ev_, Xm, models, qsets, combine, lstm)
                if "stack" in ent["methods"]:
                    st = E.StackLogistic.from_dict(ent["methods"]["stack"]["stack"])
                    missing = [m for m in st.members if m not in raw]
                    if missing:
                        raise guard.ProtectedDataError(f"{key}: stack members {missing} unavailable at scoring")
                    raw["stack"] = st.predict(raw)
                fin, thr_ = {}, {}
                for name, p in raw.items():
                    if name not in ent["methods"]:
                        continue
                    mm = ent["methods"][name]
                    fin[name] = E.platt_apply(p, mm["platt"]) if mm["platt"] else p
                    thr_[name] = {"b1": mm["threshold_b1"], "b2": mm["threshold_b2"]}
                return fin, thr_

            final, thr = finals(X)
            tag = f"{tag0}{'_in_sample_calibration_block' if tag0 == 'val' else ''}"
            # C1: role cells — primary = test-role sites (val-role in rehearsal); others labelled secondary
            cells = {"all": np.ones(len(ev), bool), **{c: ev.role == r for c, r in ROLE_CELLS.items()}}
            for cell, m in cells.items():
                if not m.any():
                    continue
                evc = subset_rows(ev, m)
                fc = {k: v[m] for k, v in final.items()}
                is_primary = ROLE_CELLS.get(cell) == primary_role
                kf = {} if is_primary else None
                results += evaluate_scores(data, lab, cand, evc, fc, fc["climatology"], thr, blk, L, ew, seed,
                                           tag=tag, cell=cell + ("*primary*" if is_primary else ""), keep_frames=kf)
                sel = selected.get(hz)
                if not (is_primary and sel and sel in fc and L == PRIMARY_LEAD):
                    continue
                # ---- primary decision statistic and §8 controls for the selected method (frozen predictions)
                p_sel, p_ref = fc[sel], fc["climatology"]
                pv = EWC.bss_pvalue(evc.y, p_sel, p_ref, site=evc.site_no, dates=evc.issue_date, basin=basin,
                                    n_boot=int(ew["n_boot"]), seed=seed)
                frames, events = kf[sel]
                # C2: ONE evaluable-onset definition for the gate, POD, placebo and controls
                gate = EWS.count_evaluable_onsets(frames, events, L, basin)
                rsel = next(r for r in reversed(results) if r["method"] == sel and r["hazard"] == hz and r["lead"] == L
                            and r["cell"].startswith(cell))
                primary[hz] = {**pv, "method": sel, "block": tag, "cell": cell, **gate,
                               "n_basin_groups": int(np.unique([basin.get(str(x), x) for x in evc.site_no]).size),
                               "pod_events_evaluable": rsel.get("b1_events_evaluable"),
                               "gate_equals_pod_set": bool(rsel.get("b1_events_evaluable") == gate["n_evaluable_onsets"]),
                               "ci_note": "two-stage CI approximate with < 10 basin groups with onsets"
                               if gate["n_basin_groups_with_onsets"] < 10 else ""}
                c = {"hazard": hz, "lead": L, "block": tag, "cell": cell, "method": sel, "n_pos_rows": int(evc.y.sum())}
                if evc.y.sum() == 0:   # no onset rows in the cell: probability controls are undefined
                    c.update(truncation_audit(cfg, data, fb, evc, var, log))
                    controls.append(c)
                    continue
                thr1 = 0.0   # frames carry margin = p - threshold_b1
                c.update({f"random_{k}": v for k, v in EWC.random_alarm_pod(frames, events, thr1, L).items()})
                c.update({f"circshift_{k}": v for k, v in EWC.circular_shift(frames, events, thr1, L, seed=seed).items()})
                c.update({f"labelperm_{k}": v for k, v in EWC.label_block_permutation(
                    evc.y, p_sel, p_ref, site=evc.site_no, dates=evc.issue_date, seed=seed).items()})
                c.update(EWC.molchan(evc.y, p_sel))
                rel, ece = EWC.reliability(evc.y, p_sel)
                rel.to_csv(run_dir / "reliability" / f"{hz}_L{L}_{tag}_{cell}_{sel}.csv", index=False)
                c["ece"] = ece
                c.update(EWC.relative_economic_value(evc.y, p_sel))
                for b_ in BASELINES:
                    if b_ in fc and b_ != sel:
                        dm = EWC.diebold_mariano((p_sel - evc.y) ** 2, (fc[b_] - evc.y) ** 2, evc.issue_date)
                        c.update({f"dm_vs_{b_}_{k}": v for k, v in dm.items()})
                if hz.startswith("H1"):   # negative-control sites: DO min never < 6 mg/L in training
                    mc = np.isin(evc.site_no, list(clean))
                    if mc.any():
                        stc = EWS.alarm_day_stats(p_sel[mc] - thr[sel]["b1"], evc.y[mc], np.ones(mc.sum(), bool),
                                                  evc.site_no[mc], 0.0, float(ew["budget_per_30d"]))
                        c.update({"negctrl_n_sites": int(np.unique(evc.site_no[mc]).size),
                                  "negctrl_n_onset_rows": int(evc.y[mc].sum()),
                                  "negctrl_alarm_days_per_30d": stc["pooled_alarm_days_per_30d"]})
                Xs = shuffled_covariates(X, ev, seed)
                fs, _ = finals(Xs)
                if sel in fs:
                    c["covshuffle_bss"] = EWS.brier_skill(evc.y, fs[sel][m], p_ref, n_boot=0)["bss"]
                c.update(truncation_audit(cfg, data, fb, evc, var, log))
                controls.append(c)
            pd.DataFrame({"site_no": ev.site_no, "issue_date": ev.issue_date, "role": ev.role, "y": ev.y,
                          **final}).to_parquet(run_dir / "event_probs" / f"{hz}_L{L}_{tag0}.parquet", index=False)
    if primary:   # C1/C3: Holm over the FIXED family (m = 3); underpowered hazards enter with p = 1
        primary = EWC.primary_decision(primary, PRIMARY_FAMILY, min_onsets=MIN_ONSETS,
                                       min_basin_groups=MIN_ONSET_BASIN_GROUPS)
        runner.write_json(run_dir / "primary_endpoint.json", {"pass": pass_name, "primary_role": primary_role,
                                                              "lead": PRIMARY_LEAD, "family": list(PRIMARY_FAMILY),
                                                              "min_onsets": MIN_ONSETS,
                                                              "min_basin_groups_with_onsets": MIN_ONSET_BASIN_GROUPS,
                                                              "results": primary})
    if controls:
        pd.DataFrame(controls).to_csv(run_dir / "controls.csv", index=False)
    if marker is not None:   # C4 / shared rule: every metric of the pass is registered
        n_reg = log_pass_rows(pass_name, split_label, run_dir, stage="scored", primary=primary, controls=controls,
                              freeze=freeze, registry_path=reg_path)
        log(f"[events] registered {n_reg} rows of ew_v2_{pass_name}_score"
            f"{' (rehearsal: scratch registry ' + str(reg_path) + ')' if reg_path else ''}")
    runner.write_json(run_dir / "score_manifest.json", {"calibration": str(cf), "calibration_sha256": cal_sha,
                                                        "pass": pass_name, "argv": getattr(a, "_argv", None),
                                                        "config_hash": frozen_config_hash(cfg),
                                                        "freeze": (freeze or {}).get("_path"),
                                                        "freeze_sha256": (freeze or {}).get("_sha256"),
                                                        "marker": str(marker) if marker else None,
                                                        "access": data.access, "quantile_model_dirs": qmodels,
                                                        "data_snapshots_verified": snap_check})
    return results, []


def quantile_inputs_multi(data, var, evl: list, qmodels: dict, naive: list, log) -> dict:
    parts = [quantile_inputs(data, var, e.block, e, qmodels, naive, log) for e in evl]
    keys = set.intersection(*(set(p) for p in parts)) if parts else set()
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def main() -> int:
    ap = runner.base_parser(__doc__)
    ap.add_argument("--phase", required=True, choices=("calibrate", "score"))
    ap.add_argument("--calibration", default=None, help="event_calibration.json (score phase)")
    ap.add_argument("--quantile-model", action="append", default=[], help="name=forecast_dir (repeatable)")
    ap.add_argument("--lstm-checkpoint", default=None)
    ap.add_argument("--gpu-prefer", type=int, default=0)
    ap.add_argument("--rehearsal-freeze", default=None,
                    help="(val only) run every protected-path check against a SCRATCH freeze record; val-role sites "
                         "are treated as unseen (review W5)")
    a = ap.parse_args()
    import sys as _sys
    a._argv = list(_sys.argv)
    if a.phase == "score" and not a.calibration:
        ap.error("--calibration is required for --phase score")
    cfg = runner.apply_overrides(runner.load_config(a.config, smoke=a.smoke), a.set)
    if a.seed is not None:
        cfg["seed"] = a.seed
    qmodels = dict(cfg.get("quantile_models") or {})
    qmodels.update(dict(x.split("=", 1) for x in a.quantile_model))
    lstm_ck = a.lstm_checkpoint or cfg.get("lstm_checkpoint")
    if lstm_ck:
        runner.setup_gpu(f"sonde_events_{a.phase}" + ("_smoke" if a.smoke else ""), prefer=a.gpu_prefer)
    run_dir = runner.run_directory(f"events_{a.phase}", cfg, a.run_name)
    (run_dir / "event_probs").mkdir(parents=True, exist_ok=True)
    log = runner.make_logger(run_dir / "log.txt")
    runner.log_config_tried(f"events_{a.phase}", cfg, run_dir)
    with runner.training_run(f"events_{a.phase}", cfg, run_dir) as run:
        lstm_sha = guard.artifact_sha256(lstm_ck) if lstm_ck else None
        fn = calibrate if a.phase == "calibrate" else score
        results, counts = fn(a, cfg, run_dir, log, qmodels, lstm_ck, lstm_sha)
        res = pd.DataFrame(results)
        res.to_csv(run_dir / "event_scores.csv", index=False)
        if counts:
            pd.DataFrame(counts).to_csv(run_dir / "event_counts.csv", index=False)
        if len(res):
            cols = ["hazard", "lead", "block", "method", "n", "n_pos", "bss", "bss_ci_low", "bss_ci_high", "auprc",
                    "b1_pooled_alarm_days_per_30d", "b1_site_alarm_days_per_30d_max", "b1_time_in_alarm",
                    "b1_pod_real", "b1_pod_placebo", "b1_pod_real_ge24h"]
            with pd.option_context("display.width", 260, "display.max_columns", 30, "display.max_rows", 600,
                                   "display.float_format", "{:.3f}".format):
                log(f"[events] {'SMOKE ' if cfg['_smoke'] else ''}{a.phase} scores (descriptive, not registered):\n"
                    f"{res[[c for c in cols if c in res.columns]].to_string(index=False)}")
        run.note(f"phase {a.phase}: {len(res)} score rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
