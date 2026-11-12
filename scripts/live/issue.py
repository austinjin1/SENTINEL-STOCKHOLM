#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Issue the EW-v2 live-trial forecasts for one issue day D (amendment ``ew_v2_live_amendment.md`` §4–§6).

Modes
-----
``live``    all roles; needs the lead-written ``FREEZE_ew_live.json`` (config + code + calibration hashes); only
            for D = today (UTC) — no retroactive issues (``--late`` writes a late issue, flagged and excluded from
            the primary analysis); appends a chained row to ``results/live/ew_v2_live_hashlog.jsonl``.
``dryrun``  train/val-role sites only, any D, separate output tree and hash log; window inputs from the
            snapshot (``--window-source snapshot``; days already in the 2026-11-08 pull) or from a dry-run fetch.
``replay``  identity check on the VAL block (WY2023, train/val-role sites, snapshot inputs): recomputes the
            calibration-time raw scores of every method with this pipeline and compares them with the stored
            ``event_probs/*_val.parquet`` (must agree to numerical precision). No test data, no labels scored.

Per hazard: frozen calibration file → artifacts (SHA-256 verified) → features with the frozen FeatureBuilder /
``sonde_events.extra_feature_blocks`` → ``sonde_events.raw_scores`` → frozen stack → frozen Platt → frozen
alarm thresholds (b1 = 1, b2 = 2 alarm-days / 30 d). Issue rows (issue-time rule, provisional values allowed):
day d observed, d not an event day, onset in (d, d+L] not excluded by the refractory rule, variable forecastable
(last value ≤ 3 d old and training climatology); H2 also needs an SC issue row.

    python -B scripts/live/issue.py --mode replay
    python -B scripts/live/issue.py --mode dryrun --date 2026-11-08 --window-source snapshot
    python -B scripts/live/issue.py --mode live            # daily, after fetch_usgs.py / fetch_gridmet.py
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import live_common as LC
import live_data as LD
import stamp as ST
from sentinel.core import paths
from sentinel.data.sonde_splits import SondeSplit
from sentinel.eval.hazards import HAZARDS, daily_onsets
from sentinel.models.sonde import common
from sentinel.models.sonde import events as E
from sentinel.models.sonde.data import climatology
from sentinel.models.sonde.features import FeatureBuilder

LOAD = {"climatology": E.climatology_from_dict, "persistence": E.persistence_from_dict,
        "logistic": E.logistic_from_dict}
REPLAY_WINDOW = ("2022-10-01", "2023-09-30")      # val block, WY2023 (train/val-role sites only)
REPLAY_FIRST_ISSUE = "2022-12-01"                  # ≥ 61 d after the window start: features fully inside it
WINDOW_START = LC.LIVE_BLOCK[0]


def _log_to(f: Path):
    f.parent.mkdir(parents=True, exist_ok=True)
    fh = open(f, "a")

    def log(m):
        line = f"{_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')} {m}"
        fh.write(line + "\n")
        fh.flush()
        print(line, flush=True)
    return log


# --------------------------------------------------------------------------- data
def make_data(cfg, hz, sites_roles: dict, end, *, window, window_frames, from_snapshot, weather_frames, log):
    m = cfg["models"][hz]
    sites = sorted(sites_roles)
    cube, sc_parts = LD.build_cube(sites, m["cube_variables"], end, window_start=window[0],
                                   window_frames=window_frames, window_from_snapshot=from_snapshot)
    cm, cs = climatology(cube)
    if m["chloride"]:
        regs = common.chloride_regressions()
        y_cl, p_cl = LD.chloride_cube(cube, sc_parts, regs)
        cube, cm, cs = LD.append_chloride(cube, cm, cs, y_cl, p_cl)
    usable = np.isfinite(cm).any(axis=(1, 2))
    if not usable.all():
        keep = np.flatnonzero(usable)
        cube = LD.Cube([cube.sites[i] for i in keep], cube.dates, cube.variables, cube.y[keep], cube.prov[keep])
        cm, cs = cm[keep], cs[keep]
    roles = np.array([sites_roles[s] for s in cube.sites])
    split = SondeSplit.load(LC.REPO / "docs/preregistration/sonde_split.json")
    if split.sha256() != cfg["split_sha256"]:
        raise PermissionError("split differs from the frozen split")
    split = dataclasses.replace(split, time_blocks={**split.time_blocks, "live": tuple(window)})
    reg = common.site_registry()
    static = reg.reindex(cube.sites)[["lat", "lon", "huc2", "elev_m", "log_drain_area_km2"]]
    W = LD.weather_array(cube, weather_frames) if m["feature_cfg"].get("use_weather") else None
    data = common.SondeData(cube, cm, cs, roles, static, split, {"sha256": cfg["split_sha256"]},
                            {"protected": False, "live_pipeline": True}, ("train", "live"), common.HORIZONS, W)
    log(f"[{hz}] cube {len(cube.sites)} sites ({int((~usable).sum())} without climatology dropped), "
        f"{cube.dates[0].date()}..{cube.dates[-1].date()}, vars {len(cube.variables)}, "
        f"weather sites {0 if W is None else int(np.isfinite(W[..., 0]).any(axis=1).sum())}")
    return data


def issue_rows(data, hz, L) -> E.EventRows:
    """Issue-time rows at the last cube day d (provisional values allowed; no future value used)."""
    hzd = HAZARDS[hz]
    t = len(data.cube.dates) - 1
    j = data.cube.var_index(hzd.variable)
    b = data.baselines(hzd.variable)
    ok = np.isfinite(b.last[:, t]) & np.isfinite(b.cm[:, b.doy[t]]) & np.isfinite(data.cube.y[:, t, j])
    if hz.startswith("H2_"):
        bs = data.baselines("spcond_uscm_mean")
        ok &= np.isfinite(bs.last[:, t]) & np.isfinite(bs.cm[:, bs.doy[t]])
    for s in np.flatnonzero(ok):
        x = data.cube.y[s, :, j].astype(float)
        ev, _ = daily_onsets(x, hzd)
        if ev[t]:
            ok[s] = False
            continue
        past = np.flatnonzero(ev[: t + 1])
        if past.size and past[-1] + hzd.refractory_days >= t + L:
            ok[s] = False
    si = np.flatnonzero(ok).astype(np.int64)
    ti = np.full(si.size, t, np.int64)
    return E.EventRows(hz, L, "live", si, ti, np.zeros(si.size, np.int8), np.asarray(data.sites)[si],
                       data.cube.dates.to_numpy()[ti], data.roles[si])


# --------------------------------------------------------------------------- models
def load_entry(cfg, hz, L):
    m = cfg["models"][hz]
    cf = Path(m["calibration"])
    cal = json.loads(cf.read_text())
    ent = cal["entries"][f"{hz}|{L}"]
    models = {}
    for name, art in ent["artifacts"].items():
        f = cf.parent / art["file"]
        if LC.sha256_file(f) != art["sha256"]:
            raise PermissionError(f"artifact {f} changed since calibration")
        models[name] = E.EventGBM.from_file(f) if name.startswith("event_gbm") else LOAD[name](json.loads(f.read_text()))
    return ent, models


def score_rows(se, data, hz, L, ev, cfg, log):
    ent, models = load_entry(cfg, hz, L)
    var = HAZARDS[hz].variable
    lab = E.hazard_labels(data, hz, (L,))        # Approved-only; used only for training-year climatology deviations
    for s, (d, y) in se.site_training_rows(data, lab, L, tuple(sorted(set(data.roles.tolist())))).items():
        if s not in models["climatology"].delta:
            E.fit_site_deviation(models["climatology"], s, d, y)
    fb = FeatureBuilder(data.cube, data.clim_mean, data.clim_std, data.static)
    fblocks = {**fb.target_block(var), **fb.context_block(exclude=var),
               **se.extra_feature_blocks(cfg["models"][hz]["feature_cfg"], data, var, log)}
    X = E.event_features(fb, data, ev, fblocks)
    for name, mdl in models.items():
        if name.startswith("event_gbm"):
            for bst in mdl.boosters:
                if list(X.columns) != list(bst.feature_name()):
                    raise ValueError(f"{hz} L{L} {name}: feature columns differ from the frozen booster")
    raw = se.raw_scores(data, hz, L, ev, X, models, {}, ["max"], None)
    if "stack" in ent["methods"]:
        st = E.StackLogistic.from_dict(ent["methods"]["stack"]["stack"])
        raw["stack"] = st.predict(raw)
    final, thr = {}, {}
    for name, p in raw.items():
        if name not in ent["methods"]:
            continue
        mm = ent["methods"][name]
        final[name] = E.platt_apply(p, mm["platt"]) if mm["platt"] else p
        thr[name] = (mm["threshold_b1"], mm["threshold_b2"])
    return raw, final, thr, X


# --------------------------------------------------------------------------- modes
def run_replay(cfg, se, out_dir: Path, log) -> dict:
    """Val-block identity check: this pipeline vs the stored calibration-time raw scores."""
    split = json.loads((LC.REPO / "docs/preregistration/sonde_split.json").read_text())
    report = {}
    for hz in cfg["models"]:
        sr = {r["site_no"]: r["role"] for r in cfg["sites"][hz] if r["role"] in ("train", "val")}
        # replay population = the same rule on train/val-role split sites (live list, already metadata-selected)
        frames = [LD.forcings_window(sorted(sr), pd.Timestamp(REPLAY_WINDOW[0]) - pd.Timedelta(days=40), REPLAY_WINDOW[1])]
        data = make_data(cfg, hz, sr, REPLAY_WINDOW[1], window=REPLAY_WINDOW, window_frames=None, from_snapshot=True,
                         weather_frames=frames, log=log)
        for L in cfg["trial"]["leads"]:
            lab = E.hazard_labels(data, hz, (L,))
            ev = E.event_rows(data, lab, L, "live", roles=("train", "val"))
            if hz.startswith("H2_"):
                ev = se.restrict_to_sc_rows(data, ev, "live")
            keep = ev.issue_date >= np.datetime64(REPLAY_FIRST_ISSUE)
            ev = se.subset_rows(ev, keep)
            raw, final, thr, _ = score_rows(se, data, hz, L, ev, cfg, log)
            stored = pd.read_parquet(Path(cfg["models"][hz]["calibration"]).parent / "event_probs" / f"{hz}_L{L}_val.parquet")
            stored["issue_date"] = pd.to_datetime(stored["issue_date"])
            stored = stored[(stored["issue_date"] >= REPLAY_FIRST_ISSUE) & stored["site_no"].isin(sr)]
            mine = pd.DataFrame({"site_no": ev.site_no, "issue_date": pd.to_datetime(ev.issue_date),
                                 **{f"mine_{k}": v for k, v in raw.items()}})
            j = mine.merge(stored, on=["site_no", "issue_date"], how="outer", indicator=True)
            both = j[j["_merge"] == "both"]
            rec = {"n_mine": int(len(mine)), "n_stored": int(len(stored)), "n_matched": int(len(both)),
                   "n_pos_matched": int(both["y"].sum())}
            for k in raw:
                if f"raw_{k}" in both:
                    a, b = both[f"mine_{k}"].to_numpy(float), both[f"raw_{k}"].to_numpy(float)
                    fin = np.isfinite(a) & np.isfinite(b)
                    rec[f"maxabs_{k}"] = float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else None
                    rec[f"n_nonfinite_mismatch_{k}"] = int((np.isfinite(a) != np.isfinite(b)).sum())
            report[f"{hz}|{L}"] = rec
            log(f"[replay] {hz} L{L}: {rec}")
    return report


def run_issue(cfg, se, a, out_root: Path, log, roles_allowed) -> dict:
    D = pd.Timestamp(a.date).normalize()
    d = D - pd.Timedelta(days=1)
    if d < pd.Timestamp(WINDOW_START) + pd.Timedelta(days=34):
        raise SystemExit("issue day too early: features would reach before the live window")
    inputs = {}
    window_frames, unknown_q, counts = None, {}, {}
    from_snapshot = a.window_source == "snapshot"
    if from_snapshot:
        if a.mode == "live":
            raise PermissionError("live issues use live inputs only")
        inputs["window_source"] = "snapshot (2026-11-08 pull)"
    else:
        fdir = out_root / "inputs" / "usgs" / D.date().isoformat()
        man = json.loads((fdir / "manifest.json").read_text())
        if LC.sha256_file(fdir / "records.parquet") != man["records_sha256"]:
            raise PermissionError("USGS fetch file changed after fetching")
        recs = pd.read_parquet(fdir / "records.parquet").to_dict("records")
        long, unknown_q = LD.ogc_long(recs)
        smap_file = (LC.REPO / "configs/live/ew_v2_live_series_map.json") if a.mode == "live" \
            else LC.DRYRUN_ROOT / "series_map_trainval.json"
        if a.mode == "live" and not smap_file.is_file():
            raise PermissionError("frozen series map missing")
        smap = json.loads(smap_file.read_text())["map"] if smap_file.is_file() else None
        long, map_info = LD.apply_series_map(long, smap, window_end=d)
        window_frames, counts = LD.wide_by_site(long)
        inputs["series_map_sha256"] = LC.sha256_file(smap_file) if smap_file.is_file() else None
        inputs["series_map"] = map_info          # incl. the keys that fell back to splice_site (review C-L3)
        inputs["n_masked_unknown_qualifier"] = int(sum(unknown_q.values()))
        inputs["usgs_records_sha256"] = man["records_sha256"]
        inputs["usgs_fetch_finished_utc"] = man["finished_utc"]
        inputs["usgs_failed_sites"] = man["n_sites_failed"]
    wf = []
    fc_end = pd.Timestamp("2026-10-20")
    if d - pd.Timedelta(days=40) <= fc_end:          # dry run in Oct/Nov 2026: frozen forcings where they exist
        wf.append(LD.forcings_window(sorted({r["site_no"] for hz in ("H2_chloride", "H4_nitrate") for r in cfg["sites"][hz]}),
                                     d - pd.Timedelta(days=40), min(d, fc_end)))
        inputs["weather_forcings_daily_until"] = str(fc_end.date())
    gdir = out_root / "inputs" / "gridmet" / D.date().isoformat()
    if (gdir / "manifest.json").exists():
        gman = json.loads((gdir / "manifest.json").read_text())
        if LC.sha256_file(gdir / "weather.parquet") != gman["weather_sha256"]:
            raise PermissionError("gridMET fetch file changed after fetching")
        wf.append(pd.read_parquet(gdir / "weather.parquet"))
        inputs["gridmet_sha256"] = gman["weather_sha256"]
        inputs["gridmet_last_date"] = gman["last_date_available"]
    else:
        log("[issue] no gridMET fetch for this day: weather features missing (NaN) where not covered")
    out, summary = [], {}
    for hz in cfg["models"]:
        sr = {r["site_no"]: r["role"] for r in cfg["sites"][hz] if r["role"] in roles_allowed}
        bg = {r["site_no"]: r["basin_group"] for r in cfg["sites"][hz]}
        data = make_data(cfg, hz, sr, d, window=(WINDOW_START, LC.LIVE_BLOCK[1]), window_frames=window_frames,
                         from_snapshot=from_snapshot, weather_frames=wf, log=log)
        var = HAZARDS[hz].variable
        jv = data.cube.var_index(var)
        for L in cfg["trial"]["leads"]:
            ev = issue_rows(data, hz, L)
            summary[f"{hz}|{L}"] = {"n_sites_configured": len(sr), "n_rows": int(len(ev))}
            if len(ev) == 0:
                continue
            raw, final, thr, X = score_rows(se, data, hz, L, ev, cfg, log)
            sel = cfg["models"][hz]["selected_method"]
            for name, p in final.items():
                t1, t2 = thr[name]
                out.append(pd.DataFrame({
                    "issue_date": D.date().isoformat(), "data_through": d.date().isoformat(), "hazard": hz, "lead": L,
                    "site_no": ev.site_no, "role": ev.role, "basin_group": [bg[s] for s in ev.site_no],
                    "method": name, "selected": name == sel, "p": np.asarray(p, float),
                    "threshold_b1": float(t1), "threshold_b2": float(t2),
                    "alarm_b1": np.asarray(p, float) >= t1, "alarm_b2": np.asarray(p, float) >= t2,
                    "last_value": data.cube.y[ev.si, ev.ti, jv].astype(float),
                    "last_value_provisional": data.cube.prov[ev.si, ev.ti, jv],
                    "n_weather_nan": np.isnan(X.filter(like="precip_sum").to_numpy(float)).sum(axis=1)
                    if any(c.startswith("precip_sum") for c in X.columns) else 0}))
            s = out[-len(final):]
            ss = [f for f in s if f["selected"].iloc[0]]
            if ss:
                summary[f"{hz}|{L}"]["n_alarms_selected_b1"] = int(ss[0]["alarm_b1"].sum())
    issue = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return {"issue": issue, "inputs": inputs, "summary": summary, "qc_counts": counts, "unknown_qualifiers": unknown_q}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("live", "dryrun", "replay"), required=True)
    ap.add_argument("--date", default=_dt.datetime.now(_dt.timezone.utc).date().isoformat())
    ap.add_argument("--window-source", choices=("fetch", "snapshot"), default="fetch")
    ap.add_argument("--late", action="store_true", help="live: issue for a past day (flagged late, not primary)")
    a = ap.parse_args()
    t_start = time.time()
    cfg, cfg_sha = LC.load_config()
    cal_shas = LC.verify_calibrations(cfg)
    static = LC.verify_static_pins(cfg)
    env_sha = LC.env_lock_sha256()
    LD.SNAPSHOT_DIR["dir"] = Path(cfg["data_pins"]["sonde_daily_dir"])
    LD.SNAPSHOT_DIR["sha"] = cfg["data_pins"]["sonde_daily_site_sha256"]
    se = LC.sonde_events_module()
    code_sha = LC.code_sha256()
    if a.mode == "replay":
        out_dir = LC.DRYRUN_ROOT / "replay"
        log = _log_to(out_dir / "replay.log")
        rep = run_replay(cfg, se, out_dir, log)
        res = {"mode": "replay", "window": REPLAY_WINDOW, "first_issue": REPLAY_FIRST_ISSUE, "code_sha256": code_sha,
               "config_sha256": cfg_sha, "calibration_sha256": cal_shas, "results": rep,
               "finished_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
        (out_dir / f"replay_{_dt.datetime.now():%Y%m%d_%H%M%S}.json").write_text(json.dumps(res, indent=1, default=str))
        return 0
    D = pd.Timestamp(a.date).normalize()
    now = pd.Timestamp.now(tz="UTC")
    if a.mode == "live":
        LC.load_freeze_live(cfg_sha)
        today = pd.Timestamp(now.date())
        if D > today:
            raise SystemExit("cannot issue for a future day")
        if D < today and not a.late:
            raise SystemExit("retroactive issue refused (use --late: flagged and excluded from the primary analysis)")
        if not (pd.Timestamp(cfg["trial"]["issue_start"]) <= D <= pd.Timestamp(cfg["trial"]["issue_end"])):
            raise SystemExit("outside the trial issue window")
        root, roles, hashlog = LC.LIVE_ROOT, ("train", "val", "test"), LC.HASHLOG
    else:
        root, roles, hashlog = LC.DRYRUN_ROOT, ("train", "val"), LC.DRYRUN_ROOT / "dryrun_hashlog.jsonl"
    idir = root / "issues" / D.date().isoformat()
    if idir.exists():
        raise FileExistsError(f"{idir} already exists (issued, or an orphan from a crash: see the hash log)")
    for old in (root / "issues").glob(f".tmp_{D.date()}_*"):      # stale temp dirs of crashed runs (never published)
        shutil.rmtree(old, ignore_errors=True)
    tdir = root / "issues" / f".tmp_{D.date()}_{os.getpid()}"       # atomic day (review W-L4): build, then rename
    tdir.mkdir(parents=True)
    log = _log_to(tdir / "issue.log")
    log(f"[issue] mode {a.mode} D {D.date()} host {socket.gethostname()} pid {os.getpid()} code {code_sha[:12]} "
        f"env {env_sha[:12]}")
    res = run_issue(cfg, se, a, root, log, roles)
    f = tdir / "issue.parquet"
    res["issue"].to_parquet(f, index=False)
    issued = pd.Timestamp.now(tz="UTC")
    deadline = pd.Timestamp(f"{D.date()} {cfg['trial']['on_time_deadline_utc']}", tz="UTC")
    man = {"schema": LC.SCHEMA_VERSION, "mode": a.mode, "issue_date": D.date().isoformat(),
           "data_through": (D - pd.Timedelta(days=1)).date().isoformat(),
           "issued_utc": issued.isoformat(timespec="seconds"), "on_time": bool(issued <= deadline and not a.late),
           "late_flag": bool(a.late), "runtime_s": round(time.time() - t_start, 1), "code_sha256": code_sha,
           "config_sha256": cfg_sha, "freeze_sha256": LC.freeze_sha(), "env_lock_sha256": env_sha,
           "calibration_sha256": cal_shas, "static_pins": static, "inputs": res["inputs"],
           "summary": res["summary"], "qc_counts": res["qc_counts"], "unknown_qualifiers": res["unknown_qualifiers"],
           "issue_sha256": LC.sha256_file(f), "n_rows": int(len(res["issue"])),
           "host": socket.gethostname(), "pid": os.getpid()}
    (tdir / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True, default=str))
    for p_ in (f, tdir / "manifest.json"):
        p_.chmod(0o444)
    files = {"issue.parquet": man["issue_sha256"], "manifest.json": LC.sha256_file(tdir / "manifest.json")}
    tdir.rename(idir)
    row = LC.append_hashlog(hashlog, {"issue_date": man["issue_date"], "status": "issued", "mode": a.mode,
                                      "issued_utc": man["issued_utc"], "on_time": man["on_time"], "files": files,
                                      "code_sha256": code_sha, "config_sha256": cfg_sha,
                                      "freeze_sha256": man["freeze_sha256"], "env_lock_sha256": env_sha,
                                      "evidence_refs": ST.pending_evidence_refs("live" if a.mode == "live" else "dryrun")})
    log(f"[issue] done: {man['n_rows']} rows, on_time {man['on_time']}, hash-log seq {row['seq']} {row['row_sha256'][:12]}")
    print(json.dumps(man["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
