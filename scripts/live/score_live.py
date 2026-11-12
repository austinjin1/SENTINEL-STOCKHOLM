#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Hardened single-look scorer of the EW-v2 live trial (amendment ``ew_v2_live_amendment.md`` §7–§9).

Looks (windows, cut-offs and gates come from the frozen config ``trial.looks``; each look is scored at most once,
own marker directory ``sentinel_runs/live/ew_v2/looks/<look>/``):
  ``prov_interim``  ≥ 2027-10-15, WY2027 issues, PROVISIONAL labels, OPERATIONAL content only (issuance coverage,
                    realised alarm-days per 30 d, provisional onset counts). No probability column is read.
  ``count_only``    ≥ 2028-06-30, WY2027 issues/targets, Approved labels; evaluable H1 onsets + basin groups only
                    (key columns only, never ``p``). Decides which final look runs.
  ``final_wy2027``  ≥ 2028-06-30; runs only if count_only found H1 powered; issues 2026-12-01..2027-09-30, targets
                    ≤ 2027-09-30. Confirmatory.
  ``final_pooled``  ≥ 2029-06-30; runs only if count_only found H1 underpowered; issues 2026-12-01..2028-09-30 (one
                    continuous block, no internal embargo), targets ≤ 2028-09-30. Confirmatory.
Primary population: on-time issued rows whose hash-log row has independent timestamp evidence ≤ D 23:59 UTC
(``verify_hashlog.primary_eligibility``); all on-time issued rows form a sensitivity analysis. Rows where any issued
method has a non-finite p are dropped for every method (reported). Labels: frozen snapshot (training block) + the
pinned label snapshot of the look (``build_label_snapshot.py``) through the frozen path.

Order (a failed check never burns a look):
  1. checks: lead-written ``FREEZE_ew_live.json`` (config, series map, issuer code hash) and
     ``FREEZE_ew_live_score.json`` (scorer import-closure hash, label-snapshot content pin per look, dates);
     calibration files + artifacts; hash chain + every issue file SHA-256; label snapshot content pin; no registry
     row of the look's experiment; no marker (CLAIM / SCORED / ABORTED) of the look;
  2. claim: O_EXCL 0444 ``CLAIM.json`` {pid, host, argv, freeze shas} + registry row ``look_claimed`` (pid, host);
  3. work inside ``try``; ANY ``BaseException`` → O_EXCL ``ABORTED.json`` + registry row ``look_aborted``, re-raised;
     an aborted look is never resumed by this script (a dated amendment is required);
  4. seal: results written once, ``SHA256SUMS``, 0444; metric rows registered; O_EXCL ``SCORED.json`` + registry row
     ``look_sealed``. Afterwards every rerun is refused (marker and registry checks).
``--rehearsal`` runs the same code on the DRY-RUN tree (train/val-role sites, provisional labels from the 2026-11-08
snapshot, scratch registry and markers under ``dryrun/looks_rehearsal``), skipping only the freeze/date checks.

    python -B scripts/live/score_live.py --look final
    python -B scripts/live/score_live.py --look prov_interim --rehearsal
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import issue as ISS
import ots_verify as OV
import live_common as LC
import live_data as LD
from sentinel.core.registry import log_result, read_registry
from sentinel.eval import ew_controls as EWC
from sentinel.eval import ew_scoring as EWS
from sentinel.eval.hazards import HAZARDS
from sentinel.models.sonde import events as E

SCORE_FREEZE = LC.REPO / "docs" / "preregistration" / "FREEZE_ew_live_score.json"
LOOKS = ("prov_interim", "count_only", "final_wy2027", "final_pooled")
APPROVED_ONLY = {"prov_interim": False, "count_only": True, "final_wy2027": True, "final_pooled": True}
FAMILY = ("H1_hypoxia", "H2_chloride", "H4_nitrate")
PRIMARY_LEAD = 3
BASELINES = ("climatology", "persistence", "logistic", "event_gbm")
CELLS = {"all": None, "test_role": "test", "train_role": "train", "val_role": "val"}
HAZARD_KEYS = {"H1_hypoxia": ("00300|00002",), "H4_nitrate": ("99133|00003",),
               "H2_chloride": ("00095|00003", "00095|00001")}          # series keys behind each hazard variable
EW = {"budget_per_30d": 1.0, "budget_sensitivity": 2.0, "k_placebo": 20, "n_boot": 2000}


SCORER_EXTRA = ("scripts/live/score_live.py", "scripts/live/ots_verify.py", "scripts/live/build_label_snapshot.py")


def labels_pin_file(look: str) -> Path:
    """Append-only, lead-written label pin of one look (review W-D3), written AFTER the label snapshot is built:
    {look, label_content_sha256, label_manifest_sha256, scorer_freeze_sha256}. The scorer freeze is never rewritten."""
    return LC.REPO / "docs" / "preregistration" / f"FREEZE_ew_live_labels_{look}.json"


def scorer_code_files() -> list[str]:
    return sorted(set(LC.code_files()) | set(SCORER_EXTRA))


def scorer_code_sha256() -> str:
    h = hashlib.sha256()
    for rel in scorer_code_files():
        h.update(rel.encode())
        h.update((LC.REPO / rel).read_bytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- checks (before any claim)
def check_hashlog(path: Path, root: Path) -> list[dict]:
    rows = LC.read_hashlog(path)
    errs = LC.verify_chain(rows)
    for r in rows:
        if r.get("status", "issued") != "issued":
            continue
        for name, sha in r["files"].items():
            f = root / "issues" / r["issue_date"] / name
            if not f.is_file() or LC.sha256_file(f) != sha:
                errs.append(f"{r['issue_date']}: {name} missing or changed")
    if errs:
        raise PermissionError(f"hash log / issue files fail verification: {errs[:5]}")
    if not rows:
        raise PermissionError("no issues in the hash log")
    return rows


def registry_has(exp: str, reg_path: Path | None) -> bool:
    try:
        rows = read_registry(reg_path) if reg_path else read_registry()
    except FileNotFoundError:
        return False
    return any(r.get("experiment") == exp for r in rows)


def pre_claim(a, cfg, cfg_sha, root, look_dir, reg_path, exp) -> dict:
    look = cfg["trial"]["looks"][a.look]
    info = {"config_sha256": cfg_sha, "calibration_sha256": LC.verify_calibrations(cfg),
            "static_pins": LC.verify_static_pins(cfg), "look_spec": look}
    if a.rehearsal:
        info["labels_dir"] = None            # rehearsal: window labels from the frozen snapshot (provisional allowed)
    else:
        fr = LC.load_freeze_live(cfg_sha, check_code=True)
        if not SCORE_FREEZE.is_file():
            raise PermissionError(f"{SCORE_FREEZE} missing (lead-written scorer freeze)")
        fs = json.loads(SCORE_FREEZE.read_text())
        if any(m in json.dumps(fs.get("status", "")) for m in ("candidate", "NOT_A_FREEZE")) or "<" in json.dumps(fs):
            raise PermissionError("scorer freeze record is a candidate or has placeholders")
        if fs.get("scorer_code_sha256") != scorer_code_sha256():
            raise PermissionError("scorer code differs from FREEZE_ew_live_score.json")
        if pd.Timestamp.now() < pd.Timestamp(look["earliest"]):
            raise PermissionError(f"look {a.look} not before {look['earliest']}")
        if a.look.startswith("final_"):
            cdir = LC.LIVE_ROOT / "looks" / "count_only"
            if not (cdir / "SCORED.json").is_file():
                raise PermissionError("a final look needs the scored count_only look")
            sums = (cdir / "SHA256SUMS").read_text().split("\n")            # W-D6: sealed result, not a loose file
            want = next((ln.split()[0] for ln in sums if ln.endswith("results/count_only.json")), None)
            sc = json.loads((cdir / "SCORED.json").read_text())
            if want is None or LC.sha256_file(cdir / "results" / "count_only.json") != want \
                    or sc.get("sha256sums") != LC.sha256_file(cdir / "SHA256SUMS"):
                raise PermissionError("count_only result does not match its sealed SHA256SUMS / SCORED.json")
            ext = json.loads((cdir / "results" / "count_only.json").read_text())["extension_triggered"]
            if (a.look == "final_pooled") != bool(ext):
                raise PermissionError(f"count_only decided extension_triggered={ext}: {a.look} is not the final look")
        ldir = LC.LIVE_ROOT / "label_snapshots" / a.look
        man = json.loads((ldir / "manifest.json").read_text())
        files = sorted((ldir / "sites").glob("*.parquet"))
        h = hashlib.sha256()
        for f in files:
            h.update(f"{f.relative_to(ldir).as_posix()}\t{LC.sha256_file(f)}\n".encode())
        pf = labels_pin_file(a.look)
        if not pf.is_file():
            raise PermissionError(f"{pf} missing (lead-written label pin, after the label snapshot)")
        pin = json.loads(pf.read_text())
        if pin.get("scorer_freeze_sha256") != LC.sha256_file(SCORE_FREEZE) \
                or man.get("scorer_freeze_sha256") != LC.sha256_file(SCORE_FREEZE):
            raise PermissionError("label snapshot / pin were not made under the current scorer freeze")
        if h.hexdigest() != pin.get("label_content_sha256") or man["content_sha256"] != pin.get("label_content_sha256") \
                or LC.sha256_file(ldir / "manifest.json") != pin.get("label_manifest_sha256"):
            raise PermissionError("label snapshot differs from its pin")
        info.update({"labels_dir": str(ldir), "labels_content_sha256": h.hexdigest(),
                     "multi_series_keys": man.get("multi_series_keys", []),
                     "freeze_live_sha256": LC.sha256_file(LC.FREEZE_LIVE),
                     "freeze_score_sha256": LC.sha256_file(SCORE_FREEZE), "freeze_live_code": fr.get("code_sha256")})
        info["ots_self_test"] = OV.self_test()      # C-D1: refuse if third-party verification is unavailable
        from sentinel.data.reserved import assert_preclaim_clean   # lead ruling 2026-11-10: pre-claim audit
        info["preclaim_audit"] = {k: len(v) if isinstance(v, list) else v
                                  for k, v in assert_preclaim_clean(exempt_dirs=(look_dir,)).items()}
        info["evidence_refs"] = OV.check_refs(LC.read_hashlog(LC.HASHLOG), "live")   # restores from the public copy
    hl = (LC.DRYRUN_ROOT / "dryrun_hashlog.jsonl") if a.rehearsal else LC.HASHLOG
    info["n_hashlog_rows"] = len(check_hashlog(hl, root))
    info["issue_files"] = preload_issue_files(root, LC.read_hashlog(hl))      # S-1: before the claim
    info["scorer_code_sha256"] = scorer_code_sha256()
    if registry_has(exp, reg_path):
        raise PermissionError(f"registry already holds rows of {exp}: look spent")
    for m in ("CLAIM.json", "SCORED.json", "ABORTED.json"):
        if (look_dir / m).exists():
            raise PermissionError(f"{look_dir / m} exists: look spent or aborted")
    return info


# --------------------------------------------------------------------------- work
ISSUE_KEY_COLS = ("issue_date", "data_through", "hazard", "lead", "site_no", "role", "basin_group", "method", "p",
                  "threshold_b1", "threshold_b2", "alarm_b1", "last_value")


def issue_file_rows(f: Path) -> int:
    """Row count of an issue file from its parquet metadata (0 for a zero-row day)."""
    import pyarrow.parquet as pq
    return int(pq.ParquetFile(f).metadata.num_rows)


def preload_issue_files(root: Path, rows: list[dict]) -> dict:
    """Review S-1: test-load EVERY issued file BEFORE the claim. A zero-row file is fine (no columns are required);
    a file with rows must hold every key column. Any other problem refuses the look before it is claimed."""
    import pyarrow.parquet as pq
    n0, n = 0, 0
    for r in rows:
        if r.get("status", "issued") != "issued":
            continue
        f = root / "issues" / r["issue_date"] / "issue.parquet"
        k = issue_file_rows(f)
        if k == 0:
            n0 += 1
            continue
        miss = [c for c in ISSUE_KEY_COLS if c not in pq.ParquetFile(f).schema_arrow.names]
        if miss:
            raise PermissionError(f"issue file {f} lacks columns {miss}: refusing before the claim")
        n += 1
    return {"n_issue_files_with_rows": n, "n_zero_row_issue_days": n0}


def load_issues(root: Path, rows: list[dict], cols_extra=()) -> pd.DataFrame:
    """Issued rows of every ``issued`` hash-log row (status rows carry no files). ``cols_extra`` = columns beyond the
    keys; the count_only look passes none, so no probability is ever read there."""
    cols = ["issue_date", "data_through", "hazard", "lead", "site_no", "role", "basin_group", "method", *cols_extra]
    out = []
    for r in rows:
        if r.get("status", "issued") != "issued":
            continue
        f = root / "issues" / r["issue_date"] / "issue.parquet"
        if issue_file_rows(f) == 0:            # review S-1: a zero-row day (e.g. late USGS values) has no columns
            continue
        df = pd.read_parquet(f, columns=cols)
        df["on_time"] = bool(r.get("on_time"))
        df["row_sha256"] = r["row_sha256"]
        out.append(df)
    if not out:
        return pd.DataFrame(columns=[*cols, "on_time", "row_sha256", "primary_eligible"])
    df = pd.concat(out, ignore_index=True)
    df["data_through"] = pd.to_datetime(df["data_through"])
    return df


def label_data(cfg, hz, roles, labels_dir, block_end, approved_only, log, rehearsal: bool = False):
    """Training block from the frozen snapshot (per-site pins) + the look's label snapshot for the live window."""
    LD.SNAPSHOT_DIR["dir"] = Path(cfg["data_pins"]["sonde_daily_dir"])
    LD.SNAPSHOT_DIR["sha"] = cfg["data_pins"]["sonde_daily_site_sha256"]
    sr = {r["site_no"]: r["role"] for r in cfg["sites"][hz] if r["role"] in roles}
    if labels_dir is None:                       # rehearsal: window from the frozen snapshot
        frames, from_snap = None, True
        end = min(pd.Timestamp(block_end), pd.Timestamp(pd.read_parquet(
            Path(cfg["data_pins"]["sonde_daily_dir"]) / "index.parquet")["last"].max()))
    else:
        frames, from_snap = {}, False
        for f in (Path(labels_dir) / "sites").glob("*.parquet"):
            w = pd.read_parquet(f)
            w.index = pd.to_datetime(w.index)
            frames[f.stem] = w
        end = pd.Timestamp(block_end)
    data = ISS.make_data(cfg, hz, sr, end, window=(LC.LIVE_BLOCK[0], str(pd.Timestamp(block_end).date())),
                         window_frames=frames, from_snapshot=from_snap, weather_frames=[], log=log)
    if rehearsal:   # dry-run days fall inside the block-start embargo; rehearsal only, no earlier data enter
        import dataclasses
        data.split = dataclasses.replace(data.split, embargo_days=0)
    lab = E.hazard_labels(data, hz, tuple(cfg["trial"]["leads"]), approved_only=approved_only)
    return data, lab


def score_look(a, cfg, root, out_dir, info, log) -> dict:
    se = LC.sonde_events_module()
    look = cfg["trial"]["looks"][a.look]
    mode = "dryrun" if a.rehearsal else "live"
    rows = LC.read_hashlog((LC.DRYRUN_ROOT / "dryrun_hashlog.jsonl") if a.rehearsal else LC.HASHLOG)
    approved_only = APPROVED_ONLY[a.look] and not a.rehearsal
    roles = ("train", "val") if a.rehearsal else ("train", "val", "test")
    i0, i1 = (pd.Timestamp(x) for x in look["issues"])
    t_end = pd.Timestamp(look.get("targets_end", look["issues"][1]))
    extra = () if a.look == "count_only" else (("alarm_b1",) if a.look == "prov_interim" else
                                               ("p", "threshold_b1", "threshold_b2", "alarm_b1", "last_value"))
    issues = load_issues(root, rows, extra)
    if a.rehearsal:   # dry run: no timestamp submission, issue dates before the trial -> window = dry-run days
        issues["on_time"] = True                    # rehearsal only: dry issues of past days count as on time
        issues["primary_eligible"] = True
        hl_dates = pd.to_datetime([r["issue_date"] for r in rows if r.get("status", "issued") == "issued"])
        i0, i1 = hl_dates.min(), hl_dates.max()           # S-1: window from the hash log (zero-row days included)
        t_end = pd.Timestamp(pd.read_parquet(Path(cfg["data_pins"]["sonde_daily_dir"]) / "index.parquet")["last"].max())
    else:
        elig = OV.primary_eligibility(mode)
        eligible = set(elig.loc[elig["primary_eligible"], "row_sha256"]) if len(elig) else set()
        issues["primary_eligible"] = issues["row_sha256"].isin(eligible)
    D = pd.to_datetime(issues["issue_date"])
    issues = issues[(D >= i0) & (D <= i1)]
    res = {"look": a.look, "approved_only_labels": approved_only, "issue_window": [str(i0.date()), str(i1.date())],
           "targets_end": str(t_end.date()), "n_hashlog_rows": len(rows),
           "n_issued_days": int(issues["issue_date"].nunique()),
           "n_primary_eligible_days": int(issues.loc[issues["primary_eligible"], "issue_date"].nunique()),
           "hazards": {}}
    # C-E2: four kinds of day in EVERY look (issued+eligible, issued-not-eligible, failed-only, missing = no row)
    win = pd.date_range(i0, i1)
    # S-1: classify days from the HASH-LOG rows (a zero-row issued day has a row but no issue rows)
    if a.rehearsal:
        elig_sha = {r["row_sha256"] for r in rows if r.get("status", "issued") == "issued"}
    else:
        elig_sha = set(elig.loc[elig["primary_eligible"], "row_sha256"]) if len(elig) else set()
    hl_issued = [r for r in rows if r.get("status", "issued") == "issued"]
    by_day = pd.DataFrame({"on_time": [bool(r.get("on_time")) or a.rehearsal for r in hl_issued],
                           "elig": [r["row_sha256"] in elig_sha for r in hl_issued],
                           "zero_rows": [issue_file_rows(root / "issues" / r["issue_date"] / "issue.parquet") == 0
                                         for r in hl_issued]},
                          index=pd.to_datetime([r["issue_date"] for r in hl_issued]))
    by_day = by_day.groupby(level=0).max()
    res["n_zero_row_issued_days"] = int(by_day.loc[(by_day.index >= i0) & (by_day.index <= i1), "zero_rows"].sum())
    failed = {pd.Timestamp(r["issue_date"]) for r in rows if r.get("status") == "failed"}
    day_class = {}
    for D in win:
        if D in by_day.index:
            day_class[D] = "issued_eligible" if (by_day.at[D, "on_time"] and by_day.at[D, "elig"]) else (
                "issued_on_time_not_eligible" if by_day.at[D, "on_time"] else "issued_late")
        else:
            day_class[D] = "failed_only" if D in failed else "missing_no_row"
    res["day_classes"] = pd.Series(day_class).value_counts().to_dict()
    res["days_issued_not_eligible"] = [str(D.date()) for D, c in day_class.items() if c == "issued_on_time_not_eligible"]
    res["days_missing_no_row"] = [str(D.date()) for D, c in day_class.items() if c == "missing_no_row"]
    n_ot = sum(c in ("issued_eligible", "issued_on_time_not_eligible") for c in day_class.values())
    frac_unattested = (res["day_classes"].get("issued_on_time_not_eligible", 0) / n_ot) if n_ot else 0.0
    res["frac_on_time_days_without_qualifying_attestation"] = frac_unattested
    co_primary = bool(frac_unattested > 0.03) and not a.rehearsal      # pre-specified bound (amendment §7)
    res["all_on_time_co_primary"] = co_primary
    primary, primary_all, controls, metrics = {}, {}, [], []
    populations = {"primary": issues["on_time"] & issues["primary_eligible"], "all_on_time": issues["on_time"]}
    for hz in cfg["models"]:
        data, lab = label_data(cfg, hz, roles, info["labels_dir"], t_end + pd.Timedelta(days=8), approved_only, log,
                               rehearsal=a.rehearsal)
        basin = {str(k): str(v) for k, v in data.split.basin_group.items()}
        cand = E.placebo_candidates(data, lab, approved_only=approved_only)
        sel = cfg["models"][hz]["selected_method"]
        for L in cfg["trial"]["leads"]:
            if a.look in ("count_only", "prov_interim") and L != PRIMARY_LEAD:
                continue
            if len(data.issue_idx("live", L)) == 0:
                res["hazards"][f"{hz}|{L}"] = {"note": "no issue dates in the block"}
                continue
            ev = E.event_rows(data, lab, L, "live", roles=roles)
            keep_t = (pd.to_datetime(ev.issue_date) + pd.Timedelta(days=L) <= t_end) & \
                     (pd.to_datetime(ev.issue_date) >= i0 - pd.Timedelta(days=1))
            ev = se.subset_rows(ev, np.asarray(keep_t))
            for pop, pmask in populations.items():
                iss = issues[(issues["hazard"] == hz) & (issues["lead"] == L) & pmask]
                isel = iss[iss["method"] == sel]
                key_ev = pd.MultiIndex.from_arrays([ev.site_no, pd.to_datetime(ev.issue_date)])
                m = key_ev.isin(pd.MultiIndex.from_arrays([isel["site_no"].to_numpy(), isel["data_through"].to_numpy()]))
                evp = se.subset_rows(ev, m)
                rec = {"n_label_valid_rows": int(len(ev)), "n_rows_with_issue": int(m.sum()),
                       "n_label_valid_without_issue": int((~m).sum()), "n_pos_rows": int(evp.y.sum())}
                tag = f"{hz}|{L}|{pop}"
                if pop == "primary" and L == PRIMARY_LEAD:      # C-E2 / W-D7, every look: onsets in each day class
                    on_day = pd.Series(lab.onset.sum(axis=0), index=data.cube.dates)
                    # issue day D (data through d = D-1) forecasts (d, d+L] = [D, D+L-1]  (off-by-one fixed)
                    nxt = lambda D_: float(on_day.reindex(pd.date_range(D_, D_ + pd.Timedelta(days=L - 1))).fillna(0).sum())
                    for cls in ("issued_eligible", "issued_on_time_not_eligible", "failed_only", "missing_no_row"):
                        ds = [D_ for D_, c in day_class.items() if c == cls]
                        rec[f"onsets_per_day_window__{cls}"] = float(np.mean([nxt(D_) for D_ in ds])) if ds else None
                        rec[f"n_days__{cls}"] = len(ds)
                if a.look in ("count_only", "prov_interim"):
                    frames, _, events, _, _ = se.site_frames(data, lab, cand, evp, np.zeros(len(evp)), "live", L)
                    rec.update(EWS.count_evaluable_onsets(frames, events, L, basin))
                    if a.look == "prov_interim" and len(evp):   # operational: realised alarm-days on valid non-onset rows
                        al = isel.set_index(["site_no", "data_through"])["alarm_b1"].reindex(
                            pd.MultiIndex.from_arrays([evp.site_no, pd.to_datetime(evp.issue_date)])).fillna(False)
                        neg = evp.y == 0
                        rec["realised_alarm_days_per_30d_pooled"] = float(30 * al.to_numpy(bool)[neg].mean()) if neg.any() else None
                    res["hazards"][tag] = rec
                    continue
                scores, thr = {}, {}
                k_ev = pd.MultiIndex.from_arrays([evp.site_no, pd.to_datetime(evp.issue_date)])
                for meth, g in iss.groupby("method"):
                    sr_ = pd.Series(g["p"].to_numpy(float), index=pd.MultiIndex.from_arrays(
                        [g["site_no"].to_numpy(), g["data_through"].to_numpy()]))
                    scores[meth] = sr_.reindex(k_ev).to_numpy(float)
                    thr[meth] = {"b1": float(g["threshold_b1"].iloc[0]), "b2": float(g["threshold_b2"].iloc[0])}
                fin = np.all([np.isfinite(v) for k, v in scores.items() if k != "threshold"], axis=0) \
                    if scores else np.zeros(len(evp), bool)
                rec["n_dropped_nonfinite_any_method"] = int((~fin).sum())
                evp = se.subset_rows(evp, fin)
                scores = {k: v[fin] for k, v in scores.items()}
                res["hazards"][tag] = rec
                if len(evp) == 0 or "climatology" not in scores:
                    continue
                for cell, role in CELLS.items():
                    cm = np.ones(len(evp), bool) if role is None else (evp.role == role)
                    if not cm.any() or (pop != "primary" and cell != "all"):
                        continue
                    evc = se.subset_rows(evp, cm)
                    fc = {k: v[cm] for k, v in scores.items()}
                    is_primary = pop == "primary" and cell == "all"
                    is_co = co_primary and pop == "all_on_time" and cell == "all"
                    kf = {} if (is_primary or is_co) else None
                    metrics += [{**r_, "population": pop} for r_ in se.evaluate_scores(
                        data, lab, cand, evc, fc, fc["climatology"], thr, "live", L, EW, 0, tag=f"live_{a.look}",
                        cell=cell + ("*primary*" if is_primary else ""), keep_frames=kf)]
                    if is_co and L == PRIMARY_LEAD and sel in fc:      # co-primary population (> 3 % unattested)
                        fr_, ev_ = kf[sel]
                        primary_all[hz] = {**EWC.bss_pvalue(evc.y, fc[sel], fc["climatology"], site=evc.site_no,
                                                            dates=evc.issue_date, basin=basin, n_boot=EW["n_boot"], seed=0),
                                           **EWS.count_evaluable_onsets(fr_, ev_, L, basin), "method": sel}
                    if not (is_primary and L == PRIMARY_LEAD and sel in fc):
                        if pop == "all_on_time" and cell == "all" and L == PRIMARY_LEAD and sel in fc:
                            res.setdefault("sensitivity_all_on_time", {})[hz] = EWC.bss_pvalue(
                                evc.y, fc[sel], fc["climatology"], site=evc.site_no, dates=evc.issue_date,
                                basin=basin, n_boot=EW["n_boot"], seed=0)
                        continue
                    p_sel, p_ref = fc[sel], fc["climatology"]
                    pv = EWC.bss_pvalue(evc.y, p_sel, p_ref, site=evc.site_no, dates=evc.issue_date, basin=basin,
                                        n_boot=EW["n_boot"], seed=0)
                    frames, events = kf[sel]
                    gate = EWS.count_evaluable_onsets(frames, events, L, basin)
                    primary[hz] = {**pv, **gate, "method": sel, "n_rows": int(len(evc)),
                                   "ci_note": "approximate (< 10 basin groups with onsets)"
                                   if gate["n_basin_groups_with_onsets"] < 10 else ""}
                    multi = {k.split("|")[0] for k in info.get("multi_series_keys", [])
                             if "|".join(k.split("|")[1:]) in HAZARD_KEYS.get(hz, ())}
                    keep_ms = ~np.isin(evc.site_no, sorted(multi))
                    if multi and keep_ms.any() and evc.y[keep_ms].sum() > 0:   # W-D5: series choice at multi-sonde sites
                        res.setdefault("sensitivity_exclude_multi_series", {})[hz] = {
                            "n_sites_excluded": len(multi), **EWC.bss_pvalue(
                                evc.y[keep_ms], p_sel[keep_ms], p_ref[keep_ms], site=evc.site_no[keep_ms],
                                dates=evc.issue_date[keep_ms], basin=basin, n_boot=EW["n_boot"], seed=0)}
                    c = {"hazard": hz, "lead": L, "method": sel, "n_pos_rows": int(evc.y.sum())}
                    if evc.y.sum() > 0:
                        c.update({f"random_{k}": v for k, v in EWC.random_alarm_pod(frames, events, 0.0, L).items()})
                        c.update({f"circshift_{k}": v for k, v in EWC.circular_shift(frames, events, 0.0, L, seed=0).items()})
                        c.update({f"labelperm_{k}": v for k, v in EWC.label_block_permutation(
                            evc.y, p_sel, p_ref, site=evc.site_no, dates=evc.issue_date, seed=0).items()})
                        c.update(EWC.molchan(evc.y, p_sel))
                        rel, c["ece"] = EWC.reliability(evc.y, p_sel)
                        rel.to_csv(out_dir / f"reliability_{hz}_L{L}.csv", index=False)
                        c.update(EWC.relative_economic_value(evc.y, p_sel))
                        for b_ in BASELINES:
                            if b_ in fc and b_ != sel:
                                dm = EWC.diebold_mariano((p_sel - evc.y) ** 2, (fc[b_] - evc.y) ** 2, evc.issue_date)
                                c.update({f"dm_vs_{b_}_{k}": v for k, v in dm.items()})
                        t1 = np.minimum(evc.ti + 1, lab.onset.shape[1] - 1)       # (b) strictly ahead
                        first_day = lab.onset[evc.si, t1] & (evc.y == 1)
                        if (~first_day).any() and evc.y[~first_day].sum() > 0:
                            c["strict_ahead_bss"] = EWS.brier_skill(evc.y[~first_day], p_sel[~first_day],
                                                                    p_ref[~first_day], n_boot=0)["bss"]
                            c["strict_ahead_n_pos_dropped"] = int(first_day.sum())
                    jj = data.cube.var_index(HAZARDS[hz].variable)          # (c) sensor-fault audit
                    g = isel.set_index(["site_no", "data_through"]).reindex(
                        pd.MultiIndex.from_arrays([evc.site_no, pd.to_datetime(evc.issue_date)]))
                    v_issue, v_label = g["last_value"].to_numpy(float), data.cube.y[evc.si, evc.ti, jj].astype(float)
                    al = g["alarm_b1"].fillna(False).to_numpy(bool)
                    for nm, msk in (("alarm_rows", al), ("onset_rows", evc.y == 1)):
                        if msk.any():
                            both = msk & np.isfinite(v_label) & np.isfinite(v_issue)
                            c[f"sensorfault_{nm}_n"] = int(msk.sum())
                            c[f"sensorfault_{nm}_frac_deleted"] = float(np.mean(~np.isfinite(v_label[msk])))
                            c[f"sensorfault_{nm}_frac_revised"] = float(np.mean(np.abs(v_label[both] - v_issue[both]) > 1e-3)) \
                                if both.any() else None
                    c["input_revision_audit"] = "separate descriptive re-issue (amendment §7 d)"
                    controls.append(c)
    if a.look == "count_only":
        h1 = res["hazards"].get(f"H1_hypoxia|{PRIMARY_LEAD}|primary", {})
        res["extension_triggered"] = bool(EWC.is_underpowered(h1.get("n_evaluable_onsets"),
                                                              h1.get("n_basin_groups_with_onsets")))
    elif a.look.startswith("final_"):
        res["primary"] = EWC.primary_decision(primary, FAMILY)
        if co_primary:            # claim only where BOTH populations reject (pre-specified, amendment §7)
            res["primary_all_on_time"] = EWC.primary_decision(primary_all, FAMILY)
            for hz in FAMILY:
                res["primary"][hz]["claim_allowed"] = bool(res["primary"][hz]["claim_allowed"]
                                                           and res["primary_all_on_time"][hz]["claim_allowed"])
        res["descriptive_only"] = bool(a.rehearsal)   # rehearsal output is never a result
        pd.DataFrame(metrics).to_csv(out_dir / "scores.csv", index=False)
        pd.DataFrame(controls).to_csv(out_dir / "controls.csv", index=False)
    (out_dir / f"{a.look}.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item()
                                                       if hasattr(o, "item") else str(o)))
    return res


def register(res, exp, split, reg_path, extra) -> int:
    n = 0

    def put(model, target, metric, value, **kw):
        nonlocal n
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if not np.isfinite(v):
            return
        log_result(workstream="M1-sonde", experiment=exp, model=model, metric=metric, value=v, split=split,
                   target=target, extra=extra, registry_path=reg_path, **kw)
        n += 1
    for k, r in res.get("hazards", {}).items():
        for m in ("n_rows_with_issue", "n_label_valid_without_issue", "n_pos_rows", "n_evaluable_onsets",
                  "n_basin_groups_with_onsets", "realised_alarm_days_per_30d_pooled", "n_dropped_nonfinite_any_method"):
            put("labels", k, m, r.get(m))
    if "extension_triggered" in res:
        put("labels", "H1_hypoxia", "extension_triggered", res["extension_triggered"])
    for hz, r in (res.get("primary") or {}).items():
        for m in ("bss", "p_one_sided", "p_holm", "reject_holm", "underpowered", "n_evaluable_onsets",
                  "n_basin_groups_with_onsets"):
            put(r.get("method", "stack"), hz, m, r.get(m), ci_low=r.get("ci_low") if m == "bss" else None,
                ci_high=r.get("ci_high") if m == "bss" else None)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--look", choices=LOOKS, required=True)
    ap.add_argument("--rehearsal", action="store_true")
    a = ap.parse_args()
    cfg, cfg_sha = LC.load_config()
    root = LC.DRYRUN_ROOT if a.rehearsal else LC.LIVE_ROOT
    base = (LC.DRYRUN_ROOT / "looks_rehearsal") if a.rehearsal else (LC.LIVE_ROOT / "looks")
    look_dir = base / a.look
    reg_path = (base / "registry_rehearsal.jsonl") if a.rehearsal else None
    exp = f"ew_v2_live_{a.look}_score"
    split = f"live_{a.look}/ew_v2_live"
    info = pre_claim(a, cfg, cfg_sha, root, look_dir, reg_path, exp)
    ident = {"pid": os.getpid(), "host": socket.gethostname()}
    claim = {"look": a.look, "argv": sys.argv, "claimed_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(), **ident,
             **info, "rehearsal": a.rehearsal}
    LC.excl_write(look_dir / "CLAIM.json", claim)
    log_result(workstream="M1-sonde", experiment=exp, model="look", metric="look_claimed", value=1.0, split=split,
               extra={**ident, "scorer_code_sha256": info["scorer_code_sha256"]}, registry_path=reg_path)
    out_dir = look_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = ISS._log_to(look_dir / "score.log")
    try:
        res = score_look(a, cfg, root, out_dir, info, log)
        files = sorted(p for p in out_dir.iterdir() if p.is_file())
        (look_dir / "SHA256SUMS").write_text("".join(f"{LC.sha256_file(p)}  results/{p.name}\n" for p in files))
        for p in files + [look_dir / "SHA256SUMS"]:
            p.chmod(0o444)
        n = register(res, exp, split, reg_path, ident)
    except BaseException as e:   # noqa: BLE001 — includes KeyboardInterrupt / SystemExit: the look is closed
        LC.excl_write(look_dir / "ABORTED.json", {**ident, "error": repr(e)[:500],
                                                  "utc": _dt.datetime.now(_dt.timezone.utc).isoformat()})
        log_result(workstream="M1-sonde", experiment=exp, model="look", metric="look_aborted", value=1.0, split=split,
                   extra={**ident, "error": repr(e)[:300]}, registry_path=reg_path)
        raise
    LC.excl_write(look_dir / "SCORED.json", {**ident, "n_registry_rows": n, "sha256sums": LC.sha256_file(look_dir / "SHA256SUMS"),
                                             "utc": _dt.datetime.now(_dt.timezone.utc).isoformat()})
    log_result(workstream="M1-sonde", experiment=exp, model="look", metric="look_sealed", value=1.0, split=split,
               extra=ident, registry_path=reg_path)
    log(f"[score_live] {a.look}: sealed, {n} metric rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
