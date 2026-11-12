#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Hardened single look for the ORIGINAL M1 forecasting track (report-only; test-role × Test-A; lead approval in
principle 2026-11-10). Wraps the existing, unedited ``sonde_predict.py`` / ``sonde_ensemble.py`` / ``sonde_score.py``.

Phases
------
``plan``       writes ``docs/preregistration/sonde_m1_testA_plan.json`` (refuses to overwrite): every prediction job
               (model, config, overrides, checkpoint + SHA-256), every ensemble (members, equal weights) and the scoring
               call (forecasts, references, strict controls, candidates), all copied from the val evidence of the
               freeze candidate (``score_cyc3.sh``, ensemble manifests, TRAINING_LOG overrides). Checkpoint SHA-256s
               are cross-checked against ``sonde_m1_freeze_candidate.json``.
``rehearsal``  runs the whole plan on the VAL block into ``sentinel_runs/sonde/m1_testlook/rehearsal_val`` and checks
               byte-level reproduction (max |Δ| of every forecast array) against the stored val forecasts of the
               candidate's evidence runs. No test data.
``look``       the single Test-A look. Before the claim it verifies:
               - the lead-written ``FREEZE_sonde.json`` (not a candidate, no placeholders) pins the plan SHA-256 and the
                 full import-closure code hash of this script plus the wrapped scripts;
               - every checkpoint SHA-256; the split; the content pins of the daily/hourly tiers;
               - no registry row of ``sonde_m1_testA_score``; no marker.
               Then: O_EXCL 0444 ``CLAIM.json`` (pid, host, argv) + ``look_claimed`` registry row with the same
               pid/host -> every predict job with ``--eval-blocks test --allow-test --include-test-sites --freeze-file``
               -> ensembles -> predictions sealed (``SHA256SUMS``, 0444, ``predictions_sealed`` row) -> scoring with
               ``--register`` -> results sealed -> ``SCORED.json`` + ``look_sealed`` row. Any ``BaseException`` after
               the claim -> ``ABORTED.json`` + ``look_aborted`` row (no resume without a dated deviation). Rescore is
               refused once ``SCORED.json`` exists.
DISCLOSURE (carried into the plan and the report): the EW-v2 Test-A pass (2026-11-09) loaded and scored DO-min labels at
test-role sites in WY2024–25 (H1 hypoxia onsets); DO forecasting labels of this look are therefore NOT blind in spirit.
The look is report-only: it reports the original track; no new-track claim rests on it.

    python -B scripts/eval/sonde_m1_testlook.py --phase plan
    python -B scripts/eval/sonde_m1_testlook.py --phase rehearsal
    python -B scripts/eval/sonde_m1_testlook.py --phase look            # only after the lead writes FREEZE_sonde.json
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from sentinel.core import paths  # noqa: E402
from sentinel.core.registry import log_result, read_registry  # noqa: E402

PY = sys.executable
RUNS = paths.RUNS_ROOT / "sonde"
PLAN = REPO / "docs" / "preregistration" / "sonde_m1_testA_plan.json"
CANDIDATE = REPO / "docs" / "preregistration" / "sonde_m1_freeze_candidate.json"
FREEZE = REPO / "docs" / "preregistration" / "FREEZE_sonde.json"
LOOK_DIR = RUNS / "m1_testlook" / "testA"
EXP = "sonde_m1_testA_score"
DISK_FLOOR_GB, LOOK_DISK_GB = 60, 15   # lead ruling 2026-11-10 (look approved up to 15 GB over M1's 10 GB cap)
ENTRY_POINTS = ("scripts/eval/sonde_m1_testlook.py", "scripts/eval/sonde_predict.py", "scripts/eval/sonde_ensemble.py",
                "scripts/eval/sonde_score.py", "scripts/eval/sonde_pi_report.py")
EXTRA_FILES = ("configs/sonde/common.yaml", "configs/sonde/gbm.yaml", "configs/sonde/lstm.yaml",
               "configs/sonde/chronos_ft.yaml", "configs/sonde/chronos_zeroshot.yaml")

# ------------------------------------------------------------------ plan content (from the candidate's val evidence)
JOBS = {   # name -> (model, config, overrides, val evidence run)
    "gbm_weather": ("gbm", "configs/sonde/gbm.yaml", ["use_weather=true"], "cyc1_gbm_weather"),
    "gbm": ("gbm", "configs/sonde/gbm.yaml", [], "full_gbm"),
    "lstm_s0": ("lstm", "configs/sonde/lstm.yaml", [], "full_lstm_s0"),
    **{f"lstm_weather_s{k}": ("lstm", "configs/sonde/lstm.yaml", ["use_weather=true", "lstm.use_weather=true"],
                              f"cyc1_lstm_weather_s{k}") for k in range(3)},
    **{f"chronos2_ft_multi_s{k}": ("chronos2_ft_multi", "configs/sonde/chronos_ft.yaml", [], f"full_chronos_ft_s{k}")
       for k in range(3)},
    "chronos2_zs_uni": ("chronos2_zs_uni", "configs/sonde/chronos_zeroshot.yaml", [], "full_chronos_zs"),
    "chronos2_zs_multi": ("chronos2_zs_multi", "configs/sonde/chronos_zeroshot.yaml", [], "full_chronos_zs"),
    "timesfm25_zs": ("timesfm25_zs", "configs/sonde/chronos_zeroshot.yaml", [], "full_pred_timesfm25_val"),
}
# Weather-input reproduction (rehearsal finding 2026-11-10): the forcings rebuild of 2026-10-27 added ERA5 forcings at
# 79 M1 sites that had NO forcings when the weather models were trained and scored on val (2026-10-24). Every weather
# job therefore masks ERA5-forced sites to NaN weather, exactly the inputs of its training and val evidence. Verified
# on val: without the mask all changed rows are at ERA5-forced sites; with it the stored val forecasts are reproduced.
WEATHER_JOB_ARGS = ["--weather-exclude-sources", "era5_seamless"]
WEATHER_JOBS = ("gbm_weather", "lstm_weather_s0", "lstm_weather_s1", "lstm_weather_s2")
# <= 2 h GPU jobs (lead rule): multivariate Chronos runs ~2.7 h on val, so it stops after the first batch flush past
# 100 min and resumes in a new process (exactly the same batches, hence identical forecasts; verified on val).
CHUNK_ARGS = ["--max-minutes", "100"]
VAL_FORECAST_DIRS = {   # where the stored val forecasts of each job live (identity check of the rehearsal)
    "gbm": "full_pred_gbm_val", "chronos2_ft_multi_s0": "full_chronos_ft_s0", "chronos2_ft_multi_s1": "full_chronos_ft_s1",
    "chronos2_ft_multi_s2": "full_chronos_ft_s2",
}
ENSEMBLES = {   # name -> [(model prefix, job)] ; equal weights (pre-specified, not tuned)
    **{f"sentinel_ens3w_s{k}": [("chronos2_ft_multi", f"chronos2_ft_multi_s{k}"), ("lstm", f"lstm_weather_s{k}"),
                                ("chronos2_zs_multi", "chronos2_zs_multi")] for k in range(3)},
    "baseline_ens_gbm_lstm": [("gbm", "gbm"), ("lstm", "lstm_s0")],
    "control3_gbmw_lstm_zsmulti": [("gbm", "gbm_weather"), ("lstm", "lstm_s0"), ("chronos2_zs_multi", "chronos2_zs_multi")],
    "control3w_gbmw_lstmw_zsmulti": [("gbm", "gbm_weather"), ("lstm", "lstm_weather_s0"),
                                     ("chronos2_zs_multi", "chronos2_zs_multi")],
}
SCORE_FORECASTS = {   # sonde_score label -> (prefix, job or ensemble)
    "gbm_weather": ("gbm", "gbm_weather"), "lstm_s0": ("lstm", "lstm_s0"),
    "chronos2_zs_uni": (None, "chronos2_zs_uni"), "chronos2_zs_multi": (None, "chronos2_zs_multi"),
    "timesfm25_zs": (None, "timesfm25_zs"),
    **{f"lstm_weather_s{k}": ("lstm", f"lstm_weather_s{k}") for k in range(3)},
    **{e: (None, e) for e in ENSEMBLES},
}
REFS = ["gbm_weather", "lstm_s0", "lstm_weather_s0", "chronos2_zs_uni", "chronos2_zs_multi", "timesfm25_zs",
        "baseline_ens_gbm_lstm", "control3_gbmw_lstm_zsmulti", "control3w_gbmw_lstmw_zsmulti"]
CANDIDATES = ["sentinel_ens3w_s0", "sentinel_ens3w_s1", "sentinel_ens3w_s2"]


def sha256_path(p: Path) -> str:
    from sentinel.models.sonde import guard
    return guard.artifact_sha256(p)


def checkpoint_of(run: str) -> tuple[str | None, str | None]:
    d = RUNS / run
    for f in ("checkpoint.json", "predict_manifest.json"):
        if (d / f).is_file():
            m = json.loads((d / f).read_text())
            return m.get("path") or m.get("checkpoint"), m.get("sha256") or m.get("checkpoint_sha256")
    return None, None


# ------------------------------------------------------------------ code hash (full import closure)
def _module_file(mod: str):
    base = REPO.joinpath(*mod.split("."))
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return base.with_suffix(".py") if base.with_suffix(".py").is_file() else None


def code_files() -> list[str]:
    todo, seen = [REPO / e for e in ENTRY_POINTS], set()
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        parts = f.relative_to(REPO).parts
        for i in range(1, len(parts)):
            init = REPO.joinpath(*parts[:i], "__init__.py")
            if init.is_file() and init not in seen:
                todo.append(init)
        tree = ast.parse(f.read_text())
        pkg = list(f.relative_to(REPO).parent.parts)
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names if a.name.startswith("sentinel")]
            elif isinstance(node, ast.ImportFrom):
                mod = ".".join(pkg[: len(pkg) - (node.level - 1)] + ([node.module] if node.module else [])) \
                    if node.level else (node.module or "")
                if mod.startswith("sentinel"):
                    mods = [mod] + [f"{mod}.{a.name}" for a in node.names]
            for m in mods:
                mf = _module_file(m)
                if mf is not None and mf not in seen:
                    todo.append(mf)
    return sorted({str(p.relative_to(REPO)) for p in seen} | set(EXTRA_FILES))


def code_sha256() -> str:
    h = hashlib.sha256()
    for rel in code_files():
        h.update(rel.encode())
        h.update((REPO / rel).read_bytes())
    return h.hexdigest()


# ------------------------------------------------------------------ phases
def phase_plan() -> int:
    if PLAN.exists():
        raise SystemExit(f"{PLAN} exists")
    cand = json.loads(CANDIDATE.read_text())
    jobs = {}
    for name, (model, cfg, sets, run) in JOBS.items():
        ck, ck_sha = checkpoint_of(run)
        if ck is not None:
            actual = sha256_path(Path(ck))
            if ck_sha and actual != ck_sha:
                raise SystemExit(f"{name}: checkpoint {ck} changed ({actual[:12]} != {ck_sha[:12]})")
            ck_sha = actual
        jobs[name] = {"model": model, "config": cfg, "set": sets, "checkpoint": ck, "checkpoint_sha256": ck_sha,
                      "val_evidence_run": run,
                      "args": (WEATHER_JOB_ARGS if name in WEATHER_JOBS else [])
                      + (CHUNK_ARGS if model.startswith("chronos2") and model.endswith("_multi") else [])}
    for m, sha in cand["weights_sha256"].items():          # members must match the candidate exactly
        j = {"chronos2_ft_multi_s0": "chronos2_ft_multi_s0", "chronos2_ft_multi_s1": "chronos2_ft_multi_s1",
             "chronos2_ft_multi_s2": "chronos2_ft_multi_s2", "lstm_weather_s0": "lstm_weather_s0",
             "lstm_weather_s1": "lstm_weather_s1", "lstm_weather_s2": "lstm_weather_s2"}.get(m)
        if j and jobs[j]["checkpoint_sha256"] != sha:
            raise SystemExit(f"{m}: plan checkpoint differs from the freeze candidate")
    rec = {"created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "look": "original M1 track, test-role sites × Test-A (WY2024–25), report-only",
           "candidate_sha256": hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
           "jobs": jobs, "ensembles": {k: {"members": v, "weights": "equal"} for k, v in ENSEMBLES.items()},
           "score": {"forecasts": SCORE_FORECASTS, "refs": REFS, "candidates": CANDIDATES, "n_boot": 2000},
           "disclosure": "EW-v2 Test-A (2026-11-09) loaded and scored DO-min labels at test-role sites in WY2024-25; "
                         "DO forecasting labels of this look are not blind in spirit; report-only original track.",
           "code_files": code_files(), "code_sha256": code_sha256()}
    PLAN.write_text(json.dumps(rec, indent=1))
    PLAN.chmod(0o444)
    print(json.dumps({"plan": str(PLAN), "sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
                      "code_sha256": rec["code_sha256"], "n_jobs": len(jobs)}, indent=1))
    return 0


GPU_MODELS = ("lstm", "timesfm25_zs")          # + every chronos2_* model
GPU_CONCURRENCY = 3     # 3 GPU jobs x 2 CPU threads = the 6-thread host rule; CPU (GBM, 6 threads) jobs run alone
MAX_RESUMES = 12        # resumable multivariate-Chronos jobs: <= 100 min per process, exit 75 = resume


def _is_gpu(j: dict) -> bool:
    return j["model"] in GPU_MODELS or j["model"].startswith("chronos2")


def run_one(name: str, j: dict, block: str, root: Path, protected: bool, log) -> Path:
    """One predict job; a resumable job (exit 75) is re-run with the same command until complete. Each process holds
    its GPU lock for <= 2 h and releases it on exit."""
    run = f"m1testlook_{root.name}_{name}"
    cmd = [PY, "-B", "scripts/eval/sonde_predict.py", "--config", j["config"], "--model", j["model"],
           "--eval-blocks", block, "--run-name", run]
    for s_ in j["set"]:
        cmd += ["--set", s_]
    cmd += j.get("args", [])
    if j["checkpoint"]:
        if j["checkpoint_sha256"] and sha256_path(Path(j["checkpoint"])) != j["checkpoint_sha256"]:
            raise PermissionError(f"{name}: checkpoint changed")
        cmd += ["--checkpoint", j["checkpoint"]]
    if protected:
        cmd += ["--allow-test", "--include-test-sites", "--freeze-file", str(FREEZE)]
    log(f"[job] {name}: {' '.join(cmd[2:])}")
    logs = []
    for attempt in range(MAX_RESUMES):
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        logs.append(f"===== process {attempt + 1} (rc {r.returncode})\n" + r.stdout + r.stderr)
        (root / f"{name}.log").write_text("\n".join(logs))
        if r.returncode == 75:
            log(f"[job] {name}: partial (process {attempt + 1}); resuming")
            continue
        if r.returncode != 0:
            raise RuntimeError(f"{name} failed (rc {r.returncode}); see {root / (name + '.log')}")
        return RUNS / run / "forecasts"
    raise RuntimeError(f"{name}: not complete after {MAX_RESUMES} processes")


def run_pool(names: list[str], plan: dict, block: str, root: Path, protected: bool, log, after=None) -> dict:
    """CPU jobs one at a time, then GPU jobs GPU_CONCURRENCY at a time (each takes its own GPU lock via
    sentinel.core.gpu). ``after(name, out)`` runs in the worker once a job is done. Re-raises the first failure
    after the running jobs end."""
    from concurrent.futures import ThreadPoolExecutor

    out = {}
    cpu = [n for n in names if not _is_gpu(plan["jobs"][n])]
    gpu = [n for n in names if _is_gpu(plan["jobs"][n])]

    def work(n):
        o = run_one(n, plan["jobs"][n], block, root, protected, log)
        if after is not None:
            after(n, o)
        return o

    for n in cpu:
        out[n] = work(n)
    errors = []
    with ThreadPoolExecutor(max_workers=GPU_CONCURRENCY) as ex:
        futs = {n: ex.submit(work, n) for n in gpu}
        for n, f in futs.items():
            try:
                out[n] = f.result()
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
    if errors:
        raise errors[0]
    return out


def run_jobs(plan: dict, block: str, root: Path, protected: bool, log) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    out = run_pool(list(plan["jobs"]), plan, block, root, protected, log)
    for ens, members in plan["ensembles"].items():
        cmd = [PY, "-B", "scripts/eval/sonde_ensemble.py", "--name", ens, "--block", block, "--out",
               str(root / "ensembles" / ens)]
        for prefix, job in members["members"]:
            cmd += ["--member", f"{prefix}@{out[job]}"]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        (root / f"ens_{ens}.log").write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            raise RuntimeError(f"ensemble {ens} failed")
        out[ens] = root / "ensembles" / ens
    return out


def compare_val(out: Path, ref: Path) -> dict:
    diffs, rowstats = {}, {}
    for f in sorted(Path(out).glob("*__val.npz")):
        tail = f.name.split("__", 1)[1]
        g = ref / f.name                       # exact file (same model prefix); never another model's file
        if not g.is_file():
            diffs[tail] = "missing in stored val forecasts"
            continue
        za, zb = np.load(f), np.load(g)
        qa, qb = za["q"], zb["q"]
        if qa.shape != qb.shape or not np.array_equal(za["site_no"], zb["site_no"]):
            diffs[tail] = f"rows differ: shape {qa.shape} vs {qb.shape}"
            continue
        d = np.abs(qa - qb).reshape(len(qa), -1)
        diffs[tail] = float(np.nanmax(d))
        bad = np.nanmax(d, axis=1) > 1e-5
        rowstats[tail] = {"n_rows": int(len(qa)), "n_rows_changed": int(bad.sum()),
                          "sites_changed": sorted(set(za["site_no"][bad].astype(str).tolist()))}
    return {"max_abs_diff_by_file": diffs, "n_files": len(diffs), "row_stats": rowstats}


def phase_rehearsal(log) -> int:
    """STREAMING val rehearsal (disk cap): each job predicts on val with exactly the plan's command (incl. resumable
    chunking and the weather rule), is compared with its stored val forecasts, and its output is deleted at once
    (identity numbers kept). Ensembles are deterministic equal-weight averages of the members (``sonde_ensemble.py``),
    so member identity implies ensemble identity; they are not re-run here. The report records the plan hash."""
    import shutil
    import threading

    plan = json.loads(PLAN.read_text())
    plan_sha = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    root = RUNS / "m1_testlook" / "rehearsal_val"
    root.mkdir(parents=True, exist_ok=True)
    rep_f = root / "identity.json"
    report = json.loads(rep_f.read_text()) if rep_f.exists() else {}
    lock = threading.Lock()

    def after(name, out):
        ref_run = VAL_FORECAST_DIRS.get(name) or plan["jobs"][name]["val_evidence_run"]
        rec = {**compare_val(out, RUNS / ref_run / "forecasts"), "plan_sha256": plan_sha,
               "job": plan["jobs"][name], "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
        with lock:
            report[name] = rec
            rep_f.write_text(json.dumps(report, indent=1, default=str))
        shutil.rmtree(Path(out).parent, ignore_errors=True)  # disk cap: delete the rehearsal forecasts
        vals = [v for v in rec["max_abs_diff_by_file"].values() if isinstance(v, float)]
        log(f"[rehearsal] {name}: max |Δ| = {max(vals or [float('nan')])}")

    todo = [n for n in plan["jobs"] if report.get(n, {}).get("plan_sha256") != plan_sha]   # resume-safe
    run_pool(todo, plan, "val", root, False, log, after=after)
    print(json.dumps(report, indent=1, default=str))
    return 0


def phase_look(log) -> int:
    if (LOOK_DIR / "SCORED.json").exists() or (LOOK_DIR / "CLAIM.json").exists() or (LOOK_DIR / "ABORTED.json").exists():
        raise SystemExit("look spent, claimed or aborted: rescore refused")
    if not FREEZE.is_file():
        raise SystemExit("FREEZE_sonde.json missing (lead-written)")
    fr = json.loads(FREEZE.read_text())
    if "<" in json.dumps(fr) or "candidate" in json.dumps(fr.get("status", "")).lower():
        raise SystemExit("FREEZE_sonde.json is a candidate or has placeholders")
    plan_sha = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    if fr.get("testlook_plan_sha256") != plan_sha or fr.get("testlook_code_sha256") != code_sha256():
        raise SystemExit("plan or import-closure code hash differs from FREEZE_sonde.json")
    plan = json.loads(PLAN.read_text())
    for name, j in plan["jobs"].items():
        if j["checkpoint"] and sha256_path(Path(j["checkpoint"])) != j["checkpoint_sha256"]:
            raise SystemExit(f"{name}: checkpoint changed")
    if any(r.get("experiment") == EXP for r in read_registry()):
        raise SystemExit(f"registry already holds {EXP}: look spent")
    import shutil
    free_gb = shutil.disk_usage(RUNS).free / 1e9   # lead ruling 2026-11-10: <= 15 GB for the look, 60 GB floor
    if free_gb < DISK_FLOOR_GB + LOOK_DISK_GB:
        raise SystemExit(f"free disk {free_gb:.0f} GB < floor {DISK_FLOOR_GB} + look budget {LOOK_DISK_GB} GB")
    ident = {"pid": os.getpid(), "host": socket.gethostname()}
    LOOK_DIR.mkdir(parents=True, exist_ok=True)
    fd = os.open(LOOK_DIR / "CLAIM.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps({**ident, "argv": sys.argv, "plan_sha256": plan_sha, "freeze_sha256":
                             hashlib.sha256(FREEZE.read_bytes()).hexdigest(),
                             "utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}))
    split = "test/sonde_split_fd4bb14fe730"
    log_result(workstream="M1-sonde", experiment=EXP, model="look", metric="look_claimed", value=1.0, split=split,
               extra=ident)
    try:
        out = run_jobs(plan, "test", LOOK_DIR / "predictions", True, log)
        sums = []
        for d in out.values():
            for f in sorted(Path(d).glob("*.npz")):
                sums.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f}")
                os.chmod(f, 0o444)
        (LOOK_DIR / "PREDICTIONS_SHA256SUMS").write_text("\n".join(sums) + "\n")
        log_result(workstream="M1-sonde", experiment=EXP, model="look", metric="predictions_sealed", value=float(len(sums)),
                   split=split, extra=ident)
        cmd = [PY, "-B", "scripts/eval/sonde_score.py", "--config", "configs/sonde/common.yaml", "--eval-blocks", "test",
               "--allow-test", "--include-test-sites", "--freeze-file", str(FREEZE), "--register",
               "--run-name", "m1_testA_score", "--refs", ",".join(plan["score"]["refs"]),
               "--candidates", ",".join(plan["score"]["candidates"])]
        for label, (prefix, src) in plan["score"]["forecasts"].items():
            cmd += ["--forecast", f"{label}={prefix + '@' if prefix else ''}{out[src]}"]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        (LOOK_DIR / "score.log").write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            raise RuntimeError("scoring failed")
        # numeric forecasts (user requirement 2026-11-10): medians + 50/80/90 % intervals with coverage and width,
        # MAE/RMSE/bias and MAE/RMSE skill vs persistence and climatology, every model x variable x horizon; per-row
        # numeric forecasts for the candidate seeds. Reads the sealed predictions only; no fitting.
        cmd = [PY, "-B", "scripts/eval/sonde_pi_report.py", "--config", "configs/sonde/common.yaml", "--eval-blocks",
               "test", "--allow-test", "--include-test-sites", "--freeze-file", str(FREEZE),
               "--out", str(LOOK_DIR / "numeric"), "--tables-for", ",".join(plan["score"]["candidates"])]
        for label, (prefix, src) in plan["score"]["forecasts"].items():
            cmd += ["--forecast", f"{label}={prefix + '@' if prefix else ''}{out[src]}"]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        (LOOK_DIR / "numeric_report.log").write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            raise RuntimeError("numeric report failed")
    except BaseException as e:  # noqa: BLE001
        fd = os.open(LOOK_DIR / "ABORTED.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps({**ident, "error": repr(e)[:400]}))
        log_result(workstream="M1-sonde", experiment=EXP, model="look", metric="look_aborted", value=1.0, split=split,
                   extra={**ident, "error": repr(e)[:200]})
        raise
    fd = os.open(LOOK_DIR / "SCORED.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps({**ident, "utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}))
    log_result(workstream="M1-sonde", experiment=EXP, model="look", metric="look_sealed", value=1.0, split=split,
               extra=ident)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("plan", "rehearsal", "look"), required=True)
    a = ap.parse_args()
    log = lambda m: print(f"{_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')} {m}", flush=True)  # noqa: E731
    return {"plan": phase_plan, "rehearsal": lambda: phase_rehearsal(log), "look": lambda: phase_look(log)}[a.phase]()


if __name__ == "__main__":
    raise SystemExit(main())
