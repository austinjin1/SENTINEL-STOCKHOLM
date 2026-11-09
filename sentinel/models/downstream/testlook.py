# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Single-pass guard and shared scoring for downstream test looks (pre-test gate 2026-11-08).

Order inside every ``scripts/train/downstream_<app>_test.py``:

1. ``verify_data(app)`` — every pinned input file (``docs/preregistration/downstream_<app>_data_pins.json``)
   must match its sha256 **before** the claim, so a data mismatch can never burn the look;
2. ``authorise(app, split_sha)`` — ``common.check_roles`` (FREEZE file, split sha, scoring-code sha), then
   ``claim_test_look``: refuses if the registry already holds a test-split row of the app, and atomically
   creates ``$SENTINEL_RUNS/downstream/<app>/test_look/TEST_LOOK.json`` (O_CREAT | O_EXCL);
3. builders read test-role rows only if ``require_authorised`` passes (freeze verifies AND this freeze's
   claim exists);
4. ``save_predictions`` writes ALL predictions (incl. per-seed) once (O_EXCL marker with sha256) **before
   any registry write**;
5. scoring + registry writes (``score_set`` → ``log_test_metric``), then ``mark_scored``.

Crash recovery: after step 4, ``--rescore-from-saved`` (``authorise_rescore``) re-scores the SAVED
predictions (sha-checked) — re-fitting is impossible because ``save_predictions`` refuses a second write and
the claim cannot be re-made. After ``mark_scored`` nothing can be re-scored.

Seed rule (RESEARCH_PLAN §15.1(iii), gate decision 2026-11-08): a claim requires, for every primary
comparison, Holm rejection with Δ > 0 for the seed-averaged model AND a paired point estimate Δ > 0 for each
of the ≥ 3 seeds on the test block. Deterministic challengers (logistic) meet it trivially (stated).
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time

from ...core import paths
from . import common as C


# --------------------------------------------------------------------------- data pins
def pins_path(app: str):
    return C.PREREG / f"downstream_{app}_data_pins.json"


def file_sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def make_pins(app: str, rel_paths: list[str], expected: dict | None = None) -> dict:
    """Pin sha256 of every input file (paths relative to SENTINEL_DATA; directories are expanded)."""
    files = {}
    for rel in rel_paths:
        p = paths.DATA_ROOT / rel
        items = sorted(q for q in p.rglob("*") if q.is_file() and q.suffix != ".log") if p.is_dir() else [p]
        for q in items:
            files[str(q.relative_to(paths.DATA_ROOT))] = file_sha(q)
    body = {"app": app, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": files, "expected": expected or {}}
    pins_path(app).write_text(json.dumps(body, indent=1, sort_keys=True))
    return body


def pip_freeze_sha() -> str:
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
    return C.sha256_text(out)


def verify_data(app: str) -> dict:
    """Pinned input files AND the pinned environment (pip freeze sha) must match — before the claim."""
    pins = json.loads(pins_path(app).read_text())
    exp_pip = pins.get("expected", {}).get("pip_freeze_sha256")
    if exp_pip and pip_freeze_sha() != exp_pip:
        raise C.ProtectedDataError(f"{app}: python environment (pip freeze) differs from the pinned one")
    bad = [rel for rel, sha in pins["files"].items()
           if not (paths.DATA_ROOT / rel).is_file() or file_sha(paths.DATA_ROOT / rel) != sha]
    if bad:
        raise C.ProtectedDataError(f"{app}: {len(bad)} pinned data files changed or missing, e.g. {bad[:3]}")
    return pins


def check_expected(app: str, key: str, value) -> None:
    exp = json.loads(pins_path(app).read_text()).get("expected", {})
    if key in exp and exp[key] != value:
        raise C.ProtectedDataError(f"{app}: {key} = {value} differs from pinned {exp[key]}")


# --------------------------------------------------------------------------- claim
def registry_test_rows(app: str) -> int:
    reg = paths.REPO_ROOT / "results" / "registry.jsonl"
    n = 0
    if reg.is_file():
        with reg.open() as fh:
            for line in fh:
                if f'"{app}' not in line:
                    continue
                r = json.loads(line)
                if r.get("workstream") == C.WORKSTREAM and str(r.get("experiment", "")).startswith(app) \
                        and str(r.get("split", "")).startswith("test"):
                    n += 1
    return n


def look_dir(app: str):
    return paths.RUNS_ROOT / "downstream" / app / "test_look"


def marker_path(app: str):
    return look_dir(app) / "TEST_LOOK.json"


def _excl_write(p, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(fd, "w") as fh:
        json.dump(rec, fh)


def claim_test_look(app: str, freeze_sha: str) -> dict:
    n = registry_test_rows(app)
    if n:
        raise C.ProtectedDataError(f"{app}: registry already holds {n} test-split rows — test block used")
    rec = {"app": app, "freeze_sha256": freeze_sha, "claimed": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "pid": os.getpid(), "host": socket.gethostname(), "sentinel_runs": str(paths.RUNS_ROOT)}
    try:
        _excl_write(marker_path(app), rec)
    except FileExistsError as e:
        raise C.ProtectedDataError(f"{app}: test look already claimed ({marker_path(app)})") from e
    return rec


def _freeze_sha(app: str) -> str:
    ff = C.freeze_file(app)
    if not ff.is_file():
        raise C.ProtectedDataError(f"{app}: {ff.name} does not exist — no test look without a freeze")
    return C.sha256_text(ff.read_text())


def authorise(app: str, split_sha: str) -> dict:
    """The ONLY entry to a test look: FREEZE + split sha + scoring-code sha, then the single-pass claim."""
    C.check_roles(app, {"train", "val", "test"}, allow_test=True, split_sha=split_sha)
    return claim_test_look(app, _freeze_sha(app))


def require_authorised(app: str, split_sha: str) -> None:
    """Every builder asked for test-role rows: the freeze must verify AND this freeze's claim must exist."""
    C.check_roles(app, {"train", "val", "test"}, allow_test=True, split_sha=split_sha)
    p = marker_path(app)
    if not p.is_file():
        raise C.ProtectedDataError(f"{app}: test rows requested without a claimed test look")
    if json.loads(p.read_text()).get("freeze_sha256") != _freeze_sha(app):
        raise C.ProtectedDataError(f"{app}: test-look claim belongs to a different freeze")


# --------------------------------------------------------------------------- crash-safe predictions
def _pred_marker(app):
    return look_dir(app) / "PREDICTIONS_SAVED.json"


def _scored_marker(app):
    return look_dir(app) / "SCORED.json"


def save_predictions(app: str, split_sha: str, pred) -> dict:
    """Write the look's predictions ONCE (before any registry write). A second call raises: no re-fit."""
    require_authorised(app, split_sha)
    if _pred_marker(app).exists():
        raise C.ProtectedDataError(f"{app}: predictions already saved — re-fitting is not allowed")
    p = look_dir(app) / "predictions.parquet"
    pred.to_parquet(p, index=False)
    rec = {"file": str(p), "sha256": file_sha(p), "n_rows": int(len(pred)), "saved": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _excl_write(_pred_marker(app), rec)
    return rec


def authorise_rescore(app: str, split_sha: str):
    """Recovery after a crash between save_predictions and mark_scored: returns the saved predictions."""
    import pandas as pd

    require_authorised(app, split_sha)
    if _scored_marker(app).exists():
        raise C.ProtectedDataError(f"{app}: test look already scored — nothing to re-score")
    if not _pred_marker(app).exists():
        raise C.ProtectedDataError(f"{app}: no saved predictions — the look crashed before predictions; report it")
    rec = json.loads(_pred_marker(app).read_text())
    if file_sha(rec["file"]) != rec["sha256"]:
        raise C.ProtectedDataError(f"{app}: saved predictions were modified")
    return pd.read_parquet(rec["file"])


def mark_scored(app: str, summary: dict) -> None:
    _excl_write(_scored_marker(app), {"scored": time.strftime("%Y-%m-%dT%H:%M:%S"), **summary})


def mode_from_args(a) -> str:
    """'test' (authorised look), 'rescore' (crash recovery from saved predictions) or 'dryrun'."""
    flags = {"test": a.allow_test, "rescore": getattr(a, "rescore_from_saved", False), "dryrun": a.dry_run_on_val}
    on = [k for k, v in flags.items() if v]
    if len(on) != 1:
        raise SystemExit("choose exactly one of --allow-test, --rescore-from-saved, --dry-run-on-val")
    return on[0]


def add_mode_args(ap) -> None:
    ap.add_argument("--allow-test", action="store_true", help="the single authorised test look")
    ap.add_argument("--rescore-from-saved", action="store_true", help="crash recovery: re-score saved predictions")
    ap.add_argument("--dry-run-on-val", action="store_true", help="identical pipeline, validation in the test role")


# --------------------------------------------------------------------------- registry
def log_test_metric(*, app: str, split_sha: str, experiment: str, model: str, metric: str, value: float,
                    split: str, n=None, ci=None, ci_method=None, is_baseline=False, data_manifests=(),
                    artifacts=(), extra=None, notes="") -> dict:
    """Registry write for the single authorised test look (split name must start with 'test')."""
    import numpy as np

    from ...core.registry import log_result

    if not split.startswith("test"):
        raise ValueError("test-look metrics use split names starting with 'test'")
    require_authorised(app, split_sha)
    if value is None or not np.isfinite(value):
        raise ValueError(f"non-finite {metric} for {model}")
    return log_result(workstream=C.WORKSTREAM, experiment=f"{app}/{experiment}", model=model, metric=metric,
                      value=float(value), split=split, n=n,
                      ci_low=None if ci is None else ci.get("ci_low"), ci_high=None if ci is None else ci.get("ci_high"),
                      ci_method=ci_method, target=None, is_baseline=is_baseline, config=None,
                      data_manifests=list(data_manifests), artifacts=list(artifacts), notes=notes,
                      extra={**(extra or {}), "test_look": True, "split_sha256": split_sha})


def prereg_snapshot(app: str, files: list[str], extra: dict | None = None) -> dict:
    """Ordering evidence without git (user instruction: no git): sha256 + mtime of each protocol, record and
    test script, appended to the registry when the FREEZE file is written (called by the lead)."""
    from ...core.registry import log_result

    snap = {}
    for rel in files:
        p = paths.REPO_ROOT / rel
        snap[rel] = {"sha256": file_sha(p), "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(p.stat().st_mtime))}
    ff = C.freeze_file(app)
    if ff.is_file():
        snap[str(ff.relative_to(paths.REPO_ROOT))] = {"sha256": file_sha(ff), "mtime": time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(ff.stat().st_mtime))}
    return log_result(workstream=C.WORKSTREAM, experiment=f"{app}/prereg_snapshot", model="prereg_snapshot",
                      metric="n_files", value=float(len(snap)), split="prereg", n=len(snap), notes="no-git ordering evidence",
                      extra={"files": snap, "snapshot_time": time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {})})


# --------------------------------------------------------------------------- shared scoring
def _metric_name(metric) -> str:
    return {C.fast_auroc: "auroc", C.fast_auprc: "auprc"}.get(metric, getattr(metric, "__name__", "metric"))


def score_set(*, app: str, mode: str, split_sha: str, set_name: str, experiment: str, y, scores: dict,
              groups: dict, primary: dict, secondary: dict | None = None, metric=None, prob_models=(),
              manifests=(), artifacts=(), per_seed: dict | None = None) -> dict:
    """Score one evaluation set. ``groups`` = {unit: cluster ids} (first = primary unit); ``primary`` /
    ``secondary`` = {challenger: [comparators]}; ``per_seed`` = {challenger: [per-seed scores]} for the seed
    rule (paired point estimate per seed vs each primary comparator). Registers only in ``test``/``rescore``."""
    import numpy as np

    metric = metric or C.fast_auroc
    mname = _metric_name(metric)
    register = mode in ("test", "rescore")
    y = np.asarray(y, float)
    unit0 = next(iter(groups))
    out = {"n": int(len(y)), "n_pos": int(np.nansum(y)), "metric": mname, "scores": {}, "paired": {}}
    for name, s in scores.items():
        bs = C.binary_scores(y, s, groups[unit0], prob=name in prob_models)
        out["scores"][name] = bs
        if register:
            for met in ("auroc", "auprc", "brier"):
                if met in bs and np.isfinite(bs[met]):
                    log_test_metric(app=app, split_sha=split_sha, experiment=experiment, model=name, metric=met,
                                    value=bs[met], split=set_name, n=bs["n"],
                                    ci={"ci_low": bs[f"{met}_ci"][0], "ci_high": bs[f"{met}_ci"][1]},
                                    ci_method=f"cluster bootstrap over {unit0}, B={C.N_BOOT}",
                                    is_baseline=name not in primary, data_manifests=manifests, artifacts=artifacts,
                                    extra={"n_pos": bs["n_pos"], "prevalence": bs["prevalence"], "rescore": mode == "rescore"})
    for kind, comp in (("primary", primary), ("secondary", secondary or {})):
        for ch, bases in comp.items():
            for b in bases:
                key = f"{ch}-{b}"
                r = {"kind": kind}
                for unit, g in groups.items():
                    r[unit] = C.paired(metric, y, scores[ch], scores[b], g)
                    if not np.isfinite(r[unit].get("diff", np.nan)):
                        r[unit]["not_estimable"] = True       # degenerate set/stratum: recorded, never registered
                    elif register:
                        d = r[unit]
                        log_test_metric(app=app, split_sha=split_sha, experiment=experiment, model=key,
                                        metric=f"{mname}_diff_paired_{unit}", value=d["diff"], split=set_name,
                                        n=int(np.isfinite(scores[b]).sum()), ci={"ci_low": d["ci_low"], "ci_high": d["ci_high"]},
                                        ci_method=f"paired cluster bootstrap over {unit}, B={C.N_BOOT}",
                                        data_manifests=manifests, extra={"p_value": d["p_value"], "kind": kind,
                                                                         "rescore": mode == "rescore"})
                if kind == "primary" and per_seed and ch in per_seed:
                    ok = np.isfinite(y) & np.isfinite(scores[b])
                    r["per_seed_diff"] = [float(metric(y[ok], np.asarray(p)[ok]) - metric(y[ok], np.asarray(scores[b])[ok]))
                                          for p in per_seed[ch]]
                out["paired"][key] = r
    return out


def holm_primary(results: dict, unit: str, unscored: tuple = ()) -> dict:
    """Holm across every primary comparison of every set on ``unit``; ``unscored`` = names of pre-registered
    family members not scored in this look, counted with p = 1 (they enlarge m, never reject)."""
    from ...eval.bootstrap import holm_bonferroni

    pv, info = {}, {}
    for s, r in results.items():
        for k, v in r["paired"].items():
            if v["kind"] == "primary":
                pv[f"{s}|{k}"] = v[unit]["p_value"]
                info[f"{s}|{k}"] = {"diff": v[unit]["diff"], "ci": [v[unit]["ci_low"], v[unit]["ci_high"]],
                                    "per_seed_diff": v.get("per_seed_diff")}
    for u in unscored:
        pv[f"unscored|{u}"] = 1.0
        info[f"unscored|{u}"] = {"diff": None, "ci": None, "per_seed_diff": None, "unscored": True}
    h = holm_bonferroni(pv)
    return {k: {"p_value": pv[k], "holm_reject": bool(h[k]), **info[k], "m": len(pv)} for k in pv}


def claim_met(holm: dict, *, min_seeds: int = 3, deterministic: bool = False) -> dict:
    """Scored primary comparisons must all be Holm-rejected with Δ > 0; seed rule: each of ≥ min_seeds
    per-seed paired point estimates > 0 (deterministic challengers: trivially met, stated)."""
    scored = {k: v for k, v in holm.items() if not v.get("unscored")}
    if not scored:   # fail closed: an empty primary family can never support a claim
        return {"holm_ok": False, "seed_rule_ok": False, "claim_met": False, "seed_rule": "no scored primary comparison"}
    holm_ok = all(v["holm_reject"] and v["diff"] is not None and v["diff"] > 0 for v in scored.values())
    if deterministic:
        seed_ok, note = True, "deterministic challenger — seed rule trivially met"
    else:
        seeds = [v.get("per_seed_diff") for v in scored.values()]
        seed_ok = all(s is not None and len(s) >= min_seeds and all(d > 0 for d in s) for s in seeds)
        note = f"each of >= {min_seeds} seeds: paired point estimate > 0 for every primary comparison"
    return {"holm_ok": bool(holm_ok), "seed_rule_ok": bool(seed_ok), "claim_met": bool(holm_ok and seed_ok),
            "seed_rule": note}
