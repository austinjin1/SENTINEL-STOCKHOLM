# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Shared plumbing for the downstream applications (RESEARCH_PLAN §3 R1–R8, §8, §14).

* **Frozen splits.** :func:`freeze_split` writes ``docs/preregistration/downstream_<app>_split.json``
  (canonical JSON; sha256 over the canonical bytes) and refuses to overwrite a frozen file with
  different content. Protocols quote the sha256; :func:`load_split` verifies it.
* **Test guard (R6).** :func:`drop_protected` removes test-role rows before any model code sees them;
  :func:`check_roles` refuses ``test`` unless ``allow_test`` and a valid
  ``docs/preregistration/FREEZE_downstream_<app>.json`` (split sha + scoring-code sha) exist. Phase 1
  never passes ``allow_test``.
* **Registry (R2).** :func:`log_metric` wraps ``sentinel.core.registry.log_result`` with workstream
  ``downstream`` and refuses any split name that mentions ``test``.
* **Evaluation (R7).** :func:`binary_scores` / :func:`regression_scores` with cluster-bootstrap CIs
  (unit stated) and :func:`paired` model-vs-baseline differences.
* **CPU budget.** :func:`cpu_slot` — at most 2 heavy downstream jobs at once (fcntl slot locks), and
  ``N_THREADS`` = 8 per job.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from ...core import paths
from ...core.registry import config_hash, log_result
from ...eval import bootstrap as B
from ...eval import metrics as M

WORKSTREAM = "downstream"
PLAN_REF = "RESEARCH_PLAN.md §8 (Goal 4 downstream applications)"
PREREG = paths.REPO_ROOT / "docs" / "preregistration"
N_THREADS = 6   # shared-host allocation 2026-10-24 22:30: one heavy job per workstream, <= 6 threads
N_BOOT = 2000
ROLES = ("train", "val", "test")
APPS = {"d1": "safeswim", "d2": "toxinwatch", "d3": "killalert", "d5": "biointegrity",
        "d6": "musselguard", "d8": "riverpulse"}


class ProtectedDataError(RuntimeError):
    """Raised when code tries to use test-role data without a valid freeze."""


# --------------------------------------------------------------------------- canonical hashing
def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def split_path(app: str) -> Path:
    return PREREG / f"downstream_{app}_split.json"


def freeze_split(app: str, assignments: dict, meta: dict, *, path: Path | None = None) -> str:
    """Freeze ``assignments`` (``{name: {unit: value}}``) for ``app``; maps whose name ends in ``role``
    may only contain train/val/test.

    The file holds ``{"app", "meta", "assignments"}`` in canonical JSON. Re-freezing identical content is
    a no-op that returns the same sha; different content raises (a frozen split is never edited — make a
    new app version and log a deviation instead).
    """
    for name, a in assignments.items():
        if not name.endswith("role"):
            continue  # e.g. cluster / fold / group maps; only *role maps are restricted to ROLES
        bad = {r for r in a.values() if r not in ROLES}
        if bad:
            raise ValueError(f"assignment {name}: unknown roles {bad}")
    body = {"app": app, "meta": meta, "assignments": {k: {str(u): r for u, r in v.items()}
                                                       for k, v in assignments.items()}}
    text = canonical_json(body)
    p = Path(path) if path else split_path(app)
    if p.exists():
        old = p.read_text()
        if old != text:
            raise ProtectedDataError(f"{p} is frozen (sha {sha256_text(old)[:12]}) and differs from the new split")
        return sha256_text(old)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return sha256_text(text)


def load_split(app: str, *, expect_sha: str | None = None, path: Path | None = None) -> tuple[dict, str]:
    p = Path(path) if path else split_path(app)
    text = p.read_text()
    sha = sha256_text(text)
    if expect_sha is not None and sha != expect_sha:
        raise ProtectedDataError(f"{p} sha {sha[:12]} != expected {expect_sha[:12]}")
    return json.loads(text), sha


# --------------------------------------------------------------------------- guard
def freeze_file(app: str) -> Path:
    return PREREG / f"FREEZE_downstream_{app}.json"


SHARED_SCORING_FILES = ("sentinel/models/downstream/common.py", "sentinel/models/downstream/weather.py",
                        "sentinel/models/downstream/crossfit.py", "sentinel/eval/metrics.py",
                        "sentinel/eval/bootstrap.py", "sentinel/eval/splits.py",
                        "sentinel/models/downstream/testlook.py")   # single-pass guard (2026-11-08)
# Files that define labels, clusters or features for each app (review 2026-10-24 W5): an edit to another
# app's module does not invalidate this app's freeze, but an edit to its label/cluster code does.
APP_SCORING_FILES = {
    "safeswim": ("sentinel/data/sources/downstream_chicago_beaches.py",),
    "toxinwatch": ("sentinel/data/sources/events_wa_toxic_algae.py",
                   # it3 national training rows (lead approval 2026-11-08; hashing only)
                   "sentinel/models/downstream/wqp_track.py", "sentinel/models/downstream/d2n_microcystin.py",
                   "sentinel/data/wqp_harmonize.py", "docs/preregistration/downstream_d2n_microcystin_split.json"),
    "killalert": ("sentinel/data/sources/events_iowa_fishkill.py",),
    "biointegrity": ("sentinel/models/bioassess/data.py", "sentinel/data/downstream_links.py"),
    "musselguard": ("sentinel/data/sources/downstream_dreissena.py", "sentinel/data/sources/wbd_huc12.py"),
    "riverpulse": ("sentinel/data/sources/downstream_metabolism.py", "sentinel/data/downstream_links.py",
                   "sentinel/data/sonde_splits.py"),
    # national WQP tracks (lead approval 2026-11-08; hashing only)
    "d1n_beaches": ("sentinel/models/downstream/wqp_track.py", "sentinel/data/wqp_harmonize.py"),
    "d2n_microcystin": ("sentinel/models/downstream/wqp_track.py", "sentinel/data/wqp_harmonize.py"),
}


def scoring_files(app: str) -> list[str]:
    files = {*SHARED_SCORING_FILES, *APP_SCORING_FILES.get(app, ()), f"sentinel/models/downstream/{app}.py",
             f"docs/preregistration/downstream_{app}.md"}
    pins = paths.REPO_ROOT / "docs" / "preregistration" / f"downstream_{app}_data_pins.json"
    if pins.is_file():   # input-data sha256 pins (pre-test gate S-W1, 2026-11-08) are part of the freeze hash
        files.add(str(pins.relative_to(paths.REPO_ROOT)))
    files |= {str(f.relative_to(paths.REPO_ROOT)) for f in (paths.REPO_ROOT / "scripts" / "train").glob(f"downstream_{app}*.py")}
    return sorted(files)


def scoring_code_sha256(app: str) -> str:
    root = paths.REPO_ROOT
    h = hashlib.sha256()
    for rel in scoring_files(app):
        fp = root / rel
        h.update(rel.encode())
        h.update(fp.read_bytes() if fp.is_file() else b"<missing>")
    return h.hexdigest()


def check_roles(app: str, roles: Iterable[str], *, allow_test: bool = False, training: bool = False,
                split_sha: str | None = None) -> None:
    roles = set(roles)
    unknown = roles - set(ROLES)
    if unknown:
        raise ValueError(f"unknown roles {unknown}")
    if "test" not in roles:
        return
    if training:
        raise ProtectedDataError("training code never uses test-role data")
    if not allow_test:
        raise ProtectedDataError(f"{app}: test-role data refused (phase 1 is validation-only; needs --allow-test "
                                 f"and {freeze_file(app).name})")
    if not split_sha:
        raise ProtectedDataError("allow_test requires the split sha256 in use (split_sha)")
    ff = freeze_file(app)
    if not ff.is_file():
        raise ProtectedDataError(f"{ff} does not exist")
    fr = json.loads(ff.read_text())
    if fr.get("split_sha256") != split_sha:
        raise ProtectedDataError("freeze split_sha256 does not match the split in use")
    if fr.get("scoring_code_sha256") != scoring_code_sha256(app):
        raise ProtectedDataError("scoring code changed since the freeze")


def drop_protected(df: pd.DataFrame, role_col: str = "role") -> pd.DataFrame:
    """Remove test-role rows (and rows without a role) — call right after assigning roles."""
    if role_col not in df:
        raise KeyError(f"no {role_col!r} column")
    keep = df[role_col].isin(["train", "val"])
    out = df.loc[keep].copy()
    out.attrs["n_dropped_protected"] = int((~keep).sum())
    return out


def assert_no_test(*frames: pd.DataFrame, role_col: str = "role") -> None:
    for f in frames:
        if role_col in f and (f[role_col] == "test").any():
            raise ProtectedDataError("test-role rows present in a phase-1 frame")


# --------------------------------------------------------------------------- CPU slots
def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextlib.contextmanager
def cpu_slot(name: str, n_slots: int = 1, poll_s: float = 10.0):
    """Hold one of ``n_slots`` workstream-wide heavy-job slots, **first come first served**.

    Default 1: the shared-host rule (CONTRIBUTING_V2, 2026-10-24 22:30) allows one CPU-heavy job per
    workstream. Waiters take a ticket (``queue/<ns-time>_<pid>``); only the oldest live ticket may try the
    slot locks, so a long-waiting job is never overtaken (tickets of dead processes are ignored/removed).
    """
    d = paths.ensure(paths.RUNS_ROOT / "downstream" / "locks")
    q = paths.ensure(d / "queue")
    ticket = q / f"{time.time_ns():020d}_{os.getpid()}"
    ticket.write_text(f"{name}\n")
    try:
        while True:
            live = []
            for t in sorted(q.iterdir()):
                try:
                    pid = int(t.name.split("_")[1])
                except (IndexError, ValueError):
                    continue
                if _pid_alive(pid):
                    live.append(t)
                else:
                    t.unlink(missing_ok=True)
            if live and live[0] == ticket:
                for k in range(n_slots):
                    fh = open(d / f"slot{k}.lock", "a+")
                    try:
                        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        fh.close()
                        continue
                    ticket.unlink(missing_ok=True)
                    fh.seek(0)
                    fh.truncate()
                    fh.write(f"{name} pid={os.getpid()} since={_dt.datetime.now().isoformat(timespec='seconds')}\n")
                    fh.flush()
                    try:
                        yield k
                    finally:
                        fh.seek(0)
                        fh.truncate()
                        fh.write(f"free (last: {name} pid={os.getpid()})\n")
                        fh.flush()
                        fcntl.flock(fh, fcntl.LOCK_UN)
                        fh.close()
                    return
            time.sleep(poll_s)
    finally:
        ticket.unlink(missing_ok=True)


_THREAD_LIMITER = None


def limit_threads(n: int = N_THREADS) -> None:
    """Cap BLAS/OpenMP pools at ``n`` threads (works after numpy is imported: threadpoolctl)."""
    global _THREAD_LIMITER
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(n)
    from threadpoolctl import threadpool_limits

    _THREAD_LIMITER = threadpool_limits(limits=n)


# --------------------------------------------------------------------------- run bookkeeping
def run_dir(app: str, name: str) -> Path:
    return paths.run_dir("downstream", f"{app}/{name}")


def log_config_tried(app: str, model: str, cfg: dict, status: str = "started") -> None:
    p = paths.ensure(paths.RUNS_ROOT / "downstream") / "configs_tried.jsonl"
    rec = {"time": _dt.datetime.now().isoformat(timespec="seconds"), "app": app, "model": model,
           "config_hash": config_hash(cfg), "config": cfg, "status": status}
    with open(p, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(json.dumps(rec, default=str) + "\n")
        fcntl.flock(fh, fcntl.LOCK_UN)


def n_configs_tried(app: str) -> int:
    p = paths.RUNS_ROOT / "downstream" / "configs_tried.jsonl"
    if not p.exists():
        return 0
    return sum(1 for line in p.read_text().splitlines() if line.strip() and json.loads(line).get("app") == app)


def log_metric(*, app: str, experiment: str, model: str, metric: str, value: float, split: str,
               n: int | None = None, ci: dict | None = None, ci_method: str | None = None,
               target: str | None = None, is_baseline: bool = False, config: dict | None = None,
               data_manifests: Iterable[str] = (), artifacts: Iterable[str] = (), notes: str = "",
               extra: dict | None = None) -> dict:
    """Registry write for downstream results; validation/development splits only in phase 1."""
    if "test" in split.lower():
        raise ProtectedDataError(f"refusing to register a test-split result ({split}) in phase 1")
    if value is None or not np.isfinite(value):
        raise ValueError(f"non-finite {metric} for {model}")
    return log_result(workstream=WORKSTREAM, experiment=f"{app}/{experiment}", model=model, metric=metric,
                      value=float(value), split=split, n=n,
                      ci_low=None if ci is None else ci.get("ci_low"),
                      ci_high=None if ci is None else ci.get("ci_high"),
                      ci_method=ci_method, target=target, is_baseline=is_baseline, config=config,
                      data_manifests=list(data_manifests), artifacts=list(artifacts), notes=notes,
                      extra=extra or {})


# --------------------------------------------------------------------------- evaluation
def _finite(*arrs):
    m = np.ones(len(arrs[0]), bool)
    for a in arrs:
        m &= np.isfinite(np.asarray(a, float))
    return m


def fast_auroc(y, s) -> float:
    """Tie-aware AUROC (Mann–Whitney), numerically equal to sklearn ``roc_auc_score``; ~50x faster in
    bootstrap loops. NaN if only one class."""
    from scipy.stats import rankdata

    y = np.asarray(y, float)
    s = np.asarray(s, float)
    n1 = y.sum()
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fast_auprc(y, s) -> float:
    """Average precision equal to sklearn ``average_precision_score`` (step-wise, ties grouped)."""
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    npos = y.sum()
    if npos == 0:
        return np.nan
    o = np.argsort(-s, kind="mergesort")
    ys, ss = y[o], s[o]
    last = np.r_[np.flatnonzero(np.diff(ss) != 0), ys.size - 1]   # end index of each distinct threshold
    tp = np.cumsum(ys)[last]
    fp = (last + 1) - tp
    prec = tp / (tp + fp)
    rec = tp / npos
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def binary_scores(y, s, groups, *, n_boot: int = N_BOOT, seed: int = 0, prob: bool = False) -> dict:
    """AUROC/AUPRC (+ Brier when ``s`` is a probability) with cluster-bootstrap CIs over ``groups``."""
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    g = np.asarray(groups)
    m = _finite(y, s)
    y, s, g = y[m], s[m], g[m]
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "prevalence": float(y.mean()) if len(y) else np.nan,
           "n_groups": int(len(np.unique(g)))}
    if len(np.unique(y)) < 2:
        out.update({"auroc": np.nan, "auprc": np.nan})
        return out

    def safe(f):
        def w(a, b):
            return f(a, b) if len(np.unique(a)) == 2 else np.nan
        return w

    for name, f in (("auroc", fast_auroc), ("auprc", fast_auprc)) + ((("brier", M.brier),) if prob else ()):
        ci = B.bootstrap_ci(safe(f), y, s, groups=g, n_boot=n_boot, seed=seed)
        out[name] = ci["point"]
        out[f"{name}_ci"] = [ci["ci_low"], ci["ci_high"]]
        out[f"{name}_n_boot_valid"] = ci["n_boot_valid"]
    return out


def regression_scores(y, p, groups, *, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    g = np.asarray(groups)
    m = _finite(y, p)
    y, p, g = y[m], p[m], g[m]
    out = {"n": int(len(y)), "n_groups": int(len(np.unique(g)))}
    for name, f in (("r2", M.nse), ("rmse", M.rmse), ("spearman", M.spearman)):
        ci = B.bootstrap_ci(f, y, p, groups=g, n_boot=n_boot, seed=seed)
        out[name] = ci["point"]
        out[f"{name}_ci"] = [ci["ci_low"], ci["ci_high"]]
    return out


def paired(metric: Callable, y, s_model, s_base, groups, *, higher_is_better: bool = True,
           n_boot: int = N_BOOT, seed: int = 0) -> dict:
    y, a, b = (np.asarray(v, float) for v in (y, s_model, s_base))
    g = np.asarray(groups)
    m = _finite(y, a, b)

    def safe(yy, pp):
        return metric(yy, pp) if len(np.unique(yy)) > 1 else np.nan

    return B.paired_bootstrap(safe, y[m], a[m], b[m], groups=g[m], n_boot=n_boot, seed=seed,
                              higher_is_better=higher_is_better)


def accuracy_at(y, p, thr: float = 0.5) -> float:
    y = np.asarray(y, float)
    return float(np.mean((np.asarray(p, float) >= thr) == (y >= 0.5)))


def write_json(path: Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return path
