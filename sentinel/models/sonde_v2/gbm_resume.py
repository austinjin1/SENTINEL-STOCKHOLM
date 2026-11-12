# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Booster-level resume, GPU guard and GPU-hour account of the GBM references (GBM-NWP and PF-GBM).

Lead ruling ``docs/preregistration/sonde_arch_v2_gbm_lead_ruling_20261110.md`` (cited as G-n):

* G-3 — the 2 h limit holds per process. Each booster is fitted whole inside one process and saved to its own file
  as soon as it is finished, with its stopping round and wall time (:class:`UnitStore`). A booster that has a record
  is never fitted again. A process starts a new booster only if its elapsed wall time plus 1.5 times the longest
  booster time it has seen so far (40 minutes while it has seen none) is at most 115 minutes (:class:`ProcessClock`;
  "seen" = fitted by this process or read by it from a booster record); otherwise it ends with exit code 75 and the
  same command continues the unit. An unfinished booster is fitted again from its start with the same seed. The
  record of a unit lists every process that worked on it. A booster of another configuration of the same arm,
  seed, fold, variable and horizon is used instead of a new fit only when its key is identical (fit rows,
  early-stopping rows, targets, objective, quantile level, seed and every setting; :func:`booster_key`); a record
  whose key differs is a hard error, never a silent refit. A booster that took more than 2 h stops the work
  (:class:`BoosterTooSlow`).
* The saved file is the model of record: the booster is read back from its file in the process that fitted it, and
  that process and every later one predict from the file (same trees: the file holds the best iteration only).
* G-6 — one GPU per process through the lock helper, at most 6 CPU threads, never a GPU that holds another user's
  process (:func:`foreign_on`, :func:`placement_check`); no fallback to the CPU anywhere in this module.
* G-7 — every GBM process writes its GPU wall time to the registry (:meth:`GBMProcess.finish`); the running total
  per stage is :func:`gpu_hours_total`; the CV projection from the step-1 stopping rounds is
  :func:`cv_projection`; :func:`cv_gate` refuses a CV GBM process without the projection on disk or when the
  projection, or the running CV total plus the projected remainder, is above the limit (300 GPU-h unless a
  lead-written registry row rules another one).

Two additions that only protect the 2 h limit and change no model: a booster still running when its process is
119 minutes old is abandoned (not saved; fitted again from its start by the next process; twice for the same
booster = :class:`BoosterTooSlow`), and the forecast step after the fits starts only while the process is at most
105 minutes old (otherwise exit 75; the next process reads the boosters from their files and predicts).
"""

from __future__ import annotations

import datetime as _dt
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from . import declared as D

_R = D.GBM_SETTINGS["resume"]
PROCESS_LIMIT_S = 60.0 * _R["process_limit_min"]              # 115 min
FIRST_ALLOWANCE_S = 60.0 * _R["first_booster_allowance_min"]  # 40 min while no booster time has been seen
FACTOR = float(_R["factor"])                                  # 1.5 x the longest booster time seen
BOOSTER_LIMIT_S = 3600.0 * _R["booster_limit_h"]              # a booster above 2 h: stop and report
ABORT_AT_S = 60.0 * _R["abort_min"]                           # process age at which a running booster is abandoned
PREDICT_LATEST_S = 60.0 * _R["predict_latest_min"]            # latest process age for the start of a forecast step
EXIT_RESUME = 75
WORKSTREAM = "M1-sonde"
GPU_EXPERIMENT = "sonde_arch_v2/gbm_gpu_hours"
BUDGET_EXPERIMENT = "sonde_arch_v2/gbm_cv_budget"
LEAD_BUDGET_EXPERIMENT = "sonde/lead_gbm_budget_ruling"


class ProcessBudgetReached(Exception):
    """G-3: this process starts no further booster (or forecast step); the same command continues (exit 75)."""


class BoosterTooSlow(RuntimeError):
    """G-3: a single booster took more than 2 h (or was abandoned twice at the process limit): stop and report."""


class ResumeError(RuntimeError):
    """A saved booster does not belong to the rows, seed or settings of this process, or its file is damaged."""


class UnitBusy(RuntimeError):
    """Another process holds the boosters of this arm, seed, fold and variable."""


class GPUError(RuntimeError):
    """G-6: the fit is not on the locked GPU, or the GPU holds another user's process."""


class BudgetError(RuntimeError):
    """G-7: no CV projection on disk, or the CV GPU-hours are above the limit without a lead ruling."""


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------- clock (G-3)

class ProcessClock:
    """Wall clock of one process and the start rule of G-3."""

    def __init__(self, t0: float | None = None, now=time.time):
        self._now = now
        self.t0 = now() if t0 is None else float(t0)
        self.longest: float | None = None

    def elapsed(self) -> float:
        return self._now() - self.t0

    def seen(self, wall_s: float) -> None:
        self.longest = float(wall_s) if self.longest is None else max(self.longest, float(wall_s))

    def need(self) -> float:
        return FIRST_ALLOWANCE_S if self.longest is None else FACTOR * self.longest

    def may_start(self) -> bool:
        return self.elapsed() + self.need() <= PROCESS_LIMIT_S

    def may_predict(self) -> bool:
        return self.elapsed() <= PREDICT_LATEST_S


class _Rounds:
    """LightGBM callback: counts the boosting rounds that ran and abandons the fit at the process limit."""

    order = 5                      # before early stopping (30): the round that triggers the stop is counted
    before_iteration = False

    def __init__(self, clock: ProcessClock):
        self.clock, self.n = clock, 0

    def __call__(self, env) -> None:
        self.n += 1
        if self.clock.elapsed() > ABORT_AT_S:
            raise _Abandoned(f"process is {self.clock.elapsed() / 60:.1f} min old after round {self.n}")


class _Abandoned(Exception):
    pass


# ----------------------------------------------------------------------------------------------- hashes, keys

def array_sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(f"{a.dtype}|{a.shape}|".encode())
        h.update(a)
    return h.hexdigest()


def frame_sha(df) -> str:
    """sha256 of a feature table: column names, dtypes and values in column order (categorical: codes + levels)."""
    import pandas as pd

    h = hashlib.sha256()
    for c in df.columns:
        col = df[c]
        if isinstance(col.dtype, pd.CategoricalDtype):
            a, tag = col.cat.codes.to_numpy(), f"{c}|categorical|{list(col.cat.categories)}|"
        else:
            a = col.to_numpy()
            tag = f"{c}|{a.dtype}|"
        h.update(tag.encode())
        h.update(np.ascontiguousarray(a))
    return h.hexdigest()


def booster_key(params: dict, *, max_rounds: int, early_stopping_rounds: int, x_fit: str, y_fit: str, x_es: str,
                y_es: str, columns: list) -> tuple[str, dict]:
    """(key, parts): everything that defines one booster. Equal keys = the same fit rows, early-stopping rows,
    targets, objective, quantile level, seed and settings (G-3)."""
    import lightgbm

    parts = {"params": {k: params[k] for k in sorted(params)}, "max_rounds": int(max_rounds),
             "early_stopping_rounds": int(early_stopping_rounds), "x_fit_sha256": x_fit, "y_fit_sha256": y_fit,
             "x_es_sha256": x_es, "y_es_sha256": y_es, "columns_sha256": _sha(list(columns)),
             "lightgbm": lightgbm.__version__}
    return _sha(parts), parts


# ----------------------------------------------------------------------------------------------- GPU guard (G-6)

def _smi(query: str) -> list[str]:
    out = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"], capture_output=True, text=True,
                         timeout=120, check=True).stdout
    return [ln for ln in out.strip().splitlines() if ln.strip()]


def gpu_apps() -> list[dict]:
    """Compute processes on every GPU: physical index, pid, uid (None if the pid is not visible), used MiB."""
    index = {}
    for ln in _smi("--query-gpu=index,uuid"):
        i, u = (x.strip() for x in ln.split(","))
        index[u] = int(i)
    out = []
    for ln in _smi("--query-compute-apps=gpu_uuid,pid,used_memory"):
        u, pid, mem = (x.strip() for x in ln.split(","))
        try:
            uid = os.stat(f"/proc/{int(pid)}").st_uid
        except OSError:
            uid = None
        out.append({"gpu": index.get(u), "pid": int(pid), "uid": uid, "mib": mem})
    return out


def foreign_on(gpu: int) -> list[dict]:
    """Processes on GPU ``gpu`` that are not this process and not provably of this user."""
    me = os.getuid()
    return [a for a in gpu_apps() if a["gpu"] == gpu and a["pid"] != os.getpid() and a["uid"] != me]


def placement_check(gpu: int, device_settings: dict, *, log=print) -> dict:
    """Technical check before the first booster of a process (synthetic numbers only, no project data, no timing):
    a short LightGBM fit with the declared device settings must show this pid on the locked GPU and on no other.
    nvidia-smi is polled DURING the fit (LightGBM releases its OpenCL context when a fit ends)."""
    import lightgbm as lgb

    bad = foreign_on(gpu)
    if bad:
        raise GPUError(f"GPU {gpu} holds a process of another user: {bad}")
    seen, stop = set(), threading.Event()

    def poll():
        while not stop.is_set():
            try:
                seen.update(a["gpu"] for a in gpu_apps() if a["pid"] == os.getpid())
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            stop.wait(0.3)

    r = np.random.default_rng(0)
    X = r.normal(size=(200_000, 40)).astype(np.float32)
    y = X[:, 0] + r.normal(size=200_000)
    p = {"objective": "quantile", "alpha": 0.5, "learning_rate": 0.05, "num_leaves": 63, "num_threads": 2, "seed": 0,
         "verbose": -1, **device_settings, "gpu_device_id": 0}
    th = threading.Thread(target=poll, daemon=True)
    th.start()
    try:
        for _ in range(3):
            lgb.train(p, lgb.Dataset(X, y), num_boost_round=150)
            if seen:
                break
    finally:
        stop.set()
        th.join()
    if seen != {gpu}:
        raise GPUError(f"the LightGBM GPU fit of pid {os.getpid()} was seen on GPU(s) {sorted(seen)}, locked GPU "
                       f"{gpu} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    log(f"[gbm] placement check passed: pid {os.getpid()} fits on GPU {gpu} only")
    return {"gpu": gpu, "seen": sorted(seen)}


# ----------------------------------------------------------------------------------------------- process (G-7a)

class GBMProcess:
    """One GBM process: its GPU, its clock and its account. :meth:`finish` writes the GPU-hour registry row."""

    def __init__(self, stage: str, tag: str, cfg, *, t0: float | None = None, clock: ProcessClock | None = None,
                 gpu: int = -1, log=print):
        self.stage, self.tag, self.cfg, self.log = stage, tag, cfg, log
        self.clock = clock or ProcessClock(t0)
        self.gpu = int(gpu)
        self.t_gpu = time.time()
        self.pid, self.utc_start = os.getpid(), _utc()
        self.fitted: list = []        # [unit, booster, wall_s]
        self.loaded: list = []        # [unit, booster]
        self.end = "error"
        self.closed = False

    @classmethod
    def start(cls, stage: str, tag: str, cfg, *, t0: float | None = None, prefer: int = 7, log=print):
        """Lock one GPU through the lock helper (device ``gpu``), 6 CPU threads, placement check."""
        gpu = -1
        if cfg.device == "gpu":
            from sentinel.models.sonde import gpu_budget, runner

            gpu_budget.widen(10000)
            gpu = int(runner.setup_gpu(f"sonde_v2_{tag}", prefer=prefer))
            runner.limit_cpu_threads(cfg.num_threads)
            log(f"[gbm] GPU {gpu} locked (LightGBM {D.GBM_SETTINGS['device_settings']}), "
                f"{cfg.num_threads} CPU threads")
            if gpu >= 0:
                placement_check(gpu, D.GBM_SETTINGS["device_settings"], log=log)
        elif cfg.device != "cpu":
            raise ValueError(cfg.device)
        return cls(stage, tag, cfg, t0=t0, gpu=gpu, log=log)

    def before_booster(self, what: str) -> None:
        """G-3 start rule and G-6 guard; raises :class:`ProcessBudgetReached` if this process must not start it."""
        c = self.clock
        if not c.may_start():
            raise ProcessBudgetReached(f"{what}: process is {c.elapsed() / 60:.1f} min old, a new booster needs "
                                       f"{c.need() / 60:.1f} min of room, limit {PROCESS_LIMIT_S / 60:.0f} min")
        if self.gpu >= 0:
            bad = foreign_on(self.gpu)
            if bad:
                raise ProcessBudgetReached(f"{what}: GPU {self.gpu} now holds a process of another user ({bad}); "
                                           "no further booster on it")

    def require_room(self, what: str) -> None:
        if not self.clock.may_predict():
            raise ProcessBudgetReached(f"{what}: process is {self.clock.elapsed() / 60:.1f} min old (latest start "
                                       f"{PREDICT_LATEST_S / 60:.0f} min)")

    def summary(self) -> dict:
        return {"pid": self.pid, "host": socket.gethostname(), "utc_start": self.utc_start, "gpu": self.gpu,
                "device": self.cfg.device,
                "device_settings": D.GBM_SETTINGS["device_settings"] if self.cfg.device == "gpu" else None,
                "num_threads": int(self.cfg.num_threads), "n_fitted": len(self.fitted), "n_loaded": len(self.loaded),
                "booster_fit_s": round(float(sum(w for *_, w in self.fitted)), 1), "end": self.end}

    def finish(self) -> dict | None:
        """Registry row with the GPU wall time of this process (lock taken → now). Compute account only."""
        if self.closed:
            return None
        self.closed = True
        from sentinel.core.registry import log_result

        hours = (time.time() - self.t_gpu) / 3600.0
        return log_result(
            workstream=WORKSTREAM, experiment=GPU_EXPERIMENT, model=self.tag, metric="gpu_hours", value=hours,
            split=self.stage, n=len(self.fitted),
            notes="GPU wall time of one GBM process, from the GPU lock to the end of the process (GBM ruling G-7a); "
                  "compute account only, no score",
            extra={**self.summary(), "utc_end": _utc(), "process_wall_s": round(self.clock.elapsed(), 1),
                   "fitted": [[u, b, round(w, 1)] for u, b, w in self.fitted]})


def gpu_hours_total(stage: str, registry_path=None) -> float:
    """Running total of GBM GPU-hours of one stage ("step1" | "cv") from the registry rows of every process."""
    from sentinel.core.registry import read_registry

    return float(sum(r["value"] for r in read_registry(registry_path)
                     if r.get("experiment") == GPU_EXPERIMENT and r.get("split") == stage))


# ----------------------------------------------------------------------------------------------- store (G-3)

def unit_lock(directory: Path):
    """Exclusive lock on the boosters of one arm, seed, fold and variable (two processes never fit the same one)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fh = open(directory / ".lock", "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise UnitBusy(f"another process works on {directory}") from None
    return fh


def _file_sha(f: Path) -> str:
    return hashlib.sha256(f.read_bytes()).hexdigest()


class UnitStore:
    """Boosters of one unit (arm, seed, fold, variable, horizon): ``<name>.txt`` (LightGBM model, best iteration
    only) + ``<name>.json`` (key, stopping round, rounds run, wall time, process, device) per booster, and
    ``unit__<config>.json`` with the rows of the unit and every process that worked on it for that configuration."""

    def __init__(self, directory: Path, *, config: str, process: GBMProcess, info: dict):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.config, self.proc = config, process
        self.unit = str(Path(*self.dir.parts[-2:]))
        self.rec_path = self.dir / f"unit__{config}.json"
        self.rec = json.loads(self.rec_path.read_text()) if self.rec_path.exists() else {"processes": []}
        old = {k: self.rec.get(k) for k in info if k in self.rec}
        if old and old != {k: info[k] for k in old}:
            diff = sorted(k for k in old if old[k] != info[k])
            raise ResumeError(f"{self.rec_path}: the unit of this process differs from the recorded one in {diff}")
        self.entry = {"pid": process.pid, "host": socket.gethostname(), "utc_start": _utc(), "gpu": process.gpu,
                      "device": process.cfg.device, "seed": int(process.cfg.seed), "fitted": [], "loaded": [],
                      "end": "running"}
        self.rec.update({"config": config, **info})
        self.rec["processes"].append(self.entry)
        self._write()

    def _write(self) -> None:
        _write_json(self.rec_path, self.rec)

    def close(self, end: str, **more) -> None:
        """End of this process's work on the unit (the first reason given stands)."""
        if self.entry["end"] != "running":
            return
        self.entry.update({"end": end, "utc_end": _utc(), **more})
        self._write()

    def has(self, name: str) -> bool:
        return (self.dir / f"{name}.json").exists()

    def get(self, name: str, key: str, parts: dict, fit, *, check=None):
        """(booster read from its file, record). ``fit(callback)`` → fitted ``lgb.Booster``; called only when the
        booster has no record. ``check``: rows on which the saved file must reproduce the fitted booster."""
        import lightgbm as lgb

        mf, jf, af = self.dir / f"{name}.txt", self.dir / f"{name}.json", self.dir / f"{name}.abandoned.json"
        what = f"{self.unit}/{name}"
        if jf.exists():
            meta = json.loads(jf.read_text())
            if meta["key"] != key:
                diff = sorted(k for k in parts if meta["key_parts"].get(k) != parts[k])
                raise ResumeError(f"{jf}: the saved booster was fitted with other rows, seed or settings "
                                  f"(differs in {diff}); it is not refitted and not used")
            if not mf.exists() or _file_sha(mf) != meta["model_sha256"]:
                raise ResumeError(f"{mf}: the model file is not the recorded one")
            self.proc.clock.seen(meta["wall_s"])
            self.proc.loaded.append([self.unit, name])
            self.entry["loaded"].append(name)
            self._write()
            return lgb.Booster(model_file=str(mf)), meta
        gone = json.loads(af.read_text()) if af.exists() else []
        if len(gone) >= 2:
            self.close("booster_abandoned_twice", booster=name)
            raise BoosterTooSlow(f"{what} was abandoned at the process limit twice ({gone}); stop and report")
        try:
            self.proc.before_booster(what)
        except ProcessBudgetReached:
            self.close("budget", next_booster=name)
            raise
        counter = _Rounds(self.proc.clock)
        t, at = time.time(), self.proc.clock.elapsed()
        try:
            booster = fit(counter)
        except _Abandoned as e:
            gone.append({"pid": self.proc.pid, "utc": _utc(), "started_at_process_s": round(at, 1),
                         "ran_s": round(time.time() - t, 1), "rounds_run": counter.n})
            _write_json(af, gone)
            self.close("abandoned_at_process_limit", booster=name)
            raise ProcessBudgetReached(f"{what}: {e}; the booster is not saved and is fitted again from its start "
                                       "by the next process") from None
        wall = time.time() - t
        best = int(booster.best_iteration or booster.current_iteration())
        tmp = mf.with_name(mf.name + f".tmp{os.getpid()}")
        booster.save_model(str(tmp), num_iteration=best)
        os.replace(tmp, mf)
        loaded = lgb.Booster(model_file=str(mf))
        diff = None
        if check is not None and len(check):
            a, b = booster.predict(check, num_iteration=best), loaded.predict(check)
            diff = float(np.max(np.abs(a - b)))
            if not diff <= 1e-6 * (1.0 + float(np.max(np.abs(a)))):
                raise ResumeError(f"{mf}: the saved file does not reproduce the fitted booster (max diff {diff})")
        meta = {"name": name, "unit": self.unit, "key": key, "key_parts": parts, "objective": parts["params"].get(
                    "objective"), "alpha": parts["params"].get("alpha"), "seed": parts["params"].get("seed"),
                "best_iteration": best, "rounds_run": int(counter.n), "n_trees": int(loaded.num_trees()),
                "wall_s": round(wall, 2), "started_at_process_s": round(at, 1), "pid": self.proc.pid,
                "host": socket.gethostname(), "utc": _utc(), "device": self.proc.cfg.device, "gpu": self.proc.gpu,
                "device_settings": D.GBM_SETTINGS["device_settings"] if self.proc.cfg.device == "gpu" else None,
                "num_threads": int(self.proc.cfg.num_threads), "fitted_for_config": self.config,
                "n_fit": self.rec.get("n_fit"), "n_es": self.rec.get("n_es"), "n_columns": self.rec.get("n_columns"),
                "file_vs_fitted_max_abs_diff": diff, "model_sha256": _file_sha(mf)}
        _write_json(jf, meta)
        self.proc.clock.seen(wall)
        self.proc.fitted.append([self.unit, name, wall])
        self.entry["fitted"].append(name)
        self._write()
        if wall > BOOSTER_LIMIT_S:
            self.close("booster_over_2h", booster=name)
            raise BoosterTooSlow(f"{what} took {wall / 3600:.2f} h (> 2 h); it is saved; stop and report")
        return loaded, meta


def read_processes(root: Path) -> list[dict]:
    """Every process that worked on the units below ``root``, per unit record (GBM ruling G-3)."""
    out = []
    for f in sorted(Path(root).rglob("unit__*.json")):
        r = json.loads(f.read_text())
        out.append({"unit": str(Path(*f.parts[-3:-1])), "config": r.get("config"), "skipped": r.get("skipped"),
                    "processes": r.get("processes", [])})
    return out


def read_metas(root: Path) -> list[dict]:
    """Every booster record below ``root`` (not the unit records, not the abandon notes)."""
    out = []
    for f in sorted(Path(root).rglob("*.json")):
        if f.name.startswith("unit__") or f.name.endswith(".abandoned.json") or ".tmp" in f.name:
            continue
        out.append(json.loads(f.read_text()))
    return out


def booster_brief(meta: dict) -> dict:
    """What a parity record keeps of a booster: stopping round, rounds run, wall time, process, device, GPU."""
    return {k: meta.get(k) for k in ("best_iteration", "rounds_run", "wall_s", "pid", "utc", "device", "gpu",
                                     "device_settings", "fitted_for_config", "model_sha256")}


# ----------------------------------------------------------------------------------------------- CV budget (G-7)

def booster_names(overrides: dict) -> list[str]:
    """File names of the boosters of one configuration (per variable and horizon)."""
    from sentinel.models.baselines.sonde_baselines import QUANTILES

    from .gbm_nwp import QLEVELS, qname

    if overrides.get("quantile_mode", "quantile") == "location_scale":
        return ["ls_loc", "ls_scale"]
    return [qname(a) for a in (QUANTILES if overrides.get("quantile_levels", 19) == 19 else QLEVELS)]


def cv_projection(step1_metas: list[dict], rows: dict, configs: dict, *, seed: int) -> dict:
    """G-7(c): CV GPU-hours projected from the step-1 boosters. Per booster of a unit with ``n_fit`` fit rows:
    (mean rounds run of the step-1 boosters) × (mean of wall seconds per round and fit row of the step-1 boosters)
    × ``n_fit``. ``rows``: {"fold|variable|h": {"n_fit", "n_es", "skipped"}}; ``configs``: {arm: {config:
    overrides}}. A booster shared by two configurations of an arm counts once in the total."""
    ok = [m for m in step1_metas if m.get("rounds_run") and m.get("n_fit")]
    if not ok:
        raise BudgetError("no step-1 booster record with rounds and rows: step 1 has not fitted its GBM")
    rounds = float(np.mean([m["rounds_run"] for m in ok]))
    s_round_row = float(np.mean([m["wall_s"] / m["rounds_run"] / m["n_fit"] for m in ok]))
    units, per, total = [], {}, 0.0
    for arm, cfgs in configs.items():
        names = {c: booster_names(o) for c, o in cfgs.items()}
        distinct = sorted(set(n for v in names.values() for n in v))
        for key, r in rows.items():
            if r.get("skipped"):
                continue
            k, v, h = key.split("|")
            s = rounds * s_round_row * int(r["n_fit"])
            units.append({"arm": arm, "seed": int(seed), "fold": int(k), "variable": v, "h": int(h),
                          "boosters": distinct, "s_per_booster": s})
            total += s * len(distinct)
            for c, nm in names.items():
                per[f"{arm}|{c}"] = per.get(f"{arm}|{c}", 0.0) + s * len(nm)
    return {"formula": "per booster: mean rounds run (step 1) x mean wall seconds per round and fit row (step 1) x "
                       "fit rows of the CV unit; a booster shared by two configurations of an arm counts once in "
                       "the total; no scaling for the number of feature columns",
            "step1": {"n_boosters": len(ok), "mean_rounds_run": rounds, "mean_best_iteration": float(np.mean(
                [m["best_iteration"] for m in ok])), "mean_s_per_round": float(np.mean(
                    [m["wall_s"] / m["rounds_run"] for m in ok])), "mean_s_per_round_and_row": s_round_row,
                "gpu_hours_total_of_boosters": float(sum(m["wall_s"] for m in ok)) / 3600.0},
            "gpu_hours_per_arm_and_config_alone": {k: v / 3600.0 for k, v in sorted(per.items())},
            "total_gpu_h": total / 3600.0, "limit_gpu_h": float(D.GBM_SETTINGS["cv_gpu_hours_limit"]),
            "units": units}


def cv_remainder(budget: dict, store_root: Path) -> float:
    """Projected GPU-hours of the CV boosters that have no record on disk yet."""
    s = 0.0
    for u in budget["units"]:
        d = Path(store_root) / f"{u['arm']}_s{u['seed']}" / f"fold{u['fold']}" / u["variable"] / f"h{u['h']}"
        s += u["s_per_booster"] * sum(1 for n in u["boosters"] if not (d / f"{n}.json").exists())
    return s / 3600.0


def lead_limit(registry_path=None) -> tuple[float, str | None]:
    """(limit in GPU-h, registry row id): 300 by G-7(d) unless the latest lead-written row rules another limit."""
    from sentinel.core.registry import read_registry

    limit, row = float(D.GBM_SETTINGS["cv_gpu_hours_limit"]), None
    for r in read_registry(registry_path):
        e = r.get("extra") or {}
        if (r.get("workstream") == WORKSTREAM and r.get("experiment") == LEAD_BUDGET_EXPERIMENT
                and e.get("written_by") == "lead" and e.get("gpu_hours_limit") is not None):
            limit, row = float(e["gpu_hours_limit"]), r.get("id")
    return limit, row


def cv_gate(budget_file: Path, store_root: Path, *, registry_path=None) -> dict:
    """G-7(c, d): called before every CV GBM process. Raises :class:`BudgetError`; returns the account."""
    budget_file = Path(budget_file)
    if not budget_file.exists():
        raise BudgetError("GBM ruling G-7(c): no CV projection on disk; run `sonde_v2_cv.py --phase gbm_budget` "
                          "after step 1 and send it to the lead before the first CV GBM process")
    b = json.loads(budget_file.read_text())
    limit, row = lead_limit(registry_path)
    spent, rem = gpu_hours_total("cv", registry_path), cv_remainder(b, store_root)
    acc = {"projection_gpu_h": b["total_gpu_h"], "spent_cv_gpu_h": spent, "projected_remainder_gpu_h": rem,
           "limit_gpu_h": limit, "lead_row": row}
    if b["total_gpu_h"] > limit or spent + rem > limit:
        raise BudgetError(f"GBM ruling G-7(d): CV GBM GPU-hours above the limit ({acc}); no further CV GBM process "
                          "starts until the lead has ruled")
    return acc
