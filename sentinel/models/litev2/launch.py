# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Launch rules of M7 architecture round v2 (lead order of 2026-11-10 20:20 EST, items A1, A2, A7).

* **Test-S KD mask on the re-run path (A1).** The re-runs of configurations 1–6 and of both no-KD twins use the
  frozen ``lite_train.py``. :func:`activate_rerun_kd_mask` wraps the frozen KD loader for this process: every
  teacher value of a chip whose pairing lies on an M2 Test-S site-date is removed and its KD weight is 0. This is the
  third and last exception to "unchanged settings" (after the 4 reserved-site chips and the label table).
* **Warm start (A2).** A v2 network starts, seed for seed, from the corrected-label re-run of configuration 3; a
  no-KD twin starts from the re-run of the no-KD twin of configuration 3; the S11 embeddings come from the same
  re-run checkpoints. :func:`init_checkpoint` is the only way a v2 job gets a leader checkpoint: it accepts a
  ``..._national2b`` re-run directory only, has no fallback, and checks the file against the sha256 registered by
  :func:`register_init_checkpoints` before the first slot job.
* **GPU-hours (A7).** :func:`gpu_job` logs every GPU job of the closed run list of the addendum's compute table in
  an append-only ledger and refuses to start a job if the logged total plus the job's own budget would pass
  :data:`CEILING_H`. A job is stopped at its budget. A job without an end record counts at its full budget.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import signal
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ...core import paths
from . import tests_mask as TM

WORKSTREAM = "M7-lite"
SEEDS = (0, 1, 2)

# ------------------------------------------------------------------------------------------------ A1: KD mask
MASK_RUN = "tests_mask_national2b"


class LaunchError(RuntimeError):
    """A launch rule of the round is broken (wrong or missing warm-start checkpoint, ceiling, run list)."""


def mask_flags() -> pd.DataFrame:
    """The Test-S flag table of national-2b, checked against the hash in its meta file."""
    d = paths.RUNS_ROOT / "lite" / MASK_RUN
    meta = json.loads((d / "meta.json").read_text())
    if hashlib.sha256((d / "flags.parquet").read_bytes()).hexdigest() != meta["flags_sha256"]:
        raise TM.MaskError("flags.parquet differs from the hash in its meta file")
    return pd.read_parquet(d / "flags.parquet")


def chip_kd_masked(chip_ids, flags: pd.DataFrame) -> np.ndarray:
    """True per chip whose teacher pairing is on an M2 Test-S site-date. A chip without a flag is refused."""
    f = flags.drop_duplicates("naip_chip_id").set_index("naip_chip_id")["kd_masked"]
    ids = pd.Series(np.asarray(chip_ids)).astype(str)
    if not ids.isin(f.index).all():
        raise TM.MaskError(f"{int((~ids.isin(f.index)).sum())} chips have no Test-S flag")
    return f.reindex(ids).to_numpy(bool)


def mask_kd_tensors(kd: dict, masked: np.ndarray) -> tuple[dict, dict]:
    """Copy of the frozen loader's KD tensors with weight 0 and no teacher value on the masked chips."""
    m = np.asarray(masked, bool)
    if len(m) != len(kd["t_w"]):
        raise TM.MaskError("the mask does not match the KD tensors")
    out = dict(kd)
    t_w = np.array(kd["t_w"], dtype=np.float32, copy=True)
    t_mu = np.array(kd["t_mu"], dtype=np.float32, copy=True)
    t_var = np.array(kd["t_var"], dtype=np.float32, copy=True)
    with_teacher = t_w.reshape(len(m), -1).sum(1) > 0
    t_w[m], t_mu[m], t_var[m] = 0.0, np.nan, np.nan
    out.update(t_w=t_w, t_mu=t_mu, t_var=t_var)
    if (t_w[m] > 0).any() or np.isfinite(t_mu[m]).any():
        raise TM.MaskError("a teacher value survived the Test-S mask")
    return out, {"n_chips": int(len(m)), "n_kd_masked": int(m.sum()),
                 "n_chips_with_teacher_before_mask": int(with_teacher.sum()),
                 "n_masked_with_teacher": int((with_teacher & m).sum()),
                 "n_chips_with_teacher": int((t_w.reshape(len(m), -1).sum(1) > 0).sum())}


def activate_rerun_kd_mask(flags: pd.DataFrame | None = None, log=print) -> None:
    """Wrap ``lite.distill.load_kd_tensors`` for this process (the re-run path): the frozen loader runs unchanged,
    then the Test-S pairings are masked. Idempotent."""
    from ..lite import distill as KD

    orig = KD.load_kd_tensors
    if getattr(orig, "tests_masked", False):
        return

    def load_kd_tensors(lab, targets_path, **kw):
        kd = orig(lab, targets_path, **kw)
        fl = mask_flags() if flags is None else flags
        out, info = mask_kd_tensors(kd, chip_kd_masked(lab["naip_chip_id"], fl))
        log(f"Test-S KD mask (lead ruling A1): {json.dumps(info)}")
        return out

    load_kd_tensors.tests_masked = True
    KD.load_kd_tensors = load_kd_tensors


# ------------------------------------------------------------------------------------------------ A2: warm start
INIT_NAME = {True: "mnv4s_raw_crop256_kd03_national2b", False: "mnv4s_raw_crop256_nokd_national2b"}   # KD → name
INIT_EXPERIMENT = "lite/v2_init_checkpoints"
INIT_MODEL = "litev2_init_checkpoints"
_RUN_RE = "natstudent_{name}_[0-9]{{8}}-[0-9]{{6}}"


def _runs_root(runs_root=None) -> Path:
    return Path(runs_root) if runs_root is not None else paths.RUNS_ROOT / "lite"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def init_run_dir(kd: bool, runs_root=None) -> Path:
    """The run directory of the corrected-label re-run a v2 network starts from (configuration 3 for ``kd``, its
    no-KD twin otherwise). Exactly one directory with all three seed checkpoints must exist; anything else is a
    hard error — there is no fallback to an as-built (``..._national2``) run."""
    name = INIT_NAME[bool(kd)]
    pat = re.compile(_RUN_RE.format(name=re.escape(name)))
    root = _runs_root(runs_root)
    dirs = sorted(d for d in root.glob(f"natstudent_{name}_*") if d.is_dir() and pat.fullmatch(d.name))
    full = [d for d in dirs if all((d / f"model_seed{s}.pt").is_file() for s in SEEDS)]
    if len(full) != 1:
        raise LaunchError(f"warm start needs exactly one finished re-run natstudent_{name}_<time> with the "
                          f"checkpoints of seeds {SEEDS}; found {len(full)} (of {len(dirs)} directories). "
                          "No as-built checkpoint is used instead.")
    return full[0]


def _init_key(d: Path, seed: int) -> str:
    return f"{d.name}/model_seed{int(seed)}.pt"


def registered_init(registry: list | None = None) -> dict:
    """{'<run>/model_seed<s>.pt': sha256} of the last registered warm-start row ({} when there is none)."""
    if registry is None:
        from ...core.registry import read_registry

        registry = read_registry()
    rows = [r for r in registry if r.get("workstream") == WORKSTREAM and r.get("experiment") == INIT_EXPERIMENT]
    return dict((rows[-1].get("extra") or {}).get("sha256") or {}) if rows else {}


def register_init_checkpoints(runs_root=None, registry: list | None = None, log_row=None) -> dict:
    """Hash the six warm-start checkpoints (two re-runs × three seeds) and write one registry row. Run once, after
    both re-runs have finished and before the first slot job; a second call with other hashes is refused."""
    shas = {}
    for kd in (True, False):
        d = init_run_dir(kd, runs_root)
        for s in SEEDS:
            shas[_init_key(d, s)] = _sha256(d / f"model_seed{s}.pt")
    have = registered_init(registry)
    if have:
        if have != shas:
            raise LaunchError("warm-start checkpoints are already registered with other hashes")
        return shas
    if log_row is None:
        from ...core.registry import log_result as log_row
    log_row(workstream=WORKSTREAM, experiment=INIT_EXPERIMENT, model=INIT_MODEL, metric="n_checkpoints",
            value=float(len(shas)), split="train", is_baseline=False,
            notes="warm-start checkpoints of the v2 networks: corrected-label re-runs of configuration 3 and of its "
                  "no-KD twin, seed-matched (lead ruling A2)",
            extra={"written_by": "M7", "sha256": shas, "init_name": {"kd": INIT_NAME[True], "nokd": INIT_NAME[False]}})
    return shas


def init_checkpoint(kd: bool, seed: int, *, init_run: str | None = None, runs_root=None,
                    registry: list | None = None) -> Path:
    """Path of the warm-start checkpoint of ``seed``. Hard errors: no finished re-run; ``init_run`` names another
    directory (an as-built run, for instance); the file is not registered or differs from its registered sha256."""
    if int(seed) not in SEEDS:
        raise LaunchError(f"seed {seed} is not one of {SEEDS}")
    d = init_run_dir(kd, runs_root)
    if init_run is not None and Path(str(init_run)).name != d.name:
        raise LaunchError(f"--init-run {init_run!r} is not the corrected-label re-run {d.name}; a v2 job loads no "
                          "other checkpoint")
    f = d / f"model_seed{int(seed)}.pt"
    want = registered_init(registry).get(_init_key(d, seed))
    if want is None:
        raise LaunchError(f"{_init_key(d, seed)} is not registered ({INIT_EXPERIMENT}); register the warm-start "
                          "checkpoints before the first slot job")
    if _sha256(f) != want:
        raise LaunchError(f"{f} differs from its registered sha256")
    return f


# ------------------------------------------------------------------------------------------------ A7: GPU-hours
CEILING_H = 45.0                                    # addendum (vii), hard ceiling of the round
JOB_CAP_H = 2.0                                     # every job ≤ 2 h through the GPU lock helper
BLOCKS = ("v2_networks", "densenet", "reruns", "slot_twins", "tabpfn", "s11_embed")   # the closed run list
FORBIDDEN_GPUS = ("2",)                             # another user's GPU (lead, 18:45 and 19:38 EST)
LEDGER = "litev2_gpu_ledger.jsonl"


class CeilingError(LaunchError):
    """Starting the job could take the round past its GPU-hour ceiling."""


def ledger_path(runs_root=None) -> Path:
    return _runs_root(runs_root) / LEDGER


def _read_ledger(fh) -> list[dict]:
    fh.seek(0)
    return [json.loads(x) for x in fh.read().splitlines() if x.strip()]


def ledger_total(records: list[dict]) -> dict:
    """Logged GPU-hours: finished jobs at their measured hours, jobs without an end record at their full budget."""
    ends = {r["job_id"]: r for r in records if r["event"] == "end"}
    spent = sum(float(ends[r["job_id"]]["hours"]) for r in records if r["event"] == "start" and r["job_id"] in ends)
    open_h = sum(float(r["budget_h"]) for r in records if r["event"] == "start" and r["job_id"] not in ends)
    per = {}
    for r in records:
        if r["event"] == "start":
            h = float(ends[r["job_id"]]["hours"]) if r["job_id"] in ends else float(r["budget_h"])
            per[r["block"]] = round(per.get(r["block"], 0.0) + h, 4)
    return {"finished_h": round(spent, 4), "open_budget_h": round(open_h, 4), "total_h": round(spent + open_h, 4),
            "ceiling_h": CEILING_H, "per_block_h": per,
            "n_jobs": sum(r["event"] == "start" for r in records), "n_open": sum(
                r["event"] == "start" and r["job_id"] not in ends for r in records)}


def _on_alarm(_sig, _frm):
    raise CeilingError("the job reached its GPU-hour budget and is stopped")


@contextlib.contextmanager
def gpu_job(name: str, block: str, budget_h: float, *, runs_root=None, clock=time.time, alarm: bool = True):
    """Run a GPU job of the closed run list under the ceiling. Refuses when the block is not in the list, the
    budget is above the job cap, the process sits on a forbidden GPU or was not started through the GPU lock helper
    (``CUDA_VISIBLE_DEVICES`` is not one index), or ``logged total + budget > CEILING_H``."""
    if block not in BLOCKS:
        raise LaunchError(f"{block!r} is not in the closed run list {BLOCKS}")
    budget_h = float(budget_h)
    if not 0 < budget_h <= JOB_CAP_H:
        raise LaunchError(f"job budget {budget_h} h is outside (0, {JOB_CAP_H}] h")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if gpu in FORBIDDEN_GPUS:
        raise LaunchError(f"GPU {gpu} belongs to another user and is never used")
    if not re.fullmatch(r"[0-9]+", gpu):               # the lock helper sets exactly one index before the job starts
        raise LaunchError("a GPU job of the round starts through the lock helper (python -m sentinel.core.gpu ...): "
                          f"CUDA_VISIBLE_DEVICES is {gpu!r}, not one GPU index")
    path = ledger_path(runs_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = clock()
    job_id = f"{name}|{os.getpid()}|{int(t0 * 1000)}"
    with open(path, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        tot = ledger_total(_read_ledger(fh))
        if tot["total_h"] + budget_h > CEILING_H + 1e-9:
            raise CeilingError(f"{name}: logged {tot['total_h']:.2f} GPU-h + budget {budget_h:.2f} h would pass the "
                               f"ceiling of {CEILING_H:.0f} GPU-h; the job does not start")
        fh.seek(0, os.SEEK_END)
        fh.write(json.dumps({"event": "start", "job_id": job_id, "name": name, "block": block, "budget_h": budget_h,
                             "gpu": gpu, "pid": os.getpid(), "t": t0,
                             "time": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(t0)),
                             "logged_before_h": tot["total_h"]}) + "\n")
        fh.flush()
    old = None
    if alarm:
        old = signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(max(1, int(budget_h * 3600)))
    status = "failed"
    try:
        yield job_id
        status = "ok"
    finally:
        if alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
        t1 = clock()
        with open(path, "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.write(json.dumps({"event": "end", "job_id": job_id, "name": name, "hours": (t1 - t0) / 3600.0,
                                 "status": status, "t": t1}) + "\n")
            fh.flush()


def ledger_summary(runs_root=None) -> dict:
    p = ledger_path(runs_root)
    if not p.is_file():
        return ledger_total([])
    with open(p) as fh:
        return ledger_total(_read_ledger(fh))


__all__ = ["LaunchError", "CeilingError", "mask_flags", "chip_kd_masked", "mask_kd_tensors", "activate_rerun_kd_mask",
           "INIT_NAME", "INIT_EXPERIMENT", "init_run_dir", "registered_init", "register_init_checkpoints",
           "init_checkpoint", "CEILING_H", "JOB_CAP_H", "BLOCKS", "ledger_path", "ledger_total", "gpu_job",
           "ledger_summary"]
