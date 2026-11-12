# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Plumbing of the M6 architecture-v2 drivers (``scripts/{data,train,eval}/sota_hydromet_v2_*.py``): configs, run
directories under ``$SENTINEL_RUNS/hydromet/sota_v2/``, the stamp guard, GPU locks and run records.

Run records written here hold counts and hashes only — never a site identifier."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
from pathlib import Path

from ...core import paths
from ...data.reserved import assert_no_reserved as shared_assert_no_reserved
from ...data.reserved import exclude_reserved as shared_exclude_reserved
from ...models.hydromet import runner as HR
from .. import reserved as R
from .. import runner as SR
from .. import sites as LS

WORKSTREAM = SR.WORKSTREAM
AMENDMENT = paths.REPO_ROOT / "docs" / "preregistration" / "hydromet_arch_v2_amendment.md"
PLAN_REF = "docs/preregistration/hydromet_arch_v2_amendment.md (M6 architecture v2)"
CFG_DIR = paths.REPO_ROOT / "configs" / "hydromet_sota" / "v2"
AMENDMENT_REL = "docs/preregistration/hydromet_arch_v2_amendment.md"
STAMP_SUFFIX = "/prereg_snapshot"
STAMP_EXPERIMENT = "hydromet_arch_v2/prereg_snapshot"     # the experiment of a lead stamp row, exactly (review F1)
STAMP_WRITER = "lead"                                     # extra.written_by of a lead stamp row (review F1)
DEVIATIONS_REL = "docs/preregistration/hydromet_arch_v2_deviations.md"
DEVIATIONS = paths.REPO_ROOT / "docs" / "preregistration" / "hydromet_arch_v2_deviations.md"
N_DYNAMIC, N_STATIC = 30, 191          # US input set v2 (amendment §3; lead rulings L5)
ROOT = paths.RUNS_ROOT / "hydromet" / "sota_v2"
EXIT_RESUME = SR.EXIT_RESUME
INNER_EXPERIMENT = "M6V2/inner_v1"
CODE_PINS_REL = "docs/preregistration/hydromet_arch_v2_code_pins.json"
CODE_PINS = paths.REPO_ROOT / "docs" / "preregistration" / "hydromet_arch_v2_code_pins.json"
CODE_PIN_CHANGE = "CODE PIN CHANGE: {label} sha256 {sha}"
CODE_PIN_LINE = re.compile(r"CODE PIN CHANGE: (\S+) sha256 ([0-9a-f]{64})")
POOL_READER = ("sentinel.hydromet_sota.gso_mt.data.load_v2_train(require_pins=True): daily label tables through "
               "scan_cube (date filter inside the file scan), grab rows through read_grab_inst")
NO_POOL_READ = "none (run records only; no pool read)"
DATA_MANIFESTS = ("sonde_daily", "forcings_daily", "streamcat", "hydromet_upstream_forcing", "hydromet_sota",
                  "hydromet_sota_v2")


class NotStampedError(RuntimeError):
    """A non-smoke arch-v2 run, or a validation number of a new arm, was requested before the lead's stamp."""


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp_status(registry_path=None) -> dict:
    """The lead's stamp (LEAD_DECISIONS_20261110.md rule 1) is a lead stamp row (:func:`_stamp_rows`) whose
    ``extra.sha256`` maps the amendment path to the sha256 of the file **as it is on disk now** (an edit after the
    stamp closes the gate again), plus the file being read-only."""
    sha = _sha(AMENDMENT)
    out = {"stamped": False, "amendment_sha256": sha, "registry_id": None, "timestamp": None, "experiment": None,
           "read_only": _read_only(AMENDMENT), "n_stamp_rows": 0}
    for r, shas in _stamp_rows(registry_path):
        if AMENDMENT_REL not in shas:
            continue
        out["n_stamp_rows"] += 1
        if shas[AMENDMENT_REL] == sha:
            out.update(registry_id=r["id"], timestamp=r.get("timestamp"), experiment=r.get("experiment"))
    out["stamped"] = bool(out["registry_id"]) and out["read_only"]
    return out


def assert_stamped(what: str = "a non-smoke arch-v2 run", registry_path=None) -> dict:
    """Every code path that trains a counted configuration, or that reads a validation-block, test-block or EA row
    for a new arm, a reference refit or a new track, calls this first. Allowed without the stamp: code, tests, data
    builds (protected rows: counts only), synthetic and train-only smoke runs."""
    st = stamp_status(registry_path)
    if not st["registry_id"]:
        raise NotStampedError(f"{what}: refused before the lead's stamp of {AMENDMENT_REL} (sha256 "
                              f"{st['amendment_sha256'][:12]}…; {st['n_stamp_rows']} stamp rows name this file, none "
                              "with its current content)")
    if not st["read_only"]:
        raise NotStampedError(f"{what}: the stamped amendment must be read-only (0444)")
    return st


def _read_only(path) -> bool:
    return not (stat.S_IMODE(os.stat(path).st_mode) & 0o222)


def _stamp_rows(registry_path=None):
    """Lead stamp rows: registry rows of workstream ``M6-hydromet`` whose experiment is **exactly**
    ``hydromet_arch_v2/prereg_snapshot``, whose ``extra.written_by`` is ``"lead"`` and that carry a path → sha256
    map (entry-1 lead rulings, F1). A row written by an arch-v2 script is never a stamp. The marker cannot be made
    unforgeable under one Unix user: the control is the lead's list of row ids, which is why every non-smoke run
    record stores the id of the row that released it."""
    reg = Path(registry_path) if registry_path else paths.REGISTRY_PATH
    if not reg.is_file():
        return
    with reg.open() as fh:
        for line in fh:
            if STAMP_SUFFIX not in line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(r, dict) or r.get("workstream") != WORKSTREAM or r.get("experiment") != STAMP_EXPERIMENT:
                continue
            extra = r.get("extra") if isinstance(r.get("extra"), dict) else {}
            if extra.get("written_by") != STAMP_WRITER:
                continue
            shas = extra.get("sha256")
            script = str((r.get("code") or {}).get("script", "")) if isinstance(r.get("code"), dict) else ""
            if isinstance(shas, dict) and r.get("id") and "sota_hydromet_v2" not in script:
                yield r, shas


def deviations_status(registry_path=None) -> dict:
    """Lead rulings L2 / L10: the deviations file is released when it is read-only and a stamp row carries the
    amendment sha256 **unchanged** together with the sha256 of the deviations file **as it is on disk now** (a new
    entry closes the gate until the lead has stamped the file again)."""
    out = {"released": False, "exists": DEVIATIONS.is_file(), "deviations_sha256": None, "read_only": False,
           "registry_id": None, "timestamp": None, "n_rows_naming_file": 0}
    if not out["exists"]:
        return out
    sha, am = _sha(DEVIATIONS), _sha(AMENDMENT)
    out.update(deviations_sha256=sha, read_only=_read_only(DEVIATIONS))
    for r, shas in _stamp_rows(registry_path):
        if DEVIATIONS_REL not in shas:
            continue
        out["n_rows_naming_file"] += 1
        if shas[DEVIATIONS_REL] == sha and shas.get(AMENDMENT_REL) == am:
            out.update(registry_id=r["id"], timestamp=r.get("timestamp"))
    out["released"] = bool(out["registry_id"]) and out["read_only"]
    return out


def assert_entry_released(what: str = "a non-smoke arch-v2 run", registry_path=None) -> dict:
    """Conditions 1–4 of :func:`assert_released` (everything but the code pin). Called directly only by the script
    that writes the code-pin record; no training or evaluation script may call it (static test)."""
    from . import data as VD

    st = assert_stamped(what, registry_path)
    dv = deviations_status(registry_path)
    if not dv["exists"]:
        raise NotStampedError(f"{what}: {DEVIATIONS_REL} does not exist (entry 1 gates the first fit; rulings L3)")
    if not dv["registry_id"]:
        raise NotStampedError(f"{what}: no lead stamp row carries the current sha256 of {DEVIATIONS_REL} "
                              f"({dv['deviations_sha256'][:12]}…) together with the amendment's; "
                              f"{dv['n_rows_naming_file']} rows name the file")
    if not dv["read_only"]:
        raise NotStampedError(f"{what}: the stamped deviations file must be read-only (0444)")
    pins = VD.verify_pins()
    psha = VD.sha256_file(VD.PINS_FILE)
    if psha not in DEVIATIONS.read_text():
        raise NotStampedError(f"{what}: the sha256 of pins.json ({psha[:12]}…) is not recorded in the stamped "
                              "deviations file")
    return {**st, "deviations_sha256": dv["deviations_sha256"], "deviations_registry_id": dv["registry_id"],
            "pins_sha256": psha, "n_pinned": len(pins["files"]),
            "changed_under_stamp": dict(pins["changed_under_stamp"])}


def assert_released(what: str = "a non-smoke arch-v2 run", registry_path=None) -> dict:
    """The guard of **every** non-smoke arch-v2 entry point (neural, GBM, P, stack, evaluation; lead rulings L10;
    entry-1 lead rulings, decision 4). Five conditions, all required:

    1. :func:`assert_stamped` — the amendment as stamped by the lead, read-only;
    2. the deviations file read-only, with a lead stamp row carrying its current sha256 (and the amendment's);
    3. :func:`sentinel.hydromet_sota.gso_mt.data.verify_pins` — every pinned table, split, role table, module and
       config file unchanged, ``pins.json`` read-only;
    4. the sha256 of ``pins.json`` written in the stamped deviations file (the pin record itself is under the lead's
       stamp);
    5. :func:`verify_code_pins` — every arch-v2 code file that pins.json does not cover is as recorded in
       ``docs/preregistration/hydromet_arch_v2_code_pins.json`` (read-only, written once).

    The returned dict carries what a run record must store (:func:`release_fields`). Smoke and synthetic runs do not
    call this; they register nothing."""
    out = assert_entry_released(what, registry_path)
    return {**out, **verify_code_pins(registry_path)}


# ----------------------------------------------------------------------------------------------------- code pin
def code_pin_targets() -> dict:
    """label → path of the arch-v2 code that ``pins.json`` does not cover (entry-1 lead rulings, decision 4): the
    runner, networks, trainer, criterion, GBM reference, gates, package files, every stack module, the ThermOx-T
    network, every training and evaluation driver and the pin script. ``gso_mt/synthetic.py`` is not listed: no
    non-smoke path imports it (static test)."""
    pk = paths.REPO_ROOT / "sentinel" / "hydromet_sota"
    files = [pk / "gso_mt" / n for n in ("__init__.py", "criterion.py", "gates.py", "gbm_ref.py", "nets.py",
                                         "runner.py", "trainer.py")]
    files += sorted((pk / "stack").glob("*.py")) + [pk / "thermox" / "__init__.py", pk / "thermox" / "nets.py"]
    files += sorted((paths.REPO_ROOT / "scripts" / "train").glob("sota_hydromet_v2_*.py"))
    files += sorted((paths.REPO_ROOT / "scripts" / "eval").glob("sota_hydromet_v2_*.py"))
    files += [paths.REPO_ROOT / "scripts" / "data" / "sota_hydromet_v2_pins.py"]
    return {str(p.relative_to(paths.REPO_ROOT)): p for p in files}


def code_pin_change_line(label: str, sha: str) -> str:
    return CODE_PIN_CHANGE.format(label=label, sha=sha)


def stamped_code_states(registry_path=None) -> dict:
    """label → the sha256 that the stamped deviations file gives a code file: the **last** whole line
    ``CODE PIN CHANGE: <label> sha256 <sha256>`` of that label (one state per label; earlier lines and the original
    pin are then no longer valid). Empty when the deviations file is not in its stamped state."""
    if not deviations_status(registry_path)["released"]:
        return {}
    out = {}
    for ln in DEVIATIONS.read_text().splitlines():
        m = CODE_PIN_LINE.fullmatch(ln)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def verify_code_pins(registry_path=None) -> dict:
    """Every file of :func:`code_pin_targets` must have the sha256 of the code-pin record (read-only, written once
    by ``scripts/data/sota_hydromet_v2_pins.py --code-pins``). After the record exists a code file changes, and a
    new driver or stack module is added, only through a stamped deviations entry that names the file and its new
    sha256 (:func:`stamped_code_states`) and the runs that are repeated."""
    if not CODE_PINS.is_file():
        raise NotStampedError(f"{CODE_PINS_REL} is missing: the code-pin record is written before the first "
                              "non-smoke fit (entry-1 lead rulings, decision 4)")
    if not _read_only(CODE_PINS):
        raise NotStampedError(f"{CODE_PINS_REL} must be read-only (0444): it is written once")
    files = json.loads(CODE_PINS.read_text())["files"]
    targets = code_pin_targets()
    gone = sorted(set(files) - set(targets))
    if gone:
        raise NotStampedError(f"code-pinned files are missing: {gone[:5]} ({len(gone)} in all)")
    changed, states = {}, stamped_code_states(registry_path)
    for label, path in targets.items():
        h, moved = _sha(path), states.get(label)
        if h != (moved or files.get(label)):
            raise NotStampedError(f"{label}: sha256 {h[:12]}… is not the code-pinned state; an arch-v2 code file "
                                  "changes only through a stamped deviations entry with the line "
                                  f"{code_pin_change_line(label, '<sha256>')!r}")
        if moved:
            changed[label] = h
    return {"code_pins_sha256": _sha(CODE_PINS), "n_code_pinned": len(files), "code_changed_under_stamp": changed}


# ------------------------------------------------------------------------------- release record of a run (F1, F5)
RELEASE_SHA_KEYS = ("amendment_sha256", "deviations_sha256", "pins_sha256", "code_pins_sha256")
RELEASE_ID_KEYS = ("amendment_registry_id", "deviations_registry_id")
RELEASE_MAP_KEYS = ("changed_under_stamp", "code_changed_under_stamp")
STATE_KEYS = ("pins_sha256", "changed_under_stamp", "code_pins_sha256")
RELEASE_FILE = "release.json"


def release_fields(stamp: dict, *, reader: str) -> dict:
    """What every non-smoke run record and registry row stores (entry-1 lead rulings: F1, decision 2 condition 3,
    decision 3, decision 4): the amendment and deviations sha256 with the ids of the lead rows that released the
    run, the pin record's sha256, ``changed_under_stamp`` as ``verify_pins`` returned it, the code-pin sha256 with
    its own change list, and the name of the reader of the pool."""
    out = {"amendment_sha256": stamp["amendment_sha256"], "amendment_registry_id": stamp["registry_id"],
           "deviations_sha256": stamp["deviations_sha256"], "deviations_registry_id": stamp["deviations_registry_id"],
           "pins_sha256": stamp["pins_sha256"], "changed_under_stamp": dict(stamp["changed_under_stamp"]),
           "code_pins_sha256": stamp["code_pins_sha256"],
           "code_changed_under_stamp": dict(stamp["code_changed_under_stamp"]), "reader": reader}
    return check_release(out)


def check_release(rel) -> dict:
    """A non-smoke run record without these fields is not valid."""
    ok = isinstance(rel, dict) and all(isinstance(rel.get(k), str) and len(rel[k]) == 64 for k in RELEASE_SHA_KEYS) \
        and all(isinstance(rel.get(k), str) and rel[k] for k in RELEASE_ID_KEYS + ("reader",)) \
        and all(isinstance(rel.get(k), dict) for k in RELEASE_MAP_KEYS)
    if not ok:
        raise NotStampedError("a non-smoke run record needs the release fields of assert_released "
                              f"({', '.join(RELEASE_SHA_KEYS + RELEASE_ID_KEYS + RELEASE_MAP_KEYS)}, reader)")
    return rel


def with_release(cfg: dict, stamp: dict | None, *, reader: str = POOL_READER) -> dict:
    """``cfg`` with the release record under ``_release`` (not part of the public configuration, so the hash of the
    shared configuration counter does not depend on the deviations file). ``stamp`` is ``None`` for a smoke run."""
    if stamp is None:
        if not cfg.get("_smoke"):
            raise NotStampedError("a non-smoke run needs the result of assert_released")
        return {**cfg, "_release": None}
    return {**cfg, "_release": release_fields(stamp, reader=reader)}


def run_release(cfg: dict) -> dict | None:
    """The release record of a run: ``None`` for a smoke run, checked and required otherwise."""
    if cfg.get("_smoke"):
        return None
    return check_release(cfg.get("_release"))


def run_config(cfg: dict, **extra) -> dict:
    """The public configuration written to a run directory, with the release record of a non-smoke run."""
    out = HR.public_config(dict(cfg, **extra))
    rel = run_release(cfg)
    if rel is not None:
        out["release"] = rel
    return out


def assert_pool_read(cfg: dict, v2) -> None:
    """Decision 3: the only non-smoke path to the pool is ``load_v2_train(require_pins=True)``, whose record carries
    the sha256 of the pin file it verified (and whose fingerprint check has passed)."""
    rel = run_release(cfg)
    if rel is not None and v2.info.get("pins_sha256") != rel["pins_sha256"]:
        raise NotStampedError("the pool of a non-smoke run was not read through load_v2_train(require_pins=True) "
                              "under the pin record of this release")


def write_release(rdir, cfg: dict) -> dict | None:
    """``release.json`` of a non-smoke run directory. A resumed run must be in the state of its first job (pin
    record, pinned files changed under a stamp, code pin): a run never spans two states."""
    import datetime as dt

    rel = run_release(cfg)
    if rel is None:
        return None
    f = Path(rdir) / RELEASE_FILE
    jobs = []
    if f.is_file():
        old = json.loads(f.read_text())
        bad = [k for k in STATE_KEYS + ("code_changed_under_stamp",) if old.get(k) != rel[k]]
        if bad:
            raise NotStampedError(f"{Path(rdir).name}: started under another state ({bad}); a run made under a "
                                  "superseded state is void and is repeated in a new run directory")
        jobs = list(old.get("jobs", []))
    jobs.append({"utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                 "deviations_sha256": rel["deviations_sha256"], "deviations_registry_id": rel["deviations_registry_id"]})
    write_json(f, {**rel, "jobs": jobs})
    return rel


def assert_combinable(dirs, release: dict, what: str) -> dict:
    """The inner evaluation and the stacks refuse to combine runs with different code-pin sha256 (decision 4); a
    run made under a superseded pin state is void (decision 2, condition 4). Every run directory must carry a
    release record whose pin record, ``changed_under_stamp`` and code-pin sha256 equal those of ``release`` (the
    state of the combining process). Returns the audit record of the combination."""
    release = check_release(release)
    code_changes = {}
    for d in map(Path, dirs):
        f = d / RELEASE_FILE
        if not f.is_file():
            raise NotStampedError(f"{what}: {d.name} has no release record (not a valid non-smoke run)")
        rec = check_release(json.loads(f.read_text()))
        bad = [k for k in STATE_KEYS if rec[k] != release[k]]
        if bad:
            raise NotStampedError(f"{what}: {d.name} was made under another state ({bad}); runs with different "
                                  "code-pin sha256 or a superseded pin state are not combined")
        code_changes[d.name] = rec["code_changed_under_stamp"]
    return {"n_runs": len(code_changes), **{k: release[k] for k in STATE_KEYS},
            "code_changed_under_stamp_by_run": code_changes}


def pinned_config_path(path) -> Path:
    """Review F4: a non-smoke run reads its configuration from a pinned file of ``configs/hydromet_sota/v2/`` and
    from nowhere else. Returns the canonical path of that file."""
    from . import data as VD

    p = Path(path)
    want = CFG_DIR / p.name
    label = f"configs/hydromet_sota/v2/{p.name}"
    if p.is_symlink() or want.is_symlink() or not want.is_file() or p.resolve() != want.resolve() \
            or label not in VD.pin_targets():
        raise NotStampedError(f"{p}: a non-smoke run reads only a pinned file of configs/hydromet_sota/v2/")
    return want


def assert_recipe(cfg: dict) -> None:
    """R-TD (lead rulings §2.2): hidden size and dropout are the published values, transcribed with a page / table
    citation in a stamped deviations entry before the first R-TD fit. Until ``rtd.yaml`` says so, no counted R-TD
    run starts."""
    if "recipe_verified" in cfg and not cfg.get("_smoke"):
        if cfg["recipe_verified"] is not True or not cfg.get("recipe_source"):
            raise NotStampedError(f"{cfg['arm']}: recipe_verified is not true / recipe_source is empty in "
                                  f"{Path(cfg['_config_file']).name}; the published recipe must be transcribed and "
                                  "stamped first (rulings §2.2)")


D2_NETS_REL = "sentinel/hydromet_sota/thermox/nets.py"


def arm_release_token(arm: str) -> str | None:
    """The line a stamped deviations entry must carry before a counted run of an arm that entry 1 does not release
    (lead rulings L4): D2 needs its own entry, which names the sha256 of the ThermOx-T network file as fitted."""
    if arm != "d2":
        return None
    f = paths.REPO_ROOT / D2_NETS_REL
    return f"D2 RELEASED: {D2_NETS_REL} sha256 {_sha(f) if f.is_file() else 'file missing'}"


def assert_arm_released(cfg: dict) -> None:
    """After :func:`assert_released`: entry 1 releases the references and D1 only. A non-smoke D2 run needs the D2
    entry in the stamped deviations file (constants of the energy balance, no Powell prior; rulings L4, L6)."""
    tok = None if cfg.get("_smoke") else arm_release_token(str(cfg.get("arm")))
    if tok and (not DEVIATIONS.is_file() or tok not in DEVIATIONS.read_text().splitlines()):
        raise NotStampedError(f"{cfg['arm']}: not released by the stamped deviations file (it must contain the "
                              f"whole line {tok!r}; rulings L4, review F6)")


D2_KILL_RULE = ROOT / "d2" / "kill_rule.json"


def d2_kill_decision(step1_s: float, step2_s: float | None, max_s: float) -> str:
    """Amendment §8: ``all_days`` if the step time with the diel sub-model on every loss day is ≤ ``max_s``; else
    ``fallback_months`` if the restricted form is ≤ ``max_s``; else ``dropped``. Without the second measurement a
    slow first step gives ``needs_step_2``."""
    if step1_s <= max_s:
        return "all_days"
    if step2_s is None:
        return "needs_step_2"
    return "fallback_months" if step2_s <= max_s else "dropped"


def record_d2_kill_rule(cfg: dict, step1_dir, step2_dir=None, *, record=None, register: bool = True) -> dict:
    """Write the outcome of the D2 kill rule (amendment §8) from the train-only smoke run(s): the measured step time
    with the diel sub-model on every loss day and, if that is above the limit, on the fallback months. The record
    holds times, memory, counts and hashes only. One registry row (``M6V2/decisions_v1``); nothing here is a stamp
    and nothing releases D2 (the D2 deviations entry does, rulings L4)."""
    import datetime as dt

    def read(d):
        if d is None:
            return None
        d = Path(d)
        info, data = json.loads((d / "train_info.json").read_text()), json.loads((d / "data_info.json").read_text())
        if not info.get("smoke") or data.get("blocks") != ["train"]:
            raise RuntimeError(f"{d.name} is not a train-only smoke run")
        return {"run": d.name, "s_per_step_median": float(info["s_per_step_median"]),
                "s_per_step_by_epoch": [h.get("s_per_step") for h in info["history"]],
                "gpu_memory": info.get("gpu_memory"), "n_params": info.get("n_params"),
                "n_locations": int(data["n_locations_used"]), "epochs": info.get("epochs")}

    kr = cfg["kill_rule"]
    s1, s2 = read(step1_dir), read(step2_dir)
    for s in (s1, s2):
        if s is not None and s["n_locations"] > 50:
            raise RuntimeError("the kill rule is measured on at most 50 train-role locations")
    dec = d2_kill_decision(s1["s_per_step_median"], None if s2 is None else s2["s_per_step_median"],
                           float(kr["max_s_per_step"]))
    nets = paths.REPO_ROOT / D2_NETS_REL
    out = {"rule": "amendment §8 (D2 kill rule); train-only smoke, grid point 1, no validation number",
           "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "max_s_per_step": float(kr["max_s_per_step"]), "batch_size": int(cfg["train"]["batch_size"]),
           "seq_len": int(cfg["train"]["seq_len"]), "fallback_months": [int(m) for m in kr["step_2_fallback_months"]],
           "step_1_all_days": s1, "step_2_fallback_months": s2, "decision": dec,
           "nets_sha256": hashlib.sha256(nets.read_bytes()).hexdigest(), "point": cfg["point"]}
    rec = Path(record) if record else D2_KILL_RULE
    rec.parent.mkdir(parents=True, exist_ok=True)
    write_json(rec, out)
    if register:
        from ...core.registry import log_result

        row = log_result(workstream=WORKSTREAM, experiment=DECISION_EXPERIMENT, model="D2 ThermOx-T (temperature stage)",
                         metric="d2_kill_rule_s_per_step", value=float((s2 or s1)["s_per_step_median"]),
                         split="train-only smoke (at most 50 train-role locations; no validation number)",
                         data_manifests=list(DATA_MANIFESTS), artifacts=[str(rec)],
                         notes=f"D2 kill rule: {dec}", extra=out)
        out["registry_id"] = row.get("id") if isinstance(row, dict) else None
        write_json(rec, out)
    return out


def resolve_d2_diel(cfg: dict, *, fallback: bool = False, record=None) -> dict:
    """Replace ``net.diel.days: rule`` of the D2 config by the months the diel sub-model runs on (``months: None`` =
    every loss day). Smoke: every day, or the fallback months with ``fallback``. Non-smoke: read from the kill-rule
    record; a missing record or a dropped arm stops the run."""
    cfg = copy.deepcopy(cfg)
    diel = dict(cfg["net"].get("diel") or {})
    if diel.pop("days", "all") == "rule" and diel.get("enabled", True):
        months = [int(m) for m in cfg["kill_rule"]["step_2_fallback_months"]]
        if cfg.get("_smoke"):
            diel["months"] = months if fallback else None
        else:
            rec = Path(record) if record else D2_KILL_RULE
            if not rec.is_file():
                raise NotStampedError(f"D2: the kill-rule record {rec} does not exist (amendment §8)")
            doc = json.loads(rec.read_text())
            dec = doc["decision"]
            if doc.get("nets_sha256") != _sha(paths.REPO_ROOT / D2_NETS_REL):
                raise NotStampedError(f"D2: {rec.name} was measured on another state of {D2_NETS_REL}; "
                                      "repeat the train-only smoke of the kill rule")
            if dec not in ("all_days", "fallback_months"):
                raise NotStampedError(f"D2 is not runnable: kill-rule decision {dec!r}")
            diel["months"] = months if dec == "fallback_months" else None
    cfg["net"]["diel"] = diel
    return cfg


def assert_pool_clean(sites, where: str = "M6 arch-v2 pool") -> int:
    """Reserved-site guard of every arch-v2 table, fit and run record (lead ruling 2026-11-10): the 729 sites of
    ``docs/preregistration/sonde_newsites_list.json`` are never read. ``sites`` is the **complete** location list (USGS
    numbers and ``WQP:<id>`` keys). Three independent checks: the shared helper (``sentinel.data.reserved``), the M6
    guard (``hydromet_sota.reserved``) and the pre-claim-audit scan for a reserved number inside any key. Returns the
    number of locations checked; raises ``ReservedSiteError`` otherwise."""
    keys = [str(s) for s in sites]
    ids = [k.split(":", 1)[-1] for k in keys]
    shared_assert_no_reserved(ids, where=where)
    if len(shared_exclude_reserved(ids)) != len(ids):
        raise R.ReservedSiteError(f"{where}: the shared helper would drop a location")
    R.assert_no_reserved(ids)
    res = R.reserved_usgs_sites()
    n = sum(LS.contains_reserved_id(k, res) for k in keys)
    if n:
        raise R.ReservedSiteError(f"{where}: a reserved number inside {n} location keys")
    return len(keys)


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path, point: str, *, base: str | None = None, smoke: bool = False) -> dict:
    """``common.yaml`` ∪ arm file ∪ grid point. An ablation / variant ``point`` is applied over the grid point
    ``base`` (the selected point; nothing is re-selected). A non-smoke call accepts only a pinned file of
    ``configs/hydromet_sota/v2/`` (review F4)."""
    import yaml

    p = Path(path) if smoke else pinned_config_path(path)
    arm = yaml.safe_load(p.read_text())
    cfg = _merge(yaml.safe_load((CFG_DIR / "common.yaml").read_text()), {k: v for k, v in arm.items()
                                                                        if k not in ("grid", "ablations")})
    grid, abl = arm.get("grid", {}), arm.get("ablations", {})
    if point in grid:
        if base not in (None, point):
            raise ValueError("--base applies to ablations and variants only")
        cfg = _merge(cfg, grid[point])
        cfg["variant_of"] = None
    elif point in abl:
        if base is None:
            if len(grid) != 1:
                raise ValueError(f"{point} needs --base (the selected grid point of {arm['arm']})")
            base = next(iter(grid))
        cfg = _merge(_merge(cfg, grid[base]), abl[point])
        cfg["variant_of"] = base
    else:
        raise ValueError(f"unknown point {point!r} in {p.name}")
    cfg.update({"point": point, "_config_file": str(p), "_smoke": bool(smoke)})
    return cfg


def train_cfg(cfg: dict, seed: int):
    from .trainer import TrainCfg

    if cfg.get("grab_share") is None:
        raise ValueError(f"{cfg['arm']}.{cfg['point']}: grab_share is a required config value (no default in code; "
                         "R-TD's is fixed at 0.5 by the lead rulings §2.2)")
    aux_w = cfg.get("aux_w")
    if cfg.get("lambda_q") is not None:                         # D2: weight of the discharge latent (amendment §6.2)
        from .nets import AUX_W

        aux_w = {**(aux_w or AUX_W), "logq": float(cfg["lambda_q"])}
    return TrainCfg(arm=f"{cfg['arm']}.{cfg['point']}", net=dict(cfg["net"]), grab_share=float(cfg["grab_share"]),
                    variables=tuple(cfg.get("variables", ("T", "DO"))), aux_w=aux_w, seed=int(seed),
                    **cfg["train"])


def run_dir(arm: str, name: str, *, smoke: bool = False) -> Path:
    d = ROOT / ("smoke" if smoke else arm) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def locked_gpu() -> int:
    """Index of the GPU that the lock helper gave this process (``python -m sentinel.core.gpu --name <job>
    --min-free-mb 10000 -- <cmd>`` sets ``CUDA_VISIBLE_DEVICES`` and holds the lock for the life of the command).
    The arch-v2 drivers never choose a GPU themselves."""
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not v.isdigit():
        raise SystemExit("no locked GPU: run through `python -m sentinel.core.gpu --name <job> --min-free-mb "
                         f"{SR.MIN_FREE_MB} -- <cmd>` (or --cpu for a debug run)")
    return int(v)


def gpu_memory_mb() -> dict | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return {"max_allocated_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                "max_reserved_mb": round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)}
    except Exception:
        return None


def training_run(cfg: dict, rdir: Path, *, seed: int, refit: bool, sites):
    """``sites``: every location the run can read (checked against the reserved list before the record opens).
    A non-smoke run needs the release record of :func:`with_release`; it goes into the run record and into
    ``release.json`` of the run directory."""
    from ...core.runlog import TrainingRun

    assert_pool_clean(sites, where=f"{cfg['arm']}.{cfg['point']} training run")
    exp = f"{'SMOKE/' if cfg.get('_smoke') else ''}M6V2/{cfg['arm']}.{cfg['point']}{'.refit' if refit else ''}"
    write_release(rdir, cfg)                                    # non-smoke: required, and one state per run
    return TrainingRun(WORKSTREAM, exp, config=run_config(cfg, seed=seed), data=list(DATA_MANIFESTS),
                       plan_ref=PLAN_REF, iteration="smoke" if cfg.get("_smoke") else "arch-v2",
                       artifacts_dir=str(rdir))


def log_config_tried(cfg: dict, rdir: Path, status: str = "started") -> None:
    """Shared append-only counter (``sentinel_runs/hydromet/configs_tried.jsonl``); smoke runs carry ``smoke: true``
    and are not counted. Seeds and refits of one point share its ``config_hash`` only if logged without the seed, so
    the point is logged without it."""
    HR.log_config_tried(f"M6V2/{cfg['arm']}.{cfg['point']}", cfg, rdir, status=status)


def log_inner(cfg: dict, info: dict, *, seed: int, rdir: Path) -> list:
    """Seed-level inner numbers → registry ``M6V2/inner_v1`` (amendment §7). Train-role inner rows only."""
    from ...core.registry import log_result

    if cfg.get("_smoke"):
        raise RuntimeError("refusing to register a smoke run")
    model = f"{cfg['arm']}.{cfg['point']}_s{seed}"
    rel = run_release(cfg)
    rows = []
    for metric, key in (("J", "best_J"), ("J_sonde", "best_j_sonde"), ("J_grab", "best_j_grab")):
        if info.get(key) is not None:
            rows.append(log_result(workstream=WORKSTREAM, experiment=INNER_EXPERIMENT, model=model, metric=metric,
                                   value=float(info[key]), split="inner (train-role inner groups x train block)",
                                   config=HR.public_config(cfg), data_manifests=list(DATA_MANIFESTS),
                                   artifacts=[str(rdir)], notes=f"seed {seed}; best epoch {info.get('best_epoch')}; "
                                   f"variant_of {cfg.get('variant_of')}",
                                   extra={"seed": seed, "arm": cfg["arm"], "point": cfg["point"],
                                          "variant_of": cfg.get("variant_of"), "by_target": info.get("best_by_target"),
                                          "release": rel}))
    return rows


def write_json(path, obj) -> None:
    HR.write_json(path, obj)


DECISION_EXPERIMENT = "M6V2/decisions_v1"


def log_decision(kind: str, model: str, payload: dict, *, rdir, release: dict, value: float = 1.0,
                 notes: str = "") -> dict:
    """A choice or a gate outcome → one registry row (``M6V2/decisions_v1``) **before** anything that depends on it:
    the selected grid point of an arm, a reference's member / form selection, the outcome of gate G-D1a. Inner
    (train-role) quantities only; ``payload`` holds names, counts and inner numbers already logged as
    ``M6V2/inner_v1`` — never a site identifier. ``release``: the release record of the deciding process."""
    from ...core.registry import log_result

    payload = {**payload, "release": check_release(release)}
    return log_result(workstream=WORKSTREAM, experiment=DECISION_EXPERIMENT, model=model, metric=kind,
                      value=float(value), split="inner (train-role inner groups x train block)",
                      data_manifests=list(DATA_MANIFESTS), artifacts=[str(rdir)], notes=notes, extra=payload)
