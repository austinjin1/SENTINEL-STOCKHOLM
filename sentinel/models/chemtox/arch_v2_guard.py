"""Single-look guard for the M4 arch-v2 sealed blocks T1_test, T6a_test and T6b_test.

New module with its own pins file, experiment key and protected-labels artefact.
``unseen_taxa.py`` and ``unseen_taxa_guard.py`` are frozen code of the earlier tracks and are
**not** edited or imported.

Revision history
----------------
* ``6e1ab9c1…f224fe`` — the version present at the partial stamp; its §8.2 guarantees were
  *required* behaviour it did not implement.
* ``9bd7d60a…c066e`` — ruling B1 (i)-(v) and C-1.
* ``c9620f37…68e9`` — addendum 01 (ruling N2-A): the two guarded auxiliary readers.
* **this version** — rulings **G-1 … G-10** of
  ``docs/preregistration/chemtox_arch_v2_lead_guard_rulings_01.md`` (sha256
  ``63b7184c6733c6ed62382c6f66b7098bdd792beb98705d0ce1d52b2424fa1a9a``), after the independent
  guard review returned ``fix_then_rereview``. Order of authority: the stamped amendment and the
  stamped lead rulings win over the guard rulings, which win over addendum 01 rev 1.

What the review found, and where each fix lives
-----------------------------------------------
======  ==========================================================================  =================
**F1**  ``verify_freeze()`` passed on a two-key JSON any workstream could write.     :func:`verify_freeze`
        It now requires the pins to be unwritable, to be vouched for by a
        lead-written registry row whose ``extra.pins_sha256`` equals the file's
        sha256, to carry every pinned key with a 64-hex value, and to carry no
        unknown entry and no null but the labels path (G-1). Every reader records
        the pins sha256 in the report it writes.
**F2**  a real-path test would read real tables once the pins existed.               tests (G-2)
**F3**  H4 applied four of the seven stamped exclusions and had no test.             :func:`load_h4_rows`
        All seven now apply, in the stamped order, each with its own counter, and    (G-3)
        the re-check is separate code.
**F4**  no sanctioned route to an H2/H3 label.                                       :func:`load_auxiliary_rows`,
        The guard now joins the target itself for exactly the returned ids, and      :func:`load_h5_pool_pairs`
        H5 comes from the pool only (G-4).
**F5**  a second look was possible through a different runs root.                    :func:`make_pins`,
        The runs root is pinned, and ``authorise()`` records a registry claim that    :func:`authorise` (G-5)
        makes a later ``authorise()`` for that block refuse from any root.
**F6**  the labels artefact was trusted by path.                                     :func:`build_protected_labels`,
        It is now write-once, hash-recorded and verified at every read, and the      :func:`load_test_block` (G-6)
        artefact's target is checked against the pinned frame.
**F7**  inputs that decide which rows come back were unpinned.                       :data:`EXPECTED_DATA_KEYS`
        ``prospective.parquet``, the ADORE split assignments and the two raw         (G-7)
        tables are pinned and recorded, and the reader refuses if the
        registry-derived spent-ADORE set differs from the pinned list.
**F8**  reports were silently overwritten.                                           :func:`_write_report` (G-8)
**F9**  the re-check was not fail-closed and covered three of seven.                 :func:`_recheck_seven` (G-9)
======  ==========================================================================  =================

Reader discipline (ruling N2, as amended by N2-A and G-10)
----------------------------------------------------------
**Five** functions here read a protected table: :func:`load_training_pool`,
:func:`build_protected_labels`, :func:`load_test_block`, :func:`load_auxiliary_rows` and
:func:`load_h4_rows`. Any other read of ``ecotox_aquatic.parquet`` or of the raw ECOTOX tables for
T1/T6 work is a protocol breach that voids the block concerned. "Physically cannot" is read as
"does not through the guard": every agent runs under one Unix account, so the controls here are
rules plus hashes plus an audit trail, not file permissions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import stat
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from sentinel.core import paths
from sentinel.models.chemtox import arch_v2 as A

__all__ = ["ProtectedBlockError", "EXPERIMENT", "CLAIM_EXPERIMENT", "BLOCKS", "PINS",
           "ADDENDUM_01_REV1", "FREEZE_RECORD", "PREREG_EXPERIMENT", "PINNED_REPO_FILES",
           "EXPECTED_DATA_KEYS", "AUX_TARGETS", "pins_path", "pins_sha256", "freeze_record_path",
           "make_pins", "verify_freeze", "block_dir", "freeze_path", "freeze_sha",
           "lead_recorded_freeze_sha", "registry_rows", "import_closure_sha", "authorise",
           "require_authorised", "build_protected_labels", "load_training_pool",
           "load_test_block", "load_auxiliary_rows", "load_h4_rows", "load_h5_pool_pairs",
           "save_predictions", "mark_scored", "verify_artifacts", "seal_dir", "scored_pass",
           "look_state", "REPORT_DIR"]

REPO = Path(__file__).resolve().parents[3]

#: The repository root as resolved at import time, used ONLY by :func:`import_closure_sha`, so the
#: code hash recorded on a claim cannot be blanked by redirecting the module's path constants.
_CODE_ROOT = Path(__file__).resolve().parents[3]

AMENDMENT = "docs/preregistration/chemtox_arch_v2_amendment.md"
RULINGS = "docs/preregistration/chemtox_arch_v2_lead_rulings.md"
GUARD_RULINGS = "docs/preregistration/chemtox_arch_v2_lead_guard_rulings_01.md"
ADDENDUM_01_REV1 = "docs/preregistration/chemtox_arch_v2_addendum_01_rev1.md"
PINS = "docs/preregistration/chemtox_arch_v2_pins.json"

#: G-1 (i) / C-1: the lead-written 0444 record of each block's FREEZE sha256.
FREEZE_RECORD = "docs/preregistration/chemtox_arch_v2_freeze_shas.json"

#: Only the lead writes rows of this experiment (C-1). The guard reads them; M4 writes none.
PREREG_EXPERIMENT = "chemtox/prereg_snapshot"

#: G-5: the guard's own single-look claim. NOT a ``chemtox/prereg_snapshot`` row.
CLAIM_EXPERIMENT = "chemtox_arch_v2_look_claim"

EXPERIMENT = A.EXPERIMENT_TEST                       #: "chemtox_arch_v2_test"
BLOCKS = ("T1_test", "T6a_test", "T6b_test")

#: Repo files the freeze covers. G-7 adds the H4 script, ``prospective.py``, the three test files,
#: the revised addendum and the twelve arm/configuration files.
PINNED_REPO_FILES = (
    AMENDMENT,
    RULINGS,
    GUARD_RULINGS,
    ADDENDUM_01_REV1,
    "docs/preregistration/chemtox_t1_clusters_v2.csv",
    "docs/preregistration/chemtox_arch_v2_spent_adore_ids_v2.json",
    "docs/preregistration/chemtox_arch_v2_report_template.md",
    "configs/chemtox/arch_v2.yaml",
    "sentinel/models/chemtox/arch_v2.py",
    "sentinel/models/chemtox/arch_v2_power.py",
    "sentinel/models/chemtox/arch_v2_guard.py",
    "sentinel/models/chemtox/arch_v2_h4.py",
    "sentinel/models/chemtox/arch_v2_arms.py",
    "sentinel/models/chemtox/arch_v2_scorer.py",
    "sentinel/models/chemtox/arch_v2_timelaw.py",
    "sentinel/models/chemtox/arch_v2_embed.py",
    "sentinel/models/chemtox/arch_v2_ted.py",
    "sentinel/models/chemtox/arch_v2_timehead.py",
    "sentinel/models/chemtox/arch_v2_mmoe.py",
    "sentinel/models/chemtox/arch_v2_rx.py",
    "sentinel/models/chemtox/arch_v2_ensemble.py",
    "sentinel/models/chemtox/arch_v2_build.py",
    "sentinel/models/chemtox/prospective.py",
    "scripts/eval/chemtox_arch_v2_h4_report.py",
    "tests/test_chemtox_arch_v2_guard.py",
    "tests/test_chemtox_arch_v2.py",
    "tests/test_chemtox_arch_v2_power.py",
)

RUNS = paths.RUNS_ROOT / "chemtox" / "arch_v2"
REPORT_DIR = RUNS / "aux_exclusions"
LABELS_PATH = paths.DATA_ROOT / "processed" / "chemtox_arch_v2" / "protected_labels_v2.parquet"
LABELS_MARKER = RUNS / "PROTECTED_LABELS.json"
TARGET = "log10_conc_mol_L"
H4_TARGET = "log10_bcf"
REGISTRY = REPO / "results" / "registry.jsonl"

#: raw ECOTOX release tables the H4 frame needs. Fixed here; never caller-supplied.
H4_RESULTS = paths.DATA_ROOT / "raw" / "ecotox" / "release_2026-09-15" / "results.parquet"
H4_TESTS = paths.DATA_ROOT / "raw" / "ecotox" / "release_2026-09-15" / "tests.parquet"

_H4_RESULT_COLUMNS = ["result_id", "test_id", "endpoint", "bcf1_mean_op", "bcf1_mean", "bcf1_unit",
                      "obs_duration_mean", "obs_duration_unit"]

#: G-4: the only target each guarded auxiliary reader returns. Anything else is refused.
AUX_TARGETS = {"H2": TARGET, "H3": TARGET, "H4": H4_TARGET}

_MARKERS = ("AUTHORISED.json", "PREDICTED.json", "SCORED.json", "ABORTED.json",
            "PROTECTED_LABELS.json")

#: G-3 / G-9: the seven exclusions of amendment §10.2, in the stamped order.
SEVEN_EXCLUSIONS = ("t2_reserved_classes", "spent_prospective", "spent_adore",
                    "held_out_unit_chemistry", "chemistry_not_in_split",
                    "same_cell_as_sealed_row", "outside_24_96h_for_T6")


class ProtectedBlockError(RuntimeError):
    """An attempt to read, score or re-score a sealed arch-v2 block outside the guard's rules."""


def file_sha(p) -> str:
    return A.sha256_file(Path(p))


def pins_path() -> Path:
    return REPO / PINS


def pins_sha256() -> str:
    return file_sha(pins_path())


def freeze_record_path() -> Path:
    return REPO / FREEZE_RECORD


def _data_paths() -> dict[str, Path]:
    """Every data file whose content decides which rows come back (G-7)."""
    return {"split": Path(A.SPLIT_PATH), "ecotox": Path(A.ECOTOX),
            "prospective": Path(A.PROSPECTIVE), "adore_splits": Path(A.ADORE_SPLITS),
            "h4_results": Path(H4_RESULTS), "h4_tests": Path(H4_TESTS),
            "labels": Path(LABELS_PATH)}


def EXPECTED_DATA_KEYS() -> tuple[str, ...]:
    """The data paths the pins must carry, as strings, in a stable order."""
    return tuple(str(p) for p in _data_paths().values())


def _check_block(block: str) -> None:
    if block not in BLOCKS:
        raise ProtectedBlockError(f"unknown block {block!r}; expected one of {BLOCKS}")


def block_dir(block: str) -> Path:
    _check_block(block)
    d = RUNS / block
    d.mkdir(parents=True, exist_ok=True)
    return d


def freeze_path(block: str) -> Path:
    """The lead-written unlock file for one block. Never written by this module."""
    _check_block(block)
    return RUNS / block / "FREEZE"


def _marker(block: str, name: str) -> Path:
    return block_dir(block) / name


def _is_unwritable(p: Path) -> bool:
    return not (stat.S_IMODE(p.stat().st_mode) & 0o222)


# ============================================================ B1 (i) / C-1  the FREEZE two-source check

def _read_freeze_record() -> dict:
    p = freeze_record_path()
    if not p.is_file():
        raise ProtectedBlockError(
            f"no lead-written FREEZE record at {p}. Under C-1 the guard checks sha256(FREEZE) "
            f"against a value the LEAD recorded in this 0444 file and in a registry row; the "
            f"workstream writes neither. No look may proceed without it.")
    if not _is_unwritable(p):
        raise ProtectedBlockError(
            f"{p} is writable (mode {oct(stat.S_IMODE(p.stat().st_mode))}); the lead's FREEZE "
            f"record must be 0444")
    try:
        rec = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise ProtectedBlockError(f"{p} is not valid JSON: {exc}") from exc
    if not isinstance(rec, dict) or not isinstance(rec.get("blocks"), dict):
        raise ProtectedBlockError(f"{p} must be a JSON object with a 'blocks' object")
    return rec


def _registry_rows(experiment: str, *, needle: str | None = None):
    """Yield parsed registry rows of one experiment. The registry path is a module constant."""
    if not Path(REGISTRY).is_file():
        return
    with open(REGISTRY) as fh:
        for line in fh:
            if experiment not in line:
                continue
            if needle is not None and needle not in line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("experiment") == experiment:
                yield row


def _registry_freeze_sha(block: str) -> str:
    _check_block(block)
    found: list[str] = []
    for row in _registry_rows(PREREG_EXPERIMENT, needle="written_by"):
        extra = row.get("extra") or {}
        if extra.get("written_by") != "lead" or str(extra.get("block")) != block:
            continue
        sha = extra.get("freeze_sha256")
        if isinstance(sha, str) and len(sha) == 64:
            found.append(sha)
    if not found:
        raise ProtectedBlockError(
            f"no registry row of experiment {PREREG_EXPERIMENT!r} with extra.written_by == 'lead', "
            f"extra.block == {block!r} and a 64-character extra.freeze_sha256. Under C-1 the "
            f"registry row is the second, independent source for the FREEZE hash and it is "
            f"missing, so the look is refused.")
    if len(set(found)) > 1:
        raise ProtectedBlockError(
            f"{len(set(found))} different FREEZE sha256 values recorded in the registry for "
            f"{block}: {sorted(set(found))}. The look is refused until the lead resolves it.")
    return found[0]


def lead_recorded_freeze_sha(block: str) -> str:
    """The lead's recorded FREEZE sha256, from BOTH sources, which must agree."""
    _check_block(block)
    rec = _read_freeze_record()
    in_file = (rec["blocks"].get(block) or {}).get("freeze_sha256")
    if not isinstance(in_file, str) or len(in_file) != 64:
        raise ProtectedBlockError(
            f"{freeze_record_path()} records no 64-character freeze_sha256 for {block}")
    in_registry = _registry_freeze_sha(block)
    if in_file != in_registry:
        raise ProtectedBlockError(
            f"the lead's two records disagree for {block}: file {in_file[:16]}, registry "
            f"{in_registry[:16]}. The look is refused until the lead resolves it.")
    return in_file


def freeze_sha(block: str) -> str:
    """sha256 of the block's FREEZE file, checked against both of the lead's records (B1 (i))."""
    f = freeze_path(block)
    if not f.is_file():
        raise ProtectedBlockError(
            f"no lead-written FREEZE file at {f}. The test look is unlocked by the LEAD, not by "
            f"this workstream: no candidate-written file substitutes for it.")
    got = file_sha(f)
    want = lead_recorded_freeze_sha(block)
    if got != want:
        raise ProtectedBlockError(
            f"{f} has sha256 {got[:16]} but the lead recorded {want[:16]}. A file named FREEZE "
            f"that the lead did not write is VOID, and a look taken on one voids the block (B1).")
    return got


# ==================================================================== G-1  pins and verification

def make_pins() -> dict:
    """Compute the pins. Written by the LEAD, never by this workstream (C-1).

    Records the resolved runs root (G-5) and every data path of G-7. The labels entry is ``null``
    until the artefact is built, which is the only null :func:`verify_freeze` tolerates.
    """
    data = {}
    for name, p in _data_paths().items():
        data[str(p)] = file_sha(p) if (name != "labels" or p.exists()) else None
    return {"files": {f: file_sha(REPO / f) for f in PINNED_REPO_FILES},
            "data": data,
            "experiment": EXPERIMENT, "claim_experiment": CLAIM_EXPERIMENT,
            "blocks": list(BLOCKS), "split_columns": list(A.SPLIT_COLUMNS),
            "holdout_unit_share": A.HOLDOUT_UNIT_SHARE, "split_seed": A.SPLIT_SEED,
            "freeze_record": FREEZE_RECORD, "prereg_experiment": PREREG_EXPERIMENT,
            "runs_root": str(Path(RUNS).resolve()),
            "seven_exclusions": list(SEVEN_EXCLUSIONS),
            "aux_targets": dict(AUX_TARGETS)}


_HEX64 = re.compile(r"^[0-9a-f]{64}$")

_PINS_SCALARS = ("experiment", "claim_experiment", "blocks", "split_columns",
                 "holdout_unit_share", "split_seed", "freeze_record", "prereg_experiment",
                 "runs_root", "seven_exclusions", "aux_targets")


def _lead_vouched_pins(sha: str) -> dict:
    """G-1 (b): a lead-written ``chemtox/prereg_snapshot`` row carrying this pins sha256."""
    for row in _registry_rows(PREREG_EXPERIMENT, needle="pins_sha256"):
        extra = row.get("extra") or {}
        if extra.get("written_by") != "lead":
            continue
        if extra.get("pins_sha256") == sha:
            return {"row_id": row.get("id"), "timestamp": row.get("timestamp"),
                    "workstream": row.get("workstream")}
    raise ProtectedBlockError(
        f"no registry row of experiment {PREREG_EXPERIMENT!r} with extra.written_by == 'lead' and "
        f"extra.pins_sha256 == {sha[:16]}…. The pins file is only a gate when the LEAD vouches for "
        f"it: a file any workstream can write is not a gate (G-1). The look and every read are "
        f"refused.")


def verify_freeze(*, require_labels: bool = False) -> dict:
    """The gate. Requires the lead, then checks every pinned hash (G-1).

    Returns the pins object with ``pins_sha256`` added, which every reader records in its report.
    """
    p = pins_path()
    if not p.is_file():
        raise ProtectedBlockError(f"no pins file at {p}: the amendment has not been stamped")
    # (a) the pins must not be writable
    if not _is_unwritable(p):
        raise ProtectedBlockError(
            f"{p} is writable (mode {oct(stat.S_IMODE(p.stat().st_mode))}): the lead writes the "
            f"pins file 0444 (C-2), so a writable pins file is not the lead's (G-1 a)")
    sha = file_sha(p)
    try:
        pinned = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise ProtectedBlockError(f"{p} is not valid JSON: {exc}") from exc
    if not isinstance(pinned, dict):
        raise ProtectedBlockError(f"{p} must be a JSON object")

    # (b) the lead must vouch for this exact file
    vouch = _lead_vouched_pins(sha)

    # (c) every pinned key present, each with a 64-hex value
    files = pinned.get("files")
    data = pinned.get("data")
    if not isinstance(files, dict) or not isinstance(data, dict):
        raise ProtectedBlockError(f"{p} must hold 'files' and 'data' objects")
    missing_files = [f for f in PINNED_REPO_FILES if f not in files]
    if missing_files:
        raise ProtectedBlockError(
            f"the pins omit {len(missing_files)} pinned repo file(s), first {missing_files[:3]} "
            f"(G-1 c)")
    expected_data = EXPECTED_DATA_KEYS()
    missing_data = [d for d in expected_data if d not in data]
    if missing_data:
        raise ProtectedBlockError(f"the pins omit pinned data path(s) {missing_data} (G-1 c)")

    # (d) no unknown entry, no null but the labels path
    unknown_files = sorted(set(files) - set(PINNED_REPO_FILES))
    unknown_data = sorted(set(data) - set(expected_data))
    if unknown_files or unknown_data:
        raise ProtectedBlockError(
            f"the pins hold unknown entries: files {unknown_files}, data {unknown_data} (G-1 d)")
    unknown_keys = sorted(set(pinned) - {"files", "data"} - set(_PINS_SCALARS))
    if unknown_keys:
        raise ProtectedBlockError(f"the pins hold unknown top-level keys {unknown_keys} (G-1 d)")
    labels_key = str(Path(LABELS_PATH))
    for key, want in list(files.items()) + list(data.items()):
        if want is None:
            if key == labels_key and not require_labels:
                continue
            raise ProtectedBlockError(
                f"the pins hold a null value for {key!r}; only the protected-labels path may be "
                f"null, and only before the first look (G-1 d, G-6)")
        if not (isinstance(want, str) and _HEX64.match(want)):
            raise ProtectedBlockError(f"the pins value for {key!r} is not a 64-hex sha256 (G-1 c)")

    # G-5: the runs root is pinned
    want_runs = pinned.get("runs_root")
    got_runs = str(Path(RUNS).resolve())
    if want_runs != got_runs:
        raise ProtectedBlockError(
            f"the pins fix the runs root at {want_runs!r} but this process resolves it to "
            f"{got_runs!r}: the single-look markers live there, so a different root would allow a "
            f"second look (G-5)")

    # the hashes themselves
    for rel, want in files.items():
        got = file_sha(REPO / rel)
        if got != want:
            raise ProtectedBlockError(f"{rel}: pinned {want[:16]}, on disk {got[:16]}")
    for abs_path, want in data.items():
        if want is None:
            continue
        got = file_sha(abs_path)
        if got != want:
            raise ProtectedBlockError(f"{abs_path}: pinned {want[:16]}, on disk {got[:16]}")

    out = dict(pinned)
    out["pins_sha256"] = sha
    out["lead_row"] = vouch
    return out


# ============================================================================= markers, inventory

def _excl_write(p: Path, rec: dict) -> None:
    """Create-or-die: the O_EXCL write is what makes a second look impossible."""
    try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise ProtectedBlockError(
            f"{p.name} already exists for this block: it has been looked at once already and a "
            f"second look is refused") from exc
    with os.fdopen(fd, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
        fh.write("\n")


def _written_inventory(block: str) -> dict:
    """B1 (iv): what was ACTUALLY written in this block's directory, not a constant."""
    d = block_dir(block)
    files = []
    for f in sorted(d.rglob("*")):
        if f.is_file():
            try:
                files.append({"name": str(f.relative_to(d)), "bytes": f.stat().st_size,
                              "sha256": file_sha(f)})
            except OSError as exc:                                  # pragma: no cover
                files.append({"name": str(f.relative_to(d)), "error": str(exc)})
    markers = {m: (d / m).is_file() for m in _MARKERS}
    return {"files": files, "n_files": len(files), "markers": markers,
            "authorised": markers["AUTHORISED.json"],
            "predictions_written": (d / "predictions.parquet").is_file(),
            "scored": markers["SCORED.json"],
            "look_spent": markers["AUTHORISED.json"],
            "look_spent_reason": (
                "AUTHORISED.json exists, so this block's single look is spent whatever happened "
                "afterwards (B1): there is no second look."
                if markers["AUTHORISED.json"] else
                "AUTHORISED.json does not exist, so no look was claimed for this block.")}


def look_state(block: str) -> dict:
    """Read-only status of a block. Safe at any time; reads no row."""
    _check_block(block)
    d = RUNS / block
    return {"block": block, "block_dir": str(d), "exists": d.is_dir(),
            "freeze_file": (d / "FREEZE").is_file(),
            "markers": {m: (d / m).is_file() for m in _MARKERS},
            "registry_rows": registry_rows(block),
            "registry_claim": _claim_rows(block),
            "labels_artefact": Path(LABELS_PATH).exists()}


def registry_rows(block: str) -> int:
    """How many result rows the registry already carries for this block."""
    _check_block(block)
    n = 0
    for row in _registry_rows(EXPERIMENT):
        if str(row.get("split", "")).startswith(block):
            n += 1
    return n


def _claim_rows(block: str) -> int:
    """G-5: how many look claims the registry carries for this block, from ANY runs root."""
    _check_block(block)
    n = 0
    for row in _registry_rows(CLAIM_EXPERIMENT):
        extra = row.get("extra") or {}
        if str(extra.get("block")) == block:
            n += 1
    return n


def _write_claim_row(rec: dict) -> dict:
    """G-5: append the look claim to the pinned registry path.

    Written directly rather than through ``sentinel.core.registry`` for two reasons: the claim must
    land in the registry path this module pins (so a test can redirect it, and so the guard reads
    back exactly what it wrote), and it must not drag the package-hash machinery into the middle of
    a look. It is **not** a row of ``chemtox/prereg_snapshot``: M4 writes none of those (C-1).
    """
    row = {"experiment": CLAIM_EXPERIMENT, "workstream": "chemtox",
           "model": "arch_v2_guard", "metric": "look_claimed", "value": 1.0,
           "split": f"{rec['block']}/claim",
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "notes": "single-look claim; survives a different runs root (G-5)",
           "extra": dict(rec)}
    Path(REGISTRY).parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY, "a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def import_closure_sha() -> dict:
    """sha256 over every repo module currently imported, plus the combined hash."""
    per = {}
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        p = Path(f).resolve()
        try:
            rel = p.relative_to(_CODE_ROOT)
        except ValueError:
            continue
        if p.suffix == ".py" and p.is_file():
            per[str(rel)] = file_sha(p)
    h = hashlib.sha256()
    for k in sorted(per):
        h.update(k.encode())
        h.update(per[k].encode())
    return {"modules": per, "n_modules": len(per), "code_sha": h.hexdigest()}


# ========================================================================= G-6  protected labels

def build_protected_labels(*, block: str) -> Path:
    """Materialise the sealed blocks' targets, 0444, write-once and hash-recorded (G-6).

    Refuses a pre-existing artefact that this guard did not write. A legitimate second or third
    look (the artefact covers all three blocks) is recognised by the write-once marker whose
    recorded sha256 equals the file's.
    """
    freeze_sha(block)                      # two-source check; refuses without the lead's records
    p = Path(LABELS_PATH)
    if p.exists():
        if not LABELS_MARKER.is_file():
            raise ProtectedBlockError(
                f"{p} already exists but {LABELS_MARKER} does not: this guard did not write it, so "
                f"it is refused. A planted protected-labels file must never be served (G-6).")
        rec = json.loads(LABELS_MARKER.read_text())
        got = file_sha(p)
        if rec.get("sha256") != got:
            raise ProtectedBlockError(
                f"{p} has sha256 {got[:16]} but the write-once marker records "
                f"{str(rec.get('sha256'))[:16]}: the artefact was changed after it was built (G-6)")
        return p
    split = pd.read_parquet(A.SPLIT_PATH, columns=["result_id", "block"])
    sealed = set(split.loc[split["block"].isin(BLOCKS), "result_id"].astype(str))
    e = pd.read_parquet(A.ECOTOX, columns=["result_id", TARGET])
    lab = e[e["result_id"].astype(str).isin(sealed)][["result_id", TARGET]].copy()
    missing = len(sealed) - len(lab)
    if missing:
        raise ProtectedBlockError(f"{missing} sealed rows have no target in {A.ECOTOX}")
    p.parent.mkdir(parents=True, exist_ok=True)
    lab.sort_values("result_id").to_parquet(p, index=False)
    os.chmod(p, 0o444)
    RUNS.mkdir(parents=True, exist_ok=True)
    _excl_write(LABELS_MARKER, {"path": str(p), "sha256": file_sha(p), "rows": int(len(lab)),
                                "built_for_block": block, "pid": os.getpid(),
                                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    return p


def _verify_protected_labels() -> dict:
    """G-6: the artefact must exist, be recorded, and still have its recorded sha256."""
    p = Path(LABELS_PATH)
    if not p.is_file():
        raise ProtectedBlockError(f"no protected-labels artefact at {p}")
    if not LABELS_MARKER.is_file():
        raise ProtectedBlockError(
            f"{p} exists but {LABELS_MARKER} does not: an unrecorded artefact is refused (G-6)")
    rec = json.loads(LABELS_MARKER.read_text())
    got = file_sha(p)
    if rec.get("sha256") != got:
        raise ProtectedBlockError(
            f"{p} has sha256 {got[:16]} but the marker records {str(rec.get('sha256'))[:16]}: the "
            f"protected labels were swapped after they were built (G-6)")
    return {"path": str(p), "sha256": got, "rows": rec.get("rows")}


# ============================================================================= authorisation

def authorise(block: str, *, note: str = "") -> dict:
    """The ONLY entry to a look. Pins verified, FREEZE verified, registry clean, claim recorded."""
    _check_block(block)
    pinned = verify_freeze()
    n = registry_rows(block)
    if n:
        raise ProtectedBlockError(
            f"{block} already carries {n} rows under experiment {EXPERIMENT}: refusing a second "
            f"look (sota_campaign.md rule 7)")
    claims = _claim_rows(block)
    if claims:
        raise ProtectedBlockError(
            f"the registry already carries {claims} look claim(s) for {block} under "
            f"{CLAIM_EXPERIMENT!r}: its single look is spent, and this holds from any runs root "
            f"(G-5). There is no second look.")
    for m in ("SCORED.json", "ABORTED.json"):
        if _marker(block, m).is_file():
            raise ProtectedBlockError(
                f"{block} already has {m}: its single look is spent and there is no second look")
    fsha = freeze_sha(block)
    closure = import_closure_sha()
    labels = build_protected_labels(block=block)          # before AUTHORISED.json, so its sha fits
    labels_rec = _verify_protected_labels()
    rec = {"block": block, "pid": os.getpid(), "host": socket.gethostname(),
           "freeze_sha256": fsha,
           "freeze_sha256_lead_record": lead_recorded_freeze_sha(block),
           "freeze_record_file": FREEZE_RECORD,
           "freeze_record_file_sha256": file_sha(freeze_record_path()),
           "code_sha": closure["code_sha"], "n_modules_hashed": closure["n_modules"],
           "pins_sha256": pinned["pins_sha256"], "lead_row": pinned["lead_row"],
           "runs_root": pinned["runs_root"],
           "protected_labels": labels_rec,
           "input_hashes": {k: file_sha(v) for k, v in _data_paths().items()
                            if Path(v).exists()},
           "note": note}
    # the registry claim goes in BEFORE the local marker, so a crash between the two still blocks
    # a second look from any runs root
    claim_row = _write_claim_row({k: rec[k] for k in
                                  ("block", "pid", "host", "freeze_sha256", "code_sha",
                                   "pins_sha256", "runs_root")})
    rec["claim_row_split"] = claim_row["split"]
    _excl_write(_marker(block, "AUTHORISED.json"), rec)
    return rec


def require_authorised(block: str) -> dict:
    """B1 (ii): re-verify the freeze on every use, then require the authorisation marker."""
    _check_block(block)
    verify_freeze()
    p = _marker(block, "AUTHORISED.json")
    if not p.is_file():
        raise ProtectedBlockError(
            f"{block} is not authorised: call authorise({block!r}) once, after the lead has "
            f"written {freeze_path(block)} and recorded its sha256")
    return json.loads(p.read_text())


# ================================================================================== the readers

def _attach_split(frame: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    """Add the split's structure columns, keeping the FRAME's value for any shared name.

    The split and ``ecotox_aquatic.parquet`` both carry ``test_cas`` and ``duration_h``; a plain
    merge renamed them to ``*_x`` / ``*_y`` and silently broke every caller that asked for
    ``test_cas`` by name. Only the split-only columns are added, so the returned frame keeps the
    pinned frame's own column names.
    """
    add = [c for c in split.columns if c == "result_id" or c not in frame.columns]
    out = frame.merge(split[add], on="result_id", how="left")
    if len(out) != len(frame):
        raise ProtectedBlockError(
            f"attaching the split changed the row count {len(frame)} -> {len(out)}")
    return out


def load_training_pool(*, columns=None) -> pd.DataFrame:
    """Pool rows only. The loader every T1/T6 fit uses; it cannot return a sealed row."""
    pinned = verify_freeze()
    split = pd.read_parquet(A.SPLIT_PATH)
    pool_ids = set(split.loc[split["block"] == "train_pool", "result_id"].astype(str))
    want = None if columns is None else sorted({*columns, "result_id"})
    e = pd.read_parquet(A.ECOTOX, columns=want)
    out = e[e["result_id"].astype(str).isin(pool_ids)].copy()
    sealed = set(split.loc[split["block"] != "train_pool", "result_id"].astype(str))
    if out["result_id"].astype(str).isin(sealed).any():
        raise ProtectedBlockError("a sealed row reached the training-pool loader")
    out = _attach_split(out, split)
    out.attrs["pins_sha256"] = pinned["pins_sha256"]
    return out


def load_test_block(block: str, *, columns=None) -> pd.DataFrame:
    """The ONLY sanctioned reader of a sealed block. Refuses after the look is over (B1 (ii))."""
    require_authorised(block)
    for m in ("SCORED.json", "ABORTED.json"):
        if _marker(block, m).is_file():
            raise ProtectedBlockError(
                f"{block} has {m}: its look is over and the block may not be read again (B1 (ii))")
    labels_rec = _verify_protected_labels()
    claim = json.loads(_marker(block, "AUTHORISED.json").read_text())
    recorded = (claim.get("protected_labels") or {}).get("sha256")
    if recorded and recorded != labels_rec["sha256"]:
        raise ProtectedBlockError(
            f"the protected labels now hash {labels_rec['sha256'][:16]} but the claim recorded "
            f"{recorded[:16]}: they were swapped after the look was authorised (G-6)")
    split = pd.read_parquet(A.SPLIT_PATH)
    ids = set(split.loc[split["block"] == block, "result_id"].astype(str))
    want = None if columns is None else sorted({*columns, "result_id"})
    e = pd.read_parquet(A.ECOTOX, columns=want)
    rows = e[e["result_id"].astype(str).isin(ids)].copy()
    lab = pd.read_parquet(LABELS_PATH)
    # G-6: the artefact's target must equal the pinned frame's target for every returned id
    ref = pd.read_parquet(A.ECOTOX, columns=["result_id", TARGET])
    ref = ref[ref["result_id"].astype(str).isin(ids)]
    chk = (ref.assign(_k=ref["result_id"].astype(str))
           .merge(lab.assign(_k=lab["result_id"].astype(str))[["_k", TARGET]], on="_k",
                  suffixes=("_pinned", "_artefact")))
    if len(chk) != len(ref):
        raise ProtectedBlockError(
            f"the protected-labels artefact covers {len(chk)} of {len(ref)} returned rows (G-6)")
    bad = int((chk[f"{TARGET}_pinned"] - chk[f"{TARGET}_artefact"]).abs().gt(0).sum())
    if bad:
        raise ProtectedBlockError(
            f"{bad} of {len(chk)} returned rows have a protected-labels target that differs from "
            f"the pinned frame's target: the artefact is not the sealed truth (G-6)")
    rows = rows.merge(lab, on="result_id", how="left", suffixes=("", "_protected"))
    out = _attach_split(rows, split[split["block"] == block])
    out.attrs["protected_labels_sha256"] = labels_rec["sha256"]
    return out


# ================================================= G-3 / G-9  the seven exclusions and the re-check

def _norm_cas_key(x) -> str:
    """A normalising CAS key: ``' 400'``, ``'400.0'``, ``'0400'`` and ``'40-0'`` all give ``400``.

    The re-check uses this where the frozen loader uses a bare ``astype(str)``, which is how
    probe P4b slipped planted chemistry past the old check (F9).
    """
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = re.sub(r"\D", "", s.replace("-", ""))
    return s.lstrip("0")


def _sealed_context() -> dict:
    """Keys of the sealed blocks, from the pinned split and cluster table. Reads no target."""
    split = pd.read_parquet(A.SPLIT_PATH)
    sealed = split[split["block"].isin(BLOCKS)]
    held_units = set(split.loc[split["block"].isin(("T1_test", "T6b_test")),
                               "holdout_unit"].astype(int))
    table = pd.read_csv(A.CLUSTER_PATH)
    raw = (table.assign(_k=table["test_cas"].astype(str))
           .drop_duplicates("_k").set_index("_k")["holdout_unit"].astype(int).to_dict())
    norm = {}
    for k, v in zip(table["test_cas"].astype(str), table["holdout_unit"].astype(int)):
        norm.setdefault(_norm_cas_key(k), v)
    return {"split": split, "sealed_result_ids": set(sealed["result_id"].astype(str)),
            "held_units": held_units, "cas_unit_raw": raw, "cas_unit_norm": norm,
            "cluster_table": table}


def _spent_sets() -> dict:
    """The two spent-id lists, with the registry-derived ADORE set checked against the pinned one."""
    spent_pro = set(pd.read_parquet(A.PROSPECTIVE, columns=["result_id"])["result_id"]
                    .dropna().astype(str))
    derived, _rep = A.spent_adore_result_ids(A.REGISTRY, A.ADORE_SPLITS)
    frozen = set(json.loads(Path(A.SPENT_IDS_PATH).read_text())["result_ids"])
    if derived != frozen:
        raise ProtectedBlockError(
            f"the spent-ADORE set derived from the registry has {len(derived)} ids but the pinned "
            f"list has {len(frozen)} ({len(derived ^ frozen)} differ): a partition was scored since "
            f"the freeze, or the list is stale. Every read is refused until the lead rules (G-7).")
    return {"prospective": spent_pro, "adore": frozen}


def _sealed_context_for_recheck() -> dict:
    """The re-check's OWN derivation of the sealed keys (G-9).

    Deliberately NOT a call to :func:`_sealed_context`: if the re-check shared the exclusions'
    derivation then one broken derivation would blind both, which is exactly what the review's
    mutations M10 and M12 exploited. The duplication is the control.
    """
    split = pd.read_parquet(A.SPLIT_PATH, columns=["result_id", "block", "holdout_unit"])
    held: set[int] = set()
    for b in ("T1_test", "T6b_test"):
        held |= set(split.loc[split["block"] == b, "holdout_unit"].astype(int))
    sealed = set()
    for b in BLOCKS:
        sealed |= set(split.loc[split["block"] == b, "result_id"].astype(str))
    tbl = pd.read_csv(A.CLUSTER_PATH, usecols=["test_cas", "holdout_unit"])
    raw, norm = {}, {}
    for k, v in zip(tbl["test_cas"].astype(str), tbl["holdout_unit"].astype(int)):
        raw.setdefault(k, v)
        norm.setdefault(_norm_cas_key(k), v)
    return {"sealed_result_ids": sealed, "held_units": held,
            "cas_unit_raw": raw, "cas_unit_norm": norm}


def _spent_sets_for_recheck() -> dict:
    """The re-check's OWN derivation of the two spent-id lists (G-9), for the same reason."""
    pro = set(pd.read_parquet(A.PROSPECTIVE, columns=["result_id"])["result_id"]
              .dropna().astype(str))
    adore = set(json.loads(Path(A.SPENT_IDS_PATH).read_text())["result_ids"])
    return {"prospective": pro, "adore": adore}


def _sealed_cells(ctx: dict) -> set[tuple[str, str, str]]:
    """(normalised CAS, species, duration) cells occupied by a sealed row. No target is read."""
    e = pd.read_parquet(A.ECOTOX, columns=["result_id", "test_cas", "latin_name", "duration_h"])
    prim = e[e["result_id"].astype(str).isin(ctx["sealed_result_ids"])]
    return {(_norm_cas_key(c), str(s), str(d))
            for c, s, d in zip(prim["test_cas"], prim["latin_name"], prim["duration_h"])}


def _sealed_pairs(ctx: dict) -> set[tuple[str, str]]:
    """(normalised CAS, species) pairs of sealed rows: the H4 cell rule (see :func:`load_h4_rows`)."""
    return {(c, s) for c, s, _d in _sealed_cells(ctx)}


def _recheck_seven(frame: pd.DataFrame, *, what: str, track: str,
                   cas_col: str, species_col: str, duration_col: str | None,
                   result_id_col: str = "result_id", class_col: str | None = "class",
                   cell_rule: str = "triple") -> dict:
    """G-9: re-check ALL SEVEN exclusions, fail-closed, in code separate from the exclusions.

    **It takes no context from the caller.** Every set is rebuilt here from the pinned files, so a
    broken exclusion step — one working from a context that is missing a held unit, a sealed id or a
    spent id — cannot also blind the re-check. Keys are compared with :func:`_norm_cas_key`, which
    is stricter than the loaders' bare ``astype(str)``. Any hit raises; the message carries counts
    only.
    """
    ctx = _sealed_context_for_recheck()
    spent = _spent_sets_for_recheck()
    out = {"rows": int(len(frame)), "cell_rule": cell_rule, "track": track}
    if not len(frame):
        return {**out, **{k: 0 for k in SEVEN_EXCLUSIONS}}

    def fail(key: str, n: int, detail: str) -> None:
        raise ProtectedBlockError(f"{what} ({track}): re-check {key} found {n} row(s) — {detail}")

    # 1 T2 reserved classes
    n = 0
    if class_col and class_col in frame.columns:
        n = int(frame[class_col].astype("string").isin(A.T2_RESERVED_CLASSES).sum())
    out["t2_reserved_classes"] = n
    if n:
        fail("t2_reserved_classes", n, "a T2 reserved class survived")

    # 2, 3 the two spent-id lists
    ids = frame[result_id_col].astype(str)
    n = int(ids.isin(spent["prospective"]).sum())
    out["spent_prospective"] = n
    if n:
        fail("spent_prospective", n, "a spent prospective result_id survived")
    n = int(ids.isin(spent["adore"]).sum())
    out["spent_adore"] = n
    if n:
        fail("spent_adore", n, "a spent ADORE result_id survived")

    # sealed result_ids: not one of the seven, but the most direct leak there is
    n = int(ids.isin(ctx["sealed_result_ids"]).sum())
    out["sealed_result_ids"] = n
    if n:
        fail("sealed_result_ids", n, "a sealed block's result_id survived")

    # 4, 5 the hold-out unit, with normalised keys and FAIL-CLOSED on unmapped chemistry
    keys = [_norm_cas_key(c) for c in frame[cas_col]]
    mapped = [ctx["cas_unit_norm"].get(k, ctx["cas_unit_raw"].get(str(raw)))
              for k, raw in zip(keys, frame[cas_col])]
    n = sum(1 for u in mapped if u is not None and int(u) in ctx["held_units"])
    out["held_out_unit_chemistry"] = n
    if n:
        fail("held_out_unit_chemistry", n, "chemistry in a held-out hold-out unit survived")
    n = sum(1 for u in mapped if u is None)
    out["chemistry_not_in_split"] = n
    if n:
        fail("chemistry_not_in_split", n,
             "chemistry that maps to no hold-out unit survived; whether it touches the hold-out "
             "is UNKNOWN, so the re-check fails closed (G-9)")

    # 6 the sealed cell
    if cell_rule == "triple":
        if duration_col is None or duration_col not in frame.columns:
            fail("same_cell_as_sealed_row", len(frame),
                 "the triple cell rule needs a duration column and none is present")
        cells = _sealed_cells(ctx)
        hit = [1 for c, s, d in zip(frame[cas_col], frame[species_col], frame[duration_col])
               if (_norm_cas_key(c), str(s), str(d)) in cells]
    else:
        pairs = _sealed_pairs(ctx)
        hit = [1 for c, s in zip(frame[cas_col], frame[species_col])
               if (_norm_cas_key(c), str(s)) in pairs]
    n = len(hit)
    out["same_cell_as_sealed_row"] = n
    if n:
        fail("same_cell_as_sealed_row", n, "a row sharing a sealed cell survived")

    # 7 the T6 window
    n = 0
    if track == "T6":
        if duration_col is None or duration_col not in frame.columns:
            fail("outside_24_96h_for_T6", len(frame),
                 "the T6 window re-check needs a duration column and none is present")
        d = pd.to_numeric(frame[duration_col], errors="coerce")
        n = int((~d.between(*A.DURATION_WINDOW).fillna(False)).sum())
    out["outside_24_96h_for_T6"] = n
    if n:
        fail("outside_24_96h_for_T6", n, "a row outside 24-96 h survived on T6")
    return out


def _write_report(name: str, rec: dict, pinned: dict) -> Path:
    """G-8: exclusive creation with a counter, 0444, under a fixed directory of the runs root."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    body = dict(rec)
    body.update({"pins_sha256": pinned["pins_sha256"],
                 "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pid": os.getpid(),
                 "host": socket.gethostname(),
                 "input_hashes": {k: file_sha(v) for k, v in _data_paths().items()
                                  if Path(v).exists()}})
    k = 0
    while True:
        p = REPORT_DIR / (f"{name}.json" if k == 0 else f"{name}.{k}.json")
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        except FileExistsError:
            k += 1
            if k > 9999:                                            # pragma: no cover
                raise ProtectedBlockError(f"too many reports named {name} in {REPORT_DIR}")
            continue
        with os.fdopen(fd, "w") as fh:
            json.dump(body, fh, indent=1, sort_keys=True, default=str)
            fh.write("\n")
        return p


def _no_caller_paths(kwargs: dict) -> None:
    """A caller may not substitute any input path or frame (addendum 01 rev 1 §3)."""
    forbidden = sorted(k for k in kwargs if k in {
        "ecotox", "prospective", "registry", "splits", "cluster_table", "split",
        "results", "tests", "h4_results", "h4_tests", "out", "output", "root"})
    if forbidden:
        raise ProtectedBlockError(
            f"caller-supplied inputs {forbidden} are refused: the guarded readers fix every path "
            f"to the pinned constants (addendum 01 rev 1 §3)")


def load_auxiliary_rows(target: str, track: str, **kwargs) -> tuple[pd.DataFrame, dict]:
    """The only sanctioned route to an H2/H3 auxiliary row set, **with its target** (G-4).

    Calls the frozen :func:`arch_v2.auxiliary_rows` with every path fixed to the pinned constants,
    re-checks all seven exclusions in separate code (G-9), then joins ``log10_conc_mol_L`` from the
    pinned frame for exactly the surviving ``result_id``s. Returns ``(rows, report)``.
    """
    _no_caller_paths(kwargs)
    if kwargs:
        raise ProtectedBlockError(f"unexpected arguments {sorted(kwargs)}")
    if target not in ("H2", "H3"):
        raise ProtectedBlockError(
            f"load_auxiliary_rows handles H2 and H3; H4 has its own reader and H5 comes from "
            f"load_h5_pool_pairs (G-4). Got {target!r}")
    pinned = verify_freeze()
    ctx = _sealed_context()
    spent = _spent_sets()
    aux, rep = A.auxiliary_rows(target, ctx["split"], track=track,
                               ecotox=A.ECOTOX, prospective=A.PROSPECTIVE,
                               registry=A.REGISTRY, splits=A.ADORE_SPLITS,
                               cluster_table=ctx["cluster_table"])
    rep = dict(rep)
    rep["guarded_by"] = "arch_v2_guard.load_auxiliary_rows"
    rep["recheck"] = _recheck_seven(aux, what=f"auxiliary {target}", track=track,
                                    cas_col="test_cas", species_col="latin_name",
                                    duration_col="duration_h", class_col="class",
                                    cell_rule="triple")
    # G-4: the guard joins the target itself, for exactly the returned ids, after the re-checks
    want = AUX_TARGETS[target]
    lab = pd.read_parquet(A.ECOTOX, columns=["result_id", want])
    keep = set(aux["result_id"].astype(str))
    lab = lab[lab["result_id"].astype(str).isin(keep)]
    before = len(aux)
    aux = aux.merge(lab, on="result_id", how="left")
    if len(aux) != before:
        raise ProtectedBlockError(f"the target join changed the row count {before} -> {len(aux)}")
    n_missing = int(aux[want].isna().sum())
    if n_missing:
        raise ProtectedBlockError(
            f"{n_missing} returned {target} row(s) have no {want} in the pinned frame (G-4)")
    rep["target_column"] = want
    rep["targets_refused"] = sorted(set(AUX_TARGETS.values()) - {want})
    rep["columns_returned"] = sorted(aux.columns)
    rep["report_path"] = str(_write_report(f"aux_{target}_{track}", rep, pinned))
    return aux, rep


def _h4_window_for_exclusion() -> tuple[float, float]:
    """The window the H4 EXCLUSION step uses. Named so a test can blind the step alone and prove
    the re-check (which reads :data:`arch_v2.DURATION_WINDOW` afresh) still refuses."""
    return tuple(A.DURATION_WINDOW)


def load_h4_rows(track: str, **kwargs) -> tuple[pd.DataFrame, dict]:
    """The only sanctioned route to the H4 (BCF-family) frame, with all seven exclusions (G-3).

    Owns every read — the two raw release tables and ``ecotox_aquatic.parquet`` — and delegates the
    join and QC rules to the pure module :mod:`sentinel.models.chemtox.arch_v2_h4`.

    The sealed-cell rule here is the **(CAS, species) pair**, not the (CAS, species, duration)
    triple: H4's ``obs_duration_mean`` is an observation duration, not the exposure duration the
    sealed rows carry, so the triple would compare incomparable quantities. The pair is a strict
    superset of the triple, so it removes **more** rows, and the choice is recorded in the report
    and in addendum 01 rev 1.
    """
    from sentinel.models.chemtox import arch_v2_h4 as H4

    _no_caller_paths(kwargs)
    if kwargs:
        raise ProtectedBlockError(f"unexpected arguments {sorted(kwargs)}")
    if track not in ("T1", "T6"):
        raise ProtectedBlockError(f"track must be 'T1' or 'T6', got {track!r}")
    pinned = verify_freeze()
    ctx = _sealed_context()
    spent = _spent_sets()

    results = pd.read_parquet(H4_RESULTS, columns=_H4_RESULT_COLUMNS)
    tests = pd.read_parquet(H4_TESTS, columns=["test_id", "test_cas", "species_number"])
    aq = pd.read_parquet(A.ECOTOX, columns=["test_cas", "chem_smiles", "latin_name",
                                            "species_number", "class"])
    smiles_by_cas = {H4.norm_cas(k): v for k, v in
                     aq.dropna(subset=["chem_smiles"]).drop_duplicates("test_cas")
                     .set_index("test_cas")["chem_smiles"].items()}
    species_by_number = (aq.dropna(subset=["latin_name"]).drop_duplicates("species_number")
                         .set_index("species_number")["latin_name"].to_dict())
    # the class join the old reader lacked (F3): T2's reserved classes are species classes
    class_by_number = (aq.dropna(subset=["class"]).drop_duplicates("species_number")
                       .set_index("species_number")["class"].to_dict())

    fam = H4.attach_basis(H4.admit_units(H4.parse_values(H4.select_bcf_family(results))))
    fam = H4.attach_duration_h(H4.join_structures(fam, tests, smiles_by_cas, species_by_number))
    fam["class"] = fam["species_number"].map(class_by_number)
    trimmed, trimmed_out = H4.trim_implausible(fam)
    rep = H4.qc_report(len(results), fam, trimmed, trimmed_out)

    # ---- the SEVEN exclusions of amendment §10.2, in the stamped order, each with its own counter
    f = trimmed
    steps: dict = {"start": int(len(f))}

    def cut(frame, key: str, mask) -> pd.DataFrame:
        m = pd.Series(mask, index=frame.index).fillna(False).astype(bool)
        out = frame[~m].copy()
        steps[f"removed_{key}"] = int(len(frame) - len(out))
        return out

    f = cut(f, "t2_reserved_classes", f["class"].astype("string").isin(A.T2_RESERVED_CLASSES))
    f = cut(f, "spent_prospective", f["result_id"].astype(str).isin(spent["prospective"]))
    f = cut(f, "spent_adore", f["result_id"].astype(str).isin(spent["adore"]))
    mapped = f["cas_norm"].map(ctx["cas_unit_norm"])
    steps["chemistry_mapped_fraction"] = round(float(mapped.notna().mean()), 4) if len(f) else 1.0
    f = cut(f, "held_out_unit_chemistry", mapped.isin(ctx["held_units"]))
    f = cut(f, "chemistry_not_in_split", f["cas_norm"].map(ctx["cas_unit_norm"]).isna())
    pairs = _sealed_pairs(ctx)
    f = cut(f, "same_cell_as_sealed_row",
            [(_norm_cas_key(c), str(s)) in pairs for c, s in zip(f["cas_norm"], f["latin_name"])])
    if track == "T6":
        lo, hi = _h4_window_for_exclusion()
        f = cut(f, "outside_24_96h_for_T6", ~f["duration_h"].between(lo, hi).fillna(False))
    else:
        steps["removed_outside_24_96h_for_T6"] = 0

    steps["admissible_rows"] = int(len(f))
    steps["admissible_chemicals"] = int(f["cas_norm"].nunique())
    steps["admissible_species"] = int(f["latin_name"].nunique())
    steps["by_basis"] = {str(k): int(v) for k, v in f["basis"].value_counts().items()}
    steps["exclusions_applied"] = list(SEVEN_EXCLUSIONS)

    out = f.copy()
    out["test_cas"] = out["cas_norm"]
    rep["admissible_for_ablation_A5"] = {track: steps}
    rep["guarded_by"] = "arch_v2_guard.load_h4_rows"
    rep["cell_rule"] = "pair (CAS, species); H4's duration is an observation duration"
    rep["recheck"] = _recheck_seven(out, what="auxiliary H4", track=track,
                                    cas_col="cas_norm", species_col="latin_name",
                                    duration_col="duration_h", class_col="class",
                                    cell_rule="pair")
    rep["target_column"] = H4_TARGET
    rep["targets_refused"] = sorted(set(AUX_TARGETS.values()) - {H4_TARGET})
    rep["columns_returned"] = sorted(out.columns)
    rep["superseded_counts"] = (
        "the pre-stamp H4 counts 10,050 (T1) and 2,984 (T6) were measured with FOUR steps by the "
        "old report script and are SUPERSEDED (G-3); H4 is re-measured once through this reader "
        "after the pins exist")
    rep["report_path"] = str(_write_report(f"aux_H4_{track}", rep, pinned))
    return out, rep


def load_h5_pool_pairs(**kwargs) -> tuple[pd.DataFrame, dict]:
    """H5 (acute-to-chronic ratio) reference pairs, from ``load_training_pool()`` rows ONLY (G-4).

    Amendment §10.1 makes H5 a **reported output, never a §15.1 label**, and G-4 fixes that it is
    computed from pool rows only: no guarded H3 row and no other row enters it. The pool is acute,
    so there is **no supervised ACR target** — the candidate emits ``log10_acr`` and it is reported.
    This function returns the pool's (chemical, species) keys with the acute target, which is all
    the protocol admits, and refuses any attempt to pass auxiliary rows in.
    """
    if kwargs:
        raise ProtectedBlockError(
            f"load_h5_pool_pairs takes no arguments; H5 comes from load_training_pool() rows only "
            f"and no guarded H3 row may enter it (G-4). Got {sorted(kwargs)}")
    pool = load_training_pool(columns=["result_id", "test_cas", "latin_name", "duration_h",
                                       TARGET])
    rep = {"guarded_by": "arch_v2_guard.load_h5_pool_pairs",
           "source": "load_training_pool only",
           "rows": int(len(pool)),
           "pairs": int(pool.groupby([pool["test_cas"].astype(str),
                                      pool["latin_name"].astype(str)]).ngroups),
           "target_column": TARGET,
           "supervised_acr_target": False,
           "note": "H5 is a reported output with an interval, never a label (amendment §10.1); the "
                   "pool is acute only, so no ACR label exists and none is fabricated"}
    return pool, rep


# =========================================================================== predictions, sealing

def _write_sums(d: Path) -> Path:
    lines = []
    for f in sorted(d.rglob("*")):
        if f.is_file() and f.name != "SHA256SUMS":
            lines.append(f"{file_sha(f)}  {f.relative_to(d)}")
    p = d / "SHA256SUMS"
    if p.exists():
        os.chmod(p, 0o644)
    p.write_text("\n".join(lines) + "\n")
    return p


def save_predictions(block: str, pred) -> dict:
    """Write-once prediction file, sealed 0444, with the directory's sums refreshed."""
    require_authorised(block)
    d = block_dir(block)
    p = d / "predictions.parquet"
    if p.exists():
        raise ProtectedBlockError(f"{p} already exists: predictions are write-once")
    pd.DataFrame(pred).to_parquet(p, index=False)
    os.chmod(p, 0o444)
    rec = {"path": str(p), "sha256": file_sha(p), "rows": int(len(pred))}
    _excl_write(_marker(block, "PREDICTED.json"), rec)
    _write_sums(d)
    return rec


def verify_artifacts(d: Path) -> dict:
    """Re-hash every file against SHA256SUMS. Must pass before a claim is written (B1 (iii))."""
    d = Path(d)
    sums = d / "SHA256SUMS"
    if not sums.is_file():
        raise ProtectedBlockError(
            f"no SHA256SUMS in {d}: there is nothing to verify, so no claim may be written")
    checked, recorded = {}, set()
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        want, name = line.split(None, 1)
        name = name.strip()
        recorded.add(name)
        f = d / name
        if not f.is_file():
            raise ProtectedBlockError(f"{f} is recorded in SHA256SUMS but missing")
        got = file_sha(f)
        if got != want:
            raise ProtectedBlockError(f"{f}: recorded {want[:16]}, on disk {got[:16]}")
        checked[name] = got
    present = {str(f.relative_to(d)) for f in d.rglob("*")
               if f.is_file() and f.name != "SHA256SUMS"}
    extra = sorted(present - recorded)
    if extra:
        raise ProtectedBlockError(
            f"{d} holds files absent from SHA256SUMS: {extra}. Every artefact must be recorded "
            f"before the claim.")
    return {"verified": len(checked), "files": checked, "sums_sha256": file_sha(sums)}


def mark_scored(block: str, summary: dict, *, verification: dict | None = None) -> None:
    """Write the single scored marker. Refuses unless verify_artifacts() has just passed."""
    require_authorised(block)
    v = verification if verification is not None else verify_artifacts(block_dir(block))
    if not isinstance(v, dict) or not v.get("verified"):
        raise ProtectedBlockError(
            "mark_scored refuses: verify_artifacts() has not passed for this block (B1 (iii))")
    rec = dict(summary)
    rec["artifact_verification"] = v
    _excl_write(_marker(block, "SCORED.json"), rec)


def seal_dir(d: Path) -> Path:
    """0444 on every file plus a SHA256SUMS; access closed at sealing."""
    d = Path(d)
    p = _write_sums(d)
    for f in sorted(d.rglob("*")):
        if f.is_file():
            os.chmod(f, 0o444)
    return p


class _Pass:
    """Handle yielded by :func:`scored_pass`. ``finish`` is the only route to ``SCORED.json``."""

    def __init__(self, block: str, claim: dict) -> None:
        self.block, self.claim = block, claim
        self.verification: dict | None = None

    def load(self, *, columns=None) -> pd.DataFrame:
        return load_test_block(self.block, columns=columns)

    def save_predictions(self, pred) -> dict:
        return save_predictions(self.block, pred)

    def finish(self, summary: dict) -> dict:
        """B1 (iii): verify every artefact, THEN write the scored marker."""
        self.verification = verify_artifacts(block_dir(self.block))
        mark_scored(self.block, summary, verification=self.verification)
        return self.verification


@contextmanager
def scored_pass(block: str, *, note: str = ""):
    """The whole look, with a ``BaseException`` abort that records what was actually written."""
    claim = authorise(block, note=note)
    handle = _Pass(block, claim)
    try:
        yield handle
    except BaseException as exc:
        d = block_dir(block)
        try:
            inv = _written_inventory(block)
            inv.update({"block": block, "error": type(exc).__name__,
                        "message": str(exc)[:2000], "pid": os.getpid(),
                        "host": socket.gethostname(),
                        "artifact_verification": handle.verification})
            target = d / "ABORTED.json"
            if target.exists():
                k = 2
                while (d / f"ABORTED.{k}.json").exists():
                    k += 1
                target = d / f"ABORTED.{k}.json"
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
            with os.fdopen(fd, "w") as fh:
                json.dump(inv, fh, indent=1, sort_keys=True)
                fh.write("\n")
        except BaseException:                                        # pragma: no cover
            pass        # never mask the original failure with a bookkeeping error
        raise
    else:
        if not _marker(block, "SCORED.json").is_file():
            raise ProtectedBlockError(
                f"the pass for {block} left no SCORED.json: call look.finish(summary), which "
                f"verifies every artefact before writing it (B1 (iii))")
        seal_dir(block_dir(block))
