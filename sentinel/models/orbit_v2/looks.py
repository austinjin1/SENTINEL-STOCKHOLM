# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Single-pass guard for the look ``M2S/test`` (arch-v2 amendment §5; lead rulings R-17, R-21, R-23, R-24).

Test-S = test-role water bodies x in-situ dates 2015-06-23 .. 2023-12-31 (amendment §1.1): cells
``testxarchive_lake`` (claims) and ``testxarchive_nonlake`` (descriptive).

R-21: only look-agnostic helpers of the v1 module ``sentinel/models/orbit/testlook.py`` are imported here (the
file hash, the O_EXCL read-only writer and the content pin of files under the processed / raw data roots). The
look list, the look folder, the claim, the open-claim check, the freeze loader and the gate of the protected-row
loader are written here; no v1 module is patched at run time and v1 ``testlook.py`` is not edited.

Order (``scripts/eval/orbitv2_test_s.py``):
1. :func:`load_freeze`: only ``docs/preregistration/FREEZE_orbit_S.json`` (status FROZEN, lead authorisation, equal
   to the reviewed candidate, both stamped files at their recorded sha256 and the lead's stamp row (R-23), split,
   data manifest, v1 weight hashes equal to those of ``FREEZE_orbit.json``, the M7 keys list (C-1), scoring-code
   hash over the import closure), then :func:`verify_data_pins`: before the claim, so a mismatch never spends the
   look;
2. :func:`claim`: refused when the registry holds any row of the look's experiment, and when the look folder holds
   anything at all (R-24); then the O_EXCL, read-only ``TEST_LOOK.json`` bound to pid and host, and a
   ``look_claimed`` registry row;
3. protected rows only inside the open claim (:func:`check_rows`), and only Test-S rows (:func:`test_s_cell`,
   :func:`require_test_s_only`): the look holds no row of the headline cells;
4. predictions written once and sealed (0444, ``SHA256SUMS``, O_EXCL ``PREDICTIONS_SAVED.json``, a
   ``predictions_sealed`` registry row). R-17: the sealed cell tables must hold the reported model, every seed
   unit and every single network of v2, every frozen single network of v1, and every claim comparator
   (:func:`sealed_columns`, :func:`check_prediction_columns`). Access closes at sealing;
5. scoring reads the sealed files only; rescoring is refused once a scoring row exists; then ``SCORED.json``.
Any BaseException between claim and seal: :func:`mark_aborted`; the look is spent.

The development dry run uses the same functions under ``root=test_looks_dryrun/<label>`` with validation rows.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ...core import paths
from ..orbit import guard as G1
from ..orbit import split as SP
from ..orbit.testlook import _excl_write, _sha_file, content_pin      # look-agnostic helpers only (R-21)
from ..sonde.guard import ProtectedDataError
from . import m7keys as M7
from . import stamp as ST
from . import train as TR

WORKSTREAM = "M2-orbit"
LOOKS = ("M2S/test",)
LOOK_EXPERIMENT = {"M2S/test": "orbit_TEST_S_{v}"}
ROOT = "test_looks"
DRYRUN_ROOT = "test_looks_dryrun"
FREEZE_FILE = paths.REPO_ROOT / "docs" / "preregistration" / "FREEZE_orbit_S.json"
# the reviewed candidate; it never unlocks a protected row
CANDIDATE_FILE = paths.REPO_ROOT / "docs" / "preregistration" / "orbit_S_freeze_candidate.json"
V1_FREEZE_FILE = G1.FREEZE_FILE
AUTH_KEYS = ("status", "authorised_by", "authorised", "candidate_sha256")
# amendment §5 ("the freeze candidate records ...") and R-20 / R-23 / C-1
FREEZE_KEYS = ("created", "look", "split_sha256", "scoring_code_sha256", "weights_sha256", "comparators",
               "data_version", "data_manifest_sha256", "data_pins", "stamped_sha256", "test_s_primary_cell",
               "power_status", "v1_weights_sha256", "v1_freeze_sha256", "config_sha256", "m7_keys_sha256",
               "sealed_models", *AUTH_KEYS)
PRIMARY_CELLS = ("scene_disjoint", "all_rows")                         # amendment §1.6
CONTROL_METRICS = ("look_claimed", "predictions_sealed", "look_aborted")

# ------------------------------------------------------------------------------------ Test-S rows (amendment §1.1)
ARCHIVE_DATE_MIN = "2015-06-23"
ARCHIVE_DATE_MAX = "2023-12-31"
CELL_LAKE = "testxarchive_lake"
CELL_NONLAKE = "testxarchive_nonlake"
CELLS = (CELL_LAKE, CELL_NONLAKE)

# ------------------------------------------------------------------------------------ scoring-code hash
# amendment §5: the hash covers the full import closure: every file hashed by the v1 freeze (a superset of the v1
# files this look imports) and every orbit_v2 file, script and configuration, the two stamped files, and the
# shared modules the Test-S path loads (tests/test_orbit_v2_looks.py compares with sys.modules of a fresh run).
V2_GLOBS = ("sentinel/models/orbit_v2/*.py", "scripts/train/orbitv2_*.py", "scripts/eval/orbitv2_*.py",
            "scripts/data/orbitv2_*.py", "configs/orbit_v2/*.yaml")
V2_SHARED = (ST.AMENDMENT, ST.RULINGS, "sentinel/models/__init__.py", "sentinel/models/sonde/__init__.py",
             "sentinel/models/sonde/guard.py", "sentinel/models/orbit/__init__.py",
             # shared data modules the orbit_v2 label reader and builder load (R-3: the project WQP rule)
             "sentinel/data/__init__.py", "sentinel/data/matchups.py", "sentinel/data/sonde_splits.py",
             "sentinel/data/sources/__init__.py")
# infrastructure that sets no prediction and no score (paths, registry, run log, GPU lock); as in the v1 hash
NOT_IN_CLOSURE = ("sentinel/core/",)
# hashed like every orbitv2 script, but never imported by the Test-S path: the keys-only count script named in the
# stamped amendment (§1.4) is a stand-alone script that runs at import and is not edited after the stamp
RUNS_AT_IMPORT = ("scripts/data/orbitv2_testS_keycounts.py",)


def frozen_files(root: Path | None = None) -> list[str]:
    root = Path(root) if root else paths.REPO_ROOT
    files = set(G1.frozen_files(root)) | set(V2_SHARED)
    for g in V2_GLOBS:
        files |= {str(f.relative_to(root)) for f in root.glob(g) if f.is_file()}
    return sorted(files)


def scoring_code_sha256(root: Path | None = None) -> str:
    root = Path(root) if root else paths.REPO_ROOT
    h = hashlib.sha256()
    for rel in frozen_files(root):
        f = root / rel
        h.update(rel.encode())
        h.update(f.read_bytes() if f.is_file() else b"<missing>")
    return h.hexdigest()


# ------------------------------------------------------------------------------------ look folder
def look_dir(look: str, root: str = ROOT) -> Path:
    if look not in LOOKS:
        raise ProtectedDataError(f"unknown test look {look!r} (this module serves {LOOKS} only)")
    return paths.RUNS_ROOT / "orbit" / root / look.replace("/", "_")


def marker(look: str, root: str = ROOT) -> Path:
    return look_dir(look, root) / "TEST_LOOK.json"


def experiment(look: str, version: str) -> str:
    if look not in LOOKS:
        raise ProtectedDataError(f"unknown test look {look!r}")
    return LOOK_EXPERIMENT[look].format(v=version)


def folder_entries(look: str, root: str = ROOT) -> list[str]:
    """Names of everything in the look folder (files and folders, any depth). Names only; nothing is opened."""
    d = look_dir(look, root)
    return sorted(q.relative_to(d).as_posix() for q in d.rglob("*")) if d.exists() else []


# ------------------------------------------------------------------------------------ data pins
# Every file the protected loader of Test-S reads (R-21). ``""`` / a folder: relative to paths.PROCESSED;
# ``RAW:`` relative to paths.RAW; ``RUNS:`` relative to paths.RUNS_ROOT (the context store pin, the M7 keys list).
DATA_PIN_SPECS = {
    "wqp_matchups": ("", ["s2_wqp_matchups/samples.parquet", "s2_wqp_matchups/matchups.parquet",
                          "s2_wqp_matchups_hist/samples.parquet", "s2_wqp_matchups_hist/matchups.parquet",
                          "s2_wqp_matchups_2026/samples.parquet", "s2_wqp_matchups_2026/matchups.parquet"]),
    "wqp_surface": ("orbit_wqp_surface", ["surface_samples.parquet", "surface_samples.meta.json"]),
    "wqp_discrete": ("wqp_discrete", ["_stations.parquet", "_qc_rejects.parquet", "_unmapped.parquet",
                                      "variable=*/data.parquet"]),
    "aquacorr_tables": ("aquacorr_global", ["s2_chip_requests.parquet", "s2_chip_plan.parquet"]),
    "chips_wqp": ("s2_chips/wqp", ["index.parquet", "index/*", "layers.json", "shards/*"]),
    "chips_aquacorr": ("s2_chips/aquacorr", ["index.parquet", "index/*", "layers.json", "shards/*"]),
    "hydrolakes": ("hydrolakes", ["hydrolakes_na_eu.parquet"]),
    "wbd_huc12": ("wbd_huc12", ["huc12.parquet"]),
    "gloria_srf": ("gloria", ["srf_info.parquet"]),
    "mdn_official_weights": ("RAW:mdn_stream_weights/MSI/x2025", ["**/*"]),
    "context_store_pin": ("RUNS:orbit/v2/context_store_pin", ["chips.parquet", "pin.json"]),
    "m7_keys_list": ("RUNS:lite/counts_arch_v2", ["c2_teacher_scored_m2_test_lake_keys.csv"]),
}
REQUIRED_PINS = tuple(DATA_PIN_SPECS)


def pin(loader_dir: str, patterns) -> dict:
    """Content pin. PROCESSED and ``RAW:`` folders go through the v1 helper unchanged; a ``RUNS:`` folder is
    hashed here in the same form (relative path, tab, sha256 of the file, per line)."""
    if not loader_dir.startswith("RUNS:"):
        return content_pin(loader_dir, patterns)
    root = paths.RUNS_ROOT / loader_dir[5:]
    files = sorted({f for p in patterns for f in root.glob(p) if f.is_file()})
    h = hashlib.sha256()
    nbytes = 0
    for f in files:
        h.update(f"{f.relative_to(root).as_posix()}\t{_sha_file(f)}\n".encode())
        nbytes += f.stat().st_size
    return {"loader_dir_rel": loader_dir, "patterns": list(patterns), "n_files": len(files), "bytes": nbytes,
            "content_sha256": h.hexdigest()}


def make_data_pins() -> dict:
    return {name: pin(d, pats) for name, (d, pats) in DATA_PIN_SPECS.items()}


def verify_data_pins(freeze: dict) -> dict:
    """Every pin of the freeze must match, every DATA_PIN_SPECS entry must be pinned with the patterns written
    here, and no pinned set may be empty."""
    pins = freeze.get("data_pins") or {}
    bad = [f"{k}: no pin" for k in DATA_PIN_SPECS if k not in pins]
    for name, p in pins.items():
        spec = DATA_PIN_SPECS.get(name)
        if spec is not None and (p.get("loader_dir_rel") != spec[0] or list(p.get("patterns", [])) != list(spec[1])):
            bad.append(f"{name}: pinned folder or patterns differ from DATA_PIN_SPECS")
            continue
        now = pin(p["loader_dir_rel"], p["patterns"])
        if now["content_sha256"] != p["content_sha256"] or now["n_files"] != p["n_files"]:
            bad.append(f"{name}: content changed ({now['n_files']} files vs {p['n_files']})")
        elif now["n_files"] == 0:
            bad.append(f"{name}: no file")
    if bad:
        raise ProtectedDataError(f"data pins do not match: {bad}")
    return {k: v["content_sha256"] for k, v in pins.items()}


# ------------------------------------------------------------------------------------ freeze loader
def stamped_sha256(root: Path | None = None) -> dict:
    root = Path(root) if root else paths.REPO_ROOT
    return {rel: _sha_file(root / rel) for rel in ST.STAMPED_FILES}


def load_freeze(freeze_file: str | Path | None = None, *, check_code: bool = True) -> dict:
    """Only the lead-written ``FREEZE_orbit_S.json`` unlocks Test-S rows. Any other path (the candidate, the v1
    freeze) is refused."""
    ff = Path(freeze_file) if freeze_file is not None else FREEZE_FILE
    if ff.resolve() != FREEZE_FILE.resolve():
        raise ProtectedDataError(f"only {FREEZE_FILE} unlocks Test-S rows (got {ff})")
    if not ff.is_file():
        raise ProtectedDataError(f"freeze file {ff} does not exist")
    try:
        freeze = json.loads(ff.read_text())
    except (OSError, ValueError) as e:
        raise ProtectedDataError(f"freeze file {ff} is not valid JSON: {e}") from e
    if not isinstance(freeze, dict):
        raise ProtectedDataError(f"freeze file {ff} must be a JSON object")
    missing = [k for k in FREEZE_KEYS if not freeze.get(k)]
    if missing:
        raise ProtectedDataError(f"freeze file {ff} lacks required keys {missing}")
    if freeze["status"] != "FROZEN":
        raise ProtectedDataError(f"freeze status {freeze['status']!r} != 'FROZEN' (a candidate never unlocks test)")
    if freeze["look"] not in LOOKS:
        raise ProtectedDataError(f"freeze look {freeze['look']!r} is not one of {LOOKS}")
    for k in ("weights_sha256", "v1_weights_sha256", "config_sha256", "stamped_sha256", "sealed_models"):
        if not isinstance(freeze[k], dict):
            raise ProtectedDataError(f"freeze file {k} must be a JSON object")
    if freeze["test_s_primary_cell"] not in PRIMARY_CELLS:
        raise ProtectedDataError(f"test_s_primary_cell must be one of {PRIMARY_CELLS} (amendment §1.6)")
    if not CANDIDATE_FILE.is_file() or _sha_file(CANDIDATE_FILE) != freeze["candidate_sha256"]:
        raise ProtectedDataError(f"candidate_sha256 does not match {CANDIDATE_FILE.name}")
    cand = json.loads(CANDIDATE_FILE.read_text())
    diff = sorted(k for k in set(cand) | set(freeze) if k not in AUTH_KEYS and cand.get(k) != freeze.get(k))
    if diff:
        raise ProtectedDataError(f"{FREEZE_FILE.name} differs from the reviewed candidate in {diff}")
    if freeze["split_sha256"] != SP.split_sha256():
        raise ProtectedDataError("freeze file split_sha256 does not match docs/preregistration/orbit_split_v3.json")
    # R-23: both stamped files at the recorded sha256, read-only, and the lead's stamp row for exactly these hashes
    if freeze["stamped_sha256"] != stamped_sha256():
        raise ProtectedDataError("stamped_sha256 does not equal the current sha256 of the amendment and the rulings")
    try:
        ST.require_stamp("M2S/test")
    except ST.StampError as e:
        raise ProtectedDataError(str(e)) from e
    # amendment §5: v1's weight hashes equal those of the lead-written v1 freeze
    if not V1_FREEZE_FILE.is_file() or _sha_file(V1_FREEZE_FILE) != freeze["v1_freeze_sha256"]:
        raise ProtectedDataError(f"v1_freeze_sha256 does not match {V1_FREEZE_FILE.name}")
    if json.loads(V1_FREEZE_FILE.read_text()).get("weights_sha256") != freeze["v1_weights_sha256"]:
        raise ProtectedDataError(f"v1_weights_sha256 differs from weights_sha256 of {V1_FREEZE_FILE.name}")
    # C-1: the keys list M7 fixed, at the ruled sha256 (load_list refuses anything else)
    if freeze["m7_keys_sha256"] != M7.LIST_SHA256:
        raise ProtectedDataError("m7_keys_sha256 is not the sha256 fixed by C-1")
    try:
        M7.load_list()
    except M7.M7ListError as e:
        raise ProtectedDataError(str(e)) from e
    man_path = paths.MANIFESTS / f"orbit_{freeze['data_version']}.json"
    if not man_path.is_file() or _sha_file(man_path) != freeze["data_manifest_sha256"]:
        raise ProtectedDataError(f"data manifest {man_path.name} missing or sha256 != frozen data_manifest_sha256")
    from ...core.manifest import verify_manifest

    problems = verify_manifest(f"orbit_{freeze['data_version']}")
    if problems:
        raise ProtectedDataError(f"frozen data files changed: {problems[:3]}")
    if check_code and freeze["scoring_code_sha256"] != scoring_code_sha256():
        raise ProtectedDataError("scoring code changed since the freeze (scoring_code_sha256 mismatch)")
    freeze["_path"] = str(ff)
    freeze["_sha256"] = _sha_file(ff)
    return freeze


def verify_weights(freeze: dict, model: str, artifact: str | Path) -> str:
    """``model``: a key of ``weights_sha256`` (v2 and comparator artifacts) or of ``v1_weights_sha256``."""
    sha = G1.artifact_sha256(artifact)
    want = freeze["weights_sha256"].get(model, freeze["v1_weights_sha256"].get(model))
    if want != sha:
        raise ProtectedDataError(f"artifact {artifact} sha256 {sha[:12]} != the frozen sha256 of {model!r}")
    return sha


# ------------------------------------------------------------------------------------ claim
def registry_rows(experiment_name: str, *, exclude_metrics=()) -> int:
    reg = paths.REPO_ROOT / "results" / "registry.jsonl"
    n = 0
    if reg.is_file():
        with reg.open() as fh:
            for line in fh:
                if experiment_name not in line:
                    continue
                r = json.loads(line)
                if r.get("experiment") == experiment_name and r.get("metric") not in exclude_metrics:
                    n += 1
    return n


def scoring_rows(experiment_name: str) -> int:
    return registry_rows(experiment_name, exclude_metrics=CONTROL_METRICS)


def control_row(experiment_name: str, metric: str) -> bool:
    reg = paths.REPO_ROOT / "results" / "registry.jsonl"
    if not reg.is_file():
        return False
    with reg.open() as fh:
        for line in fh:
            if experiment_name in line and metric in line:
                r = json.loads(line)
                if r.get("experiment") == experiment_name and r.get("metric") == metric:
                    return True
    return False


def require_no_scores(look: str, freeze: dict) -> None:
    exp = experiment(look, freeze["data_version"])
    n = scoring_rows(exp)
    if n:
        raise ProtectedDataError(f"{look}: registry already holds {n} scoring rows of {exp} — no re-scoring")


def _log_result(**k):
    from ...core.registry import log_result

    return log_result(**k)


def claim(look: str, freeze: dict, *, root: str = ROOT, register: bool = True) -> dict:
    exp = experiment(look, freeze["data_version"])
    n = registry_rows(exp)
    if n:
        raise ProtectedDataError(f"{look}: registry already holds {n} rows of {exp} — the test look was used")
    if marker(look, root).exists():
        raise ProtectedDataError(f"{look}: test look already claimed ({marker(look, root)})")
    held = folder_entries(look, root)
    if held:                                                    # R-24
        raise ProtectedDataError(f"{look}: the look folder {look_dir(look, root)} is not empty ({len(held)} entries, "
                                 f"e.g. {held[0]!r}); the claim starts only on an empty or absent folder (R-24)")
    rec = {"look": look, "experiment": exp, "freeze_file": freeze.get("_path"), "freeze_sha256": freeze["_sha256"],
           "scoring_code_sha256": freeze.get("scoring_code_sha256"), "claimed": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "pid": os.getpid(), "host": socket.gethostname()}
    try:
        _excl_write(marker(look, root), rec)
    except FileExistsError as e:
        raise ProtectedDataError(f"{look}: test look already claimed ({marker(look, root)})") from e
    if register:
        _log_result(workstream=WORKSTREAM, experiment=exp, model="orbit_v2_test_look", metric="look_claimed",
                    value=1.0, split=f"{look} claim (single pass)", config={"look": look},
                    notes="claim-time row (single-pass guard, arch-v2 amendment §5)",
                    extra={"freeze_sha256": freeze["_sha256"], "marker": str(marker(look, root))})
    return rec


def open_claims(freeze_sha: str, root: str = ROOT) -> list[str]:
    """Looks of this module claimed under this freeze by THIS process on THIS host and not yet sealed, aborted
    or scored. A marker file or an append-only registry row closes the access."""
    out = []
    for look in LOOKS:
        m = marker(look, root)
        if not m.is_file():
            continue
        rec = json.loads(m.read_text())
        if rec.get("look") != look or rec.get("freeze_sha256") != freeze_sha:
            continue
        if rec.get("pid") != os.getpid() or rec.get("host") != socket.gethostname():
            continue
        closed = any((m.parent / f).exists() for f in ("PREDICTIONS_SAVED.json", "SCORED.json", "ABORTED.json"))
        exp = rec.get("experiment")
        if root == ROOT and exp and (control_row(exp, "predictions_sealed") or control_row(exp, "look_aborted")):
            closed = True
        if not closed:
            out.append(look)
    return out


def mark_aborted(look: str, freeze: dict, reason: str, *, root: str = ROOT, register: bool = True) -> None:
    try:
        _excl_write(look_dir(look, root) / "ABORTED.json", {"aborted": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                            "reason": reason[:2000], "pid": os.getpid()})
    except FileExistsError:
        pass
    if register:
        _log_result(workstream=WORKSTREAM, experiment=experiment(look, freeze["data_version"]),
                    model="orbit_v2_test_look", metric="look_aborted", value=1.0,
                    split=f"{look} aborted before sealing",
                    notes=f"single-pass guard: aborted ({reason[:300]})", extra={"freeze_sha256": freeze["_sha256"]})


def require_open_claim(freeze: dict, root: str = ROOT) -> str:
    claims = open_claims(freeze["_sha256"], root)
    if not claims:
        raise ProtectedDataError("Test-S rows need an open M2S/test claim of this freeze "
                                 "(scripts/eval/orbitv2_test_s.py); none exists, or its predictions are already sealed")
    return claims[0]


# ------------------------------------------------------------------------------------ gate of the protected loader
def check_rows(protected_mask, *, allow_test: bool = False, training: bool = False, freeze_file=None,
               what: str = "rows") -> dict:
    """Authorise rows given their ``protected`` flags (``split.assign``). Protected rows are refused unless
    ``allow_test`` AND the lead-written Test-S freeze AND an open M2S/test claim of this process; always refused
    for training code."""
    n_prot = int(np.asarray(protected_mask, bool).sum())
    rec = {"n_protected": n_prot, "freeze_file": None, "freeze_sha256": None}
    if n_prot == 0:
        return rec
    if training:
        raise ProtectedDataError(f"training code never loads protected {what} ({n_prot} test-role or test-year "
                                 "rows); Test-S predictions come only from scripts/eval/orbitv2_test_s.py")
    if not allow_test:
        raise ProtectedDataError(f"refusing {n_prot} protected {what}: pass --allow-test (only after "
                                 f"{FREEZE_FILE.name} is written by the lead)")
    freeze = load_freeze(freeze_file)
    rec.update({"freeze_file": freeze["_path"], "freeze_sha256": freeze["_sha256"],
                "look": require_open_claim(freeze)})
    return rec


def test_s_cell(df: pd.DataFrame) -> np.ndarray:
    """Cell name per row from key columns only (role, date, in_lake, buffer_excluded): ``testxarchive_lake``,
    ``testxarchive_nonlake`` or ``""`` (not a Test-S row). Amendment §1.1."""
    day = pd.to_datetime(df["date"]).dt.normalize()
    ok = ((df["role"].astype(str) == "test").to_numpy() & ~df["buffer_excluded"].to_numpy(bool)
          & (day >= pd.Timestamp(ARCHIVE_DATE_MIN)).to_numpy() & (day <= pd.Timestamp(ARCHIVE_DATE_MAX)).to_numpy())
    return np.where(ok, np.where(df["in_lake"].to_numpy(bool), CELL_LAKE, CELL_NONLAKE), "")


test_s_cell.__test__ = False        # not a pytest test


def require_test_s_only(df: pd.DataFrame, what: str = "rows") -> None:
    """The M2S/test look holds Test-S rows and nothing else that is protected: a protected row outside Test-S
    (test role x 2024-25, a test time block of any role) stops the loader. Counts only in the message."""
    cell = test_s_cell(df)
    other = df["protected"].to_numpy(bool) & (cell == "")
    if other.any():
        raise ProtectedDataError(f"{int(other.sum())} protected {what} are not Test-S rows (test role, "
                                 f"{ARCHIVE_DATE_MIN}..{ARCHIVE_DATE_MAX}): the M2S/test loader keeps Test-S keys "
                                 "only, before any value is read")
    if (cell == "").any():
        raise ProtectedDataError(f"{int((cell == '').sum())} {what} are not Test-S rows")


def apply_c1(df: pd.DataFrame, lst: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """C-1 / R-1: rows matching M7's keys list leave every Test-S cell. ``df`` carries M7.ROW_COLS."""
    lst = M7.load_list() if lst is None else lst
    hit, rec = M7.match(df[list(M7.ROW_COLS)], lst)
    return df[~hit].reset_index(drop=True), rec


# ------------------------------------------------------------------------------------ sealed predictions
PRED_SUFFIXES = (".parquet", ".npz", ".json")
CELL_TABLE_GLOB = "cell_*.parquet"
V2_MODEL, V1_MODEL, TRACK_N_MODEL = "v2", "v1", "track_n"
COMPARATOR_PREFIX = "cmp:"


def member_columns(model: str) -> list[str]:
    """R-17: the reported model, its 3 seed units and its 9 single networks (names of ``units.members``)."""
    return ([model] + [f"{model}@unit{u}" for u in TR.UNITS]
            + [f"{model}@u{u}i{i}" for u in TR.UNITS for i in TR.INITS])


def sealed_columns(sealed_models: dict) -> list[str]:
    """Columns every sealed cell table must hold, from the freeze entry ``sealed_models`` =
    ``{"v1_networks": n, "v2": true, "track_n": bool, "comparators": [names]}``."""
    n1 = int(sealed_models.get("v1_networks", 0))
    comps = list(sealed_models.get("comparators", []))
    if n1 < 1 or not sealed_models.get("v2") or not comps:
        raise ProtectedDataError("sealed_models must name the v1 network count, v2 and the claim comparators (R-17)")
    cols = [V1_MODEL] + [f"{V1_MODEL}@s{i}" for i in range(n1)] + member_columns(V2_MODEL)
    if sealed_models.get("track_n"):
        cols += member_columns(TRACK_N_MODEL)
    return cols + [f"{COMPARATOR_PREFIX}{c}" for c in comps]


def check_prediction_columns(pred_dir: Path, required) -> int:
    """Every ``cell_*.parquet`` under ``pred_dir`` holds every required column (schema read only)."""
    import pyarrow.parquet as pq

    tables = sorted(Path(pred_dir).rglob(CELL_TABLE_GLOB))
    if not tables:
        raise ProtectedDataError(f"no {CELL_TABLE_GLOB} table to seal in {pred_dir}")
    for f in tables:
        missing = [c for c in required if c not in set(pq.read_schema(f).names)]
        if missing:
            raise ProtectedDataError(f"{f.name}: per-unit / per-network / comparator predictions missing before "
                                     f"sealing (R-17): {missing[:5]} ({len(missing)} columns)")
    return len(tables)


def seal_predictions(look: str, pred_dir: Path, root: str = ROOT, *, freeze: dict, required,
                     register: bool = True) -> dict:
    """``required``: :func:`sealed_columns` of the freeze (R-17); sealing without it is refused."""
    pred_dir = Path(pred_dir)
    if (look_dir(look, root) / "PREDICTIONS_SAVED.json").exists() or (pred_dir / "SHA256SUMS").exists():
        raise ProtectedDataError(f"{look}: predictions already sealed — they are written once")
    required = list(required or [])
    need = {V1_MODEL, f"{V1_MODEL}@s0", *member_columns(V2_MODEL)}
    if not need <= set(required) or not any(c.startswith(COMPARATOR_PREFIX) for c in required):
        raise ProtectedDataError(f"{look}: sealing needs the per-unit, per-network and comparator columns (R-17)")
    files = sorted(f for f in pred_dir.rglob("*") if f.is_file() and f.suffix in PRED_SUFFIXES)
    if not files:
        raise ProtectedDataError(f"{look}: no predictions to seal in {pred_dir}")
    n_tables = check_prediction_columns(pred_dir, required)
    sums = {f.relative_to(pred_dir).as_posix(): _sha_file(f) for f in files}
    for f in files:
        os.chmod(f, 0o444)
    sp = pred_dir / "SHA256SUMS"
    fd = os.open(sp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(fd, "w") as fh:
        fh.write("".join(f"{s}  {n}\n" for n, s in sums.items()))
    rec = {"dir": str(pred_dir), "n_files": len(files), "n_cell_tables": n_tables, "sha256sums": _sha_file(sp),
           "required_columns": required, "saved": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _excl_write(look_dir(look, root) / "PREDICTIONS_SAVED.json", rec)
    if register:
        _log_result(workstream=WORKSTREAM, experiment=experiment(look, freeze["data_version"]),
                    model="orbit_v2_test_look", metric="predictions_sealed", value=float(len(files)),
                    split=f"{look} sealed predictions",
                    notes="sealing-time row (single-pass guard, arch-v2 amendment §5, R-17); test access closed",
                    extra={"freeze_sha256": freeze["_sha256"], "sha256sums": rec["sha256sums"], "dir": rec["dir"],
                           "n_required_columns": len(required)})
    return rec


def check_sealed(look: str, root: str = ROOT) -> Path:
    if (look_dir(look, root) / "SCORED.json").exists():
        raise ProtectedDataError(f"{look}: already scored — nothing can be re-scored")
    p = look_dir(look, root) / "PREDICTIONS_SAVED.json"
    if not p.exists():
        raise ProtectedDataError(f"{look}: no sealed predictions (the look crashed before saving; report it)")
    rec = json.loads(p.read_text())
    d = Path(rec["dir"])
    if _sha_file(d / "SHA256SUMS") != rec["sha256sums"]:
        raise ProtectedDataError(f"{look}: SHA256SUMS modified")
    listed = set()
    for line in (d / "SHA256SUMS").read_text().splitlines():
        s, n = line.split("  ", 1)
        listed.add(n)
        if _sha_file(d / n) != s:
            raise ProtectedDataError(f"{look}: sealed prediction {n} modified")
    extra = {f.relative_to(d).as_posix() for f in d.rglob("*") if f.is_file() and f.suffix in PRED_SUFFIXES} - listed
    if extra:
        raise ProtectedDataError(f"{look}: unsealed files next to the sealed predictions: {sorted(extra)[:3]}")
    check_prediction_columns(d, rec.get("required_columns") or [])
    return d


def mark_scored(look: str, summary: dict, root: str = ROOT) -> None:
    _excl_write(look_dir(look, root) / "SCORED.json", {"scored": time.strftime("%Y-%m-%dT%H:%M:%S"), **summary})
