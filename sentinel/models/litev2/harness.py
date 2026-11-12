# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Shared plumbing of the v2 runs: feature store, teacher means with the Test-S mask, scoring of an evaluation
block, registry rows, and the comparators-first gate (amendment §5: "every comparator's validation rows are in the
registry before the first validation evaluation of ANCHOR-Lite, ANCHOR-Set or an ablation. The candidate's
evaluation entry point checks this and refuses otherwise"; by the lead's ruling A4 the same gate also stands before
every candidate FIT — :func:`require_gate`).

Registry: workstream ``M7-lite``; experiment :data:`EXPERIMENT` per evaluation block; model ``litev2_<name>``.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from ...core import paths
from ..lite import baselines as B
from ..lite import evaluate as E
from ..lite import runner as R
from . import labels2b as L
from . import rows as RW
from . import stamp
from . import tabular as T
from . import tests_mask as TM

EXPERIMENT = {"national_val": "litev2_val_national", "val_p2": "litev2_val_p2"}
BLOCK_TARGETS = {"national_val": tuple(L.TARGETS), "val_p2": ("turb_fnu",)}
DESCRIPTIVE = {"national_val": ("turb_fnu", "ssc"), "val_p2": ()}        # §3.4 item 3: recovered labels on val
REG_ALIAS = {"turb_fnu": "turbidity", "ssc": "tss"}                      # same metric family (MdSA, log10)
MIN_N = 10                                                               # lite_national_protocol.md N2
MODEL_PREFIX = "litev2_"
KD_RUN = "teacher_targets_litesafe_national2"                            # the existing file; nothing new is computed
MASK_RUN = "tests_mask_national2b"
DENSENET_LRS = ("1e-4", "3e-4")

P1_NAMES = tuple(f"P1_{f}" for t in ("secchi", "turbidity", "chla") for f in B.BAND_RATIO_FORMS[t])
TABULAR = ("S1_train_median", "S2_loc_knn", "S3_loc_season_gbm", "S4_gbm_rgbn", "S5_gbm_ctx", "S6_gbm_parity",
           "S7_gbm_parity_unit", "S8_gbm_parity_mt", "S9_gbm_parity_aft", *P1_NAMES, "P2a_cyfi", "P2b_cyfi_parity")
DEEP = tuple(f"{p}_lr{lr}" for p in ("P3a_densenet", "P3b_densenet_parity") for lr in DENSENET_LRS)
COMPARATORS = (*TABULAR, "S10_tabpfn", "S11_gbm_stacked", *DEEP)
CANDIDATES = ("anchor_lite", "anchor_set", "A1", "A3", "A4", "A5")


class GateError(PermissionError):
    """A candidate evaluation was requested before every comparator's validation rows are in the registry."""


# ------------------------------------------------------------------------------------------------ features
def _key(rows: pd.DataFrame) -> str:
    return hashlib.sha256("|".join(rows["uid"].astype(str)).encode()).hexdigest()[:12]


def feature_store(rows: pd.DataFrame, name: str, chips: np.ndarray | None = None) -> dict:
    """``{'X': 98 parity features, 'cyfi': 30 CyFi-style features, 'sc': [s, c] (56)}`` aligned with ``rows``;
    cached under ``$SENTINEL_RUNS/lite/litev2_features`` by the hash of the row keys."""
    d = paths.run_dir("lite", "litev2_features")
    f = d / f"{name}_{_key(rows)}.parquet"
    if f.exists():
        tab = pd.read_parquet(f)
        if tab["uid"].tolist() != rows["uid"].astype(str).tolist():
            raise RW.RowError(f"{f.name} does not match the rows")
    else:
        if chips is None:
            chips = RW.chips(rows)
        scene = RW.scene_table(rows)
        tab = pd.concat([pd.DataFrame({"uid": rows["uid"].astype(str).to_numpy()}),
                         T.feature_table(rows, chips, scene), T.cyfi_features(rows, chips),
                         T.context_block(rows, scene).add_prefix("sc__")], axis=1)
        tmp = f.with_suffix(".tmp")
        tab.to_parquet(tmp, index=False)
        tmp.rename(f)
    sc = tab[[c for c in tab.columns if c.startswith("sc__")]]
    return {"X": tab[T.parity_columns()], "cyfi": tab[[c for c in tab.columns if c.startswith("cy_")]],
            "sc": sc.rename(columns=lambda c: c[4:]), "path": f}


# ------------------------------------------------------------------------------------------------ teacher, F4
def kd_file():
    return paths.RUNS_ROOT / "lite" / KD_RUN / "teacher_targets.parquet"


def mask_flags() -> pd.DataFrame:
    """The Test-S flag table of national-2b, checked against the hash in its meta file."""
    d = paths.RUNS_ROOT / "lite" / MASK_RUN
    meta = json.loads((d / "meta.json").read_text())
    if hashlib.sha256((d / "flags.parquet").read_bytes()).hexdigest() != meta["flags_sha256"]:
        raise TM.MaskError("flags.parquet differs from the hash in its meta file")
    return pd.read_parquet(d / "flags.parquet")


def kd_masked(rows: pd.DataFrame, flags: pd.DataFrame | None = None) -> np.ndarray:
    """True for national rows whose teacher pairing lies on an M2 Test-S site-date (F4). P2 rows have no teacher
    pairing; their chip ids belong to another store and are never looked up in the national flag table."""
    f = (mask_flags() if flags is None else flags).drop_duplicates("naip_chip_id").set_index("naip_chip_id")["kd_masked"]
    nat = (rows["track"] == "national").to_numpy()
    ids = rows.loc[nat, "naip_chip_id"].astype(str)
    if not ids.isin(f.index).all():
        raise TM.MaskError(f"{int((~ids.isin(f.index)).sum())} national chips have no Test-S flag")
    out = np.zeros(len(rows), bool)
    out[nat] = f.reindex(ids).to_numpy(bool)
    return out


def teacher_means(rows: pd.DataFrame, chips: np.ndarray, flags: pd.DataFrame | None = None,
                  kd: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Chip-mean teacher μ of the four teacher targets for TRAINING rows, from the existing teacher-target file,
    with every value on a masked pairing removed (F4). Refuses any non-training row."""
    if set(rows["role"].astype(str)) - {"train", "train_p2"}:
        raise TM.MaskError("teacher values are read for training rows only")
    if kd is None:
        from ..lite import distill as KD

        kd = KD.load_kd_tensors(rows, str(kd_file()), chips=chips)
    tm = T.teacher_chip_means(kd)
    m = kd_masked(rows, flags)
    removed = {c: int((m & tm[c].notna().to_numpy()).sum()) for c in tm.columns}
    tm = tm.copy()
    tm.loc[m, :] = np.nan
    nat = (rows["track"] == "national").to_numpy()
    if flags is None:
        flags = mask_flags()
    TM.assert_masked(pd.concat([rows.loc[nat, ["naip_chip_id"]].reset_index(drop=True),
                                tm.loc[nat].reset_index(drop=True)], axis=1), flags, list(tm.columns))
    if tm.loc[~nat].notna().any().any():
        raise TM.MaskError("a P2 row carries a teacher value")
    return tm, {"n_rows": int(len(rows)), "n_kd_masked": int(m.sum()), "n_teacher_values_removed": removed,
                "n_with_teacher": {c: int(tm[c].notna().sum()) for c in tm.columns}}


def teacher_table(tr: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Chip-mean teacher μ for the v2 training rows (cached; F4-masked). Only the paired national rows reach the
    KD loader; every other row carries NaN."""
    f = paths.run_dir("lite", "litev2_features") / f"teacher_means_train_{_key(tr)}.parquet"
    flags = mask_flags()
    if f.exists():
        tm = pd.read_parquet(f)
        info = {"cached": str(f)}
    else:
        tm = pd.DataFrame(np.nan, index=range(len(tr)), columns=list(T.TEACHER_COLS))
        paired = ((tr["track"] == "national") & tr["s2_chip_id"].notna()).to_numpy()
        sub = tr.loc[paired].reset_index(drop=True)
        part, info = teacher_means(sub, RW.chips(sub), flags)
        tm.loc[paired] = part.to_numpy()
        tm.to_parquet(f, index=False)
    m = kd_masked(tr, flags)
    if len(tm) != len(tr) or tm.loc[m].notna().any().any():
        raise TM.MaskError("the teacher table is misaligned or holds a value on a masked pairing")
    info["n_kd_masked_training_rows"] = int(m.sum())
    return tm, info


def kd_tensors(tr: pd.DataFrame, chips: np.ndarray) -> tuple[dict, np.ndarray]:
    """KD tensors of the frozen loader for the v2 TRAINING rows, aligned with ``tr``, and the F4 mask. Only the
    paired national rows reach the loader (a P2 chip id is never looked up); the other rows get weight 0 and no
    cell. The mask itself is applied by ``fit.attach_kd``."""
    if set(tr["role"].astype(str)) - {"train", "train_p2"}:
        raise TM.MaskError("teacher values are read for training rows only")
    from ..lite import distill as KD

    paired = ((tr["track"] == "national") & tr["s2_chip_id"].notna()).to_numpy()
    sub = tr.loc[paired].reset_index(drop=True)
    part = KD.load_kd_tensors(sub, str(kd_file()), chips=chips[paired])
    n = len(tr)
    out = {}
    for k, v in part.items():
        v = np.asarray(v)
        fill = -1 if k == "cell_map" else (np.nan if k in ("t_mu", "t_var") else 0)
        full = np.full((n, *v.shape[1:]), fill, dtype=np.int16 if k == "cell_map" else np.float32)
        full[paired] = v
        out[k] = full
    return out, kd_masked(tr)


# ------------------------------------------------------------------------------------------------ scoring
def score_block(block: str, rows: pd.DataFrame, preds: dict, sigmas: dict | None = None) -> dict:
    """{target: score} of one model on an evaluation block. National: the frozen primary metric and population of
    the four original targets; FNU and SSC descriptively (all classes, MdSA). P2: FNU MdSA over all rows. Clusters
    are water bodies. Censored rows carry no ``y`` and so never enter a metric."""
    out = {}
    wb, sc = rows["water_body"].to_numpy(), rows["site_class"].astype(str).to_numpy()
    for t in BLOCK_TARGETS[block]:
        if t not in preds:
            continue
        y, p = rows[f"y_{t}"].to_numpy(float), np.asarray(preds[t], float)
        s = None if not sigmas or t not in sigmas else np.asarray(sigmas[t], float)
        ok = np.isfinite(y) & np.isfinite(p)
        if t in L.ORIG:
            ok &= E.primary_mask(t, sc)
        if not ok.any():
            continue
        if t in L.ORIG:
            sco = E.score_primary(t, y, p, wb, sc, sigma_log=s)
        else:
            sco = E.score(REG_ALIAS[t], y, p, wb, sigma_log=s)
            sco["population"] = "all"
        if sco["n"] == 0:
            continue
        sco["descriptive_only"] = t in DESCRIPTIVE[block]
        sco["interpretable_n"] = bool(sco["n"] >= MIN_N)
        out[t] = sco
    return out


def save_predictions(path, rows: pd.DataFrame, preds: dict, sigmas: dict | None = None) -> None:
    df = rows[["uid", "naip_chip_id", "track", "role", "water_body", "site_id", "site_class", "naip_year",
               *[f"y_{t}" for t in L.TARGETS]]].copy()
    for t, p in preds.items():
        df[f"p_{t}"] = np.asarray(p, float)
    for t, s in (sigmas or {}).items():
        df[f"s_{t}"] = np.asarray(s, float)
    df.to_parquet(path, index=False)


def log_block(block: str, name: str, scores: dict, *, cfg: dict, artifacts, is_baseline: bool, notes: str = "",
              extra: dict | None = None) -> list[dict]:
    """Registry rows of one model on one block (validation only; needs the stamp)."""
    stamp.require_stamp(f"registry rows of {name} on {block}")
    return R.log_scores(scores, model=MODEL_PREFIX + name, experiment=EXPERIMENT[block], split="val", cfg=cfg,
                        artifacts=artifacts, smoke=False, is_baseline=is_baseline, notes=notes,
                        extra={"block": block, "label_version": L.VERSION, **(extra or {})})


# ------------------------------------------------------------------------------------------------ the gate
def required_comparators(block: str, aft: tuple = ("secchi", "tss", "chla")) -> dict:
    """{comparator name: targets whose validation rows must be registered} for ``block``. S9 runs only for the
    targets of the registered list (``lite/s9_targets``); a P1 form is required for the targets it is defined for.
    National: the four original targets (FNU and SSC on ``val`` are descriptive and are not required)."""
    need = {}
    targets = [t for t in BLOCK_TARGETS[block] if t not in DESCRIPTIVE[block]]
    for c in COMPARATORS:
        if c == "S9_gbm_parity_aft":
            ts = [t for t in targets if t in aft]
        elif c.startswith("P1_"):
            form = c[3:]
            ts = [t for t in targets if form in B.BAND_RATIO_FORMS[T.P1_FORMS[t]]]
        else:
            ts = list(targets)
        if ts:
            need[c] = ts
    return need


def missing_comparators(block: str, registry: list | None = None) -> list[str]:
    """``'<comparator>/<target>'`` entries of :func:`required_comparators` without a registry row."""
    if registry is None:
        from ...core.registry import read_registry

        registry = read_registry()
    have = {(r.get("model"), r.get("target")) for r in registry
            if r.get("workstream") == R.WORKSTREAM and r.get("experiment") == EXPERIMENT[block]
            and r.get("split") == "val"}
    return [f"{c}/{t}" for c, ts in required_comparators(block).items() for t in ts
            if (MODEL_PREFIX + c, t) not in have]


def require_comparators(block: str, registry: list | None = None) -> None:
    """Called by every candidate evaluation entry point before any validation row is scored."""
    miss = missing_comparators(block, registry)
    if miss:
        raise GateError(f"{len(miss)} comparator rows of {block} are not in the registry yet "
                        f"(first: {', '.join(miss[:6])}); comparators are built first (amendment §5)")


def is_candidate(name: str) -> bool:
    """Everything that is not a comparator: ANCHOR-Lite, ANCHOR-Set, the ablations and their no-KD twins."""
    return name not in COMPARATORS


def require_gate(name: str, registry: list | None = None) -> None:
    """Stricter order (lead ruling A4, 2026-11-10 20:20 EST): no candidate fit and no candidate evaluation starts
    before every comparator validation row of BOTH blocks is in the registry. Called first thing by the fit and the
    evaluation entry points of a candidate, before any row is loaded. A comparator passes."""
    if not is_candidate(name):
        return
    if registry is None:
        from ...core.registry import read_registry

        registry = read_registry()
    for block in EXPERIMENT:
        require_comparators(block, registry)


__all__ = ["EXPERIMENT", "BLOCK_TARGETS", "DESCRIPTIVE", "MIN_N", "MODEL_PREFIX", "TABULAR", "DEEP", "COMPARATORS",
           "CANDIDATES", "P1_NAMES", "GateError", "feature_store", "kd_file", "mask_flags", "kd_masked",
           "teacher_means", "teacher_table", "kd_tensors", "score_block", "save_predictions", "log_block", "required_comparators",
           "missing_comparators", "require_comparators", "is_candidate", "require_gate"]
