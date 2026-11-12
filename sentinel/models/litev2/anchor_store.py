# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""The §4.2 anchor as ONE artefact (M7_verify ruling Q5: "the evaluation anchor and S6 are one hash-pinned
artefact"; lead order of 2026-11-10 20:20 EST, item A3 (iii)).

The anchor model is fitted once, on the v2 training rows only, and written to
``$SENTINEL_RUNS/lite/litev2_anchors/anchor_model.pkl`` together with the out-of-fold anchors of the training rows
(``anchors_train.parquet``). Its sha256 is written in one registry row (:data:`EXPERIMENT`) before any validation
row is loaded. From then on:

* the anchors of an evaluation block are ``predict_parity(model, X_block)`` of that file, written once to
  ``anchors_<block>.parquet`` (:func:`eval_anchors`);
* comparator S6 is that same table (:func:`s6_predictions`) — no second model, no second prediction;
* every reader (:func:`load_model`, :func:`load_table`) checks the file against the registered sha256 or against
  the sha256 the table was written with, and refuses a file that differs;
* the artefact is never rebuilt: :func:`build` refuses when a registry row or a file exists.

The refit table is checked before the fit (``checks.assert_refit_clean``: no validation- or test-role chip or water
body). The pickle is our own file and is loaded only after its hash has been checked.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

from ...core import paths
from . import labels2b as L
from . import tabular as T

WORKSTREAM = "M7-lite"
EXPERIMENT = "lite/v2_anchor_artefact"
MODEL = "litev2_anchor_S6"
DIR = "litev2_anchors"
MODEL_FILE, TRAIN_FILE, META_FILE = "anchor_model.pkl", "anchors_train.parquet", "meta.json"


class AnchorStoreError(RuntimeError):
    """The anchor artefact is missing, unregistered, or differs from its registered hash."""


def store_dir(root=None) -> Path:
    return Path(root) if root is not None else paths.RUNS_ROOT / "lite" / DIR


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _uid_sha(uid) -> str:
    return hashlib.sha256("|".join(pd.Series(uid).astype(str)).encode()).hexdigest()


def _registry(registry):
    if registry is None:
        from ...core.registry import read_registry

        registry = read_registry()
    return [r for r in registry if r.get("workstream") == WORKSTREAM and r.get("experiment") == EXPERIMENT]


def registered(registry: list | None = None) -> dict:
    """``extra`` of the artefact's registry row ({} when there is none). More than one row is an error."""
    rows = _registry(registry)
    if len(rows) > 1:
        raise AnchorStoreError(f"{len(rows)} registry rows for the anchor artefact; there is one artefact")
    return dict(rows[0].get("extra") or {}) if rows else {}


def _meta(root=None) -> dict:
    f = store_dir(root) / META_FILE
    return json.loads(f.read_text()) if f.is_file() else {"tables": {}}


def build(tr: pd.DataFrame, X: pd.DataFrame, *, keys: pd.DataFrame | None = None, root=None,
          registry: list | None = None, log_row=None, seeds=T.SEEDS, check_rows: bool = True) -> dict:
    """Fit the anchor model once on the training rows ``tr`` (features ``X``), write the artefact and its registry
    row, and return ``{'train': out-of-fold anchors, 'model': model, 'info': ...}``.

    ``tr`` needs uid, role, track, water_body and the label columns. No evaluation row is an input of this
    function. Refuses a second build."""
    from . import checks as C
    from . import rows as RW

    d = store_dir(root)
    if registered(registry) or (d / MODEL_FILE).exists() or (d / TRAIN_FILE).exists():
        raise AnchorStoreError("the anchor artefact exists already; it is built once and never rebuilt")
    clean = C.assert_refit_clean(tr, keys) if check_rows else {"skipped": True}
    anc = T.anchors(X, tr[list(RW.Y_COLS)], tr["water_body"].to_numpy(), seeds=seeds)
    d.mkdir(parents=True, exist_ok=True)
    (d / MODEL_FILE).write_bytes(pickle.dumps(anc["model"], protocol=4))
    anc["train"].assign(uid=tr["uid"].astype(str).to_numpy()).to_parquet(d / TRAIN_FILE, index=False)
    sha = {MODEL_FILE: _sha(d / MODEL_FILE), TRAIN_FILE: _sha(d / TRAIN_FILE)}
    extra = {"written_by": "M7", "sha256": sha, "anchor_sha256": sha[MODEL_FILE], "n_train_rows": int(len(tr)),
             "train_uid_sha256": _uid_sha(tr["uid"]), "refit_table": clean, "info": anc["info"],
             "label_version": L.VERSION}
    meta = _meta(root)
    meta.update(anchor_sha256=sha[MODEL_FILE], sha256=sha, train_uid_sha256=extra["train_uid_sha256"])
    (d / META_FILE).write_text(json.dumps(meta, indent=1))
    if log_row is None:
        from ...core.registry import log_result as log_row
    log_row(workstream=WORKSTREAM, experiment=EXPERIMENT, model=MODEL, metric="n_train_rows", value=float(len(tr)),
            split="train", n=int(len(tr)), is_baseline=True, artifacts=[str(d / MODEL_FILE), str(d / TRAIN_FILE)],
            notes="the one anchor artefact of §4.2: evaluation anchors and comparator S6 are predictions of this "
                  "file; fitted on the v2 training rows only, registered before any validation row is loaded",
            extra=extra)
    return {"train": anc["train"], "model": anc["model"], "info": {**anc["info"], "anchor_sha256": sha[MODEL_FILE]}}


def _require(registry, root) -> tuple[Path, dict]:
    d, reg = store_dir(root), registered(registry)
    if not reg:
        raise AnchorStoreError(f"the anchor artefact is not registered ({EXPERIMENT}); run the tabular comparators "
                               "first")
    for f, want in reg["sha256"].items():
        if not (d / f).is_file():
            raise AnchorStoreError(f"{d / f} is missing; the anchor artefact is not rebuilt")
        if _sha(d / f) != want:
            raise AnchorStoreError(f"{d / f} differs from its registered sha256")
    return d, reg


def load_model(registry: list | None = None, root=None) -> tuple[dict, str]:
    """The registered anchor model and its sha256 (file checked before it is unpickled)."""
    d, reg = _require(registry, root)
    return pickle.loads((d / MODEL_FILE).read_bytes()), reg["anchor_sha256"]


def first_stage(tr: pd.DataFrame, registry: list | None = None, root=None) -> dict:
    """``{'train', 'model', 'info'}`` of the registered artefact, aligned with the training rows ``tr``."""
    d, reg = _require(registry, root)
    if _uid_sha(tr["uid"]) != reg["train_uid_sha256"]:
        raise AnchorStoreError("the training rows differ from the rows the anchor artefact was fitted on")
    oof = pd.read_parquet(d / TRAIN_FILE)
    model, sha = load_model(registry, root)
    return {"train": oof[list(T.QUANTITIES)].reset_index(drop=True), "model": model,
            "info": {**reg.get("info", {}), "anchor_sha256": sha}}


def eval_anchors(block: str, uid, X: pd.DataFrame, registry: list | None = None, root=None) -> pd.DataFrame:
    """Anchors of an evaluation block: predictions of the registered model, written once to
    ``anchors_<block>.parquet``. A table that exists is read back (after its hash and its rows are checked); it is
    never predicted a second time."""
    if block == "train":
        raise AnchorStoreError("training rows take the out-of-fold anchors, never a prediction of the refitted model")
    d, reg = _require(registry, root)
    f, meta = d / f"anchors_{block}.parquet", _meta(root)
    if f.exists():
        return load_table(block, uid, registry, root)
    if block in meta.get("tables", {}):
        raise AnchorStoreError(f"{f.name} was written before and is missing now")
    model, sha = load_model(registry, root)
    out = T.predict_parity(model, X).reset_index(drop=True).assign(uid=pd.Series(uid).astype(str).to_numpy())
    out.to_parquet(f, index=False)
    meta.setdefault("tables", {})[block] = {"sha256": _sha(f), "anchor_sha256": sha, "n_rows": int(len(out)),
                                            "uid_sha256": _uid_sha(out["uid"])}
    (d / META_FILE).write_text(json.dumps(meta, indent=1))
    return out


def load_table(name: str, uid, registry: list | None = None, root=None) -> pd.DataFrame:
    """Anchor table ``name`` (``train`` or an evaluation block) aligned with ``uid``; hashes checked."""
    d, reg = _require(registry, root)
    f = d / f"anchors_{name}.parquet"
    if not f.is_file():
        raise AnchorStoreError(f"{f} is missing: run litev2_comparators.py --group tabular first")
    if name != "train":
        t = _meta(root).get("tables", {}).get(name)
        if not t or t["anchor_sha256"] != reg["anchor_sha256"] or _sha(f) != t["sha256"]:
            raise AnchorStoreError(f"{f.name} is not the table written from the registered anchor artefact")
    a = pd.read_parquet(f).set_index("uid")
    uid = pd.Series(uid).astype(str)
    if a.index.duplicated().any() or not uid.isin(a.index).all() or len(a) != len(uid):
        raise AnchorStoreError(f"{f.name} does not cover exactly the rows asked for")
    return a.loc[uid.to_numpy(), list(T.QUANTITIES)].reset_index().rename(columns={"index": "uid"})


def table_sha256(name: str, root=None) -> str:
    return _sha(store_dir(root) / f"anchors_{name}.parquet")


def s6_predictions(table: pd.DataFrame) -> dict:
    """Comparator S6 = the evaluation anchors of the block, target by target (the same numbers)."""
    return {t: table[t].to_numpy(float) for t in L.TARGETS}


__all__ = ["EXPERIMENT", "MODEL", "AnchorStoreError", "store_dir", "registered", "build", "load_model",
           "first_stage", "eval_anchors", "load_table", "table_sha256", "s6_predictions"]
