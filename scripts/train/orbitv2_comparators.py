#!/usr/bin/env python
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Parity comparator families of the architecture round v2 on dev_v5m (orbit_arch_v2_amendment.md §2.5, §3.5; lead
rulings R-10 to R-14b, R-25).

    python scripts/train/orbitv2_comparators.py --stage gbm_single --fit-scheme recent \
        --v1-selection-row <registry row id> [--max-minutes 110]

Stages, in this order (each one is resumable: a member that has a file is not fitted again; when the time budget
runs out the process exits with status 75 and the same command continues):

  knn          location kNN, k in {4, 8, 16, 32}                                                  CPU
  season       season + location GBM: 16-configuration grid, 3-seed bags                          CPU
  gbm_single   GBM family, single target: 3 feature sets x 16 configurations, 3-seed bags         CPU
  stack        out-of-fold predictions (5 role-block folds of the fit cell) of the inner-best
               single-target GBM of every target and feature set                                  CPU
  gbm_stacked  GBM family, multi-target by stacking                                               CPU
  mdn          MDN_B family: 3 inputs x 3 iteration counts x {linear, log} x {single, joint}      GPU
  select       validation MdSA of every member on the target's selection cell; one member per
               family (the FIRST validation numbers of the comparators)                           CPU
  finalize     9-seed bags of the selected GBMs, MDN_B + GBM average (R-10), trophic chains on
               out-of-fold predictions (R-11), validation-row table, registry rows, model bundle  GPU

New targets with left-censored values: every GBM and OLS member is fitted twice, with DL / sqrt(2) and on
quantified rows only (R-14b); both are family members. kNN uses quantified values. MDN_B uses the interval
likelihood (amendment §3.5).

Rows. Fit rows: the arms' fit cell (every row with a value). Early stopping: the arms' inner rows (dev_v5 rows of
the inner cell). Predictions are stored for every development row outside the fit cell. A base target is scored on
dev_v5 rows, a new target on every dev_v5m row with a quantified value. No protected row is loaded
(``devm.load(training=True)``).

Targets. The single-target GBM is fitted for every head of the A1 arms (its out-of-fold predictions are the
stacking features, and the joint MDN_B head covers every head). Every other member, the selection and the final
comparators exist for the targets with a claim: the four base targets and TP, TN, DOC (§2.5, R-10).

A member predicts every stored row: LightGBM takes missing features as they are, and a missing MDN_B input is
replaced by its fit-row median (``comparators.mdn_inputs``).

Fit scheme (lead clarification 01, K-2, sha256 34f85b83fe8580d91996d207b33d5b55da462bbb5282cc0cd2534449b79761a3).
``--fit-scheme`` is the scheme of the latest registered v1 selection when the first comparator job starts, named
by ``--v1-selection-row``; both go into every run record and registry row. Each scheme has its own folder, so
nothing of a first fit can be carried into a refit under another scheme. ``select`` and ``finalize`` run only
when the arm files name the fit scheme (R-19) and it equals the members' scheme.

Compute (K-3, R-25): at most 6 threads; a member is the unit of resumption (it is fitted from its first seed
again if a job stopped inside it, and its record names every job part that started it); every real job writes
its CPU-hours to the registry.

Every real run is behind ``stamp.require_stamp``. ``--smoke`` is a train-only test of the code path on a
dev_v5m-format cache: one configuration per family, predictions for inner rows only, no ``select`` /
``finalize``, nothing registered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.models.orbit import baselines as BL  # noqa: E402
from sentinel.models.orbit import data as D  # noqa: E402
from sentinel.models.orbit import evaluate as E  # noqa: E402
from sentinel.models.orbit import features as F1  # noqa: E402
from sentinel.models.orbit import pipeline as PL  # noqa: E402
from sentinel.models.orbit_v2 import arm as ARM  # noqa: E402
from sentinel.models.orbit_v2 import comparators as C  # noqa: E402
from sentinel.models.orbit_v2 import devm as DM  # noqa: E402
from sentinel.models.orbit_v2 import stamp as ST  # noqa: E402
from sentinel.models.orbit_v2 import targets as T  # noqa: E402

EXIT_RESUME = 75
STAGES = ("knn", "season", "gbm_single", "stack", "gbm_stacked", "mdn", "select", "finalize")
GPU_STAGES = ("mdn", "finalize")
PLAN_REF = "docs/preregistration/orbit_arch_v2_amendment.md §2.5, §3.5; lead rulings R-10..R-14b; " + PL.PLAN_REF
EXPERIMENT = "orbit_v2_cmp_{v}"
CI_METHOD = "role-block cluster bootstrap (B={b}, percentile)"
# comparator selection cells: v1's for the base targets (scripts/train/orbit_compare.py PRIMARY_TARGET_CELL) ...
BASE_CELL = {"chla": "valxval_lake", "secchi": "valxval_lake", "tss": "valxany_lake", "turbidity": "valxany_lake"}
LAKE_CELLS = ("valxval_lake", "valxany_lake")
# ... and the count rule of §2.5 for a new target (the same constants as scripts/train/orbitv2_power.py)
TRACK_N_MIN_ROWS, TRACK_N_MIN_BLOCKS = 200, 20
TRACK_N = ("tp", "tn", "doc")                      # the targets with a trophic-chain comparator (§2.5)
CHAIN_INPUTS = ("chla", "secchi")
CHAINS = ("mdn_b", "gbm")                          # R-11: both chains are members
CLAIM = ("mdn_b", "gbm", "knn", "mdn_b+gbm", "season_loc_gbm", "trophic_ols")
SMALL_LAKE_KM2 = 1.0                               # protocol §11.4, amendment §1.7
CLARIFICATION_01 = "34f85b83fe8580d91996d207b33d5b55da462bbb5282cc0cd2534449b79761a3"
MIN_FREE_GB, PROJECTED_GB = 60.0, 3.0              # R-25
ARM_SCHEMES = ("base", "recent")                   # R-19: the arms' fit scheme is one of the two
HEAD_ARM = "a1_c1"                                 # any A1 configuration: the head list does not depend on it


def full_limits() -> dict:
    """The ruled families. Only a smoke test or a unit test uses anything smaller."""
    return {"grid": tuple(range(len(C.GBM_GRID))), "feature_sets": tuple(C.GBM_FEATURE_SETS), "ks": tuple(C.KNN_KS),
            "mdn": tuple(C.mdn_members()), "max_rounds": C.GBM_MAX_ROUNDS, "threads": min(6, BL.N_THREADS),
            "member_seeds": tuple(C.GBM_MEMBER_SEEDS), "selected_seeds": tuple(C.GBM_SELECTED_SEEDS)}


LIM = full_limits()


class Budget:
    """Time limit of one job part; ``n`` counts the members it fitted."""

    def __init__(self, minutes: float, part: str = "part"):
        self.t_end, self.part, self.n = time.time() + 60.0 * minutes, part, 0

    def out(self) -> bool:
        return time.time() >= self.t_end


# ------------------------------------------------------------------------------------------------- rows

def track_n_cell(n_rows: int, n_blocks: int) -> str:
    return "valxval_lake" if n_rows >= TRACK_N_MIN_ROWS and n_blocks >= TRACK_N_MIN_BLOCKS else "valxany_lake"


def features(df: pd.DataFrame, pix: dict, feats: dict, carrays: dict, feature_sets=None) -> SimpleNamespace:
    """The comparator inputs of a set of rows. One code path for the rows the stages fit and store
    (``prepare``) and for rows predicted later from the fitted files (``predict_rows``)."""
    df = df.reset_index(drop=True)
    season = BL.location_season_matrix(df)                                 # v1's season + location features
    v1_features = pd.concat([F1.image_features(df, pix), season], axis=1)   # v1's GBM features
    sets = LIM["feature_sets"] if feature_sets is None else feature_sets
    return SimpleNamespace(n=len(df), Xseason=season.to_numpy(float),
                           X={fs: C.gbm_feature_set(fs, v1_features, feats, pix["n_pix"], carrays) for fs in sets},
                           rw3=df[[f"rw3_{b}" for b in D.BANDS]].to_numpy(float),
                           mdn={k: np.asarray(feats[k]) for k in ("cidx", "cdt", "ring")}, carrays=carrays,
                           lat=df["lat"].to_numpy(float), lon=df["lon"].to_numpy(float))


def prepare(df: pd.DataFrame, pix: dict, feats: dict, carrays: dict, fit_scheme: str, *,
            keep: str = "out") -> SimpleNamespace:
    """Everything the stages need. ``keep``: rows whose predictions are stored ("out": every row outside the fit
    cell; "inner": the inner rows only, for the smoke test)."""
    df = df.reset_index(drop=True)
    fit, inner_all = F1.fit_and_inner(df, fit_scheme)
    added = df["added"].to_numpy(bool)
    targets, head_counts = ARM.heads(df, fit, HEAD_ARM)
    y, cens = ARM.target_matrix(df, targets)
    ok, lake = ~df["buffer_excluded"].to_numpy(bool), df["in_lake"].to_numpy(bool)
    role, tb = df["role"].to_numpy(), df["tblock"].to_numpy()
    d = SimpleNamespace(df=df, n=len(df), feats=feats, carrays=carrays, fit_scheme=fit_scheme, targets=list(targets),
                        head_counts=head_counts, fit=fit, inner=inner_all & ~added, added=added, y=y, cens=cens,
                        yq=np.where(cens, np.nan, y), state=np.where(cens, 2, np.where(np.isfinite(y), 1, 0)),
                        block=df["block"].astype(str).to_numpy(), lat=df["lat"].to_numpy(float),
                        lon=df["lon"].to_numpy(float), row_id=df["row_id"].astype(str).to_numpy())
    d.cells = {"valxval_lake": (role == "val") & (tb == "val") & lake & ok,
               "valxany_lake": (role == "val") & np.isin(tb, ["train", "val"]) & lake & ok}
    if any((m & (fit | d.inner)).any() for m in d.cells.values()):
        raise C.ComparatorError("a validation-cell row is in the fit or inner cell")
    d.keep = d.inner.copy() if keep == "inner" else ~fit
    d.keep_idx = np.flatnonzero(d.keep)
    d.rows_sha = hashlib.sha256("\n".join(d.row_id[d.keep]).encode()).hexdigest()
    d.folds = np.full(d.n, -1)
    d.folds[fit] = C.role_block_folds(d.block[fit])
    x = features(df, pix, feats, carrays)
    d.Xseason, d.X, d.rw3 = x.Xseason, x.X, x.rw3
    d.claim = [t for t in d.targets if t in D.TARGETS or t in TRACK_N]
    return d


def variants(t: str) -> tuple:
    """Censoring variants of the GBM and OLS members of a target (R-14b); a base target has no censored value."""
    return tuple(C.CENSORING) if t in T.SPECS else (None,)


def labels(d, t: str, variant) -> np.ndarray:
    j = d.targets.index(t)
    return d.yq[:, j] if variant is None else C.censoring_variant(variant, d.yq[:, j], d.cens[:, j], d.y[:, j])


def expand(d, p) -> np.ndarray:
    full = np.full(d.n, np.nan)
    full[d.keep] = np.asarray(p, float)
    return full


def score_mask(d, t: str, cell: str) -> np.ndarray:
    m = d.cells[cell] & np.isfinite(d.yq[:, d.targets.index(t)])
    return m & ~d.added if t in D.TARGETS else m


def selection_cell(d, t: str) -> str:
    if t in BASE_CELL:
        return BASE_CELL[t]
    m = score_mask(d, t, "valxval_lake")
    return track_n_cell(int(m.sum()), int(np.unique(d.block[m]).size))


# ------------------------------------------------------------------------------------------ member files

def mfile(root: Path, family: str, unit: str, name: str) -> Path:
    return Path(root) / family / unit / (re.sub(r"[^A-Za-z0-9_.+-]+", "_", name).strip("_") + ".npz")


def begin(f: Path, budget: Budget) -> None:
    """Note that this job part starts the member (K-3: a member's record names every part that started it)."""
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f.with_suffix(".started"), "a") as fh:
        fh.write(budget.part + "\n")
    budget.n += 1


def save_member(f: Path, d, pred, **arrays_and_meta) -> None:
    """Predictions for the stored rows, the hash of their row ids and the member's record. ``oof``: out-of-fold
    predictions on the fit cell, stored next to them."""
    f.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(arrays_and_meta)
    extra = {k: np.asarray(meta.pop(k), float) for k in ("oof",) if k in meta}
    st = f.with_suffix(".started")
    meta["parts"] = st.read_text().split() if st.exists() else []
    tmp = f.with_name(f"{f.stem}.tmp{os.getpid()}.npz")
    np.savez(tmp, pred=np.asarray(pred, float), rows_sha=d.rows_sha, meta=json.dumps(meta, default=str), **extra)
    os.replace(tmp, f)


def dump(f: Path, obj) -> None:
    import cloudpickle

    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f"{f.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        cloudpickle.dump(obj, fh)
    os.replace(tmp, f)


def load_member(f: Path, d) -> tuple[np.ndarray, dict]:
    if not Path(f).exists():
        raise SystemExit(f"missing member file {f}: an earlier stage is not complete")
    z = np.load(f, allow_pickle=False)
    if str(z["rows_sha"]) != d.rows_sha:
        raise SystemExit(f"{f}: fitted for other rows than the ones loaded now")
    return z["pred"], json.loads(str(z["meta"]))


def nan_pred(d, cols: int | None = None) -> np.ndarray:
    return np.full(len(d.keep_idx) if cols is None else (len(d.keep_idx), cols), np.nan)


# ------------------------------------------------------------------------------------------------- GBM

def gbm_matrix(d, root: Path, fs: str, mode: str, t: str) -> np.ndarray:
    if fs == "season":
        return d.Xseason
    X = d.X[fs]
    if mode == "stacked":
        X = C.stack_features(X, load_stack(d, root, fs), t)
    return X.to_numpy(float)


def fit_bag(d, X: np.ndarray, t: str, variant, j: int, seeds, rows=None):
    """One grid configuration as a seed bag: fitted on ``rows`` (default the fit cell), each seed stopped early on
    the quantified inner rows."""
    fit = d.fit if rows is None else rows
    return C.fit_gbm_bag(X[fit], labels(d, t, variant)[fit], X[d.inner], d.yq[d.inner, d.targets.index(t)],
                         C.GBM_GRID[j], seeds, n_threads=LIM["threads"], max_rounds=LIM["max_rounds"])


def gbm_family(family: str) -> list:
    """(member name, feature set, mode, grid index) without the censoring variant."""
    if family == "season":
        return [(f"season_loc_gbm[g{j:02d}", "season", "single", j) for j in LIM["grid"]]
    mode = {"gbm_single": "single", "gbm_stacked": "stacked"}[family]
    return [(C.member_name(fs, mode, j)[:-1], fs, mode, j) for fs in LIM["feature_sets"] for j in LIM["grid"]]


def gbm_name(stem: str, variant) -> str:
    return stem + (f"|{variant}]" if variant else "]")


def run_gbm(d, root: Path, family: str, budget: Budget, log=print) -> bool:
    folder = "season" if family == "season" else "gbm"
    for t in (d.targets if family == "gbm_single" else d.claim):
        jt = d.targets.index(t)
        mats = {}
        for stem, fs, mode, j in gbm_family(family):
            for v in variants(t):
                name = gbm_name(stem, v)
                f = mfile(root, folder, t, name)
                if f.exists():
                    continue
                if budget.out():
                    return False
                if fs not in mats:
                    mats[fs] = gbm_matrix(d, root, fs, mode, t)
                X, t0 = mats[fs], time.time()
                begin(f, budget)
                spec = {"name": name, "target": t, "feature_set": fs, "mode": mode, "grid": j, "censoring": v}
                try:
                    pr = fit_bag(d, X, t, v, j, LIM["member_seeds"])
                    p = pr(X[d.keep])
                    save_member(f, d, p, **spec, rounds=pr.n_rounds, seeds=pr.seeds,
                                inner_mdsa=C.val_mdsa(expand(d, p), d.yq[:, jt], d.inner))
                except C.ComparatorError as e:
                    save_member(f, d, nan_pred(d), **spec, skipped=str(e))
                log(f"{time.strftime('%H:%M:%S')} {folder}/{t}/{name} {time.time() - t0:.0f}s")
    return True


def stack_file(root: Path, fs: str, t: str) -> Path:
    return mfile(root, "stack", fs, t)


def run_stack(d, root: Path, budget: Budget, log=print) -> bool:
    """§3.5: out-of-fold predictions of the inner-best single-target GBM (lowest MdSA on the inner rows, over the
    grid and the censoring variants) of every target, per feature set. The chosen member is fitted once more on
    the whole fit cell and kept as a model: its predictions are the stacking features of every row outside the
    fit cell, at validation and at Test-S."""
    fi = np.flatnonzero(d.fit)
    for fs in LIM["feature_sets"]:
        X = d.X[fs].to_numpy(float)
        for t in d.targets:
            f = stack_file(root, fs, t)
            if f.exists():
                continue
            if budget.out():
                return False
            begin(f, budget)
            cand = {}
            for stem, fs_, _, j in gbm_family("gbm_single"):
                for v in variants(t):
                    if fs_ == fs:
                        p, meta = load_member(mfile(root, "gbm", t, gbm_name(stem, v)), d)
                        cand[gbm_name(stem, v)] = (meta.get("inner_mdsa"), j, v)
            try:
                best = C.select_member({k: c[0] for k, c in cand.items()})
                _, j, v = cand[best]

                def fp(tr, te):
                    rows = np.zeros(d.n, bool)
                    rows[fi[tr]] = True
                    return fit_bag(d, X, t, v, j, LIM["member_seeds"], rows=rows)(X[fi[te]])
                oof = C.out_of_fold(fp, d.folds[d.fit])
                pr = fit_bag(d, X, t, v, j, LIM["member_seeds"])
                dump(f.with_suffix(".pkl"), {"predict": pr, "member": best, "feature_set": fs, "target": t})
                save_member(f, d, pr(X[d.keep]), oof=oof, member=best, grid=j, censoring=v, rounds=pr.n_rounds)
            except C.ComparatorError as e:
                save_member(f, d, nan_pred(d), skipped=str(e))
            log(f"{time.strftime('%H:%M:%S')} stack/{fs}/{t}")
    return True


def load_stack(d, root: Path, fs: str) -> dict:
    """{target: prediction for every row}: out-of-fold on the fit cell, the full-fit member elsewhere. A target
    whose single-target GBM could not be fitted has no entry."""
    out = {}
    for t in d.targets:
        f = stack_file(root, fs, t)
        p, meta = load_member(f, d)
        if "skipped" in meta:
            continue
        full = expand(d, p)
        full[d.fit] = np.load(f, allow_pickle=False)["oof"]
        out[t] = full
    return out


# ------------------------------------------------------------------------------------------------- kNN

def run_knn(d, root: Path, budget: Budget, log=print) -> bool:
    for t in d.claim:
        jt = d.targets.index(t)
        todo = [k for k in LIM["ks"] if not mfile(root, "knn", t, f"knn[k{k:02d}]").exists()]
        if not todo:
            continue
        if budget.out():
            return False
        pr = C.fit_knn(d.lat[d.fit], d.lon[d.fit], d.yq[d.fit, jt])
        for k in todo:
            begin(mfile(root, "knn", t, f"knn[k{k:02d}]"), budget)
            save_member(mfile(root, "knn", t, f"knn[k{k:02d}]"), d, pr(d.lat[d.keep], d.lon[d.keep], k),
                        name=f"knn[k{k:02d}]", target=t, k=k)
        log(f"{time.strftime('%H:%M:%S')} knn/{t}")
    return True


# ----------------------------------------------------------------------------------------------- MDN_B

def mdn_fit(d, inp: str, n_iter: int, x_log: bool, cols: list, rows: np.ndarray, device: str, seed: int = 0):
    """One trained MDN_B member (10 initialisations) on ``rows``; returns ``predict(row mask, read-out)`` [n, T]."""
    def sub(m):
        return {k: np.asarray(d.feats[k])[m] for k in ("cidx", "cdt", "ring")}
    Xf, fill = C.mdn_inputs(inp, d.rw3[rows], sub(rows), d.carrays)
    model = C.JointMDN(len(cols), n_ens=C.MDN_N_INIT, n_iter=n_iter, x_log=x_log, seed=seed, device=device)
    model.fit(Xf, d.y[rows][:, cols], d.state[rows][:, cols])

    def predict(m, readout):
        return model.predict(C.mdn_inputs(inp, d.rw3[m], sub(m), d.carrays, fill)[0], readout)
    predict.model, predict.fill = model, fill
    return predict


def mdn_units(d, head: str) -> list:
    """(unit name, target columns): the joint head covers every head of the A1 arms, a single head one target
    with a claim."""
    if head == "joint":
        return [("joint", list(range(len(d.targets))))]
    return [(t, [d.targets.index(t)]) for t in d.claim]


def mdn_trained(inp, n_iter, x_log, head) -> str:
    return f"mdn_b[{inp}|{n_iter}|{'log' if x_log else 'lin'}|{head}]"


def run_mdn(d, root: Path, device: str, budget: Budget, log=print) -> bool:
    for inp, n_iter, x_log, head in LIM["mdn"]:
        for unit, cols in mdn_units(d, head):
            name = mdn_trained(inp, n_iter, x_log, head)
            f = mfile(root, "mdn", unit, name)
            if f.exists():
                continue
            if budget.out():
                return False
            t0 = time.time()
            begin(f, budget)
            spec = {"name": name, "input": inp, "n_iter": n_iter, "x_log": x_log, "head": head,
                    "targets": [d.targets[c] for c in cols]}
            try:
                pr = mdn_fit(d, inp, n_iter, x_log, cols, d.fit, device)
                p = np.stack([pr(d.keep, r) for r in C.MDN_READOUTS])              # [read-out, n, T]
                dump(f.with_suffix(".pkl"), {"model": pr.model, "fill": pr.fill, **spec})
                save_member(f, d, p, **spec, n_fit=pr.model.n_fit)
            except C.ComparatorError as e:
                save_member(f, d, np.full((len(C.MDN_READOUTS), len(d.keep_idx), len(cols)), np.nan), **spec,
                            skipped=str(e))
            log(f"{time.strftime('%H:%M:%S')} mdn/{unit}/{name} {time.time() - t0:.0f}s")
    return True


# ------------------------------------------------------------------------------------------- selection

def family_members(d, root: Path, t: str) -> dict:
    """{family: {member name: (stored predictions, spec)}} of one target, in the declared order."""
    jt = d.targets.index(t)
    fam = {"gbm": {}, "season_loc_gbm": {}, "knn": {}, "mdn_b": {}}
    for family, key, folder in (("gbm_single", "gbm", "gbm"), ("gbm_stacked", "gbm", "gbm"),
                                ("season", "season_loc_gbm", "season")):
        for stem, _, _, _ in gbm_family(family):
            for v in variants(t):
                p, meta = load_member(mfile(root, folder, t, gbm_name(stem, v)), d)
                fam[key][gbm_name(stem, v)] = (p, meta)
    for k in LIM["ks"]:
        p, meta = load_member(mfile(root, "knn", t, f"knn[k{k:02d}]"), d)
        fam["knn"][meta["name"]] = (p, meta)
    for inp, n_iter, x_log, head in LIM["mdn"]:
        unit, col = ("joint", jt) if head == "joint" else (t, 0)
        p, meta = load_member(mfile(root, "mdn", unit, mdn_trained(inp, n_iter, x_log, head)), d)
        for r, readout in enumerate(C.MDN_READOUTS):
            fam["mdn_b"][C.mdn_name(inp, n_iter, x_log, head, readout)] = (p[r][:, col], {**meta, "readout": readout})
    return fam


def select(d, root: Path, log=print) -> dict:
    """Within each family the member with the lowest validation MdSA on the target's selection cell (§3.5)."""
    sel, rows = {}, []
    for t in d.claim:
        jt, cell = d.targets.index(t), selection_cell(d, t)
        m = score_mask(d, t, cell)
        rec = {"cell": cell, "n": int(m.sum()), "n_blocks": int(np.unique(d.block[m]).size), "families": {}}
        sel[t] = rec
        if not m.any():
            rec["skipped"] = "no quantified row in the selection cell: no comparator, no claim"
            log(f"{t}: {rec['skipped']}")
            continue
        for family, members in family_members(d, root, t).items():
            scores = {name: C.val_mdsa(expand(d, p), d.yq[:, jt], m) for name, (p, _) in members.items()}
            best = C.select_member(scores)
            rec["families"][family] = {"selected": best, "mdsa": scores[best], "spec": members[best][1],
                                       "n_members": len(scores),
                                       "n_scored": int(np.isfinite(list(scores.values())).sum())}
            rows += [{"target": t, "cell": cell, "family": family, "member": k, "mdsa": v, "selected": k == best}
                     for k, v in scores.items()]
            log(f"{t} {family}: {len(scores)} members, selected {best}")
    pd.DataFrame(rows).to_parquet(Path(root) / "member_scores.parquet", index=False)
    PL.write_json(Path(root) / "selection.json", {"fit_scheme": d.fit_scheme, "rows_sha": d.rows_sha,
                                                  "targets": d.targets, "claim_targets": d.claim, "selection": sel})
    return sel


# -------------------------------------------------------------------------------------------- finalize

def lake_area_km2(waterbody) -> np.ndarray:
    """HydroLAKES v1.0 ``Lake_area`` (km2) of in-lake units (``L<Hylak_id>``); NaN for the others. Geometry only:
    the definition of protocol §11.4 (the same table and rule as ``scripts/eval/orbit_test.py``)."""
    lakes = pd.read_parquet(paths.processed("hydrolakes") / "hydrolakes_na_eu.parquet",
                            columns=["Hylak_id", "Lake_area"])
    area = dict(zip(lakes.Hylak_id.astype(int).astype(str), lakes.Lake_area.astype(float)))
    return np.array([area.get(w[1:], np.nan) if w.startswith("L") else np.nan
                     for w in pd.Series(waterbody).astype(str)], float)


def small_lake(df: pd.DataFrame) -> np.ndarray:
    """Small-lake stratum (amendment §1.7): in-lake rows whose HydroLAKES area is below 1 km2."""
    return df["in_lake"].to_numpy(bool) & (lake_area_km2(df["waterbody"]) < SMALL_LAKE_KM2)


def ntu_class(df: pd.DataFrame, base: pd.DataFrame | None = None) -> np.ndarray:
    """True for rows whose turbidity value is of the NTU class (R-18). Added rows carry no base-target value."""
    if "turbidity_unit_class" in df.columns:
        cls = df["turbidity_unit_class"]
    elif base is not None and "turbidity_unit_class" in base.columns:
        cls = df["row_id"].astype(str).map(base.drop_duplicates("row_id").set_index("row_id")["turbidity_unit_class"])
    else:
        raise SystemExit("no turbidity unit class for the rows: the NTU-class endpoint cannot be formed (R-18)")
    return (cls.astype(str) == "NTU").to_numpy()


def _oof(d, fit_predict) -> np.ndarray:
    """Out-of-fold predictions on the fit cell; ``fit_predict(fit rows mask, predicted rows mask)``."""
    fi = np.flatnonzero(d.fit)

    def fp(tr, te):
        a, b = np.zeros(d.n, bool), np.zeros(d.n, bool)
        a[fi[tr]], b[fi[te]] = True, True
        return fit_predict(a, b)
    return C.out_of_fold(fp, d.folds[d.fit])


def finalize(d, root: Path, device: str, budget: Budget, *, ntu: np.ndarray, n_boot: int = E.N_BOOT, log=print):
    """Comparator predictions of every target with a claim: selected MDN_B, selected GBM as a 9-seed bag, kNN,
    season + location GBM as a 9-seed bag, the MDN_B + GBM average (R-10) and, for TP / TN / DOC, the trophic
    chain on out-of-fold predictions (R-11, R-14b). Returns None when the time budget ran out (the same call
    continues: every fitted piece has a file)."""
    root, fin = Path(root), Path(root) / "final"
    saved = json.loads((root / "selection.json").read_text())
    if saved["rows_sha"] != d.rows_sha or saved["targets"] != d.targets or saved["fit_scheme"] != d.fit_scheme:
        raise SystemExit("selection.json was written for other rows, targets or another fit scheme")
    sel = saved["selection"]
    done = [t for t in d.claim if sel[t]["families"]]
    if any(z not in done for z in CHAIN_INPUTS):
        raise SystemExit("no selected chl-a or Secchi comparator: the trophic chain cannot be formed")

    def rel(f):
        return str(Path(f).relative_to(root))
    preds, bundle = {t: {} for t in done}, {t: {} for t in done}
    for t in done:
        jt, fams = d.targets.index(t), sel[t]["families"]
        for family, folder in (("gbm", "gbm"), ("season_loc_gbm", "season")):
            s = fams[family]["spec"]
            f = mfile(fin, folder, t, "bag9")
            if not f.exists():
                if budget.out():
                    return None
                begin(f, budget)
                X = gbm_matrix(d, root, s["feature_set"], s["mode"], t)
                pr = fit_bag(d, X, t, s["censoring"], s["grid"], LIM["selected_seeds"])
                dump(f.with_suffix(".pkl"), {"predict": pr, "spec": s})
                save_member(f, d, pr(X[d.keep]), spec=s, rounds=pr.n_rounds, seeds=pr.seeds)
                log(f"{time.strftime('%H:%M:%S')} final {family}/{t}")
            preds[t][family] = load_member(f, d)[0]
            bundle[t][family] = {"spec": s, "file": rel(f.with_suffix(".pkl"))}
            if s["mode"] == "stacked":
                bundle[t][family]["stack_files"] = {
                    o: rel(stack_file(root, s["feature_set"], o).with_suffix(".pkl"))
                    for o in load_stack(d, root, s["feature_set"]) if o != t}
        k = fams["knn"]["spec"]["k"]
        preds[t]["knn"] = load_member(mfile(root, "knn", t, f"knn[k{k:02d}]"), d)[0]
        bundle[t]["knn"] = {"k": k, "y_fit": d.yq[d.fit, jt]}
        s = fams["mdn_b"]["spec"]
        unit, col = ("joint", jt) if s["head"] == "joint" else (t, 0)
        f = mfile(root, "mdn", unit, s["name"])
        preds[t]["mdn_b"] = load_member(f, d)[0][C.MDN_READOUTS.index(s["readout"])][:, col]
        bundle[t]["mdn_b"] = {"spec": s, "column": col, "file": rel(f.with_suffix(".pkl"))}
        preds[t]["mdn_b+gbm"] = C.average_log10(preds[t]["mdn_b"], preds[t]["gbm"])
    # R-11: out-of-fold predictions of the selected chl-a and Secchi members on the fit cell, per chain
    z = {}
    for chain in CHAINS:
        for zt in CHAIN_INPUTS:
            jz, s = d.targets.index(zt), sel[zt]["families"][chain]["spec"]
            f = mfile(fin, "oof", chain, zt)
            if not f.exists():
                if budget.out():
                    return None
                begin(f, budget)
                if chain == "gbm":
                    X = gbm_matrix(d, root, s["feature_set"], s["mode"], zt)
                    oof = _oof(d, lambda a, b: fit_bag(d, X, zt, s["censoring"], s["grid"], LIM["selected_seeds"],
                                                       rows=a)(X[b]))
                else:
                    cols, col = (list(range(len(d.targets))), jz) if s["head"] == "joint" else ([jz], 0)
                    oof = _oof(d, lambda a, b: mdn_fit(d, s["input"], s["n_iter"], s["x_log"], cols, a, device)(
                        b, s["readout"])[:, col])
                save_member(f, d, preds[zt][chain], oof=oof, spec=s)
                log(f"{time.strftime('%H:%M:%S')} out-of-fold {chain}/{zt}")
            full = expand(d, preds[zt][chain])
            full[d.fit] = np.load(f, allow_pickle=False)["oof"]
            z[chain, zt] = full
    chains = {}
    for t in [t for t in TRACK_N if t in done]:
        jt, m = d.targets.index(t), score_mask(d, t, sel[t]["cell"])
        members, scores = {}, {}
        for chain in CHAINS:
            zc, zs = z[chain, "chla"], z[chain, "secchi"]
            for v in C.CENSORING:
                name = f"trophic_ols[{chain}|{v}]"
                try:
                    coef = C.ols_fit(zc[d.fit], zs[d.fit], labels(d, t, v)[d.fit])
                    members[name] = (C.ols_predict(coef, zc[d.keep], zs[d.keep]), [float(c) for c in coef])
                    scores[name] = C.val_mdsa(expand(d, members[name][0]), d.yq[:, jt], m)
                except C.ComparatorError:
                    scores[name] = float("nan")
        best = C.select_member(scores)
        chain, v = best[len("trophic_ols["):-1].split("|")
        preds[t]["trophic_ols"] = members[best][0]
        bundle[t]["trophic_ols"] = {"member": best, "coef": members[best][1], "chain": chain, "censoring": v}
        chains[t] = {"selected": best, "scores": scores}
    # validation rows (the input of scripts/train/orbitv2_power.py and of the unit script) and their scores
    small = small_lake(d.df)
    tabs, scores = [], []
    for t in done:
        jt = d.targets.index(t)
        for cell in LAKE_CELLS:
            m = score_mask(d, t, cell)
            tab = pd.DataFrame({"target": t, "cell": cell, "row_id": d.row_id[m], "block": d.block[m],
                                "y": d.yq[m, jt], "small_lake": small[m], "ntu_class": np.asarray(ntu, bool)[m],
                                "added": d.added[m]})
            for c in CLAIM:
                tab[f"cmp:{c}"] = expand(d, preds[t][c])[m] if c in preds[t] else np.nan
            tabs.append(tab)
            if not len(tab):
                continue
            pp = {c: tab[f"cmp:{c}"].to_numpy(float) for c in CLAIM if c in preds[t]}
            sc, _ = E.score_models(tab["y"].to_numpy(float), pp, tab["block"].to_numpy(), n_boot=n_boot)
            scores.append(sc.assign(target=t, cell=cell, n=int(len(tab)), selection_cell=cell == sel[t]["cell"]))
    val = pd.concat(tabs, ignore_index=True)
    val.to_parquet(root / "val_rows.parquet", index=False)
    scores = pd.concat(scores, ignore_index=True)
    scores.to_csv(root / "val_scores.csv", index=False)
    dump(root / "bundle.pkl", {"targets": d.targets, "claim_targets": done, "fit_scheme": d.fit_scheme,
                               "comparators": bundle, "selection": sel, "chains": chains,
                               "knn_fit": {"lat": d.lat[d.fit], "lon": d.lon[d.fit]}})
    skip = ("SHA256SUMS", "comparators.json")
    sums = {rel(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*"))
            if p.is_file() and p.name not in skip and not p.name.startswith("part_")}
    (root / "SHA256SUMS").write_text("".join(f"{h}  {p}\n" for p, h in sums.items()))
    out = {"fit_scheme": d.fit_scheme, "targets": d.targets, "claim_targets": done, "head_counts": d.head_counts,
           "selection": sel, "chains": chains, "n_val_rows": int(len(val)), "rows_sha": d.rows_sha,
           "sha256sums_sha256": hashlib.sha256((root / "SHA256SUMS").read_bytes()).hexdigest()}
    PL.write_json(root / "comparators.json", out)
    return {"summary": out, "scores": scores, "val": val}


# ------------------------------------------------------------------------------- prediction of new rows

def load_pickle(f: Path):
    import cloudpickle

    with open(f, "rb") as fh:
        return cloudpickle.load(fh)


def verify_sums(root: Path) -> str:
    """Every file named in ``SHA256SUMS`` exists with the listed hash; returns the hash of ``SHA256SUMS`` (the
    value a freeze pins for the comparators)."""
    root = Path(root)
    sums = root / "SHA256SUMS"
    if not sums.is_file():
        raise SystemExit(f"{root}: no SHA256SUMS (the finalize stage is not complete)")
    lines = [ln.split("  ", 1) for ln in sums.read_text().splitlines() if ln.strip()]
    if "bundle.pkl" not in {p for _, p in lines}:
        raise SystemExit(f"{sums}: bundle.pkl is not listed")
    for h, p in lines:
        g = root / p
        if not g.is_file() or hashlib.sha256(g.read_bytes()).hexdigest() != h:
            raise SystemExit(f"{g}: missing or changed since the comparators were sealed")
    return hashlib.sha256(sums.read_bytes()).hexdigest()


def bundle_feature_sets(bundle: dict) -> list:
    return sorted({e[fam]["spec"]["feature_set"] for e in bundle["comparators"].values() for fam in ("gbm",)})


def predict_rows(root: Path, x: SimpleNamespace, bundle: dict | None = None, device: str | None = None) -> dict:
    """Comparator predictions (log10) of rows no stage has seen: ``{target: {claim comparator: [n]}}``, from the
    files named by ``bundle.pkl`` alone (the fitted bags, the stacking members, the MDN_B member with its fill
    values, the kNN table of the fit cell, the chain coefficients). ``x``: ``features`` of the rows with the
    feature sets of ``bundle_feature_sets``. The Test-S look calls this inside its open claim; the test file
    requires it to reproduce ``val_rows.parquet`` on the validation rows."""
    root = Path(root)
    b = load_pickle(root / "bundle.pkl") if bundle is None else bundle
    files, out = {}, {}

    def member(rel):
        if rel not in files:
            files[rel] = load_pickle(root / rel)
        return files[rel]
    for t in b["claim_targets"]:
        e, p = b["comparators"][t], {}
        for family in ("gbm", "season_loc_gbm"):
            s = e[family]["spec"]
            if s["feature_set"] == "season":
                X = x.Xseason
            else:
                X = x.X[s["feature_set"]]
                if s["mode"] == "stacked":
                    base = X.to_numpy(float)
                    X = C.stack_features(X, {o: member(f)["predict"](base)
                                             for o, f in e[family]["stack_files"].items()}, t)
                X = X.to_numpy(float)
            p[family] = np.asarray(member(e[family]["file"])["predict"](X), float)
        p["knn"] = C.fit_knn(b["knn_fit"]["lat"], b["knn_fit"]["lon"], e["knn"]["y_fit"])(x.lat, x.lon, e["knn"]["k"])
        m, s = member(e["mdn_b"]["file"]), e["mdn_b"]["spec"]
        model = m["model"]
        if device is not None and str(model.device) != str(device):
            model.hidden.to(device), model.out.to(device)
            model.device = device
        Xf = C.mdn_inputs(s["input"], x.rw3, x.mdn, x.carrays, m["fill"])[0]
        p["mdn_b"] = np.asarray(model.predict(Xf, s["readout"]), float)[:, e["mdn_b"]["column"]]
        p["mdn_b+gbm"] = C.average_log10(p["mdn_b"], p["gbm"])
        out[t] = p
    for t in b["claim_targets"]:
        e = b["comparators"][t].get("trophic_ols")
        if e:                                   # R-11: the chain reads the chain's own chl-a and Secchi comparator
            out[t]["trophic_ols"] = C.ols_predict(np.asarray(e["coef"], float), out["chla"][e["chain"]],
                                                  out["secchi"][e["chain"]])
    return out


# -------------------------------------------------------------------------------------------- registry

def register(res: dict, root: Path, version: str, n_boot: int, stamp_row: str, plan: dict,
             registry_path=None) -> int:
    """R-14a: the comparators' validation rows go to the registry before any arm's first validation number."""
    from sentinel.core.registry import log_result

    s = res["summary"]
    n = 0
    for r in res["scores"].itertuples():
        fam = s["selection"][r.target]["families"].get(r.model, {})
        log_result(workstream=PL.WORKSTREAM, experiment=EXPERIMENT.format(v=version), model=f"cmp:{r.model}",
                   metric="mdsa", value=float(r.mdsa), split=f"{r.cell}@{version}", n=int(r.n), is_baseline=True,
                   ci_low=float(r.mdsa_ci_low), ci_high=float(r.mdsa_ci_high), ci_method=CI_METHOD.format(b=n_boot),
                   target=r.target, data_manifests=[f"orbit_{version}"], artifacts=[str(root)],
                   config={"fit_scheme": s["fit_scheme"], "v1_selection_row": plan.get("v1_selection_row"),
                           "member": fam.get("selected"), "sha256sums_sha256": s["sha256sums_sha256"]},
                   extra={"n_clusters": int(r.n_clusters), "n_common_rows_dropped": int(r.n_dropped_nonfinite),
                          "selection_cell": bool(r.selection_cell), "family_members": fam.get("n_members"),
                          "stamp_row": stamp_row, "clarification_01_sha256": CLARIFICATION_01,
                          "chain": s["chains"].get(r.target, {}).get("selected") if r.model == "trophic_ols" else None},
                   registry_path=registry_path)
        n += 1
    return n


def cpu_hours() -> float:
    t = os.times()
    return (t.user + t.system + t.children_user + t.children_system) / 3600.0


def register_job(stage: str, part: str, budget: Budget, complete: bool, wall_h: float, device: str, version: str,
                 stamp_row: str, plan: dict, registry_path=None) -> float:
    """K-3: the CPU-hours of every comparator job are logged in the registry."""
    from sentinel.core.registry import log_result

    h = cpu_hours()
    log_result(workstream=PL.WORKSTREAM, experiment=EXPERIMENT.format(v=version), model=f"job:{part}",
               metric="cpu_hours", value=h, split=f"fit@{version}", n=int(budget.n), is_baseline=True,
               data_manifests=[f"orbit_{version}"],
               config={"fit_scheme": plan["fit_scheme"], "v1_selection_row": plan.get("v1_selection_row")},
               extra={"stage": stage, "complete": bool(complete), "wall_hours": wall_h, "device": device,
                      "threads": LIM["threads"], "pid": os.getpid(), "members_started": int(budget.n),
                      "stamp_row": stamp_row, "clarification_01_sha256": CLARIFICATION_01},
               notes="compute record of a comparator job (not a result)", registry_path=registry_path)
    return h


# ---------------------------------------------------------------------------------------------- main

def arm_fit_scheme(config_dir: Path | None = None):
    """The fit scheme named by the arm files (None until R-19 is met)."""
    import yaml

    files = sorted((config_dir or REPO / "configs" / "orbit_v2").glob("a*.yaml"))
    vals = {(yaml.safe_load(f.read_text()).get("base") or {}).get("fit_scheme") for f in files}
    if len(vals) != 1:
        raise SystemExit(f"the arm files name different fit schemes: {sorted(map(str, vals))}")
    return vals.pop()


def resolve_fit_scheme(asked: str, named, stage: str) -> str:
    """Clarification 01, K-2. Before R-19 the members are fitted with the scheme passed; once the arm files name
    a scheme, only that one is fitted, selected or reported."""
    if asked not in ARM_SCHEMES:
        raise SystemExit(f"fit scheme {asked!r}: one of {ARM_SCHEMES}")
    if stage in ("select", "finalize") and named is None:
        raise SystemExit(f"{stage}: the arm files name no fit scheme yet (R-19); no validation number before that")
    if named is not None and asked != named:
        raise SystemExit(f"--fit-scheme {asked} differs from the arm files' {named} (K-2: every family is "
                         f"refitted in full under the arms' value; nothing of the first fit is carried over)")
    return asked


def v1_selection(row_id: str, registry_path=None) -> dict:
    """The registry row of the v1 selection whose fit scheme is passed (K-2)."""
    from sentinel.core.registry import read_registry

    rows = [r for r in read_registry(registry_path) if r.get("id") == row_id]
    if len(rows) != 1 or rows[0].get("workstream") != PL.WORKSTREAM:
        raise SystemExit(f"--v1-selection-row {row_id!r}: not a registry row of {PL.WORKSTREAM}")
    return {k: rows[0].get(k) for k in ("id", "timestamp", "experiment", "model", "split")}


def mark_superseded(base: Path, scheme: str) -> list:
    """K-2: the folder of a first fit under another scheme is kept and marked superseded; nothing in it is read."""
    out = []
    for other in sorted(p for p in Path(base).iterdir() if p.is_dir() and p.name != scheme):
        if (other / "plan.json").exists():
            if not (other / "SUPERSEDED.json").exists():
                PL.write_json(other / "SUPERSEDED.json", {"superseded_by_fit_scheme": scheme,
                                                          "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            out.append(other.name)
    return out


def check_disk(path: str = "/") -> float:
    free = shutil.disk_usage(path).free / 1e9
    if free - PROJECTED_GB < MIN_FREE_GB:
        raise SystemExit(f"R-25: {free:.0f} GB free on {path}; {MIN_FREE_GB:.0f} GB must stay free after "
                         f"{PROJECTED_GB:.0f} GB of comparator files")
    return free


def smoke_limits() -> dict:
    lim = full_limits()
    lim.update(grid=(0,), feature_sets=("v1+ring+context_dates",), ks=(4,), max_rounds=20, member_seeds=(0,),
               mdn=(("12band+context+ring", 200, False, "joint"),))
    return lim


def run_stage(stage: str, d, root: Path, device: str, budget: Budget, *, ntu=None, n_boot: int = E.N_BOOT):
    """(complete, result of ``finalize`` or the selection or None)."""
    if stage == "knn":
        return run_knn(d, root, budget), None
    if stage in ("season", "gbm_single", "gbm_stacked"):
        return run_gbm(d, root, stage, budget), None
    if stage == "stack":
        return run_stack(d, root, budget), None
    if stage == "mdn":
        return run_mdn(d, root, device, budget), None
    if stage == "select":
        return True, select(d, root)
    res = finalize(d, root, device, budget, ntu=ntu, n_boot=n_boot)
    return res is not None, res


def main(argv=None) -> int:
    global LIM
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--fit-scheme", required=True, choices=list(ARM_SCHEMES))
    ap.add_argument("--v1-selection-row", default=None,
                    help="registry row id of the latest registered v1 selection (its fit scheme is --fit-scheme)")
    ap.add_argument("--max-minutes", type=float, default=110.0)
    ap.add_argument("--n-boot", type=int, default=E.N_BOOT)
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"], help="cpu for a GPU stage: smoke tests only")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data-dir", default=None, help="smoke test: a dev_v5m-format cache folder")
    a = ap.parse_args(argv)
    gpu = a.stage in GPU_STAGES
    scheme, stamp, v1row = a.fit_scheme, None, None
    if a.smoke:
        if a.stage in ("select", "finalize"):
            raise SystemExit("a smoke test computes no validation number: no select or finalize stage")
        if not a.data_dir:
            raise SystemExit("a smoke test needs --data-dir")
        if Path(a.data_dir).resolve() == DM.cache_dir().resolve() and not ST.stamp_status()["stamped"]:
            raise SystemExit("before the stamp a smoke test reads a smoke cache, not the dev_v5m cache")
        LIM = smoke_limits()
        root = paths.run_dir("orbit", "smoke/v2/comparators")
    else:
        if a.data_dir or (a.device == "cpu" and gpu):
            raise SystemExit("--data-dir and --device cpu on a GPU stage are for smoke tests")
        stamp = ST.require_stamp(f"orbitv2_comparators: stage {a.stage} on {DM.VERSION}")
        if not a.v1_selection_row:
            raise SystemExit("a real run names the v1 selection row whose fit scheme it uses (--v1-selection-row)")
        resolve_fit_scheme(scheme, arm_fit_scheme(), a.stage)
        v1row = v1_selection(a.v1_selection_row)
        check_disk()
        LIM = full_limits()
        base_dir = paths.run_dir("orbit", "v2/comparators")
        root = paths.run_dir("orbit", f"v2/comparators/{scheme}")
        superseded = mark_superseded(base_dir, scheme)
    device = a.device or ("cuda" if gpu else "cpu")
    part = f"{a.stage}_part{len(list(root.glob(f'part_{a.stage}_*.json')))}"
    print(f"orbitv2_comparators {part} pid {os.getpid()} device {device} fit scheme {scheme} "
          f"start {time.strftime('%Y-%m-%d %H:%M:%S %Z')} -> {root}", flush=True)
    if device == "cuda":
        PL.setup_gpu(f"orbit_v2_cmp_{a.stage}")
    if gpu:
        import torch

        torch.set_num_threads(2)                   # shared-host rule: GPU jobs use 2 CPU threads
    from sentinel.core.runlog import TrainingRun

    cfg = {"stage": a.stage, "fit_scheme": scheme, "v1_selection_row": v1row, "smoke": bool(a.smoke),
           "limits": {k: list(v) if isinstance(v, tuple) else v for k, v in LIM.items()}}
    budget, t0 = Budget(a.max_minutes, part), time.time()
    with TrainingRun(PL.WORKSTREAM, f"{'SMOKE/' if a.smoke else ''}orbit_v2_cmp/{scheme}/{part}", config=cfg,
                     data=[f"orbit_{DM.VERSION}"], plan_ref=PLAN_REF, iteration="arch_v2",
                     artifacts_dir=str(root)) as run:
        base = None if a.smoke else D.load_dev(DM.BASE_VERSION)[0]
        df, pix, feats, (_, carrays), info = DM.load(Path(a.data_dir) if a.data_dir else None, base=base)
        d = prepare(df, pix, feats, carrays, scheme, keep="inner" if a.smoke else "out")
        plan = {"fit_scheme": scheme, "v1_selection_row": v1row, "rows_sha": d.rows_sha, "targets": d.targets,
                "claim_targets": d.claim, "data": {"version": info.get("version"), "sha256": info["sha256"]},
                "clarification_01_sha256": CLARIFICATION_01}
        if (root / "plan.json").exists():
            was = json.loads((root / "plan.json").read_text())
            diff = [k for k in ("fit_scheme", "rows_sha", "targets", "data") if was.get(k) != plan[k]]
            if diff:
                raise SystemExit(f"{root}: the members were fitted with another {diff}")
            plan = was
        else:
            PL.write_json(root / "plan.json", plan)
        run.note(f"data {info.get('version')} table sha256 {info['sha256'][DM.TABLE][:12]}; fit scheme {scheme}"
                 f" (v1 selection row {v1row['id'] if v1row else None}); targets {d.targets}; new-target fit-cell "
                 f"counts {d.head_counts}; {int(d.fit.sum())} fit-cell rows, {int(d.inner.sum())} inner rows")
        if a.smoke:
            run.note("SMOKE TEST (train-only): one configuration per family; NOT a result; nothing registered")
        elif superseded:
            run.note(f"K-2: the fit under scheme(s) {superseded} is superseded by this one; nothing of it is read")
        ntu = ntu_class(df, base) if a.stage == "finalize" else None
        done, res = run_stage(a.stage, d, root, device, budget, ntu=ntu, n_boot=a.n_boot)
        if a.stage == "select":
            run.note("selected members: " + "; ".join(f"{t} " + ", ".join(
                f"{k}={v['selected']}" for k, v in r["families"].items()) for t, r in res.items()))
        elif a.stage == "finalize" and done:
            run.metric("n_registry_rows", register(res, root, DM.VERSION, a.n_boot, stamp["row_id"], plan))
            run.metric("n_val_rows", res["summary"]["n_val_rows"])
        wall_h = (time.time() - t0) / 3600
        run.metric("minutes", 60 * wall_h)
        run.metric("members_started", budget.n)
        if not a.smoke:
            run.metric("cpu_hours", register_job(a.stage, part, budget, done, wall_h, device, DM.VERSION,
                                                 stamp["row_id"], plan))
        PL.write_json(root / f"part_{part}.json", {"complete": bool(done), "wall_hours": wall_h,
                                                   "cpu_hours": cpu_hours(), "members_started": budget.n})
        if not done:
            run.note("time budget reached: the same command continues the stage")
            print("NOT COMPLETE: rerun the same command to continue")
            return EXIT_RESUME
    print(f"DONE stage {a.stage} ({root})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
