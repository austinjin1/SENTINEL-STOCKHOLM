#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SINGLE-LOOK M5-X test scorer (protocol ``bioassess_x_protocol.md`` §8–§9 as amended by v11). DO NOT RUN before
the lead's freeze.

The look is four separate commands (v11 item 2), all run from the pinned code snapshot (v11 item 3; the path is in
the freeze record, ``code_snapshot.dir``); single-look enforcement is in ``sentinel/models/bioassess/xtestlook.py``:

* ``--claim --authorised-by X`` — accepts ONLY the lead-written ``docs/preregistration/bioassess_x_freeze.json``
  (never the candidate record), verifies the snapshot, file/package hashes, data pins and artifacts, refuses if the
  registry holds any ``bioassess_x_test`` row — all before the claim. Then the claim (O_EXCL ``TEST_LOOK.json`` +
  registry row), the test INPUTS (labels dropped inside the loader; ``inputs/<block>.pkl``) and the unit plan
  (``plan.json``), closed by the O_EXCL ``INPUTS.json``. ``--finish-claim RUN_DIR`` repeats the input step if the
  claim process died before ``INPUTS.json``.
* ``--chunk RUN_DIR --only gpu|cpu [--shard i/n] [--max-minutes M]`` — any number, any order, in parallel, started
  again as often as needed before sealing. Every chunk re-verifies record, snapshot, data pins and artifacts, asserts
  that no row to be predicted carries a finite label, re-derives the unit plan and fits the units that do not exist
  yet. Units are write-once (one directory per (block, leaf model, fold, seed) or event fit, own sha256). A chunk
  prints unit identities, row counts and wall time only. GPU chunks run through the lock helper (``--min-free-mb``)
  on GPUs free of other users' processes and stop starting units after ``M`` <= 100 minutes. Exit 0 = nothing left
  for this selection, 3 = time limit, 4 = no GPU free, 5 = the rest is being fitted by other chunks, 6 = a unit
  failed (nothing was written for it; it is named in the chunk record and fitted again by a later chunk). A failed or
  killed chunk leaves finished units in place and does not end the look.
* ``--seal RUN_DIR`` — only when every planned unit exists and verifies: per-model prediction files are assembled,
  sealed (0444, ``SHA256SUMS``, one ``predictions_sealed`` registry row, O_EXCL ``PREDICTIONS_SAVED.json``).
* ``--score RUN_DIR --authorised-by X`` — single shot: the seal must verify; only then are the test labels loaded
  (``xdev.load_test_labels``); results are sealed and every registry row is appended in ONE locked write; then the
  O_EXCL ``SCORED.json``. Never resumed: if scoring stopped before the registry write,
  ``--rescore-from-saved RUN_DIR --authorised-by X`` recomputes everything from the sealed predictions and, where
  the stopped run left results, requires agreement within 1e-9 (v11 item 6b) — otherwise both sets are kept, no
  row is written and the lead rules.
* ``--verify-scored RUN_DIR`` — after scoring: recompute from the sealed predictions and the sealed truth files into
  a separate folder (no label load, no registry write) and compare within 1e-9.
* ``--dry-run-dev [--max-folds N] [--rehearsal-cap N]`` — the claim step of the identical code path on the
  development folds from the *candidate* record, with its own look directory and scratch registry inside the run
  directory (real markers untouched); then the same ``--chunk`` / ``--seal`` / ``--score`` commands. Labels of the
  pseudo-test rows are masked in every fitting process. ``--rehearsal-cap`` (training rows per continent and fold,
  with reduced ensemble sizes and epochs; v11 item 7) is refused outside the dry run; its numbers are not usable.

Statistics (all from the record): per block × target, C-X vs the per-target comparator on ``r2_anomaly`` (paired
location-cluster bootstrap, B = 10,000; one-sided rule p = two-sided p if Δ > 0 else 1); Holm over the 12 tests;
decision table §8 (per-seed Δ for seeds 0–2, region-unit paired CI, realised Holm-worst MDE vs 0.02). Secondary,
unadjusted: other candidates, pooled R², SENTINEL vs RF[geography], X-U by ``analysis_method``, 50-km-block CIs.

Every listed comparator (amendments v9, v10, v11 item 1): every §6 comparator of the block (``LISTED``; 16 X-L, 19
X-U, including the two rule comparators ``cxr_nobioset`` and ``cxr_swap``) is fitted in its development-chosen
variant and C-X is compared with each (same paired bootstrap, B = 10,000). Row 1 (SOTA) needs the paired CI lower
bound > 0 against every one of them (no finite estimate = not beaten); row 3 "Below" fires when the CI upper bound is
< 0 against any of them; every comparator not beaten is named in the row's reason text and in
``results/listed_comparators.json``. A record whose comparator list differs from ``LISTED``, or whose rule
comparators do not follow from its own C-X, is refused. GJAM and TRPI are "not compared, with reason"
(``results/not_compared.json``, copied from the record); neither is fitted. Confirmatory rows carry tolerance flags
(v11 item 6d) and a bootstrap p of 0 is reported as "< 2/B" (6e).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]            # the code tree this script runs from (snapshot or live)
_GPU_CHILD = os.environ.get("M5X_GPU_CHILD") == "1"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2" if _GPU_CHILD else "6"
os.environ.setdefault("SENTINEL_GPU_MIN_FREE_MB", "5000")
if sys.path[0] != str(ROOT):                          # ``sentinel`` is imported from this tree, never another one
    sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(ROOT)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and p != str(ROOT)])

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import importlib.metadata as md  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import stat  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.core.registry import file_sha256, log_result, read_registry  # noqa: E402
from sentinel.models.bioassess import xdev as XD  # noqa: E402
from sentinel.models.bioassess import xtestlook as XL  # noqa: E402
from sentinel.models.bioassess import xtrack as X  # noqa: E402

if paths.REPO_ROOT.resolve() != ROOT:
    raise ImportError(f"sentinel was imported from {paths.REPO_ROOT}, not from the tree of this script ({ROOT})")
REPO = ROOT
EXPERIMENT = XL.EXPERIMENT
R_ENV = Path.home() / "miniconda3/envs/rbioinfer"
HOLM_Z = 3.71                      # (z_{1-0.05/24} + z_0.8), protocol §8
SESOI = 0.02
# Listed comparators fitted and compared at the look (protocol amendments v9 item 1 and v11 item 1). The record must
# list exactly these.
_BENS = ("bens_rfg_tabpfn", "bens_rfg_gbm_tabpfn", "bens_rfctx_tabpfn")
RULE_COMPARATORS = ("cxr_nobioset", "cxr_swap")       # v11 item 1: derived by rule from the record's own C-X
_COMMON = ("region_mean", "rf_geography", "wapls2", "knn_community", "bioinfer", "rf_community", "gbm_community",
           "tabpfn_community", "rf_community_geography", "rf_community_context", *_BENS, *RULE_COMPARATORS)
LISTED = {"X-L": (*_COMMON, "nearest_reference"),
          "X-U": (*_COMMON, "idx_linear", "idx_linear_geo", "idx_rf", "idx_rf_geo")}
NOT_COMPARED = ("gjam", "trpi")    # "not compared, with reason" (v9 items 5-6): never fitted, never scored
LISTED_JOBS = 5                    # parallel paired bootstraps of the listed comparisons (1 thread each)
# v11 item 6: (b) a rescore from the same sealed predictions must agree within ``rescore_abs``; (c)/(d) tolerances of
# a later refit, used to flag confirmatory rows whose label-deciding quantities lie that close to their threshold.
TOLERANCE = {"delta": 2e-4, "ci": 2e-4, "per_seed_delta": 3e-4, "p": 3e-3, "rescore_abs": 1e-9}
CHUNK_MAX_MINUTES = 100            # a chunk stops starting units after its --max-minutes (<= this; jobs <= 2 h)
# v11 item 7: development rehearsal only (refused in real mode): reduced ensemble sizes / epochs under --rehearsal-cap
R_CAP_FACTOR = 3                   # v11 item 7: the R methods get this multiple of --rehearsal-cap (see look_design)
REHEARSAL = {"rf_n_estimators": 30, "gbm_n_estimators": 60, "tabpfn_n_estimators": 2, "bioset_max_epochs": 3,
             "bioset_patience": 2}
RESULT_FILES = ("confirmatory.json", "marginal.json", "listed_comparators.json", "secondary.json", "events.json",
                "not_compared.json", "event_track_status.json")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def safe_log(**kw):
    """log_result, recording a non-finite value as 'not estimable' (metric suffix, value 1) instead of failing."""
    v = kw.get("value")
    if v is None or not np.isfinite(float(v)):
        kw = {**kw, "metric": kw["metric"] + "_not_estimable", "value": 1.0, "ci_low": None, "ci_high": None,
              "notes": kw.get("notes", "") + " [value not estimable on these rows]"}
    return log_result(**kw)


# ------------------------------------------------------------------------------------------------- record checks
def r_package_versions() -> dict:
    try:
        out = subprocess.run([str(R_ENV / "bin/Rscript"), "-e",
                              'cat(as.character(packageVersion("bio.infer")), as.character(packageVersion("gjam")), '
                              'R.version.string, sep="|")'], capture_output=True, text=True, timeout=300,
                             env={**os.environ, "PATH": f"{R_ENV}/bin:{os.environ.get('PATH', '')}"})
        b, g, r = out.stdout.strip().split("|")
        return {"bio.infer": b, "gjam": g, "R": r}
    except Exception as e:                      # noqa: BLE001
        return {"error": str(e)}


def package_versions(names) -> dict:
    out = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = None
    return out


def verify_record(rec: dict, check_data: bool = True) -> list[str]:
    bad = []
    for rel, h in rec["sha256"].items():
        p = REPO / rel if not rel.startswith("/") else Path(rel)
        if not p.exists() or file_sha256(p) != h:
            bad.append(f"hash changed: {rel}")
    if check_data:
        for f, h in rec["data_sha256"].items():
            if not Path(f).exists() or file_sha256(f) != h:
                bad.append(f"data changed: {f}")
    if package_versions(rec["package_versions"]) != rec["package_versions"]:
        bad.append("python package versions changed")
    if r_package_versions() != rec["r_package_versions"]:
        bad.append("R package versions changed")
    return bad


def registry_has(experiment: str, registry_path=None) -> bool:
    return any(r.get("experiment") == experiment for r in read_registry(registry_path or XL.REGISTRY))


def rule_comparators(rec: dict, b: str) -> tuple[dict, str]:
    """Protocol v11 item 1, derived from the record's own C-X of block ``b``: ``cxr_nobioset`` = mean of C-X's
    baseline members; ``cxr_swap`` = those plus the listed single baseline (not a member; R methods are no
    ``baseline`` spec) with the highest mean development ``r2_anomaly``. Returns (specs, name of the swap member)."""
    cx = rec["candidates"][b]["confirmatory"]["spec"]
    mem = cx.get("members") or []
    base = [m for m in mem if m.get("kind") == "baseline"]
    other = [m.get("kind") for m in mem if m.get("kind") != "baseline"]
    if cx.get("kind") != "mean" or not base or other != ["bioset"]:
        raise ValueError("C-X is not one BioSet member plus baseline members")
    cs = rec["comparator_set"][b]
    names = {m["name"] for m in base}
    singles = sorted(n for n, c in cs.items() if (c.get("spec") or {}).get("kind") == "baseline" and n not in names)
    swap = max(singles, key=lambda n: float(cs[n]["dev_mean_r2_anomaly"]))
    return {"cxr_nobioset": {"kind": "mean", "members": base},
            "cxr_swap": {"kind": "mean", "members": [*base, cs[swap]["spec"]]}}, swap


def check_listed(rec: dict) -> list[str]:
    """Amendments v9 item 1 and v11 item 1: the record must list exactly the protocol's comparators per block, each
    with an executable spec that fits no "not compared" method; the two rule comparators must follow from the
    record's own C-X; the per-target comparators must be listed with the same spec; GJAM and TRPI must be recorded as
    not compared, GJAM with a failed fidelity check in both blocks; the tolerances must be the scorer's. Returns the
    faults."""
    bad = []
    cs = rec.get("comparator_set") or {}
    for b in XD.BLOCKS:
        got = cs.get(b) or {}
        miss, extra = sorted(set(LISTED[b]) - set(got)), sorted(set(got) - set(LISTED[b]))
        if miss or extra:
            bad.append(f"{b}: listed comparators differ from protocol v11 (missing {miss}, extra {extra})")
        for name, c in got.items():
            leaves = leaf_specs(c["spec"]) if isinstance(c, dict) and "spec" in c else []
            if not leaves:
                bad.append(f"{b}: listed comparator {name} has no executable spec")
            if any(lf.get("method") in NOT_COMPARED or lf.get("name") in NOT_COMPARED or lf["kind"] == "bioset"
                   for lf in leaves):
                bad.append(f"{b}: listed comparator {name} uses a model that is not a comparator")
        try:
            want, _ = rule_comparators(rec, b)
            for n, sp in want.items():
                if (got.get(n) or {}).get("spec") != sp:
                    bad.append(f"{b}: rule comparator {n} does not follow protocol v11 item 1 from the record's C-X")
            for m in want["cxr_nobioset"]["members"]:
                if (got.get(m["name"]) or {}).get("spec") != m:
                    bad.append(f"{b}: C-X baseline member {m['name']} is not listed with the same spec")
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            bad.append(f"{b}: rule comparators cannot be derived from the record ({e})")
        for t, c in (rec.get("comparators") or {}).get(b, {}).items():
            if c["name"] not in got or got[c["name"]].get("spec") != c["spec"]:
                bad.append(f"{b}/{t}: per-target comparator {c['name']} is not listed with the same spec")
    nc = rec.get("not_compared") or {}
    if sorted(nc) != sorted(NOT_COMPARED):
        bad.append(f"not_compared must hold exactly {sorted(NOT_COMPARED)}, found {sorted(nc)}")
    for b in XD.BLOCKS:
        if ((nc.get("gjam") or {}).get("fidelity") or {}).get(b, {}).get("fidelity_pass") is not False:
            bad.append(f"GJAM fidelity record for {b} missing or not failed: GJAM cannot be 'not compared'")
    if rec.get("reproducibility_tolerance") != TOLERANCE:
        bad.append("reproducibility_tolerance differs from protocol v11 item 6")
    return bad


def load_record(dry: bool) -> tuple[dict, str, str]:
    """The verified record of the look -> (record, sha256 of its file, code tree). Real mode: the lead-written
    FREEZE file only (``xtestlook.load_freeze``: equal to the candidate, snapshot verified, running from it).
    Dry run: the candidate record; its snapshot is verified when the run is started from it. Then data pins,
    artifacts, file/package hashes and the listed-comparator rules. Every command calls this first, so a mismatch
    stops a command before it does anything."""
    try:
        if dry:
            rec = json.loads(XL.CANDIDATE_FILE.read_text())
            in_snap = (REPO / XL.SNAPSHOT_MARKER).is_file()
            if in_snap:
                XL.verify_snapshot(rec)
            code = "pinned snapshot" if in_snap else "live tree (development dry run only)"
            record_sha = file_sha256(XL.CANDIDATE_FILE)
        else:
            rec = XL.load_freeze()
            code, record_sha = "pinned snapshot", rec["_sha256"]
        XL.verify_data_pins(rec)
        XL.verify_artifacts(rec)
    except XL.SingleLookError as e:
        raise SystemExit(str(e)) from e
    bad = verify_record(rec) + check_listed(rec)
    if bad:
        raise SystemExit("record verification failed: " + "; ".join(bad[:10]))
    return rec, record_sha, code


# ------------------------------------------------------------------------------------------------- specs
def leaf_specs(spec) -> list[dict]:
    if spec["kind"] == "mean":
        return [x for m in spec["members"] for x in leaf_specs(m)]
    return [spec]


def spec_id(spec) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def is_gpu_spec(spec) -> bool:
    return spec["kind"] == "bioset" or (spec["kind"] == "baseline" and spec["name"] == "tabpfn_community")


def seed_eff(spec, seed) -> int:
    stochastic = spec["kind"] == "bioset" or (spec["kind"] == "baseline" and spec["name"] in XD.STOCHASTIC)
    return seed if stochastic else 0


def model_specs(rec, b) -> dict:
    specs = {c["name"]: c["spec"] for c in [rec["candidates"][b]["confirmatory"], *rec["candidates"][b]["secondary"]]}
    specs.update({c["name"]: c["spec"] for c in rec["comparators"][b].values()})
    for name, c in (rec.get("comparator_set") or {}).get(b, {}).items():
        specs.setdefault(name, c["spec"])              # v9: every listed comparator is fitted at the look
    specs.setdefault("rf_geography", {"kind": "baseline", "name": "rf_geography", "variant": {}})
    for name, spec in (rec.get("event_track", {}).get("threshold_baselines", {}).get(b, {})).items():
        specs.setdefault(name, spec)                   # event track (v8): threshold-on-baseline models
    return specs


# ------------------------------------------------------------------------------------------------- statistics
def block50(lat, lon) -> np.ndarray:
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    r = np.floor((lat + 90) * 111.195 / 50).astype(int)
    c = np.floor((lon + 180) * 111.195 * np.cos(np.radians(lat)) / 50).astype(int)
    return np.array([f"{a}_{b}" for a, b in zip(r, c)])


def holm(p: dict) -> dict:
    keys = sorted(p, key=p.get)
    m, out, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


def _names(v) -> list[str]:
    return [str(x) for x in v] if isinstance(v, (list, tuple, np.ndarray)) else []


def label(row) -> tuple[str, str]:
    """Decision table §8 as amended by v9 and v10 (first matching row); the reason text names every listed
    comparator that is not beaten.

    ``row`` carries the family test (C-X vs the per-target comparator) and, from the listed comparisons:
    ``listed_above_candidate`` (paired CI upper bound < 0), ``listed_not_estimable`` (no finite paired estimate) and
    ``listed_not_beaten`` (CI lower bound <= 0, or not estimable). Absent keys mean "no listed comparison supplied".

    * v10: "Below" when ANY listed comparator is significantly above C-X — checked first.
    * a family Δ that cannot be estimated is labelled "Not estimable".
    * row 1 (SOTA): Holm-rejected, per-seed Δ > 0, region-unit CI > 0 and every listed comparator beaten (v9);
      otherwise a Holm-rejected test is row 2 "Above, not robust" with each failed condition named.
    * rows 3-5 against the per-target comparator as before.
    """
    above, ne = _names(row.get("listed_above_candidate")), _names(row.get("listed_not_estimable"))
    nb = _names(row.get("listed_not_beaten"))
    tie = [n for n in nb if n not in above and n not in ne]
    notes = ([f"listed comparators significantly above C-X (paired CI upper bound < 0): {', '.join(above)}"]
             if above else [])
    notes += [f"listed comparators not beaten (paired CI lower bound <= 0): {', '.join(tie)}"] if tie else []
    notes += [f"listed comparators not estimable (count as not beaten): {', '.join(ne)}"] if ne else []
    why = lambda *first: "; ".join([*first, *notes])                           # noqa: E731
    if above:                                                                  # row 3 as amended by v10
        return "Below", why()
    if not np.isfinite(row.get("delta", 0.0)) or not np.isfinite(row["ci_low"]) or not np.isfinite(row["ci_high"]):
        return "Not estimable", why("no finite paired estimate on these rows")
    if row["holm_rejected"]:
        fails = [n for n, ok in (("per-seed Δ > 0 for seeds 0-2", row["all_seeds_positive"]),
                                 ("region-unit paired CI > 0", row["region_ci_low"] > 0),
                                 ("paired CI lower bound > 0 against every listed comparator", not nb)) if not ok]
        return ("SOTA (M5-X, {}, within-region)".format(row["block"]), "") if not fails else \
            ("Above, not robust", why(*fails))
    if row["ci_high"] < 0:
        return "Below", why()
    if row["ci_low"] <= 0 <= row["ci_high"] and row["realised_mde"] > SESOI:
        return "Inconclusive (underpowered)", why()
    return "Competitive", why()


def _paired(y, pa, pb, region, cluster, n_boot: int) -> dict:
    """Paired location-cluster bootstrap of r2_anomaly(a) - r2_anomaly(b) on the rows where the label and both
    predictions are finite (``EV.paired_vs``, bootstrap seed 0). No such row -> a non-finite estimate."""
    from sentinel.models.bioassess import evaluate as EV

    ok = np.isfinite(y) & np.isfinite(pa) & np.isfinite(pb)
    if not ok.any():
        return {"n": 0, "diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    pr = EV.paired_vs(y[ok], pa[ok], pb[ok], region[ok], cluster[ok], metric="r2_anomaly", n_boot=n_boot)
    return {"n": int(ok.sum()), **{k: float(pr[k]) for k in ("diff", "ci_low", "ci_high", "p_value")}}


def paired_many(tasks: list[tuple], n_jobs: int = LISTED_JOBS) -> list[dict]:
    """``_paired`` over many (y, pa, pb, region, cluster, n_boot) tasks. Each task has its own seed-0 generator, so the
    result does not depend on ``n_jobs``; workers are processes limited to one numeric thread each."""
    if n_jobs <= 1 or len(tasks) <= 1:
        return [_paired(*t) for t in tasks]
    from joblib import Parallel, delayed, parallel_config

    with parallel_config(backend="loky", inner_max_num_threads=1):
        return list(Parallel(n_jobs=min(n_jobs, len(tasks)))(delayed(_paired)(*t) for t in tasks))


def listed_class(rec, b, name) -> str:
    return str(((rec.get("comparator_set") or {}).get(b, {}).get(name) or {}).get("class", ""))


def score_listed(models: dict, Y, region, cluster, rec: dict, b: str, n_jobs: int = LISTED_JOBS) -> list[dict]:
    """C-X vs EVERY listed comparator of block ``b`` (v9 item 2): one row per (target, comparator)."""
    listed = list((rec.get("comparator_set") or {}).get(b, {}))
    cand = rec["candidates"][b]["confirmatory"]["name"]
    nbc = int(rec["n_boot_confirmatory"])
    A = sum(models[cand].values()) / len(models[cand])
    keys, tasks = [], []
    for t in XD.TARGETS[b]:
        y, pa = Y[t].to_numpy(float), A[t].to_numpy(float)
        for c in listed:
            pb = (sum(models[c].values()) / len(models[c]))[t].to_numpy(float)
            keys.append((t, c))
            tasks.append((y, pa, pb, region, cluster, nbc))
    rows = []
    for (t, c), pr in zip(keys, paired_many(tasks, n_jobs)):
        est = bool(np.isfinite(pr["diff"]) and np.isfinite(pr["ci_low"]) and np.isfinite(pr["ci_high"]))
        rows.append({"block": b, "target": t, "candidate": cand, "comparator": c,
                     "comparator_class": listed_class(rec, b, c),
                     "is_per_target_comparator": c == rec["comparators"][b][t]["name"], "n": pr["n"],
                     "delta": pr["diff"], "ci_low": pr["ci_low"], "ci_high": pr["ci_high"],
                     "p_two_sided": pr["p_value"], "estimable": est, "beaten": bool(est and pr["ci_low"] > 0),
                     "above_candidate": bool(est and pr["ci_high"] < 0)})
    return rows


def score_blocks(pred: dict, truth: dict, meta: dict, rec: dict, n_jobs: int = LISTED_JOBS
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """pred[block][model][seed] -> frame; returns (confirmatory table, marginal table, listed-comparator table)."""
    from sentinel.models.bioassess import evaluate as EV

    nb, nbc = int(rec["n_boot"]), int(rec["n_boot_confirmatory"])
    conf, marg, listed = [], [], []
    for b, models in pred.items():
        Y, mt = truth[b], meta[b]
        region, cluster = mt["region"].to_numpy(), mt["cluster"].to_numpy()
        blk50 = block50(mt["lat"], mt["lon"])
        for name, seeds in models.items():
            mean = sum(seeds.values()) / len(seeds)
            for t in XD.TARGETS[b]:
                y, p = Y[t].to_numpy(float), mean[t].to_numpy(float)
                ok = np.isfinite(y) & np.isfinite(p)
                if ok.sum() < 3:
                    continue
                sm = EV.summary(y[ok], p[ok], region[ok])
                ci = EV.cis(y[ok], p[ok], region[ok], cluster[ok], n_boot=nb, metrics=("r2_anomaly", "r2"))
                marg.append({"block": b, "model": name, "target": t, "n": int(ok.sum()), "r2_anomaly": sm["r2_anomaly"],
                             "r2": sm["r2"], "rmse": sm["rmse"], "spearman": sm["spearman"],
                             "ci_low": ci["r2_anomaly"]["ci_low"], "ci_high": ci["r2_anomaly"]["ci_high"],
                             "r2_ci": [ci["r2"]["ci_low"], ci["r2"]["ci_high"]],
                             "per_seed_r2_anomaly": {s: EV.anomaly_r2(y[ok], f[t].to_numpy(float)[ok], region[ok])
                                                     for s, f in seeds.items()}})
        cand = rec["candidates"][b]["confirmatory"]["name"]
        lrows = score_listed(models, Y, region, cluster, rec, b, n_jobs)        # v9: every listed comparator
        listed += lrows
        for t in XD.TARGETS[b]:
            comp = rec["comparators"][b][t]["name"]
            y = Y[t].to_numpy(float)
            A, Bm = models[cand], models[comp]
            pa, pb = (sum(A.values()) / len(A))[t].to_numpy(float), (sum(Bm.values()) / len(Bm))[t].to_numpy(float)
            ok = np.isfinite(y) & np.isfinite(pa) & np.isfinite(pb)
            pr = EV.paired_vs(y[ok], pa[ok], pb[ok], region[ok], cluster[ok], metric="r2_anomaly", n_boot=nbc)
            prr = EV.paired_vs(y[ok], pa[ok], pb[ok], region[ok], region[ok], metric="r2_anomaly", n_boot=nbc)
            pr50 = EV.paired_vs(y[ok], pa[ok], pb[ok], region[ok], blk50[ok], metric="r2_anomaly", n_boot=nb)
            per_seed = {s: EV.anomaly_r2(y[ok], A[s][t].to_numpy(float)[ok], region[ok])
                        - EV.anomaly_r2(y[ok], Bm[s if s in Bm else 0][t].to_numpy(float)[ok], region[ok]) for s in A}
            signs = []
            for g in np.unique(region[ok]):
                m = region[ok] == g
                signs.append(EV.anomaly_r2(y[ok][m], pa[ok][m], region[ok][m])
                             - EV.anomaly_r2(y[ok][m], pb[ok][m], region[ok][m]))
            d = pr["diff"]
            conf.append({"block": b, "target": t, "candidate": cand, "comparator": comp, "n": int(ok.sum()), "delta": d,
                         "ci_low": pr["ci_low"], "ci_high": pr["ci_high"], "p_two_sided": pr["p_value"],
                         "p_one_sided_rule": pr["p_value"] if d > 0 else 1.0,
                         "region_ci_low": prr["ci_low"], "region_ci_high": prr["ci_high"],
                         "block50_ci": [pr50["ci_low"], pr50["ci_high"]],
                         "per_seed_delta": per_seed, "all_seeds_positive": all(v > 0 for v in per_seed.values()),
                         "region_signs_positive": int(np.sum(np.array(signs) > 0)), "n_regions": len(signs),
                         "realised_mde": HOLM_Z * (pr["ci_high"] - pr["ci_low"]) / 3.92})
            lt = [r for r in lrows if r["target"] == t]
            conf[-1].update({"n_listed": len(lt),
                             "listed_not_beaten": [r["comparator"] for r in lt if not r["beaten"]],
                             "listed_above_candidate": [r["comparator"] for r in lt if r["above_candidate"]],
                             "listed_not_estimable": [r["comparator"] for r in lt if not r["estimable"]],
                             "all_listed_beaten": bool(lt) and all(r["beaten"] for r in lt)})
    C = pd.DataFrame(conf)
    if len(C):
        adj = holm({(r.block, r.target): r.p_one_sided_rule for r in C.itertuples()})
        C["p_holm"] = [adj[(r.block, r.target)] for r in C.itertuples()]
        C["holm_rejected"] = C["p_holm"] < float(rec["alpha"])
        lab = [label(r) for _, r in C.iterrows()]
        C["label"], C["label_reason"] = [x[0] for x in lab], [x[1] for x in lab]
    return C, pd.DataFrame(marg), pd.DataFrame(listed)


def secondary(pred, truth, meta, rec) -> list[dict]:
    """Unadjusted secondary contrasts: other candidates vs comparator, C-X vs RF[geography], X-U by method."""
    from sentinel.models.bioassess import evaluate as EV

    out = []
    for b, models in pred.items():
        Y, mt = truth[b], meta[b]
        region, cluster = mt["region"].to_numpy(), mt["cluster"].to_numpy()
        cand = rec["candidates"][b]["confirmatory"]["name"]
        pairs = [(c["name"], rec["comparators"][b][t]["name"], t, "secondary candidate vs comparator")
                 for c in rec["candidates"][b]["secondary"] for t in XD.TARGETS[b]]
        pairs += [(cand, "rf_geography", t, "C-X vs RF[geography]") for t in XD.TARGETS[b]]
        for a_, b_, t, what in pairs:
            if a_ not in models or b_ not in models:
                continue
            y = Y[t].to_numpy(float)
            pa = (sum(models[a_].values()) / len(models[a_]))[t].to_numpy(float)
            pb = (sum(models[b_].values()) / len(models[b_]))[t].to_numpy(float)
            strata = {"all": np.ones(len(y), bool)}
            if b == "X-U":
                strata["ANLA"] = (mt["analysis_method"] == "ANLA").to_numpy()
                strata["non-ANLA"] = ~strata["ANLA"]
            for sname, m in strata.items():
                ok = m & np.isfinite(y) & np.isfinite(pa) & np.isfinite(pb)
                if ok.sum() < 20:
                    continue
                pr = EV.paired_vs(y[ok], pa[ok], pb[ok], region[ok], cluster[ok], metric="r2_anomaly",
                                  n_boot=int(rec["n_boot"]))
                out.append({"block": b, "what": what, "a": a_, "b": b_, "target": t, "stratum": sname,
                            "n": int(ok.sum()), "delta": pr["diff"], "ci_low": pr["ci_low"], "ci_high": pr["ci_high"],
                            "p_two_sided": pr["p_value"], "note": "secondary, unadjusted"})
    return out


# ------------------------------------------------------------------------------------------------- helpers
def seal(run_dir: Path) -> None:
    files = sorted(p for p in run_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    lines = [f"{file_sha256(p)}  {p.relative_to(run_dir)}" for p in files]
    (run_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    for p in files + [run_dir / "SHA256SUMS"]:
        p.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def setup():
    """Imports and the training design (no test row). A GPU is held only by a GPU chunk, taken through the lock
    helper (sentinel.core.gpu, --min-free-mb) among GPUs free of other users' processes."""
    import torch

    torch.set_num_threads(2)
    from sentinel.models.chemtox.tabpfn_model import load_tabpfn

    load_tabpfn()
    S, frozen, D = XD.load_context()
    return S, frozen, D


def unoccupied_gpus() -> list[int]:
    """GPU indices on which no process of ANOTHER user runs (unreadable owner = another user)."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"], capture_output=True,
                       text=True, timeout=60).stdout
    bus = {b.strip().lower(): int(i) for i, b in (ln.split(",") for ln in q.strip().splitlines())}
    a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=60).stdout
    busy = set()
    for ln in a.strip().splitlines():
        pid, b = (x.strip() for x in ln.split(","))
        try:
            mine = os.stat(f"/proc/{int(pid)}").st_uid == os.getuid()
        except (OSError, ValueError):
            mine = False
        if not mine:
            busy.add(bus.get(b.lower()))
    return sorted(set(bus.values()) - busy)


# ------------------------------------------------------------------------------------------------- look state
def look_of(run_dir: Path) -> tuple[bool, Path | None, Path | None]:
    """(dry, look directory, registry) of a run directory. A development dry run carries ``DRY_RUN`` and keeps its
    own look directory and scratch registry inside the run directory; a real look uses the repository's. A wrong
    answer cannot open anything: every later step needs the open claim of exactly this run directory."""
    dry = (run_dir / "DRY_RUN").is_file()
    return dry, (run_dir / "look" if dry else None), (run_dir / "registry_dryrun.jsonl" if dry else None)


def write_once(f: Path, text: str | bytes) -> str:
    """Write ``f`` (0444) under a temporary name and move it into place; returns its sha256."""
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f".tmp_{f.name}_{socket.gethostname()}_{os.getpid()}")
    tmp.write_bytes(text if isinstance(text, bytes) else text.encode())
    os.chmod(tmp, 0o444)
    os.replace(tmp, f)
    return file_sha256(f)


# ------------------------------------------------------------------------------------------------- units and plan
def unit_label(u: dict) -> str:
    if u["kind"] != "leaf":
        return f"{u['kind']}:{u['code']}"
    sp = u["spec"]
    return str(sp.get("name") or sp.get("config") or sp.get("method"))


def plan_units(rec: dict, folds: dict) -> list[dict]:
    """Every fit of the look, one unit per (block, leaf model, fold, effective seed) and per event fit — each
    written once to its own directory (v11 item 2.2)."""
    units, seen = [], set()
    for b in XD.BLOCKS:
        for spec in model_specs(rec, b).values():
            for lf in leaf_specs(spec):
                for k, fo in enumerate(folds[b]):
                    for s in sorted({seed_eff(lf, s) for s in rec["seeds"]}):
                        key = f"leaf_{b}_{spec_id(lf)}_f{k:02d}_s{s}"
                        if key not in seen:
                            seen.add(key)
                            units.append({"key": key, "kind": "leaf", "block": b, "spec": lf, "k": k, "g": str(fo.g),
                                          "seed": int(s), "gpu": is_gpu_spec(lf)})
    if rec.get("event_track", {}).get("status") == "active":
        from sentinel.events import bioassess_x as XE

        for (b, code) in XE.TRACK:
            for k, fo in enumerate(folds.get(b, [])):
                units.append({"key": f"evclim_{b}_{code}_f{k:02d}", "kind": "event_clim", "block": b, "code": code,
                              "k": k, "g": str(fo.g), "seed": 0, "gpu": False})
                if (b, code) in XE.POWERED:
                    units += [{"key": f"evdir_{b}_{code}_f{k:02d}_s{s}", "kind": "event_direct", "block": b,
                               "code": code, "k": k, "g": str(fo.g), "seed": int(s), "gpu": False}
                              for s in rec["seeds"]]
    return units


def make_plan(rec: dict, folds: dict, *, dry: bool, max_folds, cap) -> dict:
    return {"dry": bool(dry), "max_folds": max_folds, "rehearsal_cap": cap,
            "folds": {b: [{"g": str(fo.g), "n_train": len(fo.train), "n_test": len(fo.test),
                           "train_sha": XD.hash_ids(fo.train), "test_sha": XD.hash_ids(fo.test)} for fo in folds[b]]
                      for b in XD.BLOCKS},
            "units": plan_units(rec, folds)}


def select_units(units: list[dict], only: str | None, shard: str | None) -> list[dict]:
    sel = [u for u in units if only is None or u["gpu"] == (only == "gpu")]
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        if not 0 <= i < n:
            raise SystemExit("--shard i/n needs 0 <= i < n")
        sel = [u for j, u in enumerate(sel) if j % n == i]
    return sel


# ------------------------------------------------------------------------------------------------- rehearsal cap
def cap_fold(fo, D, cap: int):
    """Development rehearsal only: at most ``cap`` training rows per source table of the fold (seeded by fold)."""
    src = D.samples.loc[fo.train, "source"].astype(str).to_numpy()
    rng = np.random.default_rng(int(hashlib.sha256(f"{fo.block}|{fo.g}|{cap}".encode()).hexdigest()[:8], 16))
    keep = np.zeros(len(src), bool)
    for s in sorted(set(src)):
        idx = np.flatnonzero(src == s)
        keep[idx if len(idx) <= cap else rng.choice(idx, cap, replace=False)] = True
    return dataclasses.replace(fo, train=fo.train[keep])


_REHEARSAL_ON = False


def apply_rehearsal(dry: bool) -> None:
    """Reduced ensemble sizes and epochs of the capped development rehearsal. Refused outside the dry run."""
    global _REHEARSAL_ON
    if not dry:
        raise XL.SingleLookError("the rehearsal cap is refused in real mode")
    if _REHEARSAL_ON:
        return
    from sentinel.models.bioassess import xbioset as XB

    XD.RF_PARAMS["n_estimators"] = REHEARSAL["rf_n_estimators"]
    XD.GBM_PARAMS["n_estimators"] = REHEARSAL["gbm_n_estimators"]
    XD.TABPFN_PARAMS["n_estimators"] = REHEARSAL["tabpfn_n_estimators"]
    full = XB.base_cfg

    def capped(name):
        cfg = full(name)
        cfg.update(max_epochs=REHEARSAL["bioset_max_epochs"], patience=REHEARSAL["bioset_patience"])
        return cfg

    XB.base_cfg = capped
    _REHEARSAL_ON = True


# ------------------------------------------------------------------------------------------------- inputs and design
def assert_label_free(designs: dict, folds: dict) -> None:
    """v11 item 2.1: no row that is predicted carries a finite label in the process that fits and predicts."""
    for b in XD.BLOCKS:
        for fo in folds[b]:
            if set(fo.train) & set(fo.test):
                raise XL.SingleLookError(f"{b}/{fo.g}: training and predicted rows overlap")
            if np.isfinite(designs[b].Y.reindex(fo.test).to_numpy(float)).any():
                raise XL.SingleLookError(f"{b}/{fo.g}: a predicted row carries a finite label in a fitting process")


def write_inputs(rec: dict, run_dir: Path, S, frozen, D, *, dry: bool, max_folds, cap, root, reg, record_sha: str,
                 code: str) -> dict:
    """Under the open claim: the label-free inputs of the look and the unit plan, closed by ``INPUTS.json``."""
    XL.require_open_claim(root, reg, run_dir=run_dir, freeze_sha256=record_sha)
    if not dry and (max_folds or cap):
        raise XL.SingleLookError("--max-folds / --rehearsal-cap are refused in real mode")
    shas, loaded = {}, {}
    for b in XD.BLOCKS:
        if dry:                                    # pseudo-test rows of the development folds: ids only
            allf = XD.read_folds(XD.DEV_DIR / "folds_dev_v1.json", S, frozen)
            ids = sorted({str(i) for fo in allf[b] for i in fo.test})
            f = run_dir / "inputs" / f"{b}.json"
            shas[f.name] = write_once(f, json.dumps({"block": b, "test_ids": ids}))
        else:
            Dt = XD.load_test_inputs(S, b, root=root, registry_path=reg)   # labels are dropped inside the loader
            if np.isfinite(Dt.Y.to_numpy(float)).any():
                raise XL.SingleLookError("test inputs carry labels")
            f = run_dir / "inputs" / f"{b}.pkl"
            shas[f.name] = write_once(f, pickle.dumps(Dt, protocol=pickle.HIGHEST_PROTOCOL))
            loaded[b] = Dt.info
            del Dt
    _designs, folds, _ = look_design(run_dir, shas, S, frozen, D, dry=dry, max_folds=max_folds, cap=cap)
    plan = make_plan(rec, folds, dry=dry, max_folds=max_folds, cap=cap)
    info = {"run_dir": str(run_dir), "inputs": shas, "n_units": len(plan["units"]), "test_rows_loaded": loaded,
            "plan_sha256": write_once(run_dir / "plan.json", json.dumps(plan, sort_keys=True)),
            "record_sha256": record_sha, "code": code}
    XL.mark_inputs(info, root=root)
    return info


def look_design(run_dir: Path, shas: dict, S, frozen, D, *, dry: bool, max_folds, cap) -> tuple[dict, dict, dict]:
    """(design per block, folds per block, folds per block of the R methods) of a fitting process, from the
    training design and the hashed inputs of
    the look. Real: the label-free test bundles are appended to the training design. Dry: the pseudo-test rows of
    the block are in the development design, so the block's design is a copy whose labels on those rows are
    removed. Always ends with :func:`assert_label_free` and the fit guard on every fold."""
    designs, folds = {}, {}
    for b in XD.BLOCKS:
        f = run_dir / "inputs" / (f"{b}.json" if dry else f"{b}.pkl")
        if not f.is_file() or file_sha256(f) != shas.get(f.name):
            raise XL.SingleLookError(f"inputs of {b} changed or missing")
    if dry:
        allf = XD.read_folds(XD.DEV_DIR / "folds_dev_v1.json", S, frozen)
        for b in XD.BLOCKS:
            ids = pd.Index(json.loads((run_dir / "inputs" / f"{b}.json").read_text())["test_ids"])
            if any(set(fo.train) & set(ids) for fo in allf[b]):
                raise XL.SingleLookError(f"{b}: a pseudo-test row is a training row of the block")
            Y = D.Y.copy()
            Y.loc[ids] = np.nan
            designs[b] = dataclasses.replace(D, Y=Y)
            folds[b] = allf[b][:max_folds] if max_folds else allf[b]
    else:
        if max_folds or cap:
            raise XL.SingleLookError("--max-folds / --rehearsal-cap are refused in real mode")
        DD, Dts = D, {}
        for b in XD.BLOCKS:
            with open(run_dir / "inputs" / f"{b}.pkl", "rb") as fh:
                Dts[b] = pickle.load(fh)
            if np.isfinite(Dts[b].Y.to_numpy(float)).any():
                raise XL.SingleLookError("test inputs carry labels")
            DD = XD.combine(DD, Dts[b])
        for b in XD.BLOCKS:
            designs[b], folds[b] = DD, XD.test_folds(S, frozen, D, Dts[b], b)
    folds_r = folds
    if cap:
        # The rehearsal cap keeps the fits cheap. The R methods (bio.infer, and GJAM if it were a comparator) get
        # R_CAP_FACTOR times the cap: their R code builds its coefficients from the training set and fails on a
        # training set thinned to the ML cap (observed in the rehearsal of 2026-11-11: mlsolve cannot build its
        # coefficient matrix), while a full-size R fit is by far the slowest unit of a chunk. The unit plan does
        # not change: a fold keeps its identity and its predicted rows, only training rows are thinned.
        folds_r = {b: [cap_fold(fo, designs[b], R_CAP_FACTOR * int(cap)) for fo in fl] for b, fl in folds.items()}
        folds = {b: [cap_fold(fo, designs[b], int(cap)) for fo in fl] for b, fl in folds.items()}
    assert_label_free(designs, folds)
    assert_label_free(designs, folds_r)
    for b in XD.BLOCKS:
        for fo in [*folds[b], *folds_r[b]]:
            X.assert_fit_ids(XD.keys_of(fo.train), S, frozen)
    return designs, folds, folds_r


# ------------------------------------------------------------------------------------------------- fitting units
class Fitter:
    """Fits single units from the label-free designs. Holds no cache and writes no prediction: the chunk loop writes
    each finished unit once (``xtestlook.unit_write``)."""

    def __init__(self, ctx: dict, run_dir: Path, device: str, n_jobs: int = 5, allow_cpu_gpu_units: bool = False):
        self.S, self.frozen, self.designs, self.folds = ctx["S"], ctx["frozen"], ctx["designs"], ctx["folds"]
        self.folds_r = ctx.get("folds_r") or ctx["folds"]       # R methods are not thinned by the rehearsal cap
        self.run_dir, self.device, self.n_jobs = run_dir, device, n_jobs
        self.allow_cpu_gpu_units = allow_cpu_gpu_units          # --cpu-smoke (dry run) only
        self._tm, self._prep = {}, {}

    def tm(self, D):
        k = (id(D.tokens), id(D.samples))
        if k not in self._tm:
            self._tm[k] = (XD.TaxonMatrix(D), D.tokens, D.samples)
        return self._tm[k][0]

    def prep(self, D):
        from sentinel.models.bioassess import xbioset as XB

        if id(D) not in self._prep:
            self._prep[id(D)] = (XB.Prepared(D), D)
        return self._prep[id(D)][0]

    def fit(self, u: dict) -> tuple[pd.DataFrame | None, str | None]:
        """(predictions, None) of unit ``u``; (None, error text) for a failed event unit (secondary track, v7). A
        failing model unit raises; the chunk records it, writes nothing for it, goes on to the next unit and ends
        non-zero (exit 6), so the unit is fitted again by a later chunk and never enters a seal unfitted."""
        b = u["block"]
        fo = self.folds[b][u["k"]]
        if str(fo.g) != u["g"]:
            raise XL.SingleLookError(f"unit {u['key']}: fold order changed")
        X.assert_fit_ids(XD.keys_of(fo.train), self.S, self.frozen)            # fit guard (protocol §7)
        if u["kind"] == "leaf":
            if u["spec"]["kind"] == "r":                        # full training set (see look_design)
                fo = self.folds_r[b][u["k"]]
                X.assert_fit_ids(XD.keys_of(fo.train), self.S, self.frozen)
            P = self.leaf(u["spec"], b, u["k"], fo, u["seed"])
            if len(P) != len(fo.test) or set(P.index) != set(fo.test):
                raise RuntimeError(f"unit {u['key']}: predictions do not cover the fold's rows")
            return P.reindex(index=fo.test, columns=list(XD.TARGETS[b])).astype(float), None
        from sentinel.events import bioassess_x as XE

        D = self.designs[b]
        try:
            if u["kind"] == "event_clim":
                pv, cm = XE.fold_prevalence_climatology(D, b, u["code"], fo)
                return pd.DataFrame({"prevalence": pv, "climatology": cm}, index=fo.test), None
            p = XE.fold_direct(D, self.tm(D), b, u["code"], fo, u["seed"], self.n_jobs)
            return pd.DataFrame({"p": p}, index=fo.test), None
        except Exception as e:                                    # noqa: BLE001  (secondary track never stops the look)
            return None, f"{type(e).__name__}: {e}"[:2000]

    def leaf(self, spec, block, k, fo, s_eff) -> pd.DataFrame:
        if is_gpu_spec(spec) and self.device != "cuda" and not self.allow_cpu_gpu_units:
            raise RuntimeError(f"GPU unit {block}/{spec_id(spec)}/{fo.g}/seed{s_eff} needs a GPU chunk")
        D, T = self.designs[block], list(XD.TARGETS[block])
        if spec["kind"] == "baseline":
            P = pd.DataFrame(index=fo.test, columns=T, dtype=float)
            for t in T:
                v = spec["variant"].get(t, "same") if isinstance(spec.get("variant"), dict) else "same"
                P[t] = XD.fit_predict(spec["name"], block, t, D, self.tm(D), fo.train, fo.test, variant=v, seed=s_eff,
                                      device=self.device, n_jobs=self.n_jobs)
            return P
        if spec["kind"] == "bioset":
            from sentinel.models.bioassess import xbioset as XB

            preds, _ = XB.fit_fold(self.prep(D), spec["config"], block, fo, self.S, self.frozen, seeds=(s_eff,),
                                   device=self.device, pretrain_root=self.run_dir / "pretrain", fold_index=k, log=log)
            return preds[s_eff]
        if spec["kind"] == "r":
            return self.r_method(spec["method"], block, k, fo)
        raise ValueError(spec)

    def r_method(self, method, block, k, fo) -> pd.DataFrame:
        import importlib.util

        spec_ = importlib.util.spec_from_file_location("xdev_script", REPO / "scripts/eval/bioassess_x_dev.py")
        mod = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(mod)
        D = self.designs[block]
        work = self.run_dir / "r_work" / f"{method}_{block}_f{k:02d}_{socket.gethostname()}_{os.getpid()}"
        if work.exists():
            shutil.rmtree(work)
        mod.write_r_fold(work / "in", block, fo, D, self.tm(D))      # training labels + test tokens and ids only
        out = work / "out.csv"
        env = {**os.environ, "PATH": f"{R_ENV}/bin:{os.environ.get('PATH', '')}", "OMP_NUM_THREADS": "1"}
        if method == "bioinfer":
            tax = self.run_dir / "r_work" / "bioinfer_taxonomy.rds"
            if not tax.exists():
                tmp = work / "taxonomy.rds"
                pd.DataFrame({"name": sorted(D.tokens["name"].dropna().astype(str).unique())}).to_csv(
                    work / "names.csv", index=False)
                subprocess.run([str(R_ENV / "bin/Rscript"), str(REPO / "scripts/eval/bioassess_x_bioinfer.R"),
                                "taxonomy", str(work / "names.csv"), str(tmp)], check=True, env=env)
                os.replace(tmp, tax)
            cmd = [str(R_ENV / "bin/Rscript"), str(REPO / "scripts/eval/bioassess_x_bioinfer.R"), "fold",
                   str(work / "in"), str(tax), str(out)]
        else:
            cmd = [str(R_ENV / "bin/Rscript"), str(REPO / "scripts/eval/bioassess_x_gjam.R"), "fold", str(work / "in"),
                   str(out)]
        subprocess.run(cmd, check=True, env=env)
        P = pd.read_csv(out).set_index("visit_id").reindex(fo.test)
        return P.reindex(columns=list(XD.TARGETS[block]))


def chunk_state(run_dir: Path, *, dry: bool, root, reg, record_sha: str) -> tuple[dict, dict]:
    """(INPUTS marker, unit plan) of an open prediction phase; every chunk starts here."""
    XL.require_open_claim(root, reg, run_dir=run_dir, freeze_sha256=record_sha)
    info = XL.inputs_info(root)
    pf = run_dir / "plan.json"
    if info.get("run_dir") != str(run_dir) or not pf.is_file() or file_sha256(pf) != info["plan_sha256"]:
        raise XL.SingleLookError("the unit plan of this look changed or belongs to another run directory")
    plan = json.loads(pf.read_text())
    if bool(plan["dry"]) != dry:
        raise XL.SingleLookError("the unit plan belongs to the other mode")
    if not dry and (plan.get("rehearsal_cap") or plan.get("max_folds")):
        raise XL.SingleLookError("--max-folds / --rehearsal-cap are refused in real mode")
    return info, plan


def run_chunk(rec: dict, run_dir: Path, S, frozen, D, *, dry: bool, root, reg, record_sha: str, only: str | None,
              shard: str | None = None, max_minutes: float = CHUNK_MAX_MINUTES, device: str = "cpu",
              cpu_smoke: bool = False, n_jobs: int = 5) -> int:
    """One chunk of the prediction phase (v11 item 2). Prints and stores unit identities, row counts and wall time
    only. Returns 0 (nothing left for this selection), 3 (time limit) or 5 (the rest is locked by other chunks)."""
    if cpu_smoke and not dry:
        raise XL.SingleLookError("--cpu-smoke is refused in real mode")
    info, plan = chunk_state(run_dir, dry=dry, root=root, reg=reg, record_sha=record_sha)
    cap, max_folds = plan.get("rehearsal_cap"), plan.get("max_folds")
    designs, folds, folds_r = look_design(run_dir, info["inputs"], S, frozen, D, dry=dry,
                                           max_folds=max_folds, cap=cap)
    if json.dumps(make_plan(rec, folds, dry=dry, max_folds=max_folds, cap=cap), sort_keys=True) != \
            json.dumps(plan, sort_keys=True):
        raise XL.SingleLookError("the unit plan derived by this chunk differs from the plan of the claim")
    if cap:
        apply_rehearsal(dry)
    units = select_units(plan["units"], only, shard)
    F = Fitter({"S": S, "frozen": frozen, "designs": designs, "folds": folds, "folds_r": folds_r},
               run_dir, device, n_jobs=n_jobs,
               allow_cpu_gpu_units=cpu_smoke)
    t0, fitted, failed, timed_out = time.time(), [], [], False
    for u in units:
        key = u["key"]
        if XL.unit_exists(run_dir, key):
            continue
        if (time.time() - t0) / 60 >= max_minutes:
            timed_out = True
            break
        if not XL.unit_lock(run_dir, key):
            continue
        try:
            if XL.unit_exists(run_dir, key):
                continue
            XL.require_open_claim(root, reg, run_dir=run_dir, freeze_sha256=record_sha)   # nothing after sealing
            t1 = time.time()
            try:
                df, err = F.fit(u)
            except XL.SingleLookError:
                raise                      # an integrity stop is never swallowed
            except Exception as e:         # noqa: BLE001
                # A model unit that fails is NOT written: it leaves no record that could be sealed, the chunk goes
                # on to the next unit and a later chunk fits it again (v11 item 2.2). The chunk ends non-zero and
                # names the failing units, so a unit that fails every time stops the look at the seal, loudly.
                failed.append({"key": key, "wall_s": round(time.time() - t1, 2),
                               "error": f"{type(e).__name__}: {e}"[:300]})
                log("unit", key, unit_label(u), u["g"], "FAILED, not written (a later chunk fits it again):",
                    f"{type(e).__name__}: {e}"[:300])
                continue
            wall = round(time.time() - t1, 2)
            wrote = XL.unit_write(run_dir, key, df, {"unit": u, "rows": len(df) if df is not None else 0,
                                                     "wall_s": wall, "device": device, "error": err,
                                                     "record_sha256": record_sha})
            fitted.append({"key": key, "wall_s": wall, "written": bool(wrote)})
            log("unit", key, unit_label(u), u["g"], "rows", len(df) if df is not None else 0, f"{wall:.1f}s",
                "" if wrote else "(already written by another chunk)", "" if err is None else "FAILED (recorded)")
        finally:
            XL.unit_unlock(run_dir, key)
    left = [u["key"] for u in units if not XL.unit_exists(run_dir, u["key"])]
    audit = {"host": socket.gethostname(), "pid": os.getpid(), "only": only, "shard": shard, "device": device,
             "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "started": dt.datetime.fromtimestamp(t0).isoformat(),
             "minutes": round((time.time() - t0) / 60, 2), "selected": len(units), "fitted": fitted,
             "failed": failed, "left": len(left), "record_sha256": record_sha, "code_tree": str(REPO)}
    write_once(run_dir / "chunks" / f"{audit['host']}_{audit['pid']}_{time.time_ns()}.json",
               json.dumps(audit, indent=1))
    log(f"chunk done: fitted {len(fitted)}, failed {len(failed)}, left {len(left)} of {len(units)} selected,",
        f"{audit['minutes']:.1f} min")
    if failed:
        log("failing units (not written):", ", ".join(x["key"] for x in failed[:10]))
    return 0 if not left else 3 if timed_out else 6 if failed else 5


# ------------------------------------------------------------------------------------------------- sealing
def assemble(rec: dict, run_dir: Path, plan: dict, out: Path) -> dict:
    """Per-model prediction files from the write-once units (each sha-verified on read). No label is involved."""
    cache, index = {}, {}
    planned = {u["key"]: u for u in plan["units"]}
    missing = [k for k in planned if not XL.unit_exists(run_dir, k)]
    if missing:
        raise XL.SingleLookError(f"{len(missing)} of {len(planned)} planned units do not exist yet "
                                 f"(first: {missing[:3]})")
    for k, u in planned.items():
        cache[k], meta = XL.unit_read(run_dir, k)                # sha-verified
        if meta.get("unit") != u:
            raise XL.SingleLookError(f"unit {k} was fitted for a different plan")
        index[k] = {"sha256": meta["sha256"], "wall_s": meta.get("wall_s"), "rows": meta.get("rows"),
                    "device": meta.get("device"), "error": meta.get("error")}
    unit = cache.__getitem__
    n_folds = {b: len(plan["folds"][b]) for b in XD.BLOCKS}

    def model(spec, b, s):
        if spec["kind"] == "mean":
            ms = [model(m, b, s) for m in spec["members"]]
            return sum(ms) / len(ms)
        return pd.concat([unit(f"leaf_{b}_{spec_id(spec)}_f{k:02d}_s{seed_eff(spec, s)}") for k in range(n_folds[b])])

    for b in XD.BLOCKS:
        (out / b).mkdir(parents=True, exist_ok=True)
        for name, spec in model_specs(rec, b).items():
            for s in rec["seeds"]:
                model(spec, b, s).to_parquet(out / b / f"{name}__s{s}.parquet")
    ev = {"status": "not in record"}
    evu = [u for u in plan["units"] if u["kind"] != "leaf"]
    if evu:
        failed = {u["key"]: index[u["key"]]["error"] for u in evu if cache[u["key"]] is None}
        if failed:                                                # v7: recorded as "not run", never fatal
            ev = {"status": "not run", "phase": "predict", "error": json.dumps(failed)[:2000]}
        else:
            ev = {"status": "predicted"}
            for b, code in sorted({(u["block"], u["code"]) for u in evu}):
                pd.concat([unit(f"evclim_{b}_{code}_f{k:02d}") for k in range(n_folds[b])]).to_parquet(
                    out / b / f"event_clim_{code}.parquet")
                for s in sorted({u["seed"] for u in evu if u["kind"] == "event_direct"
                                 and (u["block"], u["code"]) == (b, code)}):
                    pd.concat([unit(f"evdir_{b}_{code}_f{k:02d}_s{s}") for k in range(n_folds[b])]).to_parquet(
                        out / b / f"event_direct_{code}__s{s}.parquet")
    (out / "event_track_status.json").write_text(json.dumps(ev, indent=1))
    (out / "units_index.json").write_text(json.dumps(index, indent=1, sort_keys=True))
    (out / "plan.json").write_text(json.dumps(plan, sort_keys=True))
    return ev


def do_seal(rec: dict, run_dir: Path, *, dry: bool, root, reg, record_sha: str) -> dict:
    """Seal once, only when every planned unit exists and verifies (v11 item 2.4)."""
    pred_dir = run_dir / "predictions"
    if XL.sealed_rows(reg):                         # sealing row written, marker lost: complete the same sealing
        return XL.seal_predictions(pred_dir, root=root, registry_path=reg)
    _, plan = chunk_state(run_dir, dry=dry, root=root, reg=reg, record_sha=record_sha)
    tmp = run_dir / f".predictions_tmp_{socket.gethostname()}_{os.getpid()}"
    for d in (tmp, pred_dir):                       # an unsealed leftover of a stopped seal command is rebuilt
        if d.exists():
            shutil.rmtree(d)
    assemble(rec, run_dir, plan, tmp)
    XL.require_open_claim(root, reg, run_dir=run_dir, freeze_sha256=record_sha)
    os.rename(tmp, pred_dir)
    return XL.seal_predictions(pred_dir, root=root, registry_path=reg)


def score_events(rec, pred_dir: Path, truth_dir: Path, out_dir: Path, *, split: str, reg, record_sha: str) -> dict:
    """Event track scoring from the sealed predictions and the truth files + the frozen dev calibration (secondary)."""
    from sentinel.events import bioassess_x as XE

    et = rec["event_track"]
    cf = Path(et["calibration_file"])
    if file_sha256(cf) != et["calibration_sha256"]:
        raise RuntimeError("event calibration file changed since the freeze")
    calib = pd.read_parquet(cf)
    out = {}
    for (b, code), (t, _, _) in XE.TRACK.items():
        tf = truth_dir / f"{b}.parquet"
        if not tf.is_file():
            continue
        tr = pd.read_parquet(tf)
        ids = tr.index
        cl = tr["cluster"].astype(str).to_numpy()
        yt = tr[t].to_numpy(float)
        ye = np.full(len(yt), np.nan)
        ye[np.isfinite(yt)] = XE.labels(yt[np.isfinite(yt)], b, code)
        cx = rec["candidates"][b]["confirmatory"]["name"]
        probs = {}
        for name, mname in [("C-X:" + cx, cx)] + [(m, m) for m in XE.COMPARATORS]:
            mu = sum(pd.read_parquet(pred_dir / b / f"{mname}__s{s}.parquet").reindex(ids)[t].to_numpy(float)
                     for s in rec["seeds"]) / len(rec["seeds"])
            c = calib[(calib["block"] == b) & (calib["event"] == code) & (calib["model"] == name)]
            raw = XE.derived_probability(mu, float(c["sd_all"].iloc[0]), b, code)
            probs[name] = XE.frozen_calibrated(c["p_raw"].to_numpy(float), c["y"].to_numpy(float), raw)
        clim = pd.read_parquet(pred_dir / b / f"event_clim_{code}.parquet").reindex(ids)
        probs["prevalence"], probs["climatology"] = clim["prevalence"].to_numpy(), clim["climatology"].to_numpy()
        if (b, code) in XE.POWERED:
            raw = sum(pd.read_parquet(pred_dir / b / f"event_direct_{code}__s{s}.parquet").reindex(ids)["p"].to_numpy(float)
                      for s in rec["seeds"]) / len(rec["seeds"])
            c = calib[(calib["block"] == b) & (calib["event"] == code) & (calib["model"] == "direct")]
            probs["direct"] = XE.frozen_calibrated(c["p_raw"].to_numpy(float), c["y"].to_numpy(float), raw)
        ok = np.isfinite(ye)
        pw = XE.power_flag(ye[ok], cl[ok])
        status = "descriptive only" if ((b, code) in XE.DESCRIPTIVE_ONLY or pw["flag"] != "F") else "secondary"
        cxn = "C-X:" + cx
        mets = {n: XE.metrics(ye[ok], p[ok], cl[ok], p_ref=probs["climatology"][ok]) for n, p in probs.items()}
        comps = {} if status == "descriptive only" else \
            {f"{cxn} - {n}": XE.compare(ye[ok], probs[cxn][ok], probs[n][ok], cl[ok]) for n in probs if n != cxn}
        out[f"{b}|{code}"] = {"power_test": pw, "status": status, "metrics": mets, "contrasts": comps}
        for n, v in mets.items():
            for k in ("auprc", "brier", "bss"):
                if k in v:
                    val, lo, hi = XE.metric_value(v, k)
                    safe_log(workstream="bioassess", experiment=EXPERIMENT, model=n, metric=f"event_{k}",
                             value=val, split=f"{b}_{split}", target=code, n=int(ok.sum()), registry_path=reg,
                             ci_low=lo, ci_high=hi,
                             ci_method="location-cluster bootstrap B=2000 (events.state_metrics)",
                             is_baseline=not n.startswith("C-X"),
                             notes=f"M5-X event track, SECONDARY ({status}); amendment v8",
                             extra={"power_test": pw, "record_sha256": record_sha})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "events.json").write_text(json.dumps(out, indent=1, default=lambda o: o.tolist()
                                                    if hasattr(o, "tolist") else str(o)))
    return out


def score_saved(rec, pred_dir: Path, truth_dir: Path, out_dir: Path, *, split: str, reg, record_sha: str
                ) -> pd.DataFrame:
    """Score the sealed predictions against the truth files; write the result files into ``out_dir`` and the
    registry rows into the PENDING file ``reg`` (appended to the registry in one locked write by the caller)."""
    models, truth, meta = {}, {}, {}
    for b in XD.BLOCKS:
        T = pd.read_parquet(truth_dir / f"{b}.parquet")
        ids = T.index
        truth[b] = T[list(XD.TARGETS[b])]
        meta[b] = T[["region", "cluster", "lat", "lon", "analysis_method"]]
        models[b] = {}
        for name in model_specs(rec, b):
            models[b][name] = {sd: pd.read_parquet(pred_dir / b / f"{name}__s{sd}.parquet").reindex(ids)
                               for sd in rec["seeds"]}
    C, M, L = score_blocks(models, truth, meta, rec)
    C = add_tolerance_flags(C, L, rec)                  # v11 item 6d/6e
    sec = secondary(models, truth, meta, rec)
    out_dir.mkdir(parents=True, exist_ok=True)
    C.to_json(out_dir / "confirmatory.json", orient="records", indent=1, default_handler=str)
    M.to_json(out_dir / "marginal.json", orient="records", indent=1, default_handler=str)
    nc = rec.get("not_compared") or {}
    (out_dir / "listed_comparators.json").write_text(json.dumps({
        "rule": "protocol amendments v9, v10 and v11 item 1: C-X vs every listed comparator, paired location-cluster "
                "bootstrap of "
                f"r2_anomaly on seed-mean predictions, B = {rec['n_boot_confirmatory']}; beaten = CI lower bound > 0; "
                "above_candidate = CI upper bound < 0; no finite estimate = not beaten",
        "listed": {b: list((rec.get("comparator_set") or {}).get(b, {})) for b in XD.BLOCKS},
        "not_compared": {k: v.get("status") for k, v in nc.items()},
        "rows": json.loads(L.to_json(orient="records"))}, indent=1))
    (out_dir / "not_compared.json").write_text(json.dumps(nc, indent=1, default=float))
    (out_dir / "secondary.json").write_text(json.dumps(sec, indent=1, default=float))
    common = {"n_configurations": rec["n_configurations_tried"], "selection_note": rec["selection_note"],
              "record_sha256": record_sha}
    for r in M.itertuples():
        safe_log(workstream="bioassess", experiment=EXPERIMENT, model=r.model, metric="r2_anomaly",
                 value=r.r2_anomaly, split=f"{r.block}_{split}", target=r.target, n=r.n, ci_low=r.ci_low,
                 ci_high=r.ci_high, ci_method=f"percentile location-cluster bootstrap B={rec['n_boot']}",
                 is_baseline=not r.model.startswith(("bsx_", "ens_")), registry_path=reg,
                 notes="M5-X single pass (marginal)", extra={"r2": r.r2, "rmse": r.rmse, **common})
    for r in C.itertuples():
        safe_log(workstream="bioassess", experiment=EXPERIMENT, model=f"{r.candidate} - {r.comparator}",
                 metric="delta_r2_anomaly_confirmatory", value=r.delta, split=f"{r.block}_{split}", target=r.target,
                 n=r.n, ci_low=r.ci_low, ci_high=r.ci_high,
                 ci_method=f"paired location-cluster bootstrap B={rec['n_boot_confirmatory']}", registry_path=reg,
                 notes=f"M5-X confirmatory: {r.label}" + (f" ({r.label_reason})" if r.label_reason else ""),
                 extra={"p_holm": r.p_holm, "p_one_sided_rule": r.p_one_sided_rule, "per_seed_delta": r.per_seed_delta,
                        "region_ci": [r.region_ci_low, r.region_ci_high], "block50_ci": r.block50_ci,
                        "realised_mde": r.realised_mde, "region_signs_positive": r.region_signs_positive,
                        "n_regions": r.n_regions, "n_listed": r.n_listed, "listed_not_beaten": r.listed_not_beaten,
                        "listed_above_candidate": r.listed_above_candidate,
                        "listed_not_estimable": r.listed_not_estimable, "tolerance_flags": r.tolerance_flags,
                        "p_report": r.p_report, **common})
    for r in L.itertuples():
        state = "not estimable (counts as not beaten)" if not r.estimable else "beaten" if r.beaten else \
            "above C-X" if r.above_candidate else "not beaten"
        safe_log(workstream="bioassess", experiment=EXPERIMENT, model=f"{r.candidate} - {r.comparator}",
                 metric="delta_r2_anomaly_listed_comparator", value=r.delta, split=f"{r.block}_{split}",
                 target=r.target, n=r.n, ci_low=r.ci_low, ci_high=r.ci_high,
                 ci_method=f"paired location-cluster bootstrap B={rec['n_boot_confirmatory']}", registry_path=reg,
                 notes=f"M5-X listed comparator (protocol v9-v11): {state}",
                 extra={"p_two_sided": r.p_two_sided, "class": r.comparator_class,
                        "per_target_comparator": r.is_per_target_comparator, **common})
    for k, v in nc.items():
        safe_log(workstream="bioassess", experiment=EXPERIMENT, model=k, metric="comparator_not_compared", value=1.0,
                 split=f"M5-X_{split}", registry_path=reg, is_baseline=True, notes=str(v.get("status", ""))[:500],
                 extra={"record_sha256": record_sha})
    status = json.loads((pred_dir / "event_track_status.json").read_text())
    if status.get("status") == "predicted":
        try:
            score_events(rec, pred_dir, truth_dir, out_dir, split=split, reg=reg, record_sha=record_sha)
            status = {"status": "scored"}
        except Exception as e:                                    # noqa: BLE001  (secondary track never burns the look)
            status = {"status": "not run", "phase": "score", "error": f"{type(e).__name__}: {e}"[:2000]}
            log("EVENT TRACK NOT RUN (scoring):", status["error"])
    if status.get("status") == "not run":
        safe_log(workstream="bioassess", experiment=EXPERIMENT, model="event_track", metric="event_track_not_run",
                 value=1.0, split=f"M5-X_{split}", registry_path=reg, notes="secondary event track not run: " +
                 status.get("error", ""), extra={**status, "record_sha256": record_sha})
    (out_dir / "event_track_status.json").write_text(json.dumps(status, indent=1))
    return C


# ------------------------------------------------------------------------------------------------- scoring phase
def add_tolerance_flags(C: pd.DataFrame, L: pd.DataFrame, rec: dict) -> pd.DataFrame:
    """v11 item 6d/6e: per confirmatory row, name every label-deciding quantity that lies within the refit
    tolerance of its threshold (the label stands; the flag is reported with it); ``p_report`` shows a bootstrap p of
    0 as "< 2/B"."""
    if not len(C):
        return C
    tol, alpha, B = rec["reproducibility_tolerance"], float(rec["alpha"]), int(rec["n_boot_confirmatory"])
    C = C.copy()
    key = [(r.block, r.target) for r in C.itertuples()]
    p = dict(zip(key, C["p_one_sided_rule"].astype(float)))
    lo = holm({k: max(0.0, v - tol["p"]) for k, v in p.items()})
    hi = holm({k: min(1.0, v + tol["p"]) for k, v in p.items()})
    def near(v, t):
        return bool(np.isfinite(v) and abs(v) <= t)

    flags = []
    for k, r in zip(key, C.to_dict("records")):
        f = [f"{n} within {t:g} of 0" for n, v, t in (("delta", r["delta"], tol["delta"]),
                                                       ("ci_low", r["ci_low"], tol["ci"]),
                                                       ("ci_high", r["ci_high"], tol["ci"]),
                                                       ("region_ci_low", r["region_ci_low"], tol["ci"])) if near(v, t)]
        seeds = [str(s) for s, v in (r.get("per_seed_delta") or {}).items() if near(v, tol["per_seed_delta"])]
        if seeds:
            f.append(f"per-seed delta within {tol['per_seed_delta']:g} of 0 (seeds {', '.join(seeds)})")
        if (lo[k] < alpha) != (hi[k] < alpha):
            f.append(f"Holm decision changes when every p moves by {tol['p']:g}")
        if np.isfinite(r["realised_mde"]) and abs(r["realised_mde"] - SESOI) <= HOLM_Z * 2 * tol["ci"] / 3.92:
            f.append("realised MDE within tolerance of 0.02")
        if len(L):
            m = L[(L["block"] == r["block"]) & (L["target"] == r["target"]) & L["estimable"]]
            nl = [c for c, a, b in zip(m["comparator"], m["ci_low"], m["ci_high"])
                  if near(a, tol["ci"]) or near(b, tol["ci"])]
            if nl:
                f.append(f"listed comparator CI end within {tol['ci']:g} of 0: {', '.join(nl)}")
        flags.append(f)
    C["tolerance_flags"] = flags
    C["p_report"] = [f"< {2 / B:g}" if v == 0 else f"{v:.4g}" for v in C["p_two_sided"].astype(float)]
    return C


def write_truth(rec: dict, run_dir: Path, pred_dir: Path, truth_dir: Path, *, dry: bool, root, reg) -> dict:
    """Labels and scoring metadata of the scored rows -> ``results/truth``. The ONLY place a label of a predicted row
    is loaded, and only through loaders that refuse unless the seal verifies and this process opened the scoring
    phase (``xtestlook.require_scoring``)."""
    XL.require_scoring(root, reg)
    info = XL.inputs_info(root)
    S = XD.load_split()
    D = XD.load_context()[2] if dry else None
    n = {}
    for b in XD.BLOCKS:
        cx = rec["candidates"][b]["confirmatory"]["name"]
        ids = pd.read_parquet(pred_dir / b / f"{cx}__s{rec['seeds'][0]}.parquet").index
        if dry:
            s, Yb = D.samples.loc[ids], D.Y.loc[ids, list(XD.TARGETS[b])]
        else:
            f = run_dir / "inputs" / f"{b}.pkl"
            if file_sha256(f) != info["inputs"][f.name]:
                raise XL.SingleLookError(f"inputs of {b} changed")
            with open(f, "rb") as fh:
                s = pickle.load(fh).samples.loc[ids]
            Yb = XD.load_test_labels(S, b, root=root, registry_path=reg).reindex(ids)
        T = pd.DataFrame({t: Yb[t].to_numpy(float) for t in XD.TARGETS[b]}, index=ids)
        T["region"], T["cluster"] = s["ag_eco9"].astype(str).to_numpy(), s["location_id"].astype(str).to_numpy()
        T["lat"], T["lon"] = s["lat"].to_numpy(float), s["lon"].to_numpy(float)
        am = s["analysis_method"] if "analysis_method" in s else pd.Series(None, index=ids)
        T["analysis_method"] = am.astype(object).where(am.notna(), None).astype(str).to_numpy()
        truth_dir.mkdir(parents=True, exist_ok=True)
        T.to_parquet(truth_dir / f"{b}.parquet")
        n[b] = len(T)
    return n


def _same(a, b, tol: float, path: str, out: list) -> None:
    if len(out) >= 50:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out.append(f"{path}: keys differ")
            return
        for k in a:
            _same(a[k], b[k], tol, f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: lengths differ")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, tol, f"{path}[{i}]", out)
    elif isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) \
            or not isinstance(b, (int, float)):
        if a != b:
            out.append(f"{path}: {str(a)[:60]!r} != {str(b)[:60]!r}")
    elif not (abs(a - b) <= tol or (np.isnan(a) and np.isnan(b))):
        out.append(f"{path}: numbers differ by more than {tol:g}")


def compare_results(a: Path, b: Path, tol: float) -> dict:
    """v11 item 6b: every stored number of two result folders within ``tol`` (absolute), every label and text
    exactly. A file the stopped run did not finish (absent or unparseable in ``a``) cannot be compared and is named."""
    diffs, compared, skipped = [], [], []
    for name in RESULT_FILES:
        if not (a / name).exists() and not (b / name).exists():
            continue
        try:
            ja = json.loads((a / name).read_text())
        except (OSError, ValueError):
            skipped.append(name)
            continue
        if not (b / name).is_file():
            diffs.append(f"{name}: missing in the recomputed results")
            continue
        _same(ja, json.loads((b / name).read_text()), tol, name, diffs)
        compared.append(name)
    for f in sorted((a / "truth").glob("*.parquet")) if (a / "truth").is_dir() and (b / "truth").is_dir() else []:
        try:
            ta = pd.read_parquet(f)
        except Exception:                                         # noqa: BLE001  (unfinished file of a stopped run)
            skipped.append(f"truth/{f.name}")
            continue
        tb = pd.read_parquet(b / "truth" / f.name)
        if not ta.index.equals(tb.index) or list(ta.columns) != list(tb.columns) or not all(
                (ta[c].astype(str) == tb[c].astype(str)).all() if ta[c].dtype == object
                else np.allclose(ta[c].to_numpy(float), tb[c].to_numpy(float), rtol=0, atol=tol, equal_nan=True)
                for c in ta.columns):
            diffs.append(f"truth/{f.name}: differs")
        compared.append(f"truth/{f.name}")
    return {"a": str(a), "b": str(b), "tolerance_abs": tol, "compared": compared, "not_comparable": skipped,
            "differences": diffs, "agree": not diffs}


def unit_times(pred_dir: Path) -> dict:
    """Wall seconds per unit, grouped by block | model | device (for scheduling; no prediction value)."""
    plan = {u["key"]: u for u in json.loads((pred_dir / "plan.json").read_text())["units"]}
    g = {}
    for k, v in json.loads((pred_dir / "units_index.json").read_text()).items():
        u = plan[k]
        g.setdefault(f"{u['block']}|{unit_label(u)}|{v.get('device')}", []).append(float(v.get("wall_s") or 0.0))
    return {k: {"n": len(v), "mean_s": round(float(np.mean(v)), 2), "max_s": round(float(np.max(v)), 2),
                "sum_s": round(float(np.sum(v)), 1)} for k, v in sorted(g.items())}


def do_score(rec: dict, run_dir: Path, pred_dir: Path, *, dry: bool, root, reg, record_sha: str, code: str,
             authorised_by: str, rescore: bool) -> pd.DataFrame:
    """Single-shot scoring (v11 item 2.5), called after ``xtestlook.begin_scoring``: labels -> results (sealed) ->
    every registry row in ONE locked append -> ``SCORED.json``. Never resumed; a rescore recomputes everything."""
    results = run_dir / "results"
    earlier = sorted(run_dir.glob("results_aborted_*"), key=lambda p: int(p.name.rsplit("_", 1)[1]))
    if results.exists():
        if not rescore:
            raise XL.SingleLookError("a results folder exists: scoring is single-shot (use --rescore-from-saved)")
        earlier.append(run_dir / f"results_aborted_{len(earlier) + 1}")
        os.rename(results, earlier[-1])
    plan = json.loads((pred_dir / "plan.json").read_text())
    cap = plan.get("rehearsal_cap")
    n_rows = write_truth(rec, run_dir, pred_dir, results / "truth", dry=dry, root=root, reg=reg)
    pending = results / "registry_rows.jsonl"
    C = score_saved(rec, pred_dir, results / "truth", results, split="dry_run_dev" if dry else "test", reg=pending,
                    record_sha=record_sha)
    info = {"record_sha256": record_sha, "dry_run": dry, "code": code, "code_tree": str(REPO),
            "authorised_by": authorised_by, "rescore_from_saved": rescore,
            "earlier_stopped_results": [str(p) for p in earlier],
            "folds": {b: len(plan["folds"][b]) for b in XD.BLOCKS}, "rows_scored": n_rows,
            "test_rows_loaded": XL.inputs_info(root).get("test_rows_loaded"), "max_folds": plan.get("max_folds"),
            "rehearsal": None if not cap else {
                "NOTE": "REHEARSAL — numbers not usable (training rows capped, reduced ensembles and epochs)",
                "max_training_rows_per_source": cap, **REHEARSAL},
            "listed_comparators": {b: len((rec.get("comparator_set") or {}).get(b, {})) for b in XD.BLOCKS},
            "not_compared": {k: v.get("status") for k, v in (rec.get("not_compared") or {}).items()},
            "reproducibility_tolerance": rec.get("reproducibility_tolerance"),
            "unit_wall_seconds": unit_times(pred_dir), "finished": dt.datetime.now().isoformat()}
    (results / "pass_info.json").write_text(json.dumps(info, indent=1, default=str))
    cmp = [compare_results(p, results, TOLERANCE["rescore_abs"]) for p in earlier]
    if cmp:
        (results / "rescore_comparison.json").write_text(json.dumps(cmp, indent=1))
    if any(not c["agree"] for c in cmp):            # v11 item 6b: both sets kept, no row written, the lead rules
        raise XL.SingleLookError("the rescore does not reproduce the stopped run's stored numbers within 1e-9: both "
                                 "result sets are kept, no registry row is written; the lead rules")
    seal(results)
    n = XL.flush_rows(pending, registry_path=reg)
    XL.mark_scored({"results": str(results), "n_confirmatory": len(C), "n_registry_rows": n,
                    "results_sha256sums": file_sha256(results / "SHA256SUMS")}, root=root)
    return C


def verify_sealed_results(results: Path) -> None:
    sums = results / "SHA256SUMS"
    if not sums.is_file():
        raise XL.SingleLookError("results are not sealed")
    for ln in sums.read_text().splitlines():
        h, rel = ln.split("  ", 1)
        if not (results / rel).is_file() or file_sha256(results / rel) != h:
            raise XL.SingleLookError(f"sealed result file {rel} changed or missing")


# ------------------------------------------------------------------------------------------------- commands
def _exit(e: BaseException):
    raise SystemExit(str(e)) from e


def cmd_claim(a) -> int:
    dry = bool(a.dry_run_dev)
    if not dry and (a.max_folds or a.rehearsal_cap):
        raise SystemExit("--max-folds / --rehearsal-cap are for the development dry run only (refused in real mode)")
    if not dry and not a.authorised_by:
        raise SystemExit("--authorised-by is required for the claim")
    rec, record_sha, code = load_record(dry)
    if not dry and (XL.registry_rows() or (XL.look_dir() / "TEST_LOOK.json").exists()):
        raise SystemExit(f"the M5-X test look was already claimed ({EXPERIMENT} rows or TEST_LOOK.json exist)")
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = paths.RUNS_ROOT / "bioassess" / (f"x_final_dryrun_dev_{stamp}" if dry else f"x_final_{stamp}")
    S, frozen, D = setup()                          # imports and the training design: before the claim
    run_dir.mkdir(parents=True)
    if dry:
        (run_dir / "DRY_RUN").write_text("development dry run: candidate record, development folds, own look "
                                         "directory and scratch registry\n")
    (run_dir / "claim_args.json").write_text(json.dumps({"max_folds": a.max_folds, "rehearsal_cap": a.rehearsal_cap}))
    _, root, reg = look_of(run_dir)
    by = a.authorised_by or "dry-run (development data)"
    try:
        XL.claim({**rec, "_path": rec.get("_path") or str(XL.CANDIDATE_FILE), "_sha256": record_sha}, by, run_dir,
                 root=root, registry_path=reg)
        info = write_inputs(rec, run_dir, S, frozen, D, dry=dry, max_folds=a.max_folds, cap=a.rehearsal_cap,
                            root=root, reg=reg, record_sha=record_sha, code=code)
    except XL.SingleLookError as e:
        _exit(e)
    log("claimed", run_dir, "| units planned:", info["n_units"], "| code:", code)
    print(run_dir)
    return 0


def cmd_finish_claim(run_dir: Path) -> int:
    dry, root, reg = look_of(run_dir)
    rec, record_sha, code = load_record(dry)
    try:
        c = XL.require_open_claim(root, reg, run_dir=run_dir, freeze_sha256=record_sha)
        if (XL.look_dir(root) / "INPUTS.json").exists():
            raise SystemExit("the claim already finished (INPUTS.json exists)")
        prior = json.loads((run_dir / "claim_args.json").read_text()) if (run_dir / "claim_args.json").is_file() else {}
        S, frozen, D = setup()
        for p in (run_dir / "inputs", run_dir / "plan.json"):    # unreferenced leftovers of the stopped claim
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        info = write_inputs(rec, run_dir, S, frozen, D, dry=dry, max_folds=prior.get("max_folds"),
                            cap=prior.get("rehearsal_cap"), root=root, reg=reg, record_sha=record_sha, code=code)
    except XL.SingleLookError as e:
        _exit(e)
    log("claim of", c.get("claimed"), "finished | units planned:", info["n_units"])
    return 0


def cmd_chunk(a) -> int:
    run_dir = Path(a.chunk).resolve()
    dry, root, reg = look_of(run_dir)
    if a.cpu_smoke and not dry:
        raise SystemExit("--cpu-smoke is for the development dry run only")
    if a.only not in ("gpu", "cpu") and not a.cpu_smoke:
        raise SystemExit("--chunk needs --only gpu or --only cpu")
    if not 0 < a.max_minutes <= CHUNK_MAX_MINUTES:
        raise SystemExit(f"--max-minutes must be in (0, {CHUNK_MAX_MINUTES}]")
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    gpu = a.only == "gpu" and not a.cpu_smoke
    if gpu and not _GPU_CHILD:                      # take one GPU through the lock helper, then run this command
        c = XL.open_claim(root)
        if c is None or c.get("run_dir") != str(run_dir):
            raise SystemExit("no open claim for this run directory")
        free = unoccupied_gpus()
        if os.environ.get("SENTINEL_GPUS"):
            free = [g for g in free if g in {int(x) for x in os.environ["SENTINEL_GPUS"].split(",") if x.strip()}]
        if not free:
            log("no GPU free of other users' processes")
            return 4
        cmd = [sys.executable, "-B", "-m", "sentinel.core.gpu", "--name", "m5x_final_gpu" + ("_dryrun" if dry else ""),
               "--min-free-mb", os.environ["SENTINEL_GPU_MIN_FREE_MB"], "--", sys.executable, "-B",
               str(Path(__file__).resolve()), "--chunk", str(run_dir), "--only", "gpu", "--max-minutes",
               str(a.max_minutes)] + (["--shard", a.shard] if a.shard else [])
        env = {**os.environ, "SENTINEL_GPUS": ",".join(map(str, free)), "M5X_GPU_CHILD": "1"}
        return subprocess.call(cmd, env=env, cwd=str(ROOT))
    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""     # a CPU chunk never touches a GPU
    rec, record_sha, code = load_record(dry)
    try:
        chunk_state(run_dir, dry=dry, root=root, reg=reg, record_sha=record_sha)   # before the design is loaded
        S, frozen, D = setup()
        if gpu:
            import torch

            if not torch.cuda.is_available():
                log("GPU chunk without a usable GPU")
                return 4
        log("chunk", run_dir.name, "| only:", a.only, "| shard:", a.shard, "| code:", code)
        return run_chunk(rec, run_dir, S, frozen, D, dry=dry, root=root, reg=reg, record_sha=record_sha, only=a.only,
                         shard=a.shard, max_minutes=a.max_minutes, device="cuda" if gpu else "cpu",
                         cpu_smoke=a.cpu_smoke)
    except XL.SingleLookError as e:
        _exit(e)


def cmd_seal(run_dir: Path) -> int:
    dry, root, reg = look_of(run_dir)
    rec, record_sha, _ = load_record(dry)
    try:
        s = do_seal(rec, run_dir, dry=dry, root=root, reg=reg, record_sha=record_sha)
    except XL.SingleLookError as e:
        _exit(e)
    log("sealed", s["n_files"], "prediction files:", s["sha256sums"])
    return 0


def cmd_score(run_dir: Path, authorised_by: str | None, rescore: bool) -> int:
    dry, root, reg = look_of(run_dir)
    if not dry and not authorised_by:
        raise SystemExit("--authorised-by is required for scoring")
    by = authorised_by or "dry-run (development data)"
    rec, record_sha, code = load_record(dry)
    signal.signal(signal.SIGTERM, lambda *x: (_ for _ in ()).throw(SystemExit("SIGTERM")))   # -> recorded abort
    try:
        pred_dir = XL.begin_scoring(by, run_dir, rescore=rescore, root=root, registry_path=reg)
    except XL.SingleLookError as e:
        _exit(e)
    try:                                            # from here on ANY stop is recorded; nothing is resumed
        C = do_score(rec, run_dir, pred_dir, dry=dry, root=root, reg=reg, record_sha=record_sha, code=code,
                     authorised_by=by, rescore=rescore)
    except BaseException as e:
        XL.record_abort(e, root=root, registry_path=reg)
        raise
    log("scored and sealed", run_dir)
    if len(C):
        log(C[["block", "target", "delta", "ci_low", "ci_high", "p_holm", "label"]].to_string())
        for r in C.itertuples():
            log(" ", r.block, r.target, f"listed {r.n_listed}, not beaten {len(r.listed_not_beaten)},",
                f"above C-X {len(r.listed_above_candidate)}, not estimable {len(r.listed_not_estimable)},",
                f"tolerance flags {len(r.tolerance_flags)}")
    return 0


def cmd_verify_scored(run_dir: Path) -> int:
    """After scoring: recompute from the sealed predictions and the sealed truth files into a separate folder (no
    label load, no registry write) and compare within 1e-9 (v11 item 6b). Completes a missing ``SCORED.json`` when
    the registry rows exist and the sealed results reproduce."""
    dry, root, reg = look_of(run_dir)
    rec, record_sha, _ = load_record(dry)
    try:
        if not XL.scoring_rows(reg):
            raise XL.SingleLookError("no scoring row exists: nothing to verify (use --rescore-from-saved)")
        pred_dir = XL.check_sealed(root=root, registry_path=reg, allow_scored=True)
        results = run_dir / "results"
        verify_sealed_results(results)
        out = run_dir / f"results_verify_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
        score_saved(rec, pred_dir, results / "truth", out, split="dry_run_dev" if dry else "test",
                    reg=out / "registry_rows.jsonl", record_sha=record_sha)
        cmp = compare_results(results, out, TOLERANCE["rescore_abs"])
        (out / "comparison.json").write_text(json.dumps(cmp, indent=1))
        if not cmp["agree"] or cmp["not_comparable"]:
            raise XL.SingleLookError(f"sealed results are not reproduced within 1e-9: {cmp['differences'][:5]} "
                                     f"{cmp['not_comparable']}")
        if not (XL.look_dir(root) / "SCORED.json").exists():
            XL.mark_scored({"results": str(results), "completed_by": "--verify-scored",
                            "results_sha256sums": file_sha256(results / "SHA256SUMS")}, root=root)
    except XL.SingleLookError as e:
        _exit(e)
    log("verified: sealed results reproduced within 1e-9 from the sealed predictions ->", out)
    return 0


def cmd_status(run_dir: Path) -> int:
    """Unit counts only (no prediction value is read)."""
    dry, root, _ = look_of(run_dir)
    plan = json.loads((run_dir / "plan.json").read_text())
    for name, sel in (("gpu", True), ("cpu", False)):
        us = [u for u in plan["units"] if u["gpu"] == sel]
        print(f"{name}: {sum(XL.unit_exists(run_dir, u['key']) for u in us)} of {len(us)} units written")
    d = XL.look_dir(root)
    print("dry run:", dry, "| sealed:", (d / "PREDICTIONS_SAVED.json").exists(), "| scoring started:",
          (d / "SCORING_STARTED.json").exists(), "| scored:", (d / "SCORED.json").exists())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--authorised-by")
    ap.add_argument("--claim", action="store_true", help="real look: claim, label-free inputs, unit plan")
    ap.add_argument("--dry-run-dev", action="store_true", help="the claim step on development folds (candidate record)")
    ap.add_argument("--max-folds", type=int, default=None, help="dry run only")
    ap.add_argument("--rehearsal-cap", type=int, default=None,
                    help="dry run only: max training rows per source and fold, reduced ensembles (numbers not usable)")
    ap.add_argument("--finish-claim", metavar="RUN_DIR", help="write the inputs of a claim that stopped before them")
    ap.add_argument("--chunk", metavar="RUN_DIR", help="fit units that do not exist yet")
    ap.add_argument("--only", choices=("gpu", "cpu"))
    ap.add_argument("--shard", help="i/n: every n-th selected unit starting at i")
    ap.add_argument("--max-minutes", type=float, default=CHUNK_MAX_MINUTES)
    ap.add_argument("--cpu-smoke", action="store_true", help="dry run only: fit GPU-kind units on CPU")
    ap.add_argument("--seal", metavar="RUN_DIR")
    ap.add_argument("--score", metavar="RUN_DIR")
    ap.add_argument("--rescore-from-saved", metavar="RUN_DIR")
    ap.add_argument("--verify-scored", metavar="RUN_DIR")
    ap.add_argument("--status", metavar="RUN_DIR")
    a = ap.parse_args()
    cmds = [bool(a.claim), bool(a.dry_run_dev), bool(a.finish_claim), bool(a.chunk), bool(a.seal), bool(a.score),
            bool(a.rescore_from_saved), bool(a.verify_scored), bool(a.status)]
    if sum(cmds) != 1:
        raise SystemExit("give exactly one of --claim, --dry-run-dev, --finish-claim, --chunk, --seal, --score, "
                         "--rescore-from-saved, --verify-scored, --status")
    if a.claim or a.dry_run_dev:
        return cmd_claim(a)
    if a.finish_claim:
        return cmd_finish_claim(Path(a.finish_claim).resolve())
    if a.chunk:
        return cmd_chunk(a)
    if a.seal:
        return cmd_seal(Path(a.seal).resolve())
    if a.score or a.rescore_from_saved:
        return cmd_score(Path(a.score or a.rescore_from_saved).resolve(), a.authorised_by, bool(a.rescore_from_saved))
    if a.verify_scored:
        return cmd_verify_scored(Path(a.verify_scored).resolve())
    return cmd_status(Path(a.status).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
