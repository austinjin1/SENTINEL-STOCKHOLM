#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SINGLE-PASS external evaluation of SENTINEL-BioAssess (protocol §7, §9.1). DO NOT RUN before freeze.

Modes
-----
* (default) **test pass** — reads ``docs/preregistration/bioassess_freeze.json`` (created by the lead).
  Refuses unless every frozen hash (code, shared modules, data, TabPFN weights) and package version
  matches, ``--authorised-by`` is given, no one-time marker exists and the registry holds no
  ``bioassess_test`` row. Setup (imports, GPU lock, data load) runs first; the marker is written only
  after setup succeeded (incl. a TabPFN/LightGBM/sklearn import and a TabPFN warm-up on the locked GPU).
  All test blocks of the freeze record run in one pass, in two phases: (1) every fit/prediction of every
  block (A, C, B, T) is produced and cached per unit (sha256 manifest) and the prediction tables are
  saved; (2) only then is anything scored, and registry rows are written only after all blocks are
  scored. Expected runtime of the real pass: ~6-9 h on one GPU.
* ``--dry-run-dev`` — the identical code path on development data only (dev = NRSA 2008-09,
  pseudo-test = NRSA 2013-14 for A/B/C, UK pool validation blocks as pseudo-reserved for T2); reads
  the candidate record, writes no marker, logs to a scratch registry inside the run directory.
* ``--resume RUN_DIR`` (with ``--authorised-by`` or ``--dry-run-dev``) — the pre-registered resume rule
  (protocol §9.1): after an infrastructure failure *before the first registry row*, the pass may be
  resumed ONCE, same record and authorisation, from the sha256-verified per-unit prediction cache in
  RUN_DIR/cache; cached units are never refitted, only units without a manifest entry are fitted.
* ``--score-from-predictions RUN_DIR`` — re-scores the saved prediction files of a finished pass
  (W8: GPU fits are not bit-reproducible, the saved predictions are the canonical record) into a
  scratch registry; no model is fitted.

Everything statistical comes from the record: ``tests``, ``n_boot`` (marginal CIs),
``n_boot_confirmatory`` (paired confirmatory tests), the candidates, the per-target SOTA comparators,
the H-M5 reference and the leave-BioSet-out / equal-size controls for candidate C.

Tests (models fit only on development rows):
* **A (confirmatory)** LOEO × cycle-out on NRSA 2018-19 — Holm families computed here only:
  SOTA = candidates × 6 targets vs the per-target strongest baseline; H-M5 = model A × 6 targets vs
  RF[community]. One-sided rule: p = bootstrap two-sided p if Δ > 0, else 1 (i.e. one-sided at α/2).
* **B** cycle-out on 2018-19 (all / new / re-sampled locations), **C** Test A protocol on 2023-24,
  **T1** UK pool -> 2018-19 (vs RF[geography] of Test B on the same rows), **T2** NRSA dev -> reserved
  UK — all secondary, unadjusted.
Per-seed: every stochastic model keeps seeds 0, 1, 2; an ensemble's seed-s prediction is the mean of
its members' seed-s fits; per-seed R² and per-seed Δ vs the SOTA comparator are logged.

    python -B scripts/eval/bioassess_final.py --authorised-by "<lead>"
    python -B scripts/eval/bioassess_final.py --dry-run-dev
    python -B scripts/eval/bioassess_final.py --resume <run_dir> --authorised-by "<lead>" --crash-note "<what failed>"
    python -B scripts/eval/bioassess_final.py --score-from-predictions <run_dir>
"""

from __future__ import annotations

import os

MAX_THREADS = 6
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
# GPU set = the lock helper's cap (.gpu_locks/ALLOWED); TabPFN needs ~5 GB physically free at lock time.
os.environ.setdefault("SENTINEL_GPU_MIN_FREE_MB", "5000")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sentinel.core import paths  # noqa: E402
from sentinel.core.registry import file_sha256, log_result, read_registry  # noqa: E402

FREEZE = REPO / "docs/preregistration/bioassess_freeze.json"
CANDIDATE = REPO / "docs/preregistration/bioassess_freeze_candidate.json"
MARKER = paths.RUNS_ROOT / "bioassess" / "final_scored.json"
EXPERIMENT = "bioassess_test"
SEEDS = (0, 1, 2)
N_SCORE_WORKERS = 5          # bootstrap workers (BLAS = 1 thread each; <= 6 threads in total)
TARGETS = ("log10_tp_ug_l", "log10_tn_mg_l", "log10_cond_us_cm", "log10_chloride_mg_l", "log10_turbidity_ntu",
           "ph_lab")
STOCHASTIC = {"rf_community", "rf_community_geography", "rf_geography", "rf_community_context", "rf_context",
              "gbm_community", "tabpfn_community"}
PARAM_KEY = {"rf_community": "rf", "rf_community_geography": "rf", "rf_geography": "rf", "rf_community_context": "rf",
             "rf_context": "rf", "gbm_community": "gbm", "tabpfn_community": "tabpfn",
             "knn_community": "knn_community", "wapls2": "wapls", "nearest_reference": "nearest_reference",
             "ecoregion_mean": None, "rf_taxa_only": "rf"}
MODEL_PREFIXES = ("BioSet[", "TabPFNv2[", "RandomForest[", "LightGBM[", "kNN-MAT[", "WA-PLS2[", "EcoregionMean",
                  "NearestReferenceSite")
EXPOSURE_NOTE = ("NRSA 2018-19 was used before this pass (i) by the research-phase RF baseline "
                 "(rf_blocked_baseline, fitted and scored within 2018-19) and (ii) from 2026-10-25 03:43 UTC as the "
                 "validation cycle of D5 BioIntegrity (benthic condition labels from the same samples; one oracle, "
                 "oracle_gbm_chem, takes 2018-19 chemistry as input). No M5 setting was chosen on it; M5's "
                 "candidates, comparators and settings were fixed on NRSA 2008-14 before (ii). Test C (2023-24) was "
                 "not scored by any workstream at freeze.")
log = logging.getLogger("bioassess_final")


# ------------------------------------------------------------------------------------ guards

def package_versions(names) -> dict:
    from importlib.metadata import version

    return {n: version(n) for n in names}


def verify_record(rec: dict, *, check_data: bool = True) -> list[str]:
    """List of mismatches between the record and the current environment (empty = OK)."""
    bad = [f"code:{f}" for f, h in rec["sha256"].items() if file_sha256(REPO / f) != h]
    if check_data:
        bad += [f"data:{f}" for f, h in rec.get("data_sha256", {}).items() if file_sha256(paths.DATA_ROOT / f) != h]
        ck = rec.get("tabpfn_checkpoint")
        if ck and file_sha256(Path(ck["path"]).expanduser()) != ck["sha256"]:
            bad.append("tabpfn_checkpoint")
    pv = rec.get("package_versions", {})
    cur = package_versions(pv)
    bad += [f"package:{k}={cur[k]}!={v}" for k, v in pv.items() if cur[k] != v]
    return bad


def verify_freeze(freeze_path: Path = FREEZE, marker: Path = MARKER, registry_path=None) -> dict:
    """Refuse unless the frozen record matches, no marker exists and no test row is in the registry."""
    if not Path(freeze_path).exists():
        raise SystemExit(f"refusing: no freeze record {freeze_path} (protocol §9)")
    rec = json.loads(Path(freeze_path).read_text())
    if "scripts/eval/bioassess_final.py" not in rec["sha256"]:
        raise SystemExit("refusing: freeze record does not hash this scoring script")
    bad = verify_record(rec)
    if bad:
        raise SystemExit(f"refusing: changed since freeze: {bad}")
    if Path(marker).exists():
        raise SystemExit(f"refusing: test already scored or started ({marker}); a re-run needs a new lead decision")
    if any(r.get("experiment") == EXPERIMENT for r in read_registry(registry_path)):
        raise SystemExit(f"refusing: the registry already holds '{EXPERIMENT}' rows")
    return rec


# ------------------------------------------------------------------------------------ helpers

def ensemble_members(name: str) -> list[str]:
    """'Ensemble[A+B]' -> ['A', 'B']: split on '+' outside brackets, re-attaching suffixes such as
    '+geography' / '+context' to the preceding member."""
    inner = name[len("Ensemble["):-1]
    pieces, depth, cur = [], 0, ""
    for ch in inner:
        depth += (ch == "[") - (ch == "]")
        if ch == "+" and depth == 0:
            pieces.append(cur)
            cur = ""
        else:
            cur += ch
    pieces.append(cur)
    merged: list[str] = []
    for piece in pieces:
        if merged and not piece.startswith(MODEL_PREFIXES):
            merged[-1] += "+" + piece
        else:
            merged.append(piece)
    return merged


def holm_adjust(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def is_sentinel(model: str) -> bool:
    return model.startswith(("BioSet[", "Ensemble[BioSet"))


class SafeLog:
    """``log_result`` that records NaN metrics as 'not estimable' instead of aborting the pass.

    Rows are BUFFERED: nothing reaches the registry until ``flush()``, which ``run_pass`` calls only
    after every block has been scored (W-a: a crash during scoring leaves no registry row, so the
    pre-registered resume rule still applies)."""

    def __init__(self, registry_path, common: dict):
        self.registry_path, self.common, self.not_estimable = registry_path, common, []
        self.pending: list[dict] = []

    def flush(self) -> int:
        n = 0
        for kw in self.pending:
            log_result(workstream="bioassess", experiment=EXPERIMENT, registry_path=self.registry_path, **kw)
            n += 1
        self.pending = []
        return n

    def __call__(self, **kw):
        v = kw.get("value")
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            self.not_estimable.append({k: kw.get(k) for k in ("model", "metric", "split", "target")})
            return None
        for k in ("ci_low", "ci_high"):
            if kw.get(k) is not None and not math.isfinite(float(kw[k])):
                kw[k] = None
        cfg = dict(self.common)
        cfg.update(kw.pop("config", {}) or {})
        self.pending.append({**kw, "config": cfg})
        return None


class PredCache:
    """Per-unit prediction cache (W-a). One parquet per unit (block | fold | base model, or a transfer
    seed), written atomically; ``cache/manifest.jsonl`` records each unit's sha256 the moment it is
    complete. On a resume a unit is reused only if its file matches the manifest; a unit without a
    manifest entry (the one in progress at a crash, and later ones) is fitted."""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir) / "cache"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self.dir / "manifest.jsonl"
        self.entries: dict[str, dict] = {}
        if self.manifest.exists():
            for line in self.manifest.read_text().splitlines():
                if line.strip():
                    e = json.loads(line)
                    self.entries[e["unit"]] = e
        self.n_preexisting, self.n_reused, self.n_written = len(self.entries), 0, 0

    def get(self, unit: str):
        e = self.entries.get(unit)
        if e is None:
            return None
        f = self.dir / e["file"]
        if not f.exists() or file_sha256(f) != e["sha256"]:
            raise SystemExit(f"refusing: cached unit {unit!r} fails its sha256 check ({f})")
        self.n_reused += 1
        return pd.read_parquet(f)

    def put(self, unit: str, df: pd.DataFrame) -> None:
        if unit in self.entries:
            return
        f = self.dir / (hashlib.sha256(unit.encode()).hexdigest()[:24] + ".parquet")
        tmp = f.with_suffix(".tmp")
        df.to_parquet(tmp)
        os.replace(tmp, f)
        e = {"unit": unit, "file": f.name, "sha256": file_sha256(f), "rows": len(df),
             "written": dt.datetime.now().isoformat(timespec="seconds")}
        with open(self.manifest, "a") as fh:
            fh.write(json.dumps(e) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.entries[unit] = e
        self.n_written += 1


def _seeds_to_frame(ps: dict[int, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat({f"seed{s}": ps[s] for s in SEEDS}, axis=1, names=["seed", "target"]).pipe(
        lambda d: d.set_axis([f"{a}|{b}" for a, b in d.columns], axis=1))


def _frame_to_seeds(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    out = {}
    for s in SEEDS:
        cols = [c for c in df.columns if c.startswith(f"seed{s}|")]
        out[s] = df[cols].set_axis([c.split("|", 1)[1] for c in cols], axis=1)
    return out


# ------------------------------------------------------------------------------------ prediction

class Predictor:
    """Fits a registry-named model on development rows and predicts test rows, keeping seeds."""

    def __init__(self, dev_tokens: pd.DataFrame, base_cfg: dict, device: str, cache: PredCache | None = None):
        from sentinel.models.bioassess import baselines as BL
        from sentinel.models.bioassess import bioset as BS

        self.BL = BL
        self.vocab = BS.TaxonVocab.build(dev_tokens)       # W1: development tokens only
        self.base_cfg, self.device, self.cache = base_cfg, device, cache
        self.by_reg = {v: k for k, v in BL.REGISTRY_MODEL_NAME.items()}
        self.memo: dict = {}

    def __call__(self, model, design, tr_ids, te_ids, unit: str | None = None) -> dict[int, pd.DataFrame]:
        """{seed: predictions}; memoised so a member shared by several ensembles is fitted once per fold.
        With ``unit`` (e.g. 'A|CPL') every base model's predictions are cached as unit 'A|CPL|<model>';
        ensembles are recomputed from their (cached) members."""
        if model.startswith("Ensemble["):
            parts = [self(m, design, tr_ids, te_ids, unit) for m in ensemble_members(model)]
            return {s: sum(p[s] for p in parts) / len(parts) for s in SEEDS}
        key = (model, tuple(sorted(set(design.samples["survey"]))), hash(tuple(tr_ids)), hash(tuple(te_ids)))
        uname = None if unit is None or self.cache is None else f"{unit}|{model}"
        if key not in self.memo and uname is not None:
            hit = self.cache.get(uname)
            if hit is not None:
                ps = _frame_to_seeds(hit)
                if list(ps[0].index) != list(pd.Index(te_ids)):
                    raise SystemExit(f"refusing: cached unit {uname!r} has different rows")
                self.memo[key] = ps
        if key not in self.memo:
            self.memo[key] = self._fit(model, design, tr_ids, te_ids)
        if uname is not None:
            self.cache.put(uname, _seeds_to_frame(self.memo[key]))
        return self.memo[key]

    def _fit(self, model, design, tr_ids, te_ids) -> dict[int, pd.DataFrame]:
        from sentinel.models.bioassess import cv as CV

        cols = list(TARGETS)
        if model.startswith("BioSet["):
            cand = model[len("BioSet["):model.index("]")]
            suffix = model[model.index("]") + 1:]
            geo = "context" if suffix == "+context" else (suffix == "+geography")
            cfg = CV.load_bioset_config(REPO / "configs/bioassess/bioset.yaml", cand)
            _, per_seed, _ = CV.bioset_rows(cfg, design, tr_ids, te_ids, vocab=self.vocab, seeds=SEEDS,
                                            device=self.device, geography=geo)
            return {s: pd.DataFrame(per_seed[s], index=pd.Index(te_ids), columns=cols) for s in SEEDS}
        name = self.by_reg[model]
        key = PARAM_KEY[name]
        params = dict(self.base_cfg.get(key) or {}) if key else {}
        if name in STOCHASTIC:
            return {s: self.BL.fit_predict(name, design, tr_ids, te_ids, seed=s, params=params,
                                           n_jobs=MAX_THREADS)[cols] for s in SEEDS}
        p = self.BL.fit_predict(name, design, tr_ids, te_ids, seed=0, params=params, n_jobs=MAX_THREADS)[cols]
        return {s: p for s in SEEDS}


def frame_of(design, rows, preds: dict[str, dict[int, pd.DataFrame]]) -> pd.DataFrame:
    """Canonical saved prediction table: observations, metadata, seed-mean and per-seed predictions."""
    from sentinel.eval.splits import spatial_blocks

    s = design.samples.loc[rows]
    out = pd.DataFrame({"visit_id": rows, "survey": s["survey"].to_numpy(), "location_id": s["location_id"].to_numpy(),
                        "ag_eco9": s["ag_eco9"].to_numpy(),
                        "block50": spatial_blocks(s["lat"].to_numpy(), s["lon"].to_numpy(), 50.0)})
    for t in TARGETS:
        out[f"obs|{t}"] = design.Y.loc[rows, t].to_numpy(float)
    for m, ps in preds.items():
        for t in TARGETS:
            for sd in SEEDS:
                out[f"pred_seed{sd}|{m}|{t}"] = ps[sd].loc[rows, t].to_numpy(float)
            out[f"pred|{m}|{t}"] = np.mean([out[f"pred_seed{sd}|{m}|{t}"] for sd in SEEDS], 0)
    return out


# ------------------------------------------------------------------------------------ scoring

def score_frame(F: pd.DataFrame, rec: dict, block: str, split: str, slog: SafeLog, *, confirmatory: bool,
                run_dir: Path, models: list[str]) -> dict:
    """Marginal metrics for every model, paired Δs for every candidate; Holm only when confirmatory."""
    from sentinel.eval import metrics as EM
    from sentinel.models.bioassess import evaluate as E

    nb, nbc = int(rec["n_boot"]), int(rec["n_boot_confirmatory"])
    region, cluster = F["ag_eco9"].to_numpy(), F["location_id"].to_numpy()
    block50 = F["block50"].to_numpy()
    tag = "confirmatory" if confirmatory else "secondary, unadjusted"
    note = f"SINGLE external test pass, block {block} ({tag}). {EXPOSURE_NOTE if 'nrsa1819' in split else ''}"
    from joblib import Parallel, delayed

    def marginal(m, t):
        y, p = F[f"obs|{t}"].to_numpy(float), F[f"pred|{m}|{t}"].to_numpy(float)
        return (m, t, E.summary(y, p, region), E.cis(y, p, region, cluster, n_boot=nb),
                {sd: EM.r2(y, F[f"pred_seed{sd}|{m}|{t}"].to_numpy(float)) for sd in SEEDS})

    par = Parallel(n_jobs=N_SCORE_WORKERS)
    for m, t, s, ci, seed_r2 in par(delayed(marginal)(m, t) for m in models for t in TARGETS):
        for metric in ("r2", "r2_anomaly", "r2_region_median"):
            slog(model=m, metric=metric, value=s[metric], split=split, n=s["n"], ci_low=ci[metric]["ci_low"],
                 ci_high=ci[metric]["ci_high"], target=t, is_baseline=not is_sentinel(m),
                 ci_method=f"percentile cluster bootstrap, unit = location cluster, B={nb}",
                 config={"block": block}, data_manifests=["nars_streams", "taxonomy_harmonised"],
                 artifacts=[str(run_dir)], notes=note,
                 extra={"r2_by_region": s["r2_by_region"], "ci_ecoregion_unit": ci[metric].get("ci_ecoregion"),
                        "rmse": s["rmse"], "spearman": s["spearman"], "seed_r2": seed_r2})
    fam = {"sota": {}, "h_m5": {}}
    comps = []
    for c in rec["candidates"]:
        refs = [("sota", rec["sota_comparators"][t], t) for t in TARGETS]
        refs += [("h_m5", rec["h_m5_reference"], t) for t in TARGETS]
        for label, ref in (rec.get("controls_for", {}).get(c, {}) or {}).items():
            refs += [(label, ref, t) for t in TARGETS]
        def paired(kind, ref, t):
            y = F[f"obs|{t}"].to_numpy(float)
            pc, pr = F[f"pred|{c}|{t}"].to_numpy(float), F[f"pred|{ref}|{t}"].to_numpy(float)
            B = nbc if (confirmatory and kind in ("sota", "h_m5")) else nb
            pv = E.paired_vs(y, pc, pr, region, cluster, metric="r2", n_boot=B)
            sens = {u: E.paired_vs(y, pc, pr, region, g, metric="r2", n_boot=nb)
                    for u, g in (("block50", block50), ("ecoregion", region))}
            per_seed = {sd: EM.r2(y, F[f"pred_seed{sd}|{c}|{t}"]) - EM.r2(y, F[f"pred_seed{sd}|{ref}|{t}"])
                        for sd in SEEDS}
            return kind, ref, t, B, pv, sens, per_seed

        for kind, ref, t, B, pv, sens, per_seed in par(delayed(paired)(*r) for r in refs):
            one_sided_p = pv["p_value"] if pv["diff"] > 0 else 1.0
            comps.append({"candidate": c, "kind": kind, "reference": ref, "target": t, **pv,
                          "p_one_sided_rule": one_sided_p, "per_seed_delta": per_seed,
                          "sensitivity": {u: [v["ci_low"], v["ci_high"]] for u, v in sens.items()}})
            slog(model=c, metric=f"delta_r2_vs_{kind}", value=pv["diff"], split=split, target=t,
                 ci_low=pv["ci_low"], ci_high=pv["ci_high"], is_baseline=False,
                 ci_method=f"paired percentile cluster bootstrap, unit = location cluster, B={B}",
                 config={"block": block, "reference": ref}, data_manifests=["nars_streams"], artifacts=[str(run_dir)],
                 notes=f"{note} Candidate minus {ref}.",
                 extra={"p_value_two_sided": pv["p_value"], "p_one_sided_rule": one_sided_p, "reference_model": ref,
                        "per_seed_delta": per_seed, "all_seeds_positive": all(v > 0 for v in per_seed.values()),
                        "sensitivity_ci": {u: [v["ci_low"], v["ci_high"]] for u, v in sens.items()}})
            if confirmatory and kind == "sota":
                fam["sota"][f"{c}|{t}"] = one_sided_p
            if confirmatory and kind == "h_m5" and c == rec["h_m5_family_candidate"]:
                fam["h_m5"][f"{c}|{t}"] = one_sided_p
    (run_dir / f"comparisons_{block}.json").write_text(json.dumps(comps, indent=1, default=float))
    decisions = {}
    if confirmatory:
        alpha = float(rec.get("alpha", 0.05))
        for family, pvals in fam.items():
            adj = holm_adjust(pvals)
            for key, p_adj in adj.items():
                c, t = key.split("|")
                rej = p_adj <= alpha
                comp = next(x for x in comps if x["candidate"] == c and x["target"] == t and x["kind"] == family)
                sota_label = (family == "sota" and rej and all(v > 0 for v in comp["per_seed_delta"].values()))
                decisions[f"{family}|{key}"] = {"p_raw": pvals[key], "p_holm": p_adj, "rejected": rej,
                                               "per_seed_all_positive": all(v > 0 for v in comp["per_seed_delta"].values()),
                                               "sota_label": sota_label if family == "sota" else None}
                slog(model=c, metric=f"holm_rejected_{family}", value=float(rej), split=split, target=t,
                     is_baseline=False, config={"block": block, "family": family},
                     ci_method=f"Holm step-down over {len(pvals)} tests, alpha={alpha}; one-sided rule p=two-sided p "
                               f"if delta>0 else 1",
                     notes=f"{note} Confirmatory Holm decision ({family} family).",
                     extra={"p_raw": pvals[key], "p_holm_adjusted": p_adj, "alpha": alpha, "family_size": len(pvals),
                            "per_seed_all_positive": decisions[f"{family}|{key}"]["per_seed_all_positive"],
                            "sota_label_awarded": sota_label if family == "sota" else None})
        if fam["h_m5"]:        # O2: the pre-registered H-M5 endpoint (>= 3 of 6 targets) as its own row
            n_rej = sum(1 for k, v in decisions.items() if k.startswith("h_m5|") and v["rejected"])
            slog(model=rec["h_m5_family_candidate"], metric="h_m5_n_targets_rejected", value=float(n_rej),
                 split=split, is_baseline=False, config={"block": block, "family": "h_m5"},
                 ci_method=f"count of Holm-rejected targets in the H-M5 family ({len(fam['h_m5'])} tests)",
                 notes=f"{note} H-M5 endpoint: met if >= 3 of {len(fam['h_m5'])} targets are Holm-rejected.",
                 extra={"endpoint_met": n_rej >= 3, "family_size": len(fam["h_m5"])})
        (run_dir / f"holm_{block}.json").write_text(json.dumps(decisions, indent=1, default=float))
    return decisions


def score_test_b(F, rec, survey, slog, run_dir, models):
    new = ~F["resampled"].to_numpy(bool)
    for block, mask in (("B_all", np.ones(len(F), bool)), ("B_new", new), ("B_resampled", ~new)):
        score_frame(F[mask].reset_index(drop=True), rec, block, f"{survey}/cycle_out_test_{block[2:]}", slog,
                    confirmatory=False, run_dir=run_dir, models=models)


# ------------------------------------------------------------------------------------ the pass

def warm_up_tabpfn(design, dev_ids, base_cfg: dict) -> str:
    """R2: import TabPFN (the chemtox shim and its package), LightGBM and sklearn's ensembles, and load the
    TabPFN checkpoint onto the locked GPU with one tiny fit on development rows (output discarded),
    all before the one-time marker. Model fits pass their own ``random_state``, so this changes no result."""
    import lightgbm  # noqa: F401
    import sklearn.ensemble  # noqa: F401
    import torch

    from sentinel.models.bioassess.features import DESCRIPTORS
    from sentinel.models.chemtox.tabpfn_model import load_tabpfn

    TabPFNRegressor, version = load_tabpfn()
    rows = pd.Index(dev_ids)[:64]
    X = design.samples.loc[rows, list(DESCRIPTORS)].to_numpy(np.float32)
    y = design.Y.loc[rows, TARGETS[0]].to_numpy(float)
    ok = np.isfinite(y)
    m = TabPFNRegressor(n_estimators=1, device=(base_cfg.get("tabpfn") or {}).get("device", "cuda"), random_state=0,
                        ignore_pretraining_limits=True)
    m.fit(X[ok], y[ok])
    m.predict(X[ok][:8])
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return version


def score_transfer(P1, P2, FB, design, uk, tst, reserved, test_a: str, uk_res_role: str, rec: dict, slog: SafeLog):
    """T1 (UK pool -> NRSA test cycle, vs RF[geography] of Test B on the same rows) and T2 (NRSA dev ->
    reserved UK; no comparator). Secondary, unadjusted."""
    from sentinel.eval import bootstrap as EB
    from sentinel.eval import metrics as EM
    from sentinel.models.bioassess import evaluate as E
    from sentinel.models.bioassess import transfer as TR

    region = design.samples.loc[tst, "ag_eco9"].to_numpy()
    cluster = design.samples.loc[tst, "location_id"].to_numpy()
    base = ("Transfer test (secondary, unadjusted). Deviation from protocol §7.4 (see §10): family-level RF, "
            "not the encoder.")
    t1note = f"{base} {EXPOSURE_NOTE if test_a == 'nrsa1819' else ''}"
    for t_us, t_uk in TR.SHARED.items():
        y = design.Y.loc[tst, t_us].to_numpy(float)
        p = np.mean([P1[s][t_us].to_numpy(float) for s in SEEDS], 0)
        s1 = E.summary(y, p, region)
        c1 = E.cis(y, p, region, cluster, n_boot=int(rec["n_boot"]))
        for metric in ("r2", "r2_anomaly", "r2_region_median"):
            slog(model="RandomForest[family-log-class]", metric=metric, value=s1[metric],
                 split=f"ukpool->{test_a}/transfer_test", n=s1["n"], ci_low=c1[metric]["ci_low"],
                 ci_high=c1[metric]["ci_high"], target=t_us, is_baseline=True,
                 ci_method=f"percentile cluster bootstrap, unit = location cluster, B={rec['n_boot']}",
                 data_manifests=["nars_streams", "uk_ea_ecology"], notes=f"T1. {t1note}",
                 extra={"seed_r2": {s: EM.r2(y, P1[s][t_us].to_numpy(float)) for s in SEEDS}})
        if FB is not None:      # geography-only comparator on the same rows (Test B fit, dev -> test)
            g = FB.set_index("visit_id").loc[tst, f"pred|RandomForest[geography]|{t_us}"].to_numpy(float)
            pv = E.paired_vs(y, p, g, region, cluster, metric="r2", n_boot=int(rec["n_boot"]))
            slog(model="RandomForest[family-log-class]", metric="delta_r2_vs_geography_only", value=pv["diff"],
                 split=f"ukpool->{test_a}/transfer_test", target=t_us, ci_low=pv["ci_low"],
                 ci_high=pv["ci_high"], is_baseline=True,
                 ci_method=f"paired percentile cluster bootstrap, unit = location cluster, B={rec['n_boot']}",
                 data_manifests=["nars_streams", "uk_ea_ecology"],
                 notes=f"T1 minus RandomForest[geography] (NRSA-trained). {t1note}",
                 extra={"p_value_two_sided": pv["p_value"]})
        yu = uk.Y.loc[reserved, t_uk]
        ok = yu.notna().to_numpy()
        y2 = yu.to_numpy(float)[ok]
        p2 = np.mean([P2[s][t_us].to_numpy(float) for s in SEEDS], 0)[ok]
        b = EB.bootstrap_ci(EM.r2, y2, p2, groups=uk.samples.loc[reserved[ok], "site_id"].to_numpy(),
                            n_boot=int(rec["n_boot"]))
        slog(model="RandomForest[family-log-class]", metric="r2", value=EM.r2(y2, p2),
             split=f"nrsa_dev->uk_{uk_res_role}/transfer_test", n=int(ok.sum()), ci_low=b["ci_low"],
             ci_high=b["ci_high"], target=t_us, is_baseline=True,
             ci_method=f"percentile cluster bootstrap, unit = UK site, B={rec['n_boot']}",
             data_manifests=["nars_streams", "uk_ea_ecology"],
             notes=f"T2 (no comparator exists for this direction). {base}",
             extra={"seed_r2": {s: EM.r2(y2, P2[s][t_us].to_numpy(float)[ok]) for s in SEEDS}})


def run_pass(rec: dict, *, dry: bool, run_dir: Path, registry_path, authorised_by: str, after_setup,
             resume_note: str | None = None) -> None:
    """Setup -> marker -> PHASE 1: all fits/predictions of all blocks, cached per unit and saved ->
    PHASE 2: scoring of all blocks into a buffer -> registry flush. ``resume_note`` marks a resumed pass."""
    import torch
    import yaml

    from sentinel.core.gpu import acquire_gpu
    from sentinel.core.runlog import TrainingRun
    from sentinel.models.bioassess import transfer as TR
    from sentinel.models.bioassess.data import load_nrsa
    from sentinel.models.bioassess.uk import load_uk

    torch.set_num_threads(2)
    gpu = acquire_gpu("bioassess_final" + ("_dryrun" if dry else ""))   # GPU set = lock-helper cap (ALLOWED)
    hw = {"gpu_index": gpu, "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
          "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i",
                                    str(gpu)], capture_output=True, text=True).stdout.strip()}
    if dry:   # development data only: dev = 2008-09, pseudo-test = 2013-14; UK pool_val as pseudo-reserved
        dev_s, test_a, test_c = ("nrsa0809",), "nrsa1314", "nrsa1314"
        design_ac = load_nrsa(("nrsa0809", "nrsa1314"))
        uk = load_uk(include_reserved=False)
        uk_res_role = "pool_val"
    else:
        dev_s, test_a, test_c = ("nrsa0809", "nrsa1314"), "nrsa1819", "nrsa2324"
        design_ac = load_nrsa(dev_s + (test_a,), allow_holdout=True)
        uk = load_uk(include_reserved=True)
        uk_res_role = "reserved_test"
    design_c = design_ac if test_c == test_a else load_nrsa(dev_s + (test_c,), allow_holdout=True)
    base_cfg = yaml.safe_load((REPO / "configs/bioassess/baselines.yaml").read_text())
    dev_ids = design_ac.samples.index[design_ac.samples["survey"].isin(dev_s)]
    cache = PredCache(run_dir)
    predict = Predictor(design_ac.tokens[design_ac.tokens["visit_id"].isin(set(dev_ids))], base_cfg, "cuda", cache)
    tabpfn_version = warm_up_tabpfn(design_ac, dev_ids, base_cfg)
    common = {"freeze_record": {k: rec[k] for k in ("candidates", "sota_comparators", "h_m5_reference", "tests",
                                                    "n_boot", "n_boot_confirmatory")},
              "n_configurations_tried": rec.get("n_configurations_tried"), "selection_note": rec.get("selection_note"),
              "authorised_by": authorised_by, "dry_run": dry, "hardware": hw, "resumed": resume_note}
    slog = SafeLog(registry_path, common)
    after_setup()                      # the one-time marker is written only now (real pass)
    models = list(rec["models"])
    tag = "final_test_pass" + ("_dryrun_dev" if dry else "") + ("_resume" if resume_note else "")
    with TrainingRun("bioassess", tag, config={**common, "tabpfn_version": tabpfn_version},
                     data=["nars_streams", "taxonomy_harmonised", "uk_ea_ecology"],
                     plan_ref="docs/preregistration/bioassess_protocol.md §7, §9.1", artifacts_dir=str(run_dir)) as run:
        if resume_note:
            run.note(f"RESUME ({resume_note}); {cache.n_preexisting} cached units found")
        # ---------------- PHASE 1: every fit / prediction of every block; nothing is scored here
        frames, saved = {}, {}
        for block, design, survey in (("A", design_ac, test_a), ("C", design_c, test_c)):
            if block not in rec["tests"]:
                continue
            dev = design.samples.index[design.samples["survey"].isin(dev_s)]
            tst = design.samples.index[design.samples["survey"] == survey]
            preds = {m: {s: pd.DataFrame(np.nan, index=tst, columns=list(TARGETS)) for s in SEEDS} for m in models}
            for g in sorted(design.samples.loc[tst, "ag_eco9"].unique()):
                tr = dev[design.samples.loc[dev, "ag_eco9"] != g]
                te = tst[design.samples.loc[tst, "ag_eco9"] == g]
                if set(design.samples.loc[tr, "location_id"]) & set(design.samples.loc[te, "location_id"]):
                    raise AssertionError("location overlap in LOEO x cycle fold")
                for m in models:
                    ps = predict(m, design, tr, te, unit=f"{block}|{g}")
                    for s in SEEDS:
                        preds[m][s].loc[te] = ps[s].to_numpy()
                log.info("block %s ecoregion %s predicted", block, g)
            frames[block] = (frame_of(design, tst, preds), survey)
        if "B" in rec["tests"]:
            design = design_ac
            dev = design.samples.index[design.samples["survey"].isin(dev_s)]
            tst = design.samples.index[design.samples["survey"] == test_a]
            F = frame_of(design, tst, {m: predict(m, design, dev, tst, unit="B|all") for m in models})
            F["resampled"] = F["location_id"].isin(set(design.samples.loc[dev, "location_id"])).to_numpy()
            frames["B"] = (F, test_a)
            log.info("block B predicted")
        P1 = P2 = None
        if "T" in rec["tests"]:
            pool = uk.samples.index[uk.samples["role"] == "pool_fit"]
            reserved = uk.samples.index[uk.samples["role"] == uk_res_role]
            tst_t = design_ac.samples.index[design_ac.samples["survey"] == test_a]
            dev_t = design_ac.samples.index[design_ac.samples["survey"].isin(dev_s)]
            P1, P2 = {}, {}
            for s in SEEDS:
                hit = cache.get(f"T1|seed{s}")
                P1[s] = hit if hit is not None else TR.uk_to_us(uk, design_ac, pool, tst_t, seed=s, n_jobs=MAX_THREADS)
                cache.put(f"T1|seed{s}", P1[s])
                hit = cache.get(f"T2|seed{s}")
                P2[s] = hit if hit is not None else TR.us_to_uk(design_ac, uk, dev_t, reserved, seed=s,
                                                                 n_jobs=MAX_THREADS)
                cache.put(f"T2|seed{s}", P2[s])
            log.info("block T predicted")
        for block, (F, _) in frames.items():
            F.to_parquet(run_dir / f"pred_test{block}.parquet")
            saved[f"pred_test{block}.parquet"] = file_sha256(run_dir / f"pred_test{block}.parquet")
        if P1 is not None:
            pd.concat({f"seed{s}": P1[s] for s in SEEDS}, axis=1).to_parquet(run_dir / "pred_T1.parquet")
            pd.concat({f"seed{s}": P2[s] for s in SEEDS}, axis=1).to_parquet(run_dir / "pred_T2.parquet")
            saved.update({f: file_sha256(run_dir / f) for f in ("pred_T1.parquet", "pred_T2.parquet")})
        (run_dir / "predictions_manifest.json").write_text(json.dumps(
            {"files": saved, "cache_units": len(cache.entries), "cache_reused": cache.n_reused,
             "cache_written": cache.n_written, "completed": dt.datetime.now().isoformat(timespec="seconds")},
            indent=1))
        run.note(f"phase 1 done: all predictions saved ({len(cache.entries)} cached units; "
                 f"{cache.n_reused} reused, {cache.n_written} fitted in this process)")
        # ---------------- PHASE 2: scoring (buffered), then one registry flush
        for block in ("A", "C"):
            if block in frames:
                F, survey = frames[block]
                score_frame(F, rec, block, f"{survey}/loeo_x_cycle_test", slog,
                            confirmatory=(block == rec["confirmatory_test"]), run_dir=run_dir, models=models)
        if "B" in frames:
            score_test_b(frames["B"][0], rec, test_a, slog, run_dir, models)
        if P1 is not None:
            score_transfer(P1, P2, frames["B"][0] if "B" in frames else None, design_ac, uk, tst_t, reserved,
                           test_a, uk_res_role, rec, slog)
        (run_dir / "not_estimable.json").write_text(json.dumps(slog.not_estimable, indent=1))
        n = slog.flush()
        run.note(f"phase 2 done: {n} registry rows written; {len(slog.not_estimable)} metrics not estimable "
                 f"(recorded, not logged)")


def verify_resume(run_dir: Path, authorised_by: str | None, *, dry: bool, freeze_path: Path = FREEZE,
                  marker: Path = MARKER, registry_path=None) -> dict:
    """The pre-registered resume rule (protocol §9.1): once, same record, same authorisation, only if no
    registry row of the pass exists. Returns the record to resume with."""
    run_dir = Path(run_dir)
    if not (run_dir / "freeze_record.json").exists() or not (run_dir / "cache" / "manifest.jsonl").exists():
        raise SystemExit(f"refusing resume: {run_dir} has no freeze_record.json / cache manifest")
    rec = json.loads((run_dir / "freeze_record.json").read_text())
    if (run_dir / "resume.json").exists():
        raise SystemExit("refusing resume: this pass was already resumed once; a further run needs a new lead decision")
    if dry:
        reg = run_dir / "scratch_registry.jsonl"
        if reg.exists() and reg.read_text().strip():
            raise SystemExit("refusing resume: the scratch registry already holds rows")
        return rec
    if not Path(freeze_path).exists() or json.loads(Path(freeze_path).read_text()) != rec:
        raise SystemExit("refusing resume: the run directory's record differs from bioassess_freeze.json")
    bad = verify_record(rec)
    if bad:
        raise SystemExit(f"refusing resume: changed since freeze: {bad}")
    if not Path(marker).exists():
        raise SystemExit("refusing resume: no one-time marker (nothing to resume)")
    mk = json.loads(Path(marker).read_text())
    if Path(mk.get("run_dir", "")) != run_dir:
        raise SystemExit(f"refusing resume: the marker belongs to {mk.get('run_dir')}")
    if "finished" in mk or "resumed" in mk:
        raise SystemExit("refusing resume: the pass finished or was already resumed")
    if mk.get("authorised_by") != authorised_by:
        raise SystemExit("refusing resume: a resume needs the same authorisation as the crashed pass")
    if any(r.get("experiment") == EXPERIMENT for r in read_registry(registry_path)):
        raise SystemExit(f"refusing resume: the registry already holds '{EXPERIMENT}' rows")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--authorised-by", default=None)
    g.add_argument("--dry-run-dev", action="store_true")
    g.add_argument("--score-from-predictions", default=None, metavar="RUN_DIR")
    ap.add_argument("--resume", default=None, metavar="RUN_DIR",
                    help="pre-registered one-time resume after an infrastructure failure (needs --crash-note)")
    ap.add_argument("--crash-note", default=None, help="what failed (logged with the resume)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    if a.resume and (a.score_from_predictions or not a.crash_note or not (a.authorised_by or a.dry_run_dev)):
        ap.error("--resume needs --crash-note and either --authorised-by (same as the crashed pass) or --dry-run-dev")
    if a.score_from_predictions:
        src = Path(a.score_from_predictions)
        rec = json.loads((src / "freeze_record.json").read_text())
        run_dir = paths.run_dir("bioassess", f"rescore_{stamp}")
        slog = SafeLog(run_dir / "scratch_registry.jsonl", {"rescored_from": str(src)})
        for f in sorted(src.glob("pred_test[ABC].parquet")):
            block = f.stem[-1]
            F = pd.read_parquet(f)
            survey = F["survey"].iloc[0]
            if block == "B":
                score_test_b(F, rec, survey, slog, run_dir, rec["models"])
            else:
                score_frame(F, rec, block, f"{survey}/loeo_x_cycle_test", slog,
                            confirmatory=(block == rec["confirmatory_test"]), run_dir=run_dir, models=rec["models"])
        slog.flush()
        print(f"re-scored into {run_dir / 'scratch_registry.jsonl'}")
        return 0
    if a.resume:
        run_dir = Path(a.resume)
        rec = verify_resume(run_dir, a.authorised_by, dry=a.dry_run_dev)
        note = {"resumed": dt.datetime.now().isoformat(), "crash_note": a.crash_note}
        (run_dir / "resume.json").write_text(json.dumps(note, indent=1))

        def mark_resumed():
            if not a.dry_run_dev:
                MARKER.write_text(json.dumps({**json.loads(MARKER.read_text()), **note}, indent=1))

        reg = run_dir / "scratch_registry.jsonl" if a.dry_run_dev else None
        run_pass(rec, dry=a.dry_run_dev, run_dir=run_dir, registry_path=reg,
                 authorised_by=a.authorised_by or "dry-run", after_setup=mark_resumed, resume_note=a.crash_note)
        if not a.dry_run_dev:
            MARKER.write_text(json.dumps({**json.loads(MARKER.read_text()), "finished": dt.datetime.now().isoformat()},
                                         indent=1))
        print(f"resumed pass complete: {run_dir}")
        return 0
    if a.dry_run_dev:
        rec = json.loads(CANDIDATE.read_text())
        bad = verify_record(rec, check_data=True)
        if bad:
            log.warning("candidate record differs from the environment (dry run continues): %s", bad)
        run_dir = paths.run_dir("bioassess", f"final_dryrun_dev_{stamp}")
        (run_dir / "freeze_record.json").write_text(json.dumps(rec, indent=1))
        run_pass(rec, dry=True, run_dir=run_dir, registry_path=run_dir / "scratch_registry.jsonl",
                 authorised_by="dry-run", after_setup=lambda: None)
        print(f"dry run complete: {run_dir}")
        return 0
    if not a.authorised_by:
        ap.error("--authorised-by is required for the test pass")
    rec = verify_freeze()
    run_dir = paths.run_dir("bioassess", f"final_{stamp}")
    (run_dir / "freeze_record.json").write_text(json.dumps(rec, indent=1))

    def write_marker():
        MARKER.parent.mkdir(parents=True, exist_ok=True)
        MARKER.write_text(json.dumps({"started": dt.datetime.now().isoformat(), "authorised_by": a.authorised_by,
                                      "run_dir": str(run_dir)}, indent=1))

    run_pass(rec, dry=False, run_dir=run_dir, registry_path=None, authorised_by=a.authorised_by,
             after_setup=write_marker)
    MARKER.write_text(json.dumps({**json.loads(MARKER.read_text()), "finished": dt.datetime.now().isoformat()},
                                 indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
