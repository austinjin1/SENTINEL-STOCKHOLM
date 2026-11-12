#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Development cross-validation of the M3 architecture round v2 (arms D1, D3, D4, the §6 parity list, GEO-8).

Governing texts: docs/preregistration/microbe_arch_v2_amendment.md (lead stamp 2026-11-10) and
docs/preregistration/microbe_arch_v2_lead_rulings.md. Development rows only: ``load_design()`` refuses holdout
events and ``microbe_geo.load_dev_targets`` refuses protected ids; the protected label file is never opened.

Every mode first passes ``archv2.gate`` for every (model, scheme, family) it would touch — amendment stamp, the
deviations-file release where one is needed (for a +QMP twin also its base model's), released scheme, +QMP rules,
slot #8, the logged G10 choice for ``niche_s_qmp`` — BEFORE any data is loaded.

Modes (one per call):

  (default)            out-of-fold predictions of ``--models`` on ``--schemes`` for ``--families``; one fit pass
                       serves both families. Output ``$SENTINEL_RUNS/microbe/cv_archv2/<scheme>/<family>/``.
                       Registry: workstream 'M3-microbe', experiment 'microbe_arch_v2_cv', split
                       'neon_dev/<scheme>_cv'. A failed run gets a registry row ``run_failed``.
  --ensemble A B ..    late fusion of existing out-of-fold files (protocol §10.1), per family.
  --tier0-best         per-target best of GlobalMean / Ridge[DOY+location] / RF[DOY+location] on development LODO.
  --calibration M ..   the calibration diagnostic of lead ruling A1 on existing out-of-fold files.
  --smoke-train-only   NicheInvert-S fitted on the training rows of fold 0 and inverted on those same rows:
                       first-step failure share (kill rule §2.5) and CPU time. No held-out row, no metric.

Fold shards of the inversion-based models (lead ruling of 2026-11-10, execution log E14) and, one model per
call, of the five long comparators / block models of ``SHARD_COMPARATORS`` (lead ruling B, execution log E20):

  --models M .. --shard-folds I ..   one process fits the listed outer folds (0-based) of the listed models with
                       the same code path per fold as the unsharded run, at one thread (the thread count of
                       these models, sharded or not). It writes
                       the fold's predictions, its label-free diagnostics and its own CPU time under
                       ``<scheme>/_shards/<model>/`` -- for the comparators of ``SHARD_COMPARATORS``, which run
                       one family set per call, ``<scheme>/_shards/<model>__<families>/`` (lead ruling on
                       xgb_v4mr of 2026-11-10 §V-3/§V-4) -- and nothing else: no metric is computed, printed
                       or logged.
  --models M .. --merge-shards       once, after every fold is present: sums the CPU time over the shards, refuses
                       to score if the sum is over the model's budget (registry row ``run_failed``, no metric),
                       otherwise writes and scores the merged table exactly as the unsharded run does.

    python -B scripts/train/microbe_arch_v2_cv.py --models niche_s --families P G
    python -B scripts/train/microbe_arch_v2_cv.py --models niche_s xgb_nichez_context --shard-folds 0 6 12 18
    python -B scripts/train/microbe_arch_v2_cv.py --models niche_s --merge-shards
    python -m sentinel.core.gpu --name m3_tabpfn_v4 --min-free-mb 10500 -- \
        python -B scripts/train/microbe_arch_v2_cv.py --models tabpfn_v4clr --families P G --device cuda
"""

from __future__ import annotations

import os

MAX_THREADS = 6   # shared-host rule: one CPU-heavy job, <= 6 threads
# The inversion-based models run at ONE thread, sharded or not (lead ruling 2026-11-10, log E14/E15: the thread
# count is not a model setting; the train-only timing showed no wall-time gain from more BLAS threads).
NICHE_THREADS = 1
# Long comparators and block models whose fold shards the lead approved (ruling B, execution log E20): same code
# path per fold, one model at a time, <= 6 one-thread shard processes at one time. Like the inversion-based models
# they run at ONE thread sharded or not, so the unsharded table and the merged table are the same bit for bit (the
# thread count is not a model setting). None of them had a table from this runner before this rule.
SHARD_COMPARATORS = ("rf_clr_tuned", "xgb_clr_tuned", "xgb_clr_stack", "xgb_v4mr", "xgb_v4clr_3000")
# numpy's OpenBLAS reads its thread count when it is loaded: 6 for the dense algebra of the inversion and of
# weighted averaging (each call is wrapped in threadpool_limits(n_jobs)); OpenMP / MKL stay at 1 (XGBoost and the
# forests get their threads through n_jobs).
os.environ["OPENBLAS_NUM_THREADS"] = str(MAX_THREADS)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np   # loads OpenBLAS with 6 threads before microbe_cv resets the variable
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.core.registry import log_result
from sentinel.core.runlog import TrainingRun

log = logging.getLogger("microbe_arch_v2_cv")
PLAN_REF = "microbe_arch_v2_amendment §2-§7; microbe_arch_v2_lead_rulings §2"
CPU_BUDGET_H = {"niche_s": 6.0, "niche_s_tax": 6.0, "niche_s_a0": 6.0, "niche_s_qmp": 6.0, "xgb_v4mr": 30.0,
                "xgb_v4mr_stack": 12.0}          # amendment §2.5, §3.3 (per track); xgb_v4mr: 30.0 per family run
# (lead ruling on xgb_v4mr of 2026-11-10 §V-2/§V-4, registry row fa48ff4c72bf -- at most 30 CPU-h for each target
# family, 60 CPU-h in all; xgb_v4mr runs one family per call (§V-3), so each family's merge sums that family's
# shards only, and no other entry of this table changes).
COUNTED = ("niche_s", "xgb_v4mr", "xgb_v4mr_stack", "niche_s_a0", "niche_s_tax", "xgb_nichez_context",
           "niche_s_qmp")                        # niche_s_qmp: dev01 lead ruling D8
CODE_FILES = ("sentinel/models/microbe/niche.py", "sentinel/models/microbe/archv2.py",
              "scripts/train/microbe_arch_v2_cv.py")   # hashed into every shard; the merge needs one code state


def out_dir(scheme: str, family: str, tag: str = "") -> Path:
    return paths.ensure(paths.RUNS_ROOT / "microbe" / f"cv_archv2{tag}" / scheme / family)


def confound_file(tag: str = "") -> Path:
    return paths.RUNS_ROOT / "microbe" / f"cv_archv2{tag}" / "qmp_confound.json"


def shard_set(name: str, families=()) -> str:
    """The shard-set folder name. The comparators of ``SHARD_COMPARATORS`` run one family set per call (lead
    ruling on xgb_v4mr of 2026-11-10 §V-3: one family at a time, each family its own shard set and its own
    merge-and-score step, its own CPU figure), so their shard set carries the families it was fitted for. The
    inversion-based models keep ONE shard set that serves both families: there a fold is fitted once (ruling of
    log E14), and their folder name does not change."""
    return f"{name}__{''.join(families)}" if families and name in SHARD_COMPARATORS else name


def shard_dir(scheme: str, name: str, tag: str = "", families=()) -> Path:
    return paths.RUNS_ROOT / "microbe" / f"cv_archv2{tag}" / scheme / "_shards" / shard_set(name, families)


def shardable(name: str, AV) -> bool:
    return AV.backend(name) == "niche" or name in SHARD_COMPARATORS


def n_threads(name: str, AV) -> int:
    return NICHE_THREADS if shardable(name, AV) else MAX_THREADS


def code_sha256() -> dict:
    import hashlib

    return {f: hashlib.sha256((paths.REPO_ROOT / f).read_bytes()).hexdigest() for f in CODE_FILES}


def cpu_hours() -> float:
    t = os.times()
    return (t.user + t.system + t.children_user + t.children_system) / 3600.0


class _FamilyRun:
    """Prefixes the metrics that ``score_and_log`` writes to the TrainingRun with the target family."""

    def __init__(self, run, family):
        self.run, self.family = run, family

    def metric(self, name, value):
        self.run.metric(f"{self.family}:{name}", value)

    def note(self, text):
        self.run.note(text)


def gate_all(names, a, AV) -> dict:
    """Every gate of the call, before any data is loaded. Returns the stamp status."""
    st = None
    for scheme in a.schemes:
        for fam in a.families:
            for n in names:
                st = AV.gate(n, scheme, fam, confound_done=confound_file(a.tag).is_file(), slot8_row=a.slot8_row,
                             g10_row=a.g10_row)
    return st["stamp"]


def load(scheme: str, AV):
    """(development design with the 17 master dimensions, folds). The folds must equal the existing track's."""
    from sentinel.models.microbe import splits as SP
    from sentinel.models.microbe.design import load_design

    design = AV.attach_master(load_design())
    folds = SP.make_folds(design.samples, scheme)
    old = paths.RUNS_ROOT / "microbe" / "cv" / scheme / "folds.json"
    mine = [f.to_dict() for f in folds]
    if old.is_file() and json.loads(old.read_text()) != mine:
        raise RuntimeError(f"folds of {scheme} differ from the existing track ({old}); parity (amendment §6) broken")
    return design, folds, mine


def reference(scheme: str, family: str, tag: str) -> pd.DataFrame | None:
    """RF[DOY+location] out-of-fold predictions (MSE-skill reference): the existing track for P, this round for G."""
    p = (paths.RUNS_ROOT / "microbe" / "cv" / scheme if family == "P" else out_dir(scheme, family, tag)) / "context_rf.parquet"
    return pd.read_parquet(p) if p.is_file() else None


def log_failure(name, scheme, families, cfg, exc, AV) -> None:
    log_result(workstream="M3-microbe", experiment=AV.EXPERIMENT, model=AV.label(name), metric="run_failed", value=1.0,
               split=f"neon_dev/{scheme}_cv", is_baseline=AV.is_baseline(name), config=cfg,
               data_manifests=AV.MANIFESTS, notes=f"families {'+'.join(families)}: {type(exc).__name__}: {exc}"[:500])


def save_and_score(name, scheme, a, design, per_seed: dict, cfg, run, AV, MCV, extra_notes="") -> dict:
    """Write the family files and the registry rows of one model. ``per_seed``: {seed: frame of all targets}."""
    seeds = sorted(per_seed)
    res_all = {}
    for fam in a.families:
        targets = list(AV.FAMILIES[fam])
        out = out_dir(scheme, fam, a.tag)
        oof = sum(per_seed[s][targets] for s in seeds) / len(seeds)
        if "fold" in per_seed[seeds[0]]:
            oof["fold"] = per_seed[seeds[0]]["fold"]
        oof.to_parquet(out / f"{name}.parquet")
        if len(seeds) > 1:
            for s in seeds:
                per_seed[s][targets].to_parquet(out / f"{name}_seed{s}.parquet")
        res = MCV.score_and_log(design, oof[targets], name=name, label=AV.label(name), scheme=scheme, targets=targets,
                                is_baseline=AV.is_baseline(name), config={**cfg, "family": fam}, n_boot=a.n_boot,
                                per_seed={s: per_seed[s][targets] for s in seeds} if len(seeds) > 1 else None,
                                artifacts=[str(out / f"{name}.parquet")], run=_FamilyRun(run, fam),
                                ref=None if name == "context_rf" else reference(scheme, fam, a.tag),
                                extra_notes=" | ".join(x for x in (a.notes, extra_notes, f"family {fam}",
                                                                   f"role {AV.role(name)}") if x),
                                experiment=AV.EXPERIMENT, manifests=AV.MANIFESTS)
        (out / f"{name}_scores.json").write_text(json.dumps(res, indent=1, default=float))
        log.info("%s %s family %s: mean R2 %.4f %s", name, scheme, fam,
                 float(np.mean([r["r2"] for r in res.values()])) if res else float("nan"),
                 {t: round(r["r2"], 3) for t, r in res.items()})
        res_all[fam] = res
    return res_all


def refuse_existing(name, scheme, a, AV) -> None:
    for fam in a.families:
        f = out_dir(scheme, fam, a.tag) / f"{name}.parquet"
        if f.exists() and (name in COUNTED or not a.overwrite):
            raise AV.GateError(
                f"{f} exists. A counted model is never rerun once a development number exists (amendment §2.5, "
                "rulings 'Reruns'); a comparator needs --overwrite.")


def model_setup(name, scheme, a, stamp, AV) -> tuple[dict, list, list]:
    """(registry config, seeds, target columns) of one model run; the same for a shard and for the whole run."""
    targets = [t for fam in a.families for t in AV.FAMILIES[fam]]
    seeds = list(a.seeds) if AV.stochastic(name) else [0]
    cfg = {"model": name, "scheme": scheme, "families": list(a.families), "seeds": seeds, "role": AV.role(name),
           "backend": AV.backend(name), "stamp_row": stamp["row_id"],
           "amendment_sha256": stamp["amendment_sha256"], "n_jobs": n_threads(name, AV), "device": a.device}
    if AV.backend(name) == "niche":
        from sentinel.models.microbe import niche as NI

        v = "niche_s" if name == NI.CONTROL else name
        cfg["params"] = {k: (list(x) if isinstance(x, tuple) else x) for k, x in {**NI.DEFAULTS, **NI.VARIANTS[v]}.items()}
    return cfg, seeds, targets


def finish(name, scheme, a, design, per_seed, diags, used, cfg, run, AV, MCV, extra_notes="") -> None:
    """Diagnostics file, CPU time and the scored family files of one finished model (whole run or merged shards)."""
    run.metric("cpu_hours", used)
    notes = f"{used:.2f} CPU-h"
    if diags:
        for fam in a.families:
            (out_dir(scheme, fam, a.tag) / f"{name}_diag.json").write_text(json.dumps(diags, indent=1, default=float))
        if "test_first_step_fail_share" in diags[0]:
            w = np.array([d["n_test"] - d["n_test_without_reads"] for d in diags], float)
            fail = float(np.sum(w * np.array([d["test_first_step_fail_share"] for d in diags])) / max(w.sum(), 1))
            run.metric("first_step_fail_share", fail)
            notes += f"; first-step failure share of held-out inversions {fail:.4f}"
    save_and_score(name, scheme, a, design, per_seed, cfg, run, AV, MCV,
                   extra_notes="; ".join(x for x in (notes, extra_notes) if x))


def run_model(name, scheme, a, design, folds, stamp, AV, MCV) -> None:
    refuse_existing(name, scheme, a, AV)
    cfg, seeds, targets = model_setup(name, scheme, a, stamp, AV)
    params = {}
    budget = CPU_BUDGET_H.get(name)
    c0 = cpu_hours()

    def on_fold(i, f, d):
        used = cpu_hours() - c0
        extra = ""
        if d and "test_first_step_fail_share" in d:
            extra = f" n_eff {d['n_eff']} first-step fail {d['test_first_step_fail_share']:.3f} units {d['n_units']}"
        log.info("%s fold %d/%d %s done, %.2f CPU-h%s", name, i + 1, len(folds), f.name, used, extra)
        if budget is not None and used > budget:
            raise RuntimeError(f"{name}: CPU budget of {budget} h per track exceeded after fold {i + 1} "
                               f"({used:.2f} CPU-h); no development metric was computed (amendment kill rule)")

    with TrainingRun(MCV.WORKSTREAM, f"{AV.EXPERIMENT}:{name}:{scheme}:{'+'.join(a.families)}", config=cfg,
                     data=AV.MANIFESTS, plan_ref=PLAN_REF, iteration=a.iteration,
                     artifacts_dir=str(out_dir(scheme, a.families[0], a.tag).parent)) as run:
        try:
            per_seed, diags = AV.oof(name, design, folds, targets=targets, seeds=seeds, n_jobs=cfg["n_jobs"],
                                     device=a.device, params=params, on_fold=on_fold)
            finish(name, scheme, a, design, per_seed, diags, cpu_hours() - c0, cfg, run, AV, MCV)
        except Exception as exc:   # every run gets a registry row, failed runs included
            log_failure(name, scheme, a.families, cfg, exc, AV)
            raise


# ------------------------------------------------------------------------------------------------ fold shards

def _shard_cfg(cfg: dict) -> dict:
    """The part of a run's config that every shard and the merge must share (the stamp row id may move when a
    later lead stamp row repeats the amendment hash, so it is recorded but not compared)."""
    return json.loads(json.dumps({k: v for k, v in cfg.items() if k != "stamp_row"}, default=float))


def run_shard(names, scheme, a, design, folds, stamp, AV) -> None:
    """The listed outer folds of the listed inversion-based models, or of ONE model of ``SHARD_COMPARATORS``, one
    fold at a time through ``archv2.oof`` (the code path of the unsharded run). Writes predictions, label-free
    diagnostics and CPU time. No metric, no label comparison, no registry row except ``run_failed`` after an
    exception. Nothing here depends on the shard or on the order of the folds: every seed comes from ``--seeds``
    and every inner split is a function of the fold's training rows."""
    from threadpoolctl import threadpool_limits

    from sentinel.models.microbe import niche as NI

    sel = sorted(set(int(i) for i in a.shard_folds))
    if not sel or sel[0] < 0 or sel[-1] >= len(folds):
        raise AV.GateError(f"--shard-folds must be fold numbers 0..{len(folds) - 1}; got {a.shard_folds}")
    if a.overwrite:
        raise AV.GateError("a fold shard is fitted once; --overwrite does not apply to --shard-folds")
    bad = [n for n in names if not shardable(n, AV)]
    if bad:
        raise AV.GateError(f"fold shards are approved for the inversion-based models (lead ruling, log E14) and for "
                           f"{SHARD_COMPARATORS} (lead ruling B, log E20) only; not for {bad}")
    if len(set(names)) > 1 and any(n in SHARD_COMPARATORS for n in names):
        raise AV.GateError(f"one model at a time: {[n for n in names if n in SHARD_COMPARATORS]} is sharded in a "
                           "call of its own (lead ruling B, log E20)")
    names = sorted(dict.fromkeys(names), key=lambda n: n == NI.CONTROL)   # the control last: it reuses the fold
    for name in names:                                                   # of niche_s and pays no inversion
        refuse_existing(name, scheme, a, AV)
        done = [i for i in sel if any(shard_dir(scheme, name, a.tag, a.families).glob(f"fold{i:02d}*"))]
        if done:
            raise AV.GateError(f"{name}: shard files of folds {done} exist under "
                               f"{shard_dir(scheme, name, a.tag, a.families)}; a "
                               "fold is fitted once")
    code = code_sha256()
    with threadpool_limits(limits=NICHE_THREADS):
        for i in sel:
            for name in names:
                cfg, seeds, targets = model_setup(name, scheme, a, stamp, AV)
                d = paths.ensure(shard_dir(scheme, name, a.tag, a.families))
                c0, t0 = cpu_hours(), time.time()
                try:
                    per_seed, diags = AV.oof(name, design, [folds[i]], targets=targets, seeds=seeds,
                                             n_jobs=n_threads(name, AV), device=a.device, params={})
                except Exception as exc:
                    log_failure(name, scheme, a.families, {**cfg, "shard_fold": i}, exc, AV)
                    raise
                used = cpu_hours() - c0
                for sd in seeds:
                    frame = per_seed[sd]
                    frame["fold"] = i
                    tmp = d / f"fold{i:02d}_seed{sd}.parquet.tmp"
                    frame.to_parquet(tmp)
                    os.replace(tmp, d / f"fold{i:02d}_seed{sd}.parquet")
                meta = {"model": name, "fold": i, "fold_name": folds[i].name, "n_folds": len(folds), "seeds": seeds,
                        "targets": targets, "n_test": int(len(folds[i].test)), "cpu_hours": used,
                        "wall_s": time.time() - t0, "n_threads": n_threads(name, AV), "pid": os.getpid(),
                        "shard_folds": sel, "config": _shard_cfg(cfg), "stamp_row": stamp["row_id"],
                        "code_sha256": code, "diag": ({**diags[0], "fold": i} if diags else None)}
                tmp = d / f"fold{i:02d}.json.tmp"
                tmp.write_text(json.dumps(meta, indent=1, default=float))
                os.replace(tmp, d / f"fold{i:02d}.json")      # written last: this file marks the fold as complete
                log.info("%s shard fold %d/%d %s written, %.3f CPU-h", name, i + 1, len(folds), folds[i].name, used)


def merge_shards(name, scheme, a, design, folds, stamp, AV, MCV) -> None:
    """One merge-and-score step: every fold present, one code state, the CPU time summed over the shards within the
    model's budget — otherwise no prediction file is read and no metric is computed."""
    if a.overwrite:
        raise AV.GateError("fold shards are merged and scored once; --overwrite does not apply to --merge-shards")
    refuse_existing(name, scheme, a, AV)
    cfg, seeds, targets = model_setup(name, scheme, a, stamp, AV)
    d = shard_dir(scheme, name, a.tag, a.families)
    missing = [i for i in range(len(folds)) if not (d / f"fold{i:02d}.json").is_file()]
    if missing:
        raise AV.GateError(f"{name}: shard folds {missing} are missing under {d}; the merge needs all {len(folds)}")
    metas = [json.loads((d / f"fold{i:02d}.json").read_text()) for i in range(len(folds))]
    for i, m in enumerate(metas):
        want = {"model": name, "fold": i, "fold_name": folds[i].name, "n_folds": len(folds), "seeds": seeds,
                "targets": targets, "n_threads": n_threads(name, AV), "config": _shard_cfg(cfg),
                "code_sha256": metas[0]["code_sha256"]}
        diff = [k for k, v in want.items() if m.get(k) != v]
        if diff:
            raise AV.GateError(f"{name}: shard of fold {i} differs from this merge call or from the shard of fold 0 "
                               f"in {diff}; all folds need one configuration and one code state")
    used = float(sum(m["cpu_hours"] for m in metas))
    budget = CPU_BUDGET_H.get(name)
    cfg = {**cfg, "sharded": True, "n_shard_processes": len({m["pid"] for m in metas}),
           "shard_cpu_hours": [round(float(m["cpu_hours"]), 4) for m in metas], "code_sha256": metas[0]["code_sha256"]}
    with TrainingRun(MCV.WORKSTREAM, f"{AV.EXPERIMENT}:{name}:{scheme}:{'+'.join(a.families)}", config=cfg,
                     data=AV.MANIFESTS, plan_ref=PLAN_REF, iteration=a.iteration,
                     artifacts_dir=str(out_dir(scheme, a.families[0], a.tag).parent)) as run:
        try:
            log.info("%s: %d shard folds, %.2f CPU-h in total", name, len(metas), used)
            if budget is not None and used > budget:
                raise RuntimeError(f"{name}: CPU budget of {budget} h per track exceeded by the sum over the fold "
                                   f"shards ({used:.2f} CPU-h); no development metric was computed (amendment "
                                   "kill rule)")
            per_seed = {}
            for sd in seeds:
                parts = []
                for i, f in enumerate(folds):
                    part = pd.read_parquet(d / f"fold{i:02d}_seed{sd}.parquet")
                    if list(part.index) != list(np.asarray(f.test)) or not (part["fold"] == i).all():
                        raise RuntimeError(f"{name}: shard of fold {i} seed {sd} does not hold the fold's test rows")
                    parts.append(part)
                per_seed[sd] = pd.concat(parts)
            diags = [m["diag"] for m in metas if m.get("diag")]
            finish(name, scheme, a, design, per_seed, diags, used, cfg, run, AV, MCV,
                   extra_notes=f"merged from {len(metas)} fold shards at {n_threads(name, AV)} thread")
        except Exception as exc:   # every run gets a registry row, failed runs included
            log_failure(name, scheme, a.families, cfg, exc, AV)
            raise


def run_ensemble(members, scheme, a, design, stamp, AV, MCV) -> None:
    name = "ens_" + "+".join(members)
    label = "Ens[" + "+".join(AV.label(m) for m in members) + "]"
    cfg = {"model": name, "members": list(members), "scheme": scheme, "families": list(a.families),
           "rule": "unweighted mean of member OOF", "stamp_row": stamp["row_id"]}
    with TrainingRun(MCV.WORKSTREAM, f"{AV.EXPERIMENT}:{name}:{scheme}:{'+'.join(a.families)}", config=cfg,
                     data=AV.MANIFESTS, plan_ref="microbe_protocol §10.1; amendment §6, §7.5", iteration=a.iteration,
                     artifacts_dir=str(out_dir(scheme, a.families[0], a.tag).parent)) as run:
        for fam in a.families:
            targets = list(AV.FAMILIES[fam])
            out = out_dir(scheme, fam, a.tag)
            oof = AV.ensemble([pd.read_parquet(out / f"{m}.parquet") for m in members], targets)
            oof.to_parquet(out / f"{name}.parquet")
            res = MCV.score_and_log(design, oof, name=name, label=label, scheme=scheme, targets=targets,
                                    is_baseline=all(AV.is_baseline(m) for m in members),
                                    config={**cfg, "family": fam}, n_boot=a.n_boot, per_seed=None,
                                    artifacts=[str(out / f"{name}.parquet")], run=_FamilyRun(run, fam),
                                    ref=reference(scheme, fam, a.tag),
                                    extra_notes=f"late fusion of OOF predictions | family {fam}",
                                    experiment=AV.EXPERIMENT, manifests=AV.MANIFESTS)
            (out / f"{name}_scores.json").write_text(json.dumps(res, indent=1, default=float))
            log.info("%s %s family %s: %s", name, scheme, fam, {t: round(r["r2"], 3) for t, r in res.items()})


def run_tier0_best(scheme, a, design, stamp, AV, MCV) -> None:
    """Per target the best development-LODO scorer of the three tier-0 models (protocol §10 amendment), family G."""
    from sentinel.models.microbe import evaluate as E

    tier0 = ("global_mean", "context_ridge", "context_rf")
    S = design.samples
    for fam in a.families:
        if fam != "G":
            raise AV.GateError("tier0_best of family P exists on the existing track (cv/lodo/tier0_best.parquet)")
        targets, out = list(AV.FAMILIES[fam]), out_dir(scheme, fam, a.tag)
        oofs = {m: pd.read_parquet(out / f"{m}.parquet") for m in tier0}
        choice = {}
        for t in targets:
            sc = {m: float(E.summary(S.loc[o.index, t].to_numpy(float), o[t].to_numpy(float),
                                     S.loc[o.index, "domain_id"].astype(str).to_numpy())["r2"]) for m, o in oofs.items()}
            choice[t] = {"model": max(sc, key=sc.get), "lodo_r2": sc}
        (out / "tier0_best_choice.json").write_text(json.dumps(choice, indent=1))
        oof = pd.DataFrame({t: oofs[choice[t]["model"]][t] for t in targets})
        oof.to_parquet(out / "tier0_best.parquet")
        cfg = {"model": "tier0_best", "scheme": scheme, "family": fam, "choice": {t: c["model"] for t, c in choice.items()},
               "stamp_row": stamp["row_id"]}
        with TrainingRun(MCV.WORKSTREAM, f"{AV.EXPERIMENT}:tier0_best:{scheme}:{fam}", config=cfg, data=AV.MANIFESTS,
                         plan_ref="microbe_protocol §10; amendment §6", iteration=a.iteration,
                         artifacts_dir=str(out)) as run:
            res = MCV.score_and_log(design, oof, name="tier0_best", label="tier0_best", scheme=scheme, targets=targets,
                                    is_baseline=True, config=cfg, n_boot=a.n_boot, per_seed=None,
                                    artifacts=[str(out / "tier0_best.parquet")], run=_FamilyRun(run, fam), ref=None,
                                    extra_notes=f"per-target best tier-0 model on development LODO | family {fam}",
                                    experiment=AV.EXPERIMENT, manifests=AV.MANIFESTS)
            (out / "tier0_best_scores.json").write_text(json.dumps(res, indent=1, default=float))


def run_calibration(names, scheme, a, design, stamp, AV, MCV) -> None:
    """Lead ruling A1: within-domain slope (observed on predicted) and SD(predicted)/SD(observed) per target."""
    S = design.samples
    for name in names:
        cfg = {"model": name, "scheme": scheme, "families": list(a.families), "diagnostic": "calibration",
               "stamp_row": stamp["row_id"]}
        with TrainingRun(MCV.WORKSTREAM, f"{AV.EXPERIMENT}:calibration:{name}:{scheme}", config=cfg, data=AV.MANIFESTS,
                         plan_ref="lead rulings §2.2 A1", iteration=a.iteration,
                         artifacts_dir=str(out_dir(scheme, a.families[0], a.tag).parent)) as run:
            for fam in a.families:
                out = out_dir(scheme, fam, a.tag)
                f = out / f"{name}.parquet"
                if not f.is_file() and fam == "P":
                    f = paths.RUNS_ROOT / "microbe" / "cv" / scheme / f"{name}.parquet"
                oof = pd.read_parquet(f)
                dom = S.loc[oof.index, "domain_id"].astype(str).to_numpy()
                res = {}
                for t in AV.FAMILIES[fam]:
                    c = AV.calibration(S.loc[oof.index, t].to_numpy(float), oof[t].to_numpy(float), dom)
                    res[t] = c
                    common = dict(workstream=MCV.WORKSTREAM, experiment=AV.EXPERIMENT, model=AV.label(name),
                                  split=f"neon_dev/{scheme}_cv", target=t, is_baseline=AV.is_baseline(name),
                                  config={**cfg, "family": fam}, data_manifests=AV.MANIFESTS, artifacts=[str(f)],
                                  n=c["n"], notes=f"calibration diagnostic (lead ruling A1) | family {fam}")
                    for m in ("slope_within_domain", "sd_ratio", "sd_ratio_within_domain", "slope_pooled"):
                        if np.isfinite(c[m]):
                            log_result(metric=f"calib_{m}", value=c[m], **common)
                    run.metric(f"{fam}:slope_within_domain_{t}", c["slope_within_domain"])
                    run.metric(f"{fam}:sd_ratio_{t}", c["sd_ratio"])
                (out / f"{name}_calibration.json").write_text(json.dumps(res, indent=1, default=float))
                log.info("calibration %s family %s: %s", name, fam,
                         {t: (round(c["slope_within_domain"], 2), round(c["sd_ratio"], 2)) for t, c in res.items()})


def smoke_train_only(a, design, folds, AV) -> dict:
    """Kill rule §2.5: first-step failures of the inversion on TRAINING events. No held-out row is touched and no
    prediction is compared with a label."""
    from threadpoolctl import threadpool_limits

    from sentinel.models.microbe import features as F
    from sentinel.models.microbe import niche as NI

    S, f = design.samples, folds[0]
    p = {**NI.DEFAULTS}
    ut = NI.unit_table(design)
    tr = ut.rows(f.train)
    Y = S.loc[f.train, list(NI.MASTER)].to_numpy(float)
    ctx, _ = F.context_features(S, f.train)
    out = {"fold": f.name, "n_train": int(len(tr)), "by_n_eff": {}}
    c0, t0 = cpu_hours(), time.time()
    out["n_threads"] = NICHE_THREADS
    with threadpool_limits(limits=NICHE_THREADS):
        model = NI.fit_niche(ut.counts[tr], Y, ctx, S.loc[f.train, "site_id"].astype(str).to_numpy(),
                             ut.profile(int(p["kmer"])), p, (ut.genus, ut.family))
        out["fit_cpu_h"], out["fit_wall_s"] = cpu_hours() - c0, time.time() - t0
        for ne in p["n_eff_grid"]:
            c1, t1 = cpu_hours(), time.time()
            r = NI.infer(model, ut.counts[tr], ctx, float(ne), params=p)
            wr = r["has_reads"]
            out["by_n_eff"][str(ne)] = {
                "first_step_fail_share": float((wr & ~r["first_ok"]).sum() / max(wr.sum(), 1)),
                "n_with_reads": int(wr.sum()), "mean_newton_steps": float(r["n_steps"][wr].mean()),
                "exact_hessian_share": float(r["exact_hessian"][wr].mean()),
                "infer_cpu_h": cpu_hours() - c1, "infer_wall_s": time.time() - t1}
    out["n_units"] = int(len(model.vocab))
    out["max_first_step_fail_share"] = max(v["first_step_fail_share"] for v in out["by_n_eff"].values())
    out["kill_rule_ok"] = bool(out["max_first_step_fail_share"] <= 0.05)
    # one outer fold = inner_folds x (fit + grid inversions of 1/k of the rows) + final fit + test inversion
    per_fold = (p["inner_folds"] + 1) * out["fit_cpu_h"] + sum(v["infer_cpu_h"] for v in out["by_n_eff"].values())
    out["projected_cpu_h_per_track"] = float(per_fold * len(folds))
    d = paths.ensure(paths.RUNS_ROOT / "microbe" / f"cv_archv2{a.tag}")
    (d / "niche_s_smoke_train_only.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=[])
    ap.add_argument("--ensemble", nargs="+", default=None, metavar="MEMBER")
    ap.add_argument("--tier0-best", action="store_true")
    ap.add_argument("--calibration", nargs="+", default=None, metavar="MODEL")
    ap.add_argument("--smoke-train-only", action="store_true")
    ap.add_argument("--schemes", nargs="+", default=["lodo"])
    ap.add_argument("--families", nargs="+", default=["P", "G"], choices=["P", "G"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--iteration", default="arch_v2")
    ap.add_argument("--tag", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--overwrite", action="store_true", help="comparators only; counted models are never rerun")
    ap.add_argument("--slot8-row", default=None, help="registry row id of the lead's written D1-gate evaluation")
    ap.add_argument("--g10-row", default=None, help="registry row id of the logged G10 choice (niche_s_qmp only)")
    ap.add_argument("--shard-folds", nargs="+", type=int, default=None, metavar="I",
                    help="with --models: fit these outer folds (0-based) only; predictions and CPU time, no metric")
    ap.add_argument("--merge-shards", action="store_true",
                    help="with --models: merge the fold shards of each model (all folds present) and score once")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if sum(map(bool, (a.models, a.ensemble, a.tier0_best, a.calibration, a.smoke_train_only))) != 1:
        ap.error("exactly one of --models / --ensemble / --tier0-best / --calibration / --smoke-train-only")
    if (a.shard_folds is not None or a.merge_shards) and not a.models:
        ap.error("--shard-folds and --merge-shards go with --models")
    if a.shard_folds is not None and a.merge_shards:
        ap.error("--shard-folds or --merge-shards, not both")
    if (a.shard_folds is not None or a.merge_shards) and len(a.schemes) != 1:
        ap.error("fold shards work on one scheme per call")

    from sentinel.models.microbe import archv2 as AV
    from sentinel.models.microbe import stamp as ST

    # ---- gates first: nothing is loaded and nothing is written before they pass
    if a.smoke_train_only:
        stamp = ST.stamp_status()                      # allowed before the stamp: no development number
        names = []
    elif a.calibration:
        ST.require_release("calibration diagnostic", "CALIBRATION")
        names = list(a.calibration)
    elif a.tier0_best:
        names = ["global_mean", "context_ridge", "context_rf"]
    else:
        names = list(a.models or a.ensemble)
    if names:
        if a.models and any(AV.backend(n) == "base" for n in names) and "P" in a.families:
            raise AV.GateError("baseline tables of the existing track run on family G only here (their family-P "
                               "out-of-fold files are under cv/<scheme>/); pass --families G")
        stamp = gate_all(names, a, AV)

    import microbe_cv as MCV   # after numpy: its OPENBLAS_NUM_THREADS=1 no longer applies to the loaded library
    import torch

    torch.set_num_threads(2 if a.device != "cpu" else NICHE_THREADS if a.shard_folds is not None else MAX_THREADS)
    for scheme in a.schemes:
        design, folds, fold_dicts = load(scheme, AV)
        if a.smoke_train_only:
            out = smoke_train_only(a, design, folds, AV)
            print(json.dumps(out, indent=1, default=float))
            return 0 if out["kill_rule_ok"] else 3
        if a.shard_folds is not None:                  # predictions and CPU time only
            run_shard(list(a.models), scheme, a, design, folds, stamp, AV)
            return 0
        (out_dir(scheme, a.families[0], a.tag).parent / "folds.json").write_text(json.dumps(fold_dicts))
        if a.merge_shards:
            for m in a.models:
                merge_shards(m, scheme, a, design, folds, stamp, AV, MCV)
        elif a.ensemble:
            run_ensemble(a.ensemble, scheme, a, design, stamp, AV, MCV)
        elif a.tier0_best:
            run_tier0_best(scheme, a, design, stamp, AV, MCV)
        elif a.calibration:
            run_calibration(a.calibration, scheme, a, design, stamp, AV, MCV)
        else:
            for m in a.models:
                run_model(m, scheme, a, design, folds, stamp, AV, MCV)
    return 0


if __name__ == "__main__":
    sys.exit(main())
