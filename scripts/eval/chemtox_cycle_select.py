#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Aggregate improvement-cycle-2 searches, pick one configuration per family, build the ensembles.

Reads every ``shard*_results.json`` under ``$SENTINEL_RUNS/chemtox/cycle2/<family>_<challenge>_<split>_extra-<kind>/``
(written by ``scripts/train/chemtox_cycle.py``), selects the best configuration per family by
**mean RMSE over ADORE's official CV folds 0–4**, and scores the unweighted-mean and NNLS-stacked
ensembles of the per-family best out-of-fold predictions, always printing the components' own
scores next to the ensemble's. Optionally also pulls the phase-1 families (chemprop, TabPFN,
embeddings) from the cycle-1 search directories so the ensemble can mix both cycles.

With ``--freeze`` it writes the frozen-configuration block and its sha256 for stage 2 of
``docs/preregistration/chemtox_prospective_plan.md``. Nothing here reads the ADORE test rows or any
prospective row.

    python -B scripts/eval/chemtox_cycle_select.py --families xgboost rf lightgbm catboost \
        readacross gpboost chemeleon_lightgbm chemeleon_tabpfn --log-registry
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.core.registry import package_hash
from sentinel.data.sources import adore as A
from sentinel.models.chemtox import cv as CV
from sentinel.models.chemtox import cycle as CYC
from sentinel.models.chemtox import ensemble as EN
from sentinel.models.chemtox.data import ChemToxDataset, FeatureSpec

log = logging.getLogger("chemtox_cycle_select")
EXPERIMENT = "adore_chemtox_cycle2"
#: name suffix per extra-training-data variant, so a frozen model's name says what it trained on
EXTRA_SUFFIX = {"none": "", "ecotox": "_aug", "adore_taxa": "_multitaxa",
                "adore_taxa_all": "_multitaxa", "both": "_aug_multitaxa"}
#: Families that are **comparators**, never candidates for a "SOTA" claim: the published method's
#: two learners and the regulatory read-across. Every variant of them (re-tuned, augmented,
#: multi-taxa, mw_logp) stays a comparator — a claim must beat the best of them, so a strong
#: variant of a baseline must never be able to pass itself off as the candidate that beat it.
COMPARATOR_FAMILIES = ("xgboost", "rf", "readacross")
#: The published method itself (Gasser et al. 2024), as frozen for the ADORE pass. It is carried
#: into the prospective freeze unchanged under the names ``xgboost_published`` / ``rf_published``,
#: so the comparator plan §15.1 (i) asks for is the published configuration and not a re-tuned
#: tree. The re-tuned trees are separate entries (``xgboost`` / ``rf``) with the same budget as
#: every other cycle-2 family.
PHASE1_SELECTION = "select/{challenge}_{split}/selection.json"
PUBLISHED_FALLBACK = {
    "xgboost_published": {
        "features": {"fingerprint": "MACCS", "published": True, "chemical": "four"},
        "params": {"n_estimators": 800, "eta": 0.03, "max_depth": 6, "min_child_weight": 1,
                   "subsample": 0.8},
        "n_configurations": 32},
    "rf_published": {
        "features": {"fingerprint": "MACCS", "published": True, "chemical": "four"},
        "params": {"n_estimators": 300, "max_depth": 20, "max_features": 0.3},
        "n_configurations": 8},
}


def model_family_of(name: str) -> str:
    """``xgboost_published`` -> ``xgboost`` (the family that implements it)."""
    base = name
    for suf in ("_mwlogp", "_published", "_aug_multitaxa", "_multitaxa", "_aug"):
        if base.endswith(suf):
            base = base[: -len(suf)]
    return base


def explicit_config_cv(family: str, tag: str, challenge: str, split: str) -> float | None:
    """Masked-CV RMSE of a single explicit configuration run (``--config-json --tag <tag>``)."""
    d = (paths.RUNS_ROOT / "chemtox" / "cycle2" /
         f"{family}_{challenge}_{split}_extra-none_dm_cfg-{tag}")
    recs = [r for r in load_records(d) if "cv_mean" in r and len(r.get("fold_metrics", [])) >= 5]
    return min((r["cv_mean"]["rmse"] for r in recs), default=None)


def published_baselines(challenge: str, split: str) -> dict:
    """The phase-1 frozen published-baseline configurations, read from phase 1's selection.json."""
    import json as _json

    path = paths.RUNS_ROOT / "chemtox" / PHASE1_SELECTION.format(challenge=challenge, split=split)
    out = {k: dict(v) for k, v in PUBLISHED_FALLBACK.items()}
    if path.exists():
        frozen = _json.loads(path.read_text()).get("frozen", {})
        for new, old in (("xgboost_published", "xgboost"), ("rf_published", "rf")):
            if old in frozen and frozen[old].get("features"):
                out[new] = {"features": dict(frozen[old]["features"]),
                            "params": dict(frozen[old]["params"]),
                            "n_configurations": frozen[old].get("n_configurations",
                                                                out[new]["n_configurations"]),
                            "source": str(path)}
    return out


def cycle_dirs(family: str, challenge: str, split: str, *, require_masking: bool = True
               ) -> list[Path]:
    """Cycle-2 run directories of one family.

    ``require_masking`` keeps only the deployment-masked runs (directory suffix ``_dm``). The
    unmasked variant measures a feature the deployed model will not have (see
    ``cycle.DEPLOYMENT_MISSING_COLUMNS``), so it must never enter selection by accident.
    """
    base = paths.RUNS_ROOT / "chemtox" / "cycle2"
    dirs = [p for p in base.glob(f"{family}_{challenge}_{split}_extra-*") if p.is_dir()]
    if require_masking:
        dirs = [p for p in dirs if p.name.endswith("_dm")]
    return sorted(dirs)


def variant_extra(dir_name: str) -> str:
    """Training-data variant of a run directory: ``..._extra-ecotox_dm`` -> ``ecotox``.

    Handles the explicit-configuration directories too (``..._extra-none_dm_cfg-<tag>``).
    """
    tail = dir_name.split("extra-")[-1] if "extra-" in dir_name else "none"
    return re.split(r"_(?:dm|nodm)(?:_cfg-|$)", tail, maxsplit=1)[0]


def is_masked_dir(dir_name: str) -> bool:
    """True for a deployment-masked run directory (search ``*_dm`` or explicit ``*_dm_cfg-*``)."""
    return re.search(r"_dm(?:_cfg-|$)", dir_name) is not None


def explicit_dirs(family: str, tags: list[str], challenge: str, split: str) -> list[Path]:
    """Masked single-configuration run directories (``chemtox_cycle.py --config-json --tag``)."""
    base = paths.RUNS_ROOT / "chemtox" / "cycle2"
    out = []
    for t in tags:
        d = base / f"{family}_{challenge}_{split}_extra-none_dm_cfg-{t}"
        if not d.is_dir():
            raise FileNotFoundError(f"explicit configuration directory missing: {d}")
        out.append(d)
    return out


def phase1_dir(family: str, challenge: str, split: str) -> Path:
    return paths.RUNS_ROOT / "chemtox" / "search" / f"{family}_{challenge}_{split}"


def load_records(d: Path) -> list[dict]:
    recs: list[dict] = []
    for f in sorted(d.glob("shard*_results.json")):
        recs.extend(json.loads(f.read_text()).get("records", []))
    return recs


def full_records(recs: list[dict], *, min_folds: int = 5) -> list[dict]:
    """Configurations scored on all five official folds, unscreened (selection is only from these)."""
    out, seen = [], set()
    for r in recs:
        if "cv_mean" not in r or r.get("screen"):
            continue
        if len(r.get("fold_metrics", [])) < min_folds or r["index"] in seen:
            continue
        seen.add(r["index"])
        out.append(r)
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument("--phase1-families", nargs="*", default=[],
                    help="cycle-1 families to pull from $SENTINEL_RUNS/chemtox/search")
    ap.add_argument("--explicit", nargs="*", default=[], metavar="FAMILY=TAG[,TAG]",
                    help="a phase-1 frozen configuration re-run under masked CV as explicit "
                         "single configurations (chemtox_cycle.py --config-json --tag TAG); the "
                         "best of the listed tags is the family's frozen configuration, e.g. "
                         "chemprop=v1frozen,v1_chemprop_mwlogp")
    ap.add_argument("--challenge", default="t-F2F")
    ap.add_argument("--split", default=A.PRIMARY_SPLIT)
    ap.add_argument("--target", default="molar")
    ap.add_argument("--budget", type=int, default=CYC.DEFAULT_BUDGET)
    ap.add_argument("--allow-unmasked", action="store_true",
                    help="also read runs without deployment masking (not for selection)")
    ap.add_argument("--ensemble-members", nargs="*", default=None,
                    help="families to put in the ensembles (default: all with out-of-fold files)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--log-registry", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--out", default=None)
    return ap.parse_args(argv)


def oof_source(summary: dict, name: str) -> Path | None:
    """Where an entry's out-of-fold predictions came from (recorded in the frozen block).

    ``name`` is the summary key (the frozen name, e.g. ``lightgbm_aug``); the file is named after
    the implementing family (``oof_lightgbm_015.npy``), exactly as the CV runner writes it.
    """
    s = summary.get(name) or {}
    family = s.get("family", name)
    variant, idx = s.get("best_variant"), s.get("best_index")
    if idx is None:
        return None
    if variant:
        return paths.RUNS_ROOT / "chemtox" / "cycle2" / variant / \
            f"oof_{family}_{int(idx):03d}.npy"
    return paths.RUNS_ROOT / "chemtox" / "search" / f"{family}_t-F2F_split_occurrence" / \
        f"oof_{family}_{int(idx):03d}.npy"


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.out) if args.out else paths.run_dir(
        "chemtox", f"cycle2_select/{args.challenge}_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = ChemToxDataset.load(args.challenge, args.split, target=args.target,
                             specs=FeatureSpec(fingerprint="MACCS"))
    ds.assert_no_chemical_leakage(log=log)
    y, chem, taxon = ds.y, ds.chem, ds.taxon
    folds = ds.assignment.where(ds.assignment.str.fullmatch(r"\d+").fillna(False))

    summary: dict[str, dict] = {}
    oof: dict[str, np.ndarray] = {}

    # Tuning parity (lead decision 2026-11-08): every training-data variant of a family
    # (``extra-none``, ``extra-ecotox``, ``extra-adore_taxa_all``, ...) is its own 32-configuration
    # search and becomes its own entry under its pre-registered name (``lightgbm``,
    # ``lightgbm_aug``, ``gpboost_multitaxa``, ...). Taking the best over variants would give a
    # family 64+ configurations against the comparators' 32.
    entries: list[tuple[str, str, list[Path], int, str]] = []   # (name, family, dirs, budget, kind)
    for family in args.families:
        dirs = cycle_dirs(family, args.challenge, args.split,
                          require_masking=not args.allow_unmasked)
        if not dirs:
            log.warning("%s: no cycle-2 run directory", family)
            continue
        for d in dirs:
            ex = variant_extra(d.name)
            entries.append((f"{family}{EXTRA_SUFFIX.get(ex, '_' + ex)}", family, [d],
                            args.budget, "search"))
    for spec_ in args.explicit:
        family, _, tags = spec_.partition("=")
        tag_list = [t for t in tags.split(",") if t]
        if not tag_list:
            raise SystemExit(f"--explicit {spec_!r}: expected FAMILY=TAG[,TAG]")
        # a phase-1 winner (selected from 32 configurations in phase 1) re-run under masked CV;
        # the masked runs choose only between the listed explicit variants of it (plan §4.3)
        entries.append((family, family, explicit_dirs(family, tag_list, args.challenge, args.split),
                        args.budget, "explicit"))
    names = [e[0] for e in entries]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise RuntimeError(f"two run directories map to the same frozen name: {dup}")

    for key, family, dirs, budget, kind in entries:
        best, best_dir, n_done, n_failed, n_screened = None, None, 0, 0, 0
        per_variant = {}
        for d in dirs:
            recs = load_records(d)
            ok = full_records(recs)
            n_done += len(ok)
            n_failed += len([r for r in recs if "error" in r])
            n_screened += len({r["index"] for r in recs if r.get("screen") and "cv_mean" in r})
            if not ok:
                continue
            b = min(ok, key=lambda r: r["cv_mean"]["rmse"])
            per_variant[d.name] = {"n_completed": len(ok), "best_index": b["index"],
                                   "best_cv_rmse": b["cv_mean"]["rmse"],
                                   "n_screened": len({r["index"] for r in recs
                                                      if r.get("screen") and "cv_mean" in r})}
            if best is None or b["cv_mean"]["rmse"] < best["cv_mean"]["rmse"]:
                best, best_dir = b, d
        if best is None:
            log.warning("%s: no completed configuration", key)
            continue
        p = best_dir / f"oof_{family}_{best['index']:03d}.npy"
        if p.exists():
            oof[key] = np.load(p)
        else:
            log.warning("%s: no out-of-fold file at %s — it cannot enter an ensemble", key, p)
        summary[key] = {
            "family": family, "extra": variant_extra(best_dir.name), "selection_kind": kind,
            "masked": is_masked_dir(best_dir.name),
            "variants": per_variant, "best_variant": best_dir.name,
            "best_index": best["index"], "best_features": best["features"],
            "best_params": {k: v for k, v in best["params"].items()},
            "cv_mean": best["cv_mean"], "cv_sd": best["cv_sd"], "pooled": best["pooled"],
            "n_configurations": budget, "n_completed": n_done, "n_failed": n_failed,
            "n_screened": n_screened, "n_variants": len(per_variant),
            "n_full_cv_runs_across_variants": n_done,
            "all_cv_rmse": sorted(r["cv_mean"]["rmse"] for d in dirs
                                  for r in full_records(load_records(d))),
            "cycle": 2,
        }
        # §4.3 of the prospective pre-registration: what each chemical block scores on masked CV
        by_block: dict[str, list[float]] = {}
        by_option: dict[str, list[float]] = {}
        for d in dirs:
            for r in full_records(load_records(d)):
                f = r["features"]
                by_block.setdefault(str(f.get("chemical", "four")), []).append(
                    r["cv_mean"]["rmse"])
                for opt in ("rdkit_descriptors", "rdkit_descriptors_full", "chemeleon"):
                    if opt in f:
                        by_option.setdefault(f"{opt}={bool(f[opt])}", []).append(
                            r["cv_mean"]["rmse"])
        summary[key]["cv_rmse_by_chemical_block"] = {
            k: {"n": len(v), "mean": float(np.mean(v)), "best": float(np.min(v))}
            for k, v in sorted(by_block.items())}
        summary[key]["cv_rmse_by_feature_option"] = {
            k: {"n": len(v), "mean": float(np.mean(v)), "best": float(np.min(v))}
            for k, v in sorted(by_option.items())}
        log.info("%-20s best CV RMSE %.4f (%s, config %d; %d completed, %d failed)", key,
                 best["cv_mean"]["rmse"], best_dir.name, best["index"], n_done, n_failed)
        for k, v in summary[key]["cv_rmse_by_chemical_block"].items():
            log.info("    chemical=%-8s n=%-3d mean %.4f best %.4f", k, v["n"], v["mean"],
                     v["best"])

    for family in args.phase1_families:
        d = phase1_dir(family, args.challenge, args.split)
        ok = full_records(load_records(d))
        if not ok:
            log.warning("%s: no phase-1 records in %s", family, d)
            continue
        best = min(ok, key=lambda r: r["cv_mean"]["rmse"])
        p = d / f"oof_{family}_{best['index']:03d}.npy"
        if p.exists():
            oof[family] = np.load(p)
        # phase-1 parameters may carry data-frame placeholders (multitask / ECOTOX pretraining);
        # they are kept here so the freeze can refuse them explicitly rather than drop them
        summary[family] = {"family": family, "extra": "none", "masked": False,
                           "best_index": best["index"], "best_features": best["features"],
                           "best_params": dict(best["params"]),
                           "cv_mean": best["cv_mean"], "cv_sd": best["cv_sd"],
                           "pooled": best["pooled"], "n_configurations": 32,
                           "n_completed": len(ok), "cycle": 1}
        log.info("%-20s best CV RMSE %.4f (cycle 1, config %d)", family,
                 best["cv_mean"]["rmse"], best["index"])

    # N4: an ensemble member's out-of-fold predictions must come from a **masked** run, otherwise
    # the stack weights would be fitted on predictions made with a feature the deployed model does
    # not have. Phase-1 (unmasked) out-of-fold files are refused unless --allow-unmasked says so.
    unmasked_members = [k for k in oof
                        if not is_masked_dir(str(summary.get(k, {}).get("best_variant", "")))]
    if unmasked_members and not args.allow_unmasked:
        raise RuntimeError(
            f"these families have no deployment-masked out-of-fold predictions: {unmasked_members}. "
            f"Re-run them through scripts/train/chemtox_cycle.py (masking is the default; a "
            f"phase-1 frozen configuration can be reproduced with --config-json) or drop them from "
            f"--families. Mixing masked and unmasked out-of-fold predictions in the stack is the "
            f"defect round 2 of the review flagged as N4.")
    if args.ensemble_members:
        missing = [m for m in args.ensemble_members if m not in oof]
        if missing:
            raise RuntimeError(
                f"requested ensemble members without out-of-fold predictions: {missing}. The "
                f"ensemble must be built from exactly the models it is selected with; re-run those "
                f"families or correct --ensemble-members.")
    members = {k: v for k, v in oof.items()
               if args.ensemble_members is None or k in args.ensemble_members}
    ens: dict[str, dict] = {}
    if len(members) >= 2:
        avg = EN.average_ensemble(members, y, chem=chem, taxon=taxon)
        stk = EN.stacked_ensemble(members, y, chem=chem, taxon=taxon, folds=folds.to_numpy())
        for r in (avg, stk):
            ens[r.name] = r.to_dict()
            log.info("%s: CV RMSE %.4f over %d rows; components %s", r.name, r.metrics["rmse"],
                     int(r.mask.sum()), {k: round(v["rmse"], 4)
                                         for k, v in r.component_metrics.items()})
        np.save(out_dir / "oof_ensemble_mean.npy", avg.pred)
        np.save(out_dir / "oof_ensemble_stack.npy", stk.pred)

    # paired out-of-fold comparisons against the reproduced published baseline (CV evidence for the
    # freeze decision; it is NOT a test claim and is labelled as such everywhere)
    paired: dict[str, dict] = {}
    if "xgboost" in oof:
        from sentinel.eval.bootstrap import paired_bootstrap

        def rmse(yy, pp):
            return float(np.sqrt(np.mean((np.asarray(pp, float) - np.asarray(yy, float)) ** 2)))

        ref = np.asarray(oof["xgboost"], dtype=float)
        candidates: dict[str, np.ndarray] = {k: np.asarray(v, dtype=float)
                                             for k, v in oof.items() if k != "xgboost"}
        for name in ens:
            f = out_dir / f"oof_ensemble_{'mean' if 'mean' in name else 'stack'}.npy"
            if f.exists():
                candidates[name] = np.load(f)
        for name, pred in candidates.items():
            m = np.isfinite(y) & np.isfinite(pred) & np.isfinite(ref)
            ci = paired_bootstrap(rmse, y[m], pred[m], ref[m],
                                  groups=np.asarray(chem, dtype=object).astype(str)[m],
                                  n_boot=1000, seed=0, higher_is_better=False)
            ci["ci_method"] = "paired cluster bootstrap by chemical, 1000 resamples (CV out-of-fold)"
            ci["n_rows"] = int(m.sum())
            paired[name] = ci
            log.info("paired vs xgboost (CV out-of-fold): %-22s dRMSE %+.4f [%+.4f, %+.4f] p=%.3f",
                     name, ci["diff"], ci["ci_low"], ci["ci_high"], ci.get("p_value", float("nan")))

    payload = {"challenge": args.challenge, "split": args.split, "target": args.target,
               "cycle": 2, "families": summary, "ensembles": ens,
               "paired_vs_xgboost_cv": paired,
               "selection": "mean RMSE over ADORE official CV folds 0-4, ADORE rows only"}
    if args.freeze:
        frozen: dict[str, dict] = {}
        for key, s in summary.items():
            family = s.get("family", key)
            dropped = [k for k, v in s["best_params"].items() if str(v).startswith("<DataFrame")]
            if dropped:
                raise RuntimeError(
                    f"{key}: the winning configuration carries data frames in its parameters "
                    f"({dropped}). Freezing it would silently score a different model (the frames "
                    f"are rebuilt from the challenge files and the exclusion rule). Re-run that "
                    f"family with the rows supplied through --extra, which applies the prospective "
                    f"guard, or exclude it from the freeze.")
            # the summary key already is the pre-registered name (family + training-data suffix)
            extra = s.get("extra") or variant_extra(str(s.get("best_variant", "")))
            frozen[key] = {"features": dict(s["best_features"]),
                           "params": {k: v for k, v in s["best_params"].items()
                                      if not str(k).startswith("_")},
                           "extra": extra, "family": family,
                           "extra_weight": s["best_params"].get("_extra_weight", 1.0),
                           "role": ("comparator" if family in COMPARATOR_FAMILIES
                                    else "candidate"),
                           "seeds": list(args.seeds), "cv_rmse": s["cv_mean"]["rmse"],
                           "n_configurations": s["n_configurations"], "cycle": s["cycle"],
                           "selection_kind": s.get("selection_kind", "phase1"),
                           "best_variant": s.get("best_variant"),
                           "variants_tried": list(s.get("variants", {})),
                           "n_full_cv_runs": s.get("n_completed")}
        for name, r in ens.items():
            comps = list(r["components"])
            missing = [c for c in comps if c not in frozen]
            if missing:
                raise RuntimeError(
                    f"{name}: components {missing} have no frozen entry — the ensemble would be "
                    f"scored over fewer models than it was selected with")
            frozen[name] = {"ensemble": True, "components": comps,
                            "component_families": [frozen[c]["family"] for c in comps],
                            "method": "unweighted mean" if "mean" in name
                            else "non-negative least squares on training-portion out-of-fold "
                                 "predictions",
                            "oof_sources": {c: str(oof_source(summary, c)) for c in comps},
                            "cv_rmse": r["metrics"]["rmse"], "seeds": list(args.seeds),
                            "role": "candidate"}
        pub_tag = {"xgboost_published": "pub_xgb", "rf_published": "pub_rf"}
        for name, cfg in published_baselines(args.challenge, args.split).items():
            fam = model_family_of(name)
            frozen[name] = {**cfg, "seeds": list(args.seeds), "extra": "none", "cycle": 1,
                            "family": fam, "role": "comparator",
                            "cv_rmse": explicit_config_cv(fam, pub_tag[name], args.challenge,
                                                          args.split),
                            "note": "reproduced published baseline (Gasser et al. 2024)"}
            # the same baseline with only the chemical properties a new chemical can have. The
            # published spec uses experimental water solubility and melting point, which are
            # missing for every prospective row; scoring both variants and comparing against the
            # better one keeps the baseline at full strength (RESEARCH_PLAN §15.5 guard 4).
            variant = {**cfg, "features": {**cfg["features"], "chemical": "mw_logp"},
                       "seeds": list(args.seeds), "extra": "none", "cycle": 1,
                       "family": model_family_of(name), "role": "comparator",
                       "note": ("reproduced published baseline, chemical block reduced to the "
                                "RDKit-computable properties (MW, logP)")}
            variant["cv_rmse"] = explicit_config_cv(model_family_of(name),
                                                    pub_tag[name] + "_mwlogp", args.challenge,
                                                    args.split)
            frozen[f"{name}_mwlogp"] = variant
        # the §15.1 (i) reference: the published variant with the best **masked** CV RMSE, fixed
        # here (inside the hashed block) rather than resolved at scoring time
        pub_cv = {k: v.get("cv_rmse") for k, v in frozen.items()
                  if "_published" in k and isinstance(v.get("cv_rmse"), (int, float))}
        if not pub_cv:
            raise RuntimeError(
                "no masked-CV score for any reproduced published baseline: run the four published "
                "configurations through chemtox_cycle.py --config-json first (plan §4.3), "
                "otherwise the §15.1 (i) comparator cannot be fixed before the test")
        frozen["__published_reference__"] = min(pub_cv, key=pub_cv.get)
        frozen["__published_reference_cv__"] = pub_cv
        frozen["__allow_unmasked__"] = bool(args.allow_unmasked)
        log.info("published reference (best masked CV): %s %s", frozen["__published_reference__"],
                 pub_cv)
        payload["frozen"] = frozen
        payload["frozen_hash"] = hashlib.sha256(
            json.dumps(frozen, sort_keys=True, default=str).encode()).hexdigest()[:16]
        payload["code_hashes"] = {"sentinel_package": package_hash()}
        for f in ("scripts/eval/chemtox_score_prospective.py", "scripts/train/chemtox_cycle.py",
                  "scripts/eval/chemtox_cycle_select.py",
                  "sentinel/models/chemtox/prospective.py"):
            fp = paths.REPO_ROOT / f
            payload["code_hashes"][Path(f).name] = (
                hashlib.sha256(fp.read_bytes()).hexdigest()[:16] if fp.exists() else "missing")
        log.info("frozen-configuration hash: %s", payload["frozen_hash"])
        for k, v in payload["code_hashes"].items():
            log.info("  %-34s %s", k, v)
    CV.write_json(out_dir / "selection.json", payload)
    log.info("wrote %s", out_dir / "selection.json")

    if args.log_registry:
        for key, s in summary.items():
            CV.log_metrics(ds, experiment=EXPERIMENT, model=key, kind="cv",
                           metrics=s["cv_mean"],
                           config={"challenge": args.challenge, "split": args.split,
                                   "model": key, "family": s.get("family", key),
                                   "extra": s.get("extra"), "features": s["best_features"],
                                   "params": s["best_params"], "cycle": s["cycle"],
                                   "selection": "best of the equal-budget random search "
                                                "(mean RMSE over official CV folds 0-4)"},
                           n=int(s["pooled"]["n"]),
                           is_baseline=s.get("family", key) in COMPARATOR_FAMILIES,
                           notes=("improvement cycle 2 search winner; CV only, no test row of any "
                                  "kind was read"),
                           extra={"n_configurations": s["n_configurations"],
                                  "n_completed": s["n_completed"], "cv_sd": s["cv_sd"],
                                  "pooled_out_of_fold": s["pooled"],
                                  "variants": s.get("variants"),
                                  "best_variant": s.get("best_variant")},
                           artifacts=[str(out_dir / "selection.json")])
        for name, r in ens.items():
            CV.log_metrics(ds, experiment=EXPERIMENT, model=f"cycle2_{name}", kind="cv",
                           metrics=r["metrics"],
                           config={"challenge": args.challenge, "split": args.split,
                                   "components": r["components"],
                                   "method": "unweighted mean" if "mean" in name
                                   else "leave-one-fold-out non-negative least squares"},
                           n=r["n_scored"], is_baseline=False,
                           notes=("ENSEMBLE of " + ", ".join(r["components"]) +
                                  "; component CV scores in extra.component_metrics"),
                           extra={"component_metrics": r["component_metrics"],
                                  "weights": r["weights"], "n_configurations": "see components"},
                           artifacts=[str(out_dir / "selection.json")])
        log.info("registry updated")

    table = pd.DataFrame({k: v.get("cv_mean", {}) for k, v in summary.items()}).T
    with pd.option_context("display.width", 220, "display.float_format", lambda v: f"{v:.4f}"):
        print("\n=== cycle 2 — CV (ADORE official folds 0-4) ===")
        print(table.sort_values("rmse") if "rmse" in table.columns else table)
        if ens:
            print("\n=== ensembles (CV) ===")
            print(pd.DataFrame({k: v["metrics"] for k, v in ens.items()}).T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
