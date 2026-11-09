#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Score the frozen ChemTox models **once** on the prospective test set. Sealed.

This is the only script that reads ``$SENTINEL_DATA/processed/chemtox_prospective/prospective.parquet``
target values. It is sealed three ways, like the ADORE pass before it:

1. ``--i-have-approval`` is required; without it the script exits before the file is opened.
2. ``--expect-data-hash`` must equal the sha256 of the prospective file (the value frozen in
   ``docs/preregistration/chemtox_prospective_plan.md`` §1.4); any difference exits non-zero.
3. ``--expect-config-hash`` must equal the hash of the frozen configuration block in
   ``selection.json`` (stage 2 of the same document), so a configuration edited after freezing
   cannot be scored silently.

A model that already has a prospective entry in the registry is refused unless ``--allow-rescore``
is passed, which records ``supersedes``.

Training data (frozen in the pre-registration §2): all ADORE ``t-F2F`` rows — both partitions of
``split_occurrence`` — because every prospective chemical is absent from ADORE by CAS *and*
InChIKey; plus, for configurations whose frozen ``extra`` says so, ECOTOX rows that ADORE does not
contain. Before any fit, ``prospective.assert_prospective_chemicals_absent`` is run over the
training frame and raises if a single prospective chemical is present.

    python -B scripts/eval/chemtox_score_prospective.py --i-have-approval \
        --expect-data-hash <sha256> --expect-config-hash <16hex> --accelerator gpu
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.core.registry import file_sha256, log_result, package_hash, registry_frame
from sentinel.core.runlog import TrainingRun
from sentinel.data.sources import adore as A
from sentinel.eval.bootstrap import holm_bonferroni, paired_bootstrap
from sentinel.models.chemtox import cv as CV
from sentinel.models.chemtox import cycle as CYC
from sentinel.models.chemtox import evaluate as E
from sentinel.models.chemtox import prospective as PRO
from sentinel.models.chemtox.data import ChemToxDataset, FeatureSpec

log = logging.getLogger("chemtox_prospective_test")
EXPERIMENT = "chemtox_prospective_test"
WORKSTREAM = "chemtox"
TARGET_NAME = "log10_LC50_EC50_mol_per_L"
#: the primary endpoint: fish, novel chemicals, inside ADORE's chemical domain
PRIMARY_TRACK = "fish_novel"
#: the co-reported strict endpoint: the same rows restricted to parent-structure novelty. The
#: pre-registered decision rule (§4.4 d) requires an improvement to keep its sign here.
STRICT_TRACK = "fish_novel_strict"
TRACKS = ("fish_novel", "fish_novel_strict", "crust_novel", "crust_novel_strict",
          "fish_novel_all", "crust_novel_all", "temporal", "noncarbon", "algae_novel")
EXTRA_COLUMNS = ("chem_rdkit_can_smiles", "chem_inchikey", "chem_Morgan_fp", "chem_MACCS_fp",
                 "tax_genus", "tax_family", "tax_order", "tax_class", "reference_number")
#: Fallback list of comparator names, used only when the frozen block carries no ``role`` field.
#: The authoritative source is ``role`` in the frozen configuration (written by
#: ``chemtox_cycle_select.py --freeze``): every variant of the published method's two learners and
#: of the read-across comparator is a comparator, never a candidate.
BASELINE_FAMILIES = ("xgboost", "rf", "readacross")
#: trivial baselines computed here rather than frozen
TRIVIAL = ("global_mean", "species_mean", "species_duration_mean")
#: the comparators the Holm-corrected decision rule is applied to (plan §15.1 i: the strongest
#: published method and the strongest simple baseline)
HOLM_REFERENCES = ("published_reference", "best_baseline")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--i-have-approval", action="store_true",
                    help="required; the prospective file is not opened without it")
    ap.add_argument("--expect-data-hash", required=True)
    ap.add_argument("--expect-config-hash", required=True)
    ap.add_argument("--selection", default=None,
                    help="selection.json with the frozen block (default: cycle2_select run dir)")
    ap.add_argument("--models", nargs="*", default=None, help="subset of the frozen models")
    ap.add_argument("--tracks", nargs="*", default=list(TRACKS))
    ap.add_argument("--challenge", default="t-F2F")
    ap.add_argument("--split", default=A.PRIMARY_SPLIT)
    ap.add_argument("--target", default="molar")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--accelerator", default="cpu", choices=("cpu", "gpu"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--train-partition-only", action="store_true",
                    help="sensitivity entry: train on the official training partition only")
    ap.add_argument("--allow-rescore", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="build everything and run the guards, but do not read any target value")
    ap.add_argument("--preflight-rows", type=int, default=3000,
                    help="dry run only: fit each model on this many training rows and predict "
                         "every prospective row, to prove the one-shot pass cannot crash "
                         "(training targets only; no prospective target is read)")
    return ap.parse_args(argv)


def set_threads(n: int) -> int:
    n = max(1, min(int(n), 8))
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(n)
    import torch

    torch.set_num_threads(n)
    from sentinel.models.chemtox import gbdt as G
    from sentinel.models.chemtox import gpboost_model as GP
    from sentinel.models.chemtox import xgb as XG

    XG.XGB_DEFAULTS["n_jobs"] = n
    G.LGBM_DEFAULTS["n_jobs"] = n
    G.CATBOOST_DEFAULTS["thread_count"] = n
    GP.GPBOOST_DEFAULTS["num_threads"] = n
    return n


def frozen_hash(frozen: dict) -> str:
    return hashlib.sha256(json.dumps(frozen, sort_keys=True, default=str).encode()).hexdigest()[:16]


def existing_entries(models: list[str], tracks: list[str]) -> list[tuple[str, str]]:
    """(model, track) pairs that already have a prospective entry in the registry.

    Uses :func:`sentinel.core.registry.registry_frame` (``lookup`` resolves exactly one entry and
    raises on an empty registry, which is not what a guard needs). Superseded entries are dropped
    by ``registry_frame``, so a re-run recorded with ``supersedes`` does not block the next one.
    """
    df = registry_frame()
    if df is None or len(df) == 0 or "experiment" not in getattr(df, "columns", []):
        return []
    sub = df[(df["workstream"] == WORKSTREAM) & (df["experiment"] == EXPERIMENT)]
    if sub.empty:
        return []
    have = set(zip(sub["model"].astype(str), sub["split"].astype(str)))
    return [(m, tr) for m in models for tr in tracks
            if (m, f"prospective/{tr}/test") in have]


def supersede_id(model: str, track: str) -> str | None:
    """The id of the entry a re-score supersedes (most recent for this model and track)."""
    df = registry_frame()
    if df is None or len(df) == 0:
        return None
    sub = df[(df["workstream"] == WORKSTREAM) & (df["experiment"] == EXPERIMENT)
             & (df["model"] == model) & (df["split"] == f"prospective/{track}/test")]
    return None if sub.empty else str(sub.iloc[-1]["id"])


def track_role(track: str) -> str:
    """How a track is labelled wherever it appears (pre-registration §1.3, §1.3a, §4.4)."""
    if track == PRIMARY_TRACK:
        return "PRIMARY"
    if track == STRICT_TRACK:
        return "CO-PRIMARY (strict parent-novel)"
    return "secondary"


def track_mask(pro: pd.DataFrame, track: str) -> np.ndarray:
    """Rows of one pre-registered track.

    ``fish_novel`` and ``crust_novel`` are the ADORE-chemical-domain subsets (the primary endpoint
    is ``fish_novel``); ``*_strict`` additionally keeps only the rows whose chemical is novel at the
    **parent-structure** level (pre-registration §1.3a — the co-reported endpoint, 1,362 rows / 236
    chemicals for fish); the ``*_all`` names are the same tracks without the domain restriction,
    i.e. including organometallics and carbon-bearing inorganics (§1.3).
    """
    base = track
    strict = base.endswith("_strict")
    if strict:
        base = base[: -len("_strict")]
    allcarbon = base.endswith("_all") or track.endswith("_all")
    if base.endswith("_all"):
        base = base[: -len("_all")]
    m = (pro["prospective_track"] == base).to_numpy()
    if not allcarbon and base in ("fish_novel", "crust_novel"):
        m &= pro["in_adore_domain"].fillna(False).to_numpy().astype(bool)
    if strict:
        m &= pro["parent_novel"].fillna(False).to_numpy().astype(bool)
    return m


def build_training_frame(ds: ChemToxDataset, extra_kind: str, *, train_only: bool,
                         extra_weight: float = 1.0
                         ) -> tuple[pd.DataFrame, np.ndarray, dict, np.ndarray | None]:
    """ADORE rows (+ optional extra rows), with the prospective-chemical guard applied.

    Returns (rows, y, info, sample_weight); ``sample_weight`` is ``None`` unless the frozen
    configuration gave the extra rows a weight other than 1, in which case it is 1 for the ADORE
    rows and ``extra_weight`` for the extra ones — the same weighting the CV run used.
    """
    mask = ds.train_mask() if train_only else (ds.train_mask() | ds.test_mask())
    idx = np.flatnonzero(mask)
    rows = ds.rows(idx)
    y = ds.y[idx]
    info = {"n_adore_rows": int(len(rows)), "partition": "train only" if train_only else "train+test",
            "extra": extra_kind}
    if extra_kind and extra_kind != "none":
        ex = CYC.load_extra_rows(extra_kind, challenge=ds.challenge, split=ds.split)
        if len(ex):
            add = ex.rows.reindex(columns=rows.columns)
            y_add = pd.to_numeric(add[ds.TARGETS[ds.target]], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(y_add)
            rows = pd.concat([rows, add.loc[ok]], ignore_index=True)
            y = np.concatenate([y, y_add[ok]])
        info["extra_info"] = ex.info
        info["n_rows_total"] = int(len(rows))
    PRO.assert_prospective_chemicals_absent(rows, what="prospective-pass training frame")
    weights = None
    if extra_weight != 1.0 and info.get("n_rows_total"):
        n_adore = info["n_adore_rows"]
        weights = np.concatenate([np.ones(n_adore),
                                  np.full(len(rows) - n_adore, float(extra_weight))])
        info["extra_weight"] = float(extra_weight)
    ok = np.isfinite(y)
    return (rows.loc[ok].reset_index(drop=True), y[ok], info,
            None if weights is None else weights[ok])


#: name suffixes that describe the training data or the role, not the model family
NAME_SUFFIXES = ("_trainpart", "_aug_multitaxa", "_multitaxa", "_aug", "_published", "_mwlogp")


def model_family(name: str, frozen: dict | None = None) -> str:
    """Registry/model name -> the model family that implements it.

    ``xgboost_published`` is the published configuration of the ``xgboost`` family, ``*_aug`` /
    ``*_multitaxa`` say which extra training rows a frozen model used, and ``_trainpart`` marks the
    sensitivity entry trained on the training partition only. When the frozen block carries an
    explicit ``family`` field it wins over the name.
    """
    if frozen and isinstance(frozen.get(name), dict) and frozen[name].get("family"):
        return str(frozen[name]["family"])
    base = name
    changed = True
    while changed:
        changed = False
        for suf in NAME_SUFFIXES:
            if base.endswith(suf):
                base = base[: -len(suf)]
                changed = True
    return base


def fit_predict(family: str, spec: FeatureSpec, params: dict, train_rows: pd.DataFrame,
                y_train: np.ndarray, pro: pd.DataFrame, *, seed: int, device: str,
                chemeleon_device: str, frozen: dict | None = None,
                sample_weight=None) -> np.ndarray:
    """Fit one model on the training frame and predict every prospective row."""
    from sentinel.models.chemtox.data import FeatureBuilder

    builder = FeatureBuilder(spec, chemeleon_device=chemeleon_device)
    Xtr = builder.fit_transform(train_rows)
    Xpr = builder.transform(pro)
    p = dict(params)
    fam = model_family(family, frozen)
    if fam in ("tabpfn", "chemeleon_tabpfn", "embeddings"):
        p["device"] = "cuda" if device == "gpu" else "cpu"
    if fam.startswith("chemprop"):
        p["accelerator"] = device
    model = CYC.make_model(fam, seed)(p)
    model.fit(Xtr, y_train, train_rows=train_rows, valid_rows=None,
              feature_names=builder.names_, builder=builder, sample_weight=sample_weight)
    return np.asarray(model.predict(Xpr, rows=pro), dtype=float)


def trivial_predictions(train_rows: pd.DataFrame, y_train: np.ndarray, pro: pd.DataFrame
                        ) -> dict[str, np.ndarray]:
    """Global / species / species+duration mean baselines, fitted on the training rows only."""
    sp = train_rows[A.SPECIES_KEY].astype("string").fillna("<NA>")
    du = train_rows["result_obs_duration_mean"].astype("string").fillna("<NA>")
    g = float(np.nanmean(y_train))
    by_sp = pd.Series(y_train).groupby(sp.to_numpy()).mean()
    by_spdu = pd.Series(y_train).groupby((sp + "|" + du).to_numpy()).mean()
    psp = pro[A.SPECIES_KEY].astype("string").fillna("<NA>")
    pdu = pro["result_obs_duration_mean"].astype("string").fillna("<NA>")
    out = {
        "global_mean": np.full(len(pro), g),
        "species_mean": psp.map(by_sp).astype(float).fillna(g).to_numpy(),
    }
    key = (psp + "|" + pdu)
    out["species_duration_mean"] = (key.map(by_spdu).astype(float)
                                    .fillna(psp.map(by_sp).astype(float)).fillna(g).to_numpy())
    return out


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def paired_delta_ci(y, pa, pb, chem, *, n_boot: int, seed: int = 0) -> dict:
    """Paired chemical-cluster bootstrap of RMSE(a) - RMSE(b) on identical rows.

    Uses the shared ``sentinel.eval.bootstrap.paired_bootstrap`` (the same function the ADORE test
    pass used), so the statistic, the p-value definition and the resampling unit are identical
    across the two test sets. Negative ``diff`` means the model has the lower RMSE.
    """
    t = paired_bootstrap(rmse, np.asarray(y, float), np.asarray(pa, float), np.asarray(pb, float),
                         groups=np.asarray(chem, dtype=object).astype(str), n_boot=n_boot,
                         seed=seed, higher_is_better=False)
    t["ci_method"] = f"paired cluster bootstrap by chemical, {n_boot} resamples"
    return t


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.i_have_approval:
        log.error("refusing to run: the prospective test set is scored once, with the "
                  "coordinator's approval (--i-have-approval)")
        return 2
    threads = set_threads(args.threads)

    data_path = PRO.prospective_path()
    sha = file_sha256(data_path)
    if sha != args.expect_data_hash:
        log.error("prospective data hash mismatch: file %s, expected %s", sha, args.expect_data_hash)
        return 3
    sel_path = Path(args.selection) if args.selection else (
        paths.RUNS_ROOT / "chemtox" / f"cycle2_select/{args.challenge}_{args.split}" / "selection.json")
    payload = json.loads(Path(sel_path).read_text())
    frozen = payload.get("frozen")
    if not frozen:
        log.error("%s has no frozen block; run chemtox_cycle_select.py --freeze first", sel_path)
        return 4
    fh = frozen_hash(frozen)
    if fh != args.expect_config_hash:
        log.error("frozen-configuration hash mismatch: %s, expected %s", fh, args.expect_config_hash)
        return 5
    log.info("seals passed: data %s, configurations %s", sha[:16], fh)

    out_dir = paths.run_dir("chemtox", "prospective_test")
    handler = logging.FileHandler(out_dir / "score.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)

    pro = pd.read_parquet(data_path)
    log.info("prospective rows %d; tracks %s", len(pro),
             pro["prospective_track"].value_counts().to_dict())

    # which models, in which order (baselines first so the comparisons exist)
    suffix = "_trainpart" if args.train_partition_only else ""

    def role_of(name: str) -> str:
        cfg = frozen.get(name, {})
        if cfg.get("role") in ("comparator", "candidate"):
            return cfg["role"]
        fam = cfg.get("family") or model_family(name, frozen)
        return "comparator" if fam in BASELINE_FAMILIES else "candidate"

    meta = {k: v for k, v in frozen.items() if k.startswith("__")}
    frozen = {k: v for k, v in frozen.items() if not k.startswith("__")}
    comparator_frozen = [n for n in frozen if not frozen[n].get("ensemble")
                         and role_of(n) == "comparator"]
    candidate_frozen = [n for n in frozen if not frozen[n].get("ensemble")
                        and role_of(n) == "candidate"]
    names = comparator_frozen + candidate_frozen          # comparators first
    ens_names = [n for n in frozen if frozen[n].get("ensemble")]
    if args.models:
        names = [n for n in names if n in args.models]
        ens_names = [n for n in ens_names if n in args.models]
    log.info("comparators: %s", comparator_frozen)
    log.info("candidates: %s; ensembles: %s", candidate_frozen, ens_names)
    # the published method is the §15.1 (i) reference; when both chemical blocks were frozen the
    # better of the two on masked CV is the reference, and that choice is recorded
    published_reference = meta.get("__published_reference__")
    pub_cv = meta.get("__published_reference_cv__", {})
    if published_reference is None or published_reference not in frozen:
        log.error("the frozen block does not name a published reference "
                  "(__published_reference__): re-freeze with chemtox_cycle_select.py --freeze, "
                  "which fixes it from the masked CV scores. Plan §15.1 (i) cannot be evaluated "
                  "without it, and the scorer will not pick one after the fact.")
        return 7
    log.info("published reference (fixed in the frozen block, masked CV %s): %s of %s",
             pub_cv.get(published_reference), published_reference, pub_cv)

    # the single-pass guard, applied BEFORE anything is fitted or scored: if a model already has a
    # prospective entry for a requested track, this pass would be a second look at that test set
    guarded = [f"{n}{suffix}" for n in names + ens_names] + [f"{n}{suffix}" for n in TRIVIAL]
    existing = existing_entries(guarded, list(args.tracks))
    if existing and not args.allow_rescore:
        log.error("refusing to run: %d model x track combinations already have prospective entries "
                  "(%s...); the prospective set is scored once per frozen configuration. Pass "
                  "--allow-rescore only for a documented, superseding re-run.",
                  len(existing), existing[:4])
        return 6

    # every ensemble must be scorable before anything is fitted: its components must be in this
    # pass and its out-of-fold matrices must exist (round-3 review finding N5)
    stack_weights: dict[str, dict[str, float]] = {}
    for ename in ens_names:
        comps = list(frozen[ename].get("components", []))
        if len(comps) < 2:
            log.error("%s: a frozen ensemble needs at least two components, has %s", ename, comps)
            return 9
        unknown = [c for c in comps if c not in names]
        if unknown:
            log.error("%s: components %s are not scored in this pass (models: %s)", ename, unknown,
                      names)
            return 9
        if "mean" not in ename:
            stack_weights[ename] = nnls_weights(frozen, ename, args)
            log.info("%s weights (from the frozen out-of-fold sources): %s", ename,
                     {c: round(v, 3) for c, v in stack_weights[ename].items()})

    datasets: dict[tuple, ChemToxDataset] = {}

    def dataset_for(spec: FeatureSpec) -> ChemToxDataset:
        key = tuple(sorted(spec.to_dict().items()))
        if key not in datasets:
            ds = ChemToxDataset.load(args.challenge, args.split, target=args.target, specs=spec,
                                     extra_columns=EXTRA_COLUMNS)
            ds.assert_no_chemical_leakage(log=log)
            datasets[key] = ds
        return datasets[key]

    results: dict[str, dict] = {}
    preds: dict[str, np.ndarray] = {}
    train_info: dict[str, dict] = {}
    preflight_failures: list[str] = []
    config_common = {"data_sha256": sha, "frozen_hash": fh, "selection": str(sel_path),
                     "training": "all ADORE t-F2F rows" if not args.train_partition_only
                     else "ADORE t-F2F training partition only",
                     "threads": threads, "accelerator": args.accelerator,
                     "tracks_scored": list(args.tracks), "n_boot": int(args.n_boot),
                     "seeds_override": list(args.seeds) if args.seeds else None,
                     "model_name_suffix": suffix or None,
                     "primary_track": PRIMARY_TRACK,
                     "primary_subset": "in_adore_domain (and the parent_novel subset reported "
                                       "beside it, pre-registration §1.3a)"}
    if suffix:
        log.info("sensitivity pass: models are recorded as <name>%s", suffix)

    with TrainingRun(WORKSTREAM, f"{EXPERIMENT}:scoring", config=config_common,
                     data=["adore_raw", "adore", "ecotox", "chemtox_prospective"],
                     plan_ref="docs/preregistration/chemtox_prospective_plan.md",
                     artifacts_dir=str(out_dir)) as run:
        for name in names:
            cfg = frozen[name]
            spec = FeatureSpec(**cfg["features"])
            ds = dataset_for(spec)
            train_rows, y_train, tinfo, sw = build_training_frame(
                ds, cfg.get("extra", "none"), train_only=args.train_partition_only,
                extra_weight=float(cfg.get("extra_weight", 1.0) or 1.0))
            train_info[name] = tinfo
            seeds = args.seeds if args.seeds else cfg.get("seeds", [0])
            if args.dry_run:
                # pre-flight: build the matrices the real pass would build (no fit, no target read)
                from sentinel.models.chemtox.data import FeatureBuilder

                builder = FeatureBuilder(spec, chemeleon_device="cuda"
                                         if args.accelerator == "gpu" else "cpu")
                Xtr = builder.fit_transform(train_rows)
                Xpr = builder.transform(pro)
                ok = bool(np.isfinite(Xpr).all()) and Xtr.shape[1] == Xpr.shape[1]
                train_info[name]["preflight"] = {
                    "n_features": int(Xtr.shape[1]), "train_shape": list(Xtr.shape),
                    "prospective_shape": list(Xpr.shape), "finite_and_aligned": ok}
                log.info("dry run: %-22s %d training rows (%s), %d features, prospective %s, "
                         "finite+aligned %s, %d seeds", name, len(train_rows), tinfo.get("extra"),
                         Xtr.shape[1], Xpr.shape, ok, len(seeds))
                if not ok:
                    log.error("%s: pre-flight failed — matrices misaligned or non-finite", name)
                    preflight_failures.append(name)
                # fit and predict for real on a subsample: the ADORE pass was aborted twice by
                # crashes *after* scores had been printed (chemtox_test_plan.md §6b), so the whole
                # fit/predict path is exercised here, where no prospective target is read
                if args.preflight_rows > 0:
                    k = min(int(args.preflight_rows), len(train_rows))
                    rs = np.random.default_rng(0)
                    take = np.sort(rs.choice(len(train_rows), size=k, replace=False))
                    sub = train_rows.iloc[take].reset_index(drop=True)
                    try:
                        pp = fit_predict(name, spec, cfg["params"], sub, y_train[take], pro,
                                         seed=seeds[0], device=args.accelerator,
                                         chemeleon_device="cuda" if args.accelerator == "gpu"
                                         else "cpu", frozen=frozen,
                                         sample_weight=None if sw is None else sw[take])
                        finite = bool(np.isfinite(pp).all())
                        train_info[name]["preflight"].update(
                            {"fit_rows": k, "predictions_finite": finite,
                             "prediction_mean": float(np.mean(pp)),
                             "prediction_sd": float(np.std(pp))})
                        log.info("dry run: %-22s fit on %d rows, %d finite predictions "
                                 "(mean %.2f, sd %.2f)", name, k, int(np.isfinite(pp).sum()),
                                 float(np.mean(pp)), float(np.std(pp)))
                        if not finite:
                            log.error("%s: pre-flight predictions are not all finite", name)
                            preflight_failures.append(name)
                    except Exception as exc:  # noqa: BLE001 — that is what this check is for
                        log.error("%s: PRE-FLIGHT FIT FAILED (%s: %s)", name,
                                  type(exc).__name__, exc)
                        train_info[name]["preflight"]["fit_error"] = f"{type(exc).__name__}: {exc}"
                        preflight_failures.append(name)
                continue
            per_seed = []
            for seed in seeds:
                p = fit_predict(name, spec, cfg["params"], train_rows, y_train, pro, seed=seed,
                                device=args.accelerator,
                                chemeleon_device="cuda" if args.accelerator == "gpu" else "cpu",
                                frozen=frozen, sample_weight=sw)
                per_seed.append(p)
                log.info("%s seed %d fitted on %d training rows", name, seed, len(train_rows))
            P = np.vstack(per_seed)
            preds[name] = P.mean(axis=0)
            results[name] = {"seed_predictions": P, "seeds": list(seeds)}
            np.save(out_dir / f"pred_{name}{suffix}.npy", P)

        # trivial and ensemble predictions
        if not args.dry_run and preds:
            base_spec = FeatureSpec(**frozen[names[0]]["features"]) if names else FeatureSpec()
            ds0 = dataset_for(base_spec)
            tr0, y0, _, _ = build_training_frame(ds0, "none",
                                                 train_only=args.train_partition_only)
            for k, v in trivial_predictions(tr0, y0, pro).items():
                preds[k] = v
                results[k] = {"seed_predictions": v[None, :], "seeds": [0], "trivial": True}
            for ename in ens_names:
                comps = list(frozen[ename]["components"])
                missing = [c for c in comps if c not in preds]
                if missing:
                    raise RuntimeError(
                        f"{ename}: components {missing} were not scored in this pass. An ensemble "
                        f"must be scored over exactly the models it was selected with; fix the "
                        f"frozen block instead of scoring a different ensemble.")
                w = ({c: 1.0 / len(comps) for c in comps} if "mean" in ename
                     else stack_weights[ename])
                # per-seed ensembles: seed s of the ensemble combines seed s of every stochastic
                # component (a deterministic component contributes its single prediction), so
                # plan §15.1 (iii) can be evaluated for the ensemble as well
                n_seeds = max(results[c]["seed_predictions"].shape[0] for c in comps)
                rows_ = []
                for s in range(n_seeds):
                    acc = np.zeros(len(pro))
                    for c in comps:
                        sp = results[c]["seed_predictions"]
                        acc += w[c] * sp[min(s, sp.shape[0] - 1)]
                    rows_.append(acc)
                P = np.vstack(rows_)
                preds[ename] = P.mean(axis=0)
                results[ename] = {"seed_predictions": P, "seeds": list(range(n_seeds)),
                                  "weights": w, "components": comps}
                np.save(out_dir / f"pred_{ename}{suffix}.npy", P)
                log.info("%s over %d seeds, weights %s", ename, n_seeds,
                         {c: round(w[c], 3) for c in w})

        # ---- metrics per track
        report: dict = {"data_sha256": sha, "frozen_hash": fh, "tracks": {}, "models": names,
                        "ensembles": ens_names, "train_info": train_info,
                        "package_hash": package_hash()}
        if args.dry_run:
            report["preflight_failures"] = preflight_failures
            report["ensembles_verified"] = {e: {"components": frozen[e]["components"],
                                                "weights": stack_weights.get(e)}
                                            for e in ens_names}
            CV.write_json(out_dir / "dry_run.json", report)
            if preflight_failures:
                log.error("dry run FAILED for %d model(s): %s — fix them before the single pass",
                          len(preflight_failures), preflight_failures)
                return 8
            log.info("dry run complete for %d models; no target value was read", len(names))
            return 0
        y_all = pd.to_numeric(pro["result_conc1_mean_mol_log"], errors="coerce").to_numpy(float)
        chem_all = pro["test_cas"].astype("string").fillna("<NA>").to_numpy()
        taxon_all = pro["tax_gs"].astype("string").fillna("<NA>").to_numpy()
        # candidates are the models a claim can be made for; everything else is a comparator
        comparator_names = set(comparator_frozen) | set(TRIVIAL)
        candidate_models = [n for n in list(preds) if n not in comparator_names]
        report["candidate_models"] = candidate_models
        for track in args.tracks:
            m = track_mask(pro, track)
            if m.sum() == 0:
                continue
            y, chem, taxon = y_all[m], chem_all[m], taxon_all[m]
            tr: dict = {"n_rows": int(m.sum()), "n_chemicals": int(pd.unique(chem).size),
                        "n_species": int(pd.unique(taxon).size), "models": {}}
            # "the strongest simple baseline" (plan §15.1 i), chosen on this track's own scores —
            # deliberately conservative: it makes the claim harder, never easier
            baseline_rmse = {n: rmse(y, preds[n][m]) for n in comparator_names if n in preds}
            if baseline_rmse:
                best_base = min(baseline_rmse, key=baseline_rmse.get)
                preds["best_baseline"] = preds[best_base]
                results["best_baseline"] = results[best_base]
                tr["best_baseline"] = {"model": best_base, "rmse": baseline_rmse[best_base],
                                       "all_baselines": baseline_rmse}
            if published_reference in preds:
                preds["published_reference"] = preds[published_reference]
                results["published_reference"] = results[published_reference]
                tr["published_reference"] = {"model": published_reference,
                                            "rmse": rmse(y, preds[published_reference][m])}
            for name, pred in list(preds.items()):
                if name in ("best_baseline", "published_reference"):
                    continue
                p = pred[m]
                met = E.report(y, p, chem=chem, taxon=taxon)
                ci = E.bootstrap_report(y, p, chem=chem, taxon=taxon, metrics=("rmse",),
                                        n_boot=args.n_boot)
                entry = {"metrics": met, "rmse_ci": ci.get("rmse"),
                         "seed_rmse": [float(np.sqrt(np.mean((sp[m] - y) ** 2)))
                                       for sp in results[name]["seed_predictions"]]}
                for comparator in sorted(comparator_names) + ["best_baseline",
                                                               "published_reference"]:
                    if comparator in preds and comparator != name:
                        entry[f"vs_{comparator}"] = paired_delta_ci(
                            y, p, preds[comparator][m], chem, n_boot=args.n_boot)
                        # plan §15.1 (iii): the gain must hold in every seed, not only in the mean
                        per_seed_delta = [
                            rmse(y, sp[m]) - rmse(y, preds[comparator][m])
                            for sp in results[name]["seed_predictions"]]
                        entry[f"vs_{comparator}"]["per_seed_delta"] = [float(d)
                                                                       for d in per_seed_delta]
                        if len(per_seed_delta) > 1:
                            entry[f"vs_{comparator}"]["all_seeds_better"] = bool(
                                all(d < 0 for d in per_seed_delta))
                        else:
                            # one prediction only (a deterministic model, an ensemble of seed-means
                            # or a trivial baseline): the rule does not apply and is not faked
                            entry[f"vs_{comparator}"]["all_seeds_better"] = None
                            entry[f"vs_{comparator}"]["seed_rule"] = (
                                "not applicable: a single deterministic prediction")
                # pre-registered breakdowns (§4.1): structure class, metal-bearing, duration
                cuts = {
                    "structure_class": pro["structure_class"].astype("string").fillna("<NA>"),
                    "has_metal": pro["has_metal"].astype("string").fillna("<NA>"),
                    "duration_h": pro["result_obs_duration_mean"].astype("string").fillna("<NA>"),
                    "parent_novel": pro["parent_novel"].astype("string").fillna("<NA>"),
                }
                entry["breakdowns"] = {}
                for cut_name, series in cuts.items():
                    vals = series.to_numpy()[m]
                    entry["breakdowns"][cut_name] = {
                        str(v): {"n": int((vals == v).sum()),
                                 "rmse": rmse(y[vals == v], p[vals == v])}
                        for v in pd.unique(vals) if (vals == v).sum() >= 10}
                tr["models"][name] = entry
                log.info("[%s] %-22s RMSE %.4f [%.3f, %.3f]", track, name, met["rmse"],
                         entry["rmse_ci"]["ci_low"] if entry["rmse_ci"] else float("nan"),
                         entry["rmse_ci"]["ci_high"] if entry["rmse_ci"] else float("nan"))
            # Holm across models for the comparison against the reproduced XGBoost baseline
            # Holm family: the candidate models only. The reproduced baselines, the trivial
            # baselines and the read-across comparator are comparators, not candidates, so they do
            # not inflate the correction (they are reported with their own CIs either way).
            for ref in HOLM_REFERENCES:
                pvals = {n: v[f"vs_{ref}"]["p_value"] for n, v in tr["models"].items()
                         if n in candidate_models and f"vs_{ref}" in v
                         and np.isfinite(v[f"vs_{ref}"].get("p_value", np.nan))}
                if not pvals:
                    continue
                rejected = holm_bonferroni(pvals)
                # the p-value is two-sided, so a rejection only counts when the candidate is the
                # *better* model (negative RMSE difference)
                rejected = {n: bool(b and tr["models"][n][f"vs_{ref}"]["diff"] < 0)
                            for n, b in rejected.items()}
                tr[f"holm_rejected_vs_{ref}"] = {n: bool(b) for n, b in rejected.items()}
                tr[f"holm_family_vs_{ref}"] = sorted(pvals)
                for n, b in rejected.items():
                    tr["models"][n][f"vs_{ref}"]["holm_rejected"] = bool(b)
            for alias in ("best_baseline", "published_reference"):
                preds.pop(alias, None)
                results.pop(alias, None)
            report["tracks"][track] = tr
            if "xgboost_published" not in preds:
                log.warning("the reproduced published baseline is not in this pass: the primary "
                            "comparison of plan §15.1 (i) cannot be made")

            for name, entry in tr["models"].items():
                if name in ("best_baseline", "published_reference"):
                    continue
                cfg = frozen.get(name, {})
                log_result(
                    workstream=WORKSTREAM, experiment=EXPERIMENT, model=f"{name}{suffix}",
                    target=TARGET_NAME, metric="rmse", value=entry["metrics"]["rmse"],
                    ci_low=(entry["rmse_ci"] or {}).get("ci_low"),
                    ci_high=(entry["rmse_ci"] or {}).get("ci_high"),
                    ci_method=(entry["rmse_ci"] or {}).get("ci_method"),
                    split=f"prospective/{track}/test", n=int(m.sum()),
                    config={**config_common, "model": name, "frozen": cfg,
                            "track": track,
                            "primary": bool(track == PRIMARY_TRACK),
                            "train_info": train_info.get(name)},
                    data_manifests=["adore_raw", "adore", "ecotox", "chemtox_prospective"],
                    is_baseline=(name in comparator_names or results[name].get("trivial", False)),
                    notes=("prospective test, single pre-registered pass ("
                           + track_role(track) + " track"
                           + (", UNDERPOWERED" if track == "temporal" else "") + ")"),
                    extra={k: v for k, v in entry.items() if k != "seed_predictions"},
                    artifacts=[str(out_dir / f"pred_{name}{suffix}.npy")],
                    supersedes=(supersede_id(f"{name}{suffix}", track)
                                if args.allow_rescore else None),
                )
        CV.write_json(out_dir / "prospective_results.json", report)
        if PRIMARY_TRACK in report["tracks"]:
            best = min(report["tracks"][PRIMARY_TRACK]["models"].items(),
                       key=lambda kv: kv[1]["metrics"]["rmse"])
            run.metric("primary_best_rmse", best[1]["metrics"]["rmse"])
            run.note(f"primary track best: {best[0]} RMSE {best[1]['metrics']['rmse']:.4f}")
    log.info("wrote %s", out_dir / "prospective_results.json")
    return 0


def nnls_weights(frozen: dict, ename: str, args) -> dict[str, float]:
    """Refit the stack's non-negative weights on the training-portion out-of-fold predictions.

    The out-of-fold files are the ones the freeze recorded (``oof_sources`` in the frozen block,
    keyed by the component's frozen name), so the weights are fitted on exactly the predictions the
    selection used. A missing file is a hard error: an ensemble must never be scored with silently
    different weights or fewer members.
    """
    from scipy.optimize import nnls

    sources = frozen[ename].get("oof_sources") or {}
    comps = list(frozen[ename]["components"])
    missing = [c for c in comps if c not in sources or not Path(str(sources[c])).exists()]
    if missing:
        raise RuntimeError(
            f"{ename}: out-of-fold predictions missing for {missing} "
            f"({[str(sources.get(c)) for c in missing]}). Re-freeze with "
            f"chemtox_cycle_select.py --freeze after those families have a masked 5-fold run.")
    ds = ChemToxDataset.load(args.challenge, args.split, specs=FeatureSpec(fingerprint="MACCS"))
    y = ds.y
    P = np.column_stack([np.load(str(sources[c])) for c in comps])
    m = np.isfinite(y) & np.isfinite(P).all(axis=1)
    w, _ = nnls(P[m], y[m])
    s = w.sum()
    w = w / s if s > 0 else np.full(len(comps), 1.0 / len(comps))
    return {c: float(wi) for c, wi in zip(comps, w)}


if __name__ == "__main__":
    raise SystemExit(main())
