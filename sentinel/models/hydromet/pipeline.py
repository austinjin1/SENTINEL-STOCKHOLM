# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Predict-from-artifacts and score-from-prediction-files for SENTINEL-HydroMet — the code behind
scripts/eval/hydromet_predict.py and scripts/eval/hydromet_score.py (and the only path to test data).

* **Artifact names.** Baselines: ``<name>`` (``models/<name>.pkl``). Seeded deep models:
  ``<model>_s<seed>`` (from the checkpoint's own config), so every seed has its own entry in
  FREEZE_hydromet.json ``weights_sha256`` and its own prediction file (review C2).
* **Provenance on protected data** (review C3c). Prediction files scored on test blocks / test-role
  sites must carry ``artifact_sha256 == freeze.weights_sha256[name]`` and the current freeze sha256.
* **Seed ensembles.** The evaluated SENTINEL model is the mean of its ≥ 2 seed predictions in model
  space (protocol §6).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as E
from . import guard
from .data import HydroMetData, from_model, to_model

SEED_RE = re.compile(r"(.+)_s(\d+)")


def checkpoint_name(path) -> str:
    """``<model>_s<seed>`` for an EA-LSTM checkpoint (model name and seed read from the checkpoint)."""
    import torch

    d = torch.load(path, map_location="cpu", weights_only=False)
    base = d.get("info", {}).get("model_name") or Path(path).stem.replace("_best", "")
    base = SEED_RE.fullmatch(base).group(1) if SEED_RE.fullmatch(base) else base
    return f"{base}_s{int(d['cfg']['seed'])}"


def discover_artifacts(run_dirs) -> dict[str, Path]:
    arts: dict[str, Path] = {}
    import json

    for rd in map(Path, run_dirs):
        found = {p.stem: p for p in sorted((rd / "models").glob("*.pkl"))}
        sel = rd / "selection.json"
        # an EA-LSTM root: only the SELECTED grid point's seed runs are artifacts (never the other points)
        sub = sorted(rd.glob(f"g{json.loads(sel.read_text())['selected_index']}_s*")) if sel.is_file() else [rd]
        for d in sub:
            found.update({checkpoint_name(p): p for p in sorted(d.glob("*_best.pt"))})
        for k, p in found.items():
            if k in arts:
                raise ValueError(f"duplicate artifact name {k}: {arts[k]} and {p}")
            arts[k] = p
    return arts


def predict_artifacts(data: HydroMetData, artifacts: dict, site_idx, t_idx, block: str, out_dir, *, lag: int,
                      access: dict, smoke: bool = False, device: str = "cpu", log=print, track_r_fold: str | None = None,
                      ) -> list[Path]:
    """Predict with saved artifacts (no fitting). On protected data every artifact's sha256 must equal
    the frozen ``weights_sha256[name]``. Writes one npz per artifact with full provenance."""
    from .baselines import BaselineBase
    from .features import FeatureSet

    if access.get("protected"):
        if smoke:
            raise guard.ProtectedDataError("smoke artifacts are never applied to protected data")
        for name, p in artifacts.items():   # track-R artifacts are frozen under fold-unique keys
            guard.verify_weights(access["freeze"], f"trackr_{track_r_fold}/{name}" if track_r_fold else name, p)
    files, fs = [], None
    for name, p in artifacts.items():
        if str(p).endswith(".pkl"):
            fs = fs or FeatureSet.build(data, lag=lag)
            preds, dist = BaselineBase.load(p).predict(data, fs, site_idx, t_idx), None
            seed = None
        else:
            from .lstm import HydroMetEALSTM

            model = HydroMetEALSTM.load(p)
            if model.cfg.forcing_lag_days != lag:
                raise ValueError(f"{name}: checkpoint forcing lag differs from the run config")
            preds, dist = model.predict(data, site_idx, t_idx, device=device, return_dist=True)
            seed = int(model.cfg.seed)
        m = SEED_RE.fullmatch(name)
        meta = {"smoke": bool(smoke), "forcing_lag_days": lag, "split_sha256": data.split.sha256(),
                "artifact": str(p), "artifact_sha256": guard.artifact_sha256(p),
                "freeze_sha256": access.get("freeze_sha256"), "seed": seed,
                "base_model": m.group(1) if (m and seed is not None) else name, "track_r_fold": track_r_fold}
        f = E.save_predictions(Path(out_dir) / f"{name}__{block}.npz", model=name, preds=preds,
                               sites=[data.sites[i] for i in site_idx], dates=data.dates[t_idx], block=block,
                               dist=dist, meta=meta)
        files.append(f)
        log(f"[predict] {name}: {len(site_idx)} sites x {len(t_idx)} days -> {f}")
    return files


def _track_r_folds() -> dict:
    from . import tracks

    return tracks.load_folds()


def load_prediction_set(pred_dirs, data: HydroMetData, site_idx, t_idx, block: str, *, smoke: bool,
                        access: dict) -> tuple[dict, dict]:
    """Load ``*__<block>.npz`` from ``pred_dirs``; checks split hash, smoke flag, unique names and, on
    protected data, artifact/freeze provenance."""
    preds, sources, folds_seen = {}, {}, {}
    prot = bool(access.get("protected"))
    for d in pred_dirs:
        for f in sorted(Path(d).glob(f"*__{block}.npz")):
            pf = E.load_predictions(f)
            h = pf["header"]
            if h.get("split_sha256") != data.split.sha256():
                raise ValueError(f"{f}: produced under a different split")
            if bool(h.get("smoke")) != bool(smoke):
                raise ValueError(f"{f}: smoke flag {h.get('smoke')} != this scoring run")
            name = h["model"]
            fold = h.get("track_r_fold")
            if prot:
                want = access["freeze"]["weights_sha256"].get(f"trackr_{fold}/{name}" if fold else name)
                if not want or h.get("artifact_sha256") != want:
                    raise guard.ProtectedDataError(f"{f}: artifact_sha256 does not match frozen weights_sha256[{name!r}]")
                if h.get("freeze_sha256") != access.get("freeze_sha256"):
                    raise guard.ProtectedDataError(f"{f}: not produced under the current freeze file")
            scoring_r = (data.info.get("track_r") or {}).get("fold")
            if bool(fold) != bool(scoring_r) or (scoring_r and scoring_r not in ("cv", "test")):
                raise ValueError(f"{f}: track-R fold {fold!r} cannot be scored in this run (track_r={scoring_r!r}); "
                                 "track-R files are scored only together (--track-r cv, or the M6R/test look)")
            if fold:   # protocol §10: every site in a fold file must belong to that held-out region
                body = _track_r_folds()
                wrong = [s for s in pf["sites"] if body["fold_of"].get(s) != fold]
                if wrong:
                    raise ValueError(f"{f}: {len(wrong)} sites outside held-out region {fold} (e.g. {wrong[:3]})")
            if name in preds:
                if not fold or fold in folds_seen.get(name, set()):
                    raise ValueError(f"duplicate model name {name} ({f} and {sources[name]})")
                new = E.align_predictions(pf, data, site_idx, t_idx)
                for v in new:   # disjoint held-out regions: fill
                    old = preds[name].get(v)
                    preds[name][v] = new[v] if old is None else np.where(np.isfinite(old), old, new[v])
                folds_seen[name].add(fold)
                sources[name] = sources[name] + ";" + str(f)
                continue
            folds_seen[name] = {fold} if fold else set()
            preds[name], sources[name] = E.align_predictions(pf, data, site_idx, t_idx), str(f)
    return preds, sources


# Protocol v0.7 (cycle 2): fixed, unfitted per-seed blends in model space (no weight is estimated anywhere):
#   <blend>_s<k> = mean(<a>_s<k>, <b>_s<k>); the evaluated <blend> is then the usual seed ensemble.
BLENDS = {"ealstm_gbmblend": ("ealstm", "forcing_gbm_bag")}


def seed_blends(preds: dict, blends: dict = BLENDS) -> dict:
    """Per-seed equal-weight model-space blends of two seeded model families (seeds present in both)."""
    out = {}
    for name, (a, b) in blends.items():
        for k in sorted({int(m.group(2)) for n in preds if (m := SEED_RE.fullmatch(n)) and m.group(1) == a}):
            pa, pb = preds.get(f"{a}_s{k}"), preds.get(f"{b}_s{k}")
            if pa is None or pb is None or f"{name}_s{k}" in preds:
                continue
            out[f"{name}_s{k}"] = {v: from_model(v, 0.5 * (to_model(v, pa[v]) + to_model(v, pb[v]))).astype("float32")
                                   for v in set(pa) & set(pb)}
    return out


def seed_ensembles(preds: dict) -> tuple[dict, dict]:
    """Protocol §6: ``<model>`` = mean of its ``<model>_s<k>`` predictions in model space (log for log
    targets), back-transformed; needs ≥ 2 seeds. Returns (ensembles, {model: [seed names]})."""
    groups: dict = {}
    for name in preds:
        m = SEED_RE.fullmatch(name)
        if m:
            groups.setdefault(m.group(1), []).append(name)
    out, members = {}, {}
    for base, names in groups.items():
        if len(names) < 2 or base in preds:
            continue
        vs = set.intersection(*(set(preds[n]) for n in names))
        out[base] = {v: from_model(v, np.mean([to_model(v, preds[n][v]) for n in names], axis=0)).astype("float32")
                     for v in vs}
        members[base] = sorted(names)
    return out, members


def score(data: HydroMetData, preds: dict, site_idx, t_idx, block: str, roles, out_dir, *, refs=(), n_boot=2000,
          min_days=60, primary=False, access=None, log=print, fixed_comparators: dict | None = None,
          primary_models=None) -> dict:
    """Per-site + summary + pooled (+ primary) analysis on common cells; writes CSV/parquet to out_dir."""
    from .runner import write_json

    access = access or {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = {**preds, **seed_blends(preds)}
    ens, members = seed_ensembles(preds)
    preds = {**preds, **ens}
    suffix = f"{block}_{'-'.join(roles)}"
    site_df, info = E.site_metrics(data, preds, site_idx, t_idx, block, min_days=min_days)
    site_df.to_parquet(out_dir / f"site_metrics_{suffix}.parquet", index=False)
    write_json(out_dir / f"rows_{suffix}.json", info)
    write_json(out_dir / "seed_members.json", members)
    summ = E.summarize(site_df, refs=refs, n_boot=n_boot)
    summ.to_csv(out_dir / f"summary_{suffix}.csv", index=False)
    pooled = E.pooled_metrics(data, preds, site_idx, t_idx, block, min_days=min_days, n_boot=n_boot)
    pooled.to_csv(out_dir / f"pooled_{suffix}.csv", index=False)
    sec = E.secondary_comparisons(site_df, n_boot=n_boot)
    sec.to_csv(out_dir / f"secondary_{suffix}.csv", index=False)
    res = {"site": site_df, "summary": summ, "pooled": pooled, "members": members, "primary": None,
           "comparators": None, "secondary": sec}
    if len(sec):
        with pd.option_context("display.width", 250, "display.max_columns", 20):
            log("[score] secondary paired comparisons (descriptive; basin-group bootstrap):\n"
                + sec[["analysis", "model", "reference", "var", "median_model", "median_reference", "diff", "ci_low",
                       "ci_high", "n_sites", "n_groups"]].round(3).to_string(index=False))
    if primary:
        if access.get("protected"):   # the look's freeze block (passed by hydromet_test_look.py) or the freeze
            comps = fixed_comparators or access["freeze"].get("comparators")
            if not comps:
                raise guard.ProtectedDataError("protected-data primary comparison needs frozen 'comparators'")
            if not primary_models:
                raise guard.ProtectedDataError("protected-data primary comparison needs the frozen model family")
        elif fixed_comparators:   # pre-declared comparators (track R, protocol §10): no selection here
            comps = {v: dict(f) for v, f in fixed_comparators.items()}
        else:
            if tuple(roles) != ("val",) or block != "val":
                raise RuntimeError("comparators are selected on val-role sites x val block only")
            comps = E.select_comparators(site_df, role="val")
        write_json(out_dir / "comparators.json", comps)
        prim = E.primary_comparison(site_df, comps, seed_models=members, n_boot=n_boot,
                                    **({"models": tuple(primary_models)} if primary_models else {}))
        prim.to_csv(out_dir / f"primary_{suffix}.csv", index=False)
        res.update({"primary": prim, "comparators": comps})
        log(f"[score] primary comparison (comparators {comps}):\n"
            f"{_primary_table(prim) if len(prim) else '(no SENTINEL seed ensembles to compare)'}")
    return res


def _primary_table(prim: pd.DataFrame) -> str:
    cols = ["model", "var", "gbm_comparator", "gbm_diff", "gbm_ci_low", "gbm_ci_high", "physics_comparator",
            "physics_diff", "physics_ci_low", "physics_ci_high", "p_iut", "holm_reject", "gbm_every_seed_positive",
            "physics_every_seed_positive", "won", "gbm_n_sites", "gbm_n_groups"]
    with pd.option_context("display.width", 300, "display.max_columns", 30):
        return prim[[c for c in cols if c in prim]].round(3).to_string(index=False)
