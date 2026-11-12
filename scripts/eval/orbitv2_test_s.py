#!/usr/bin/env python
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M2S/test — the single look of SENTINEL v2 M2 at the Test-S cells (``testxarchive_lake``, ``testxarchive_nonlake``).

Sources: ``docs/preregistration/orbit_arch_v2_amendment.md`` §1.6, §1.7, §2.6, §2.7, §5, §6; lead rulings R-10,
R-15, R-17, R-17a, R-18, R-18b, R-21, R-24; clarification 01 (K-1). The look-specific parts are
``sentinel/models/orbit_v2/looks.py`` (claim, sealing, markers; v1's ``testlook.py`` is not used), the row loader
``sentinel/models/orbit_v2/testrows.py`` and this script; the decision rules are ``orbit_v2/claims.py``.

Order of a real look (nothing is retried; a second start is refused):
  1. ``looks.load_freeze()`` (only the lead-written FREEZE_orbit_S.json), ``--allow-test``, the protected-row
     loader exists, data pins, every artefact at its frozen sha256, the fit scheme of the comparators equals the
     arms', the development table gives scene keys. Any failure here stops BEFORE the claim.
  2. ``looks.claim`` (refuses a registry row, a marker or a non-empty look folder).
  3. Test-S rows through the loader (open claim only), predictions of every frozen model and member, one long
     table per cell, ``looks.seal_predictions`` (access closes; R-17 columns required).
  4. Scores from the sealed files alone: the four confirmatory families on the primary cell fixed in the freeze,
     the other cell as the sensitivity analysis, descriptive tables (§6.4).
  5. Registry rows, ``SCORED.json``.

``--dry-run LABEL`` runs steps 2-4 on VALIDATION rows (``valxany_lake`` in the place of the lake cell) from the
candidate file's artefacts, under ``test_looks_dryrun/LABEL``, with no registry row, and checks that the
predictions made here equal the ones stored by the training scripts. It reads no protected row.

Sealed long table ``cell_<cell>.parquet``: one row per (row, target) with a label. ``y`` log10 (NaN if
left-censored; then ``y_dl``); strata columns; ``v1`` (the frozen ensemble as v1 reports it), ``v1@s<i>`` (each
frozen network alone: its mixture median, without the ensemble's blend); ``v2``, ``v2@unit<u>``, ``v2@u<u>i<i>``
(mixture medians of the pooled mixtures); ``track_n...`` likewise (§2.7; a copy of v2's when it is the same
arm); ``cmp:<name>``; ``q|<model>|<level>`` mixture quantiles of v1 and v2.

Not in this version (the freeze candidate cannot be written before they are): the event track (R-13a) and the
MDN_B interval comparison.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.core.runlog import TrainingRun  # noqa: E402
from sentinel.models.orbit import conformal as CF  # noqa: E402
from sentinel.models.orbit import data as D  # noqa: E402
from sentinel.models.orbit import evaluate as E  # noqa: E402
from sentinel.models.orbit import pipeline as PL  # noqa: E402
from sentinel.models.orbit_v2 import arm as ARM  # noqa: E402
from sentinel.models.orbit_v2 import claims as CL  # noqa: E402
from sentinel.models.orbit_v2 import devm as DM  # noqa: E402
from sentinel.models.orbit_v2 import looks as LK  # noqa: E402
from sentinel.models.orbit_v2 import testcells as TC  # noqa: E402
from sentinel.models.orbit_v2 import train as TR  # noqa: E402
from sentinel.models.orbit_v2 import units as U  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load("orbitv2_comparators", "scripts/train/orbitv2_comparators.py")     # comparator prediction from the bundle
V1T = _load("orbit_test_v1", "scripts/eval/orbit_test.py")                   # v1's frozen baseline predictor

LOOK = LK.LOOKS[0]
LOADER = "sentinel.models.orbit_v2.testrows"
PLAN_REF = "docs/preregistration/orbit_arch_v2_amendment.md §5, §6; lead rulings R-10, R-15, R-17..R-18b, R-21, R-24"
INTERVAL_QS = V1T.INTERVAL_QS
REAL_CELLS = {"lake": LK.CELL_LAKE, "nonlake": LK.CELL_NONLAKE}
DRY_CELLS = {"lake": "valxany_lake", "nonlake": "valxany_nonlake"}
ARM_MODELS = (LK.V2_MODEL, LK.TRACK_N_MODEL)
PARITY_TOL = 2e-3            # log10; stored validation mixtures are float32
ROW_COLS = ("row_id", "block", "waterbody", "lat", "lon", "date", "role", "in_lake", "buffer_excluded", "added",
            "source", "platform", *TC.KEY_COLS)
META_COLS = ("row_id", "block", "waterbody", "scene_disjoint", "small_lake", "ntu_class", "source", "year",
             "n_context", "dist_train_km", "added")


def _log_result(**k):
    from sentinel.core.registry import log_result

    return log_result(**k)


def qcol(model: str, q: float) -> str:
    return f"q|{model}|{q:g}"


# ------------------------------------------------------------------------------------------------ artefacts

def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else paths.RUNS_ROOT / p


def artifact_files(art: dict) -> list:
    """[(key of weights_sha256 / v1_weights_sha256, file)] of every network and fitted object the look loads."""
    out = [(Path(c).parent.name, resolve(c)) for c in art["v1"]["satwq_checkpoints"]]
    if art["v1"].get("blend_with"):
        out.append(("fitted_baselines", resolve(art["v1"]["fitted_baselines"])))
    for m in ARM_MODELS:
        if art.get(m):
            out += [(f"{m}@u{u}i{i}", U.network_dir(resolve(art[m]["root"]), art[m]["arm"], u, i) / "model.pt")
                    for u in TR.UNITS for i in TR.INITS]
    return out


def check_artifacts(freeze: dict) -> dict:
    """Step 1: the artefact list is complete, agrees with ``sealed_models`` and every file has its frozen hash."""
    art, sm = freeze.get("artifacts"), freeze["sealed_models"]
    if not isinstance(art, dict) or not art.get("v1") or not art.get(LK.V2_MODEL) or not art.get("comparators"):
        raise SystemExit("the freeze names no artifacts for v1, v2 and the comparators")
    n1 = len(art["v1"]["satwq_checkpoints"])
    if n1 != int(sm.get("v1_networks", -1)) or bool(art.get(LK.TRACK_N_MODEL)) != bool(sm.get("track_n")):
        raise SystemExit("artifacts and sealed_models disagree on the v1 networks or the Track N arm")
    if sorted(sm.get("comparators", [])) != sorted(S.CLAIM):
        raise SystemExit(f"sealed_models must name the claim comparators {sorted(S.CLAIM)}")
    for key, f in artifact_files(art):
        LK.verify_weights(freeze, key, f)
    root = resolve(art["comparators"]["root"])
    if S.verify_sums(root) != freeze["weights_sha256"].get("comparators"):
        raise SystemExit(f"{root}: SHA256SUMS is not the frozen comparator set")
    bundle = S.load_pickle(root / "bundle.pkl")
    schemes = {"comparators": bundle["fit_scheme"]}
    import torch

    for m in ARM_MODELS:
        if art.get(m):
            ck = torch.load(U.network_dir(resolve(art[m]["root"]), art[m]["arm"], 0, 0) / "model.pt",
                            map_location="cpu", weights_only=False)
            schemes[m] = ck["config"]["fit_scheme"]
    if len(set(schemes.values())) != 1:
        raise SystemExit(f"fit schemes differ: {schemes} (clarification 01, K-2)")
    return {"n_v1": n1, "has_track_n": bool(art.get(LK.TRACK_N_MODEL)), "fit_scheme": schemes["comparators"],
            "bundle": bundle, "comparator_root": root}


# ------------------------------------------------------------------------------------------- predictions

def predict_v1(art: dict, df: pd.DataFrame, pix: dict, device) -> dict:
    """v1 as frozen: ``{"v1": [N, 4], "v1@s<i>": [N, 4], "q": {level: [N, 4]}}`` (log10, D.TARGETS order)."""
    from sentinel.events import orbit as EO                       # v1's own ensemble routine
    from sentinel.models.orbit import baselines as BL
    from sentinel.models.orbit import features as F1

    mixes = [EO.satwq_mixture_embedding([resolve(c)], df, pix, device)[0] for c in art["satwq_checkpoints"]]
    ens = (np.concatenate([m[0] / len(mixes) for m in mixes], -1), np.concatenate([m[1] for m in mixes], -1),
           np.concatenate([m[2] for m in mixes], -1))
    q = TR.mixture_median(ens, INTERVAL_QS)
    med = q[:, :, INTERVAL_QS.index(0.5)].copy()
    blend = art.get("blend_with")
    if blend:
        import cloudpickle

        fitted = cloudpickle.loads(resolve(art["fitted_baselines"]).read_bytes())["fitted"]
        Ximg, Xctx = F1.image_features(df, pix), BL.location_season_matrix(df)
        for j, t in enumerate(D.TARGETS):
            med[:, j] = 0.5 * med[:, j] + 0.5 * np.asarray(V1T.baseline_predict(blend, t, fitted, df, Ximg, Xctx), float)
    out = {LK.V1_MODEL: med, "q": {lv: q[:, :, k] for k, lv in enumerate(INTERVAL_QS)}}
    for i, m in enumerate(mixes):
        out[f"{LK.V1_MODEL}@s{i}"] = U.medians(m)
    return out


def arm_networks(spec: dict, df: pd.DataFrame, pix: dict, feats: dict, ctx_cache, device) -> tuple[list, dict]:
    """(targets, {(unit, init): (w, mu, sg)}) of the arm's 9 frozen networks on the rows of ``df``, predicted as
    ``scripts/train/orbitv2_arm.py`` predicts its validation rows."""
    import torch

    from sentinel.models.orbit import satwq as V1

    arm, root = spec["arm"], resolve(spec["root"])
    targets, nets, every = None, {}, np.ones(len(df), bool)
    for u in TR.UNITS:
        for i in TR.INITS:
            f = U.network_dir(root, arm, u, i) / "model.pt"
            ck = torch.load(f, map_location="cpu", weights_only=False)
            if (ck["arm"], ck["seed"]) != (arm, TR.network_seed(u, i)):
                raise SystemExit(f"{f}: not the network of arm {arm}, unit {u}, init {i}")
            targets = list(ck["targets"]) if targets is None else targets
            if list(ck["targets"]) != targets:
                raise SystemExit(f"{f}: its heads differ from the first network's")
            v1 = ck["config"]["v1"]
            model = ARM.build_model(arm, v1, len(targets)).to(device)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            norm = PL.Norm.from_dict(ck["norm"])
            ctx_off = (ARM.ARMS[arm]["model"] == "satwq_v1"
                       and bool(v1["finetune"]["aug"].get("ctx_off", False)))
            ds = ARM.make_set(arm, df, pix, feats, ctx_cache, norm, every, targets)
            nets[(u, i)] = V1.predict_mixture(model, ds, device, norm.y_mean, norm.y_std, ctx_off=ctx_off)
    return targets, nets


def arm_columns(model: str, nets: dict) -> dict:
    """R-17 columns of one arm: reported model, 3 seed units, 9 networks ([N, T] log10 medians) and quantiles."""
    mix = U.arm_mixture(nets)
    q = TR.mixture_median(mix, INTERVAL_QS)
    out = {model: q[:, :, INTERVAL_QS.index(0.5)], "q": {lv: q[:, :, k] for k, lv in enumerate(INTERVAL_QS)}}
    for u in TR.UNITS:
        out[f"{model}@unit{u}"] = U.medians(U.unit_mixture(nets, u))
    for (u, i), m in nets.items():
        out[f"{model}@u{u}i{i}"] = U.medians(m)
    return out


def predict_comparators(root: Path, bundle: dict, df, pix, feats, carrays, device) -> dict:
    x = S.features(df, pix, feats, carrays, S.bundle_feature_sets(bundle))
    return S.predict_rows(root, x, bundle, None if str(device) == "cpu" else str(device))


def subset(df: pd.DataFrame, pix: dict, feats: dict, mask: np.ndarray):
    """Rows of ``mask`` in their stored order (the context cache is not per row and stays whole)."""
    n = len(df)
    cut = lambda d: {k: (v[mask] if hasattr(v, "__len__") and len(v) == n else v) for k, v in d.items()}  # noqa: E731
    return df[mask].reset_index(drop=True), cut(pix), cut(feats)


def long_tables(df: pd.DataFrame, feats: dict, cells: dict, preds: dict, meta: dict) -> dict:
    """{cell: long table}. ``preds``: {"v1": dict, "v2": (targets, dict), "track_n": (targets, dict) | None,
    "cmp": {target: {name: [N]}}}; ``meta``: per-row arrays of META_COLS not in ``df``."""
    base, n = list(D.TARGETS), len(df)
    t2, c2 = preds[LK.V2_MODEL]
    tn = preds.get(LK.TRACK_N_MODEL)
    targets = base + [t for t in t2 if t not in base]
    targets += [t for t in (tn[0] if tn else []) if t not in targets] + [t for t in preds["cmp"] if t not in targets]
    y, cens = ARM.target_matrix(df, targets)
    added = df["added"].to_numpy(bool)
    frame = {"row_id": df["row_id"].astype(str).to_numpy(), "block": df["block"].astype(str).to_numpy(),
             "waterbody": df["waterbody"].astype(str).to_numpy(), "source": df["source"].astype(str).to_numpy(),
             "year": pd.to_datetime(df["date"]).dt.year.to_numpy(int), "added": added,
             "n_context": (np.asarray(feats["cidx"]) >= 0).sum(1), **meta}
    nan = np.full(n, np.nan)
    tabs = {c: [] for c in cells}
    for j, t in enumerate(targets):
        drop = added if t in base else np.zeros(n, bool)                   # §5: no added row under a base target
        has = np.isfinite(y[:, j]) & ~drop
        col = {"target": np.full(n, t), **{k: frame[k] for k in META_COLS},
               "y": np.where(cens[:, j], np.nan, y[:, j]), "censored": cens[:, j],
               "y_dl": np.where(cens[:, j], y[:, j], np.nan)}
        if t in base:
            jb = base.index(t)
            col.update({k: v[:, jb] for k, v in preds[LK.V1_MODEL].items() if k != "q"})
            col.update({qcol(LK.V1_MODEL, lv): v[:, jb] for lv, v in preds[LK.V1_MODEL]["q"].items()})
        else:
            col.update({k: nan for k in preds[LK.V1_MODEL] if k != "q"})
            col.update({qcol(LK.V1_MODEL, lv): nan for lv in INTERVAL_QS})
        for model, pr in ((LK.V2_MODEL, (t2, c2)), (LK.TRACK_N_MODEL, tn)):
            if pr is None:
                continue
            ja = pr[0].index(t) if t in pr[0] else None
            col.update({k: (v[:, ja] if ja is not None else nan) for k, v in pr[1].items() if k != "q"})
            col.update({qcol(model, lv): (v[:, ja] if ja is not None else nan) for lv, v in pr[1]["q"].items()})
        for c in S.CLAIM:
            col[CL.cmp_col(c)] = np.asarray(preds["cmp"].get(t, {}).get(c, nan), float)
        tab = pd.DataFrame(col)
        for cell, m in cells.items():
            tabs[cell].append(tab[has & m])
    return {c: pd.concat(v, ignore_index=True) for c, v in tabs.items()}


def predict_and_seal(freeze: dict, info: dict, rows, dev: pd.DataFrame, cells: dict, *, root: str, register: bool,
                     device, run=None, ntu_base=None) -> dict:
    """Step 3. ``rows`` = (table, pixels, features, context cache) holding the rows of every cell; ``dev``: the
    development table (keys and coordinates only: reference scenes, train-role points)."""
    df, pix, feats, ctx_cache = rows
    missing = [c for c in ROW_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"the row table lacks {missing}")
    art = freeze["artifacts"]
    use = np.zeros(len(df), bool)
    for m in cells.values():
        use |= np.asarray(m, bool)
    cells = {c: np.asarray(m, bool)[use] for c, m in cells.items()}
    df, pix, feats = subset(df, pix, feats, use)
    ref, ref_rec = TC.reference_scenes(dev, info["fit_scheme"])
    disjoint, dj_rec = TC.scene_disjoint(TC.row_keys(df), ref)
    meta = {"scene_disjoint": disjoint, "small_lake": S.small_lake(df), "ntu_class": S.ntu_class(df, ntu_base),
            "dist_train_km": TC.dist_train_km(df, TC.train_points(dev))}
    preds = {LK.V1_MODEL: predict_v1(art["v1"], df, pix, device)}
    nets = {}
    for model in ARM_MODELS:
        if not art.get(model):
            continue
        same = model == LK.TRACK_N_MODEL and art[model] == art[LK.V2_MODEL]        # §2.7: nothing is added
        nets[model] = nets[LK.V2_MODEL] if same else arm_networks(art[model], df, pix, feats, ctx_cache, device)
        preds[model] = (nets[model][0], arm_columns(model, nets[model][1]))
    preds["cmp"] = predict_comparators(info["comparator_root"], info["bundle"], df, pix, feats, ctx_cache[1], device)
    tabs = long_tables(df, feats, cells, preds, meta)
    pred_dir = LK.look_dir(LOOK, root) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=False)
    for cell, tab in tabs.items():
        tab.to_parquet(pred_dir / f"cell_{cell}.parquet", index=False)
    rec = {"cells": {c: {"n_rows": int(m.sum()), "n_table_rows": int(len(tabs[c]))} for c, m in cells.items()},
           "reference_scenes": ref_rec, "scene_disjoint": dj_rec, "targets": sorted(set(tabs[next(iter(tabs))]["target"]))}
    PL.write_json(pred_dir / "predict.json", rec)
    sealed = LK.seal_predictions(LOOK, pred_dir, root, freeze=freeze, required=LK.sealed_columns(freeze["sealed_models"]),
                                 register=register)
    if run is not None:
        run.note(f"predictions sealed: {sealed['n_files']} files, sha256sums {sealed['sha256sums'][:12]}; rows per "
                 f"cell {rec['cells']} (counts only)")
    return {"record": rec, "rows": (df, feats), "nets": nets, "tables": tabs}


# ------------------------------------------------------------------------------------------------ scoring

def flatten(fams: dict, analysis: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one row per endpoint x comparator, one row per endpoint x member) of ``claims.all_families``."""
    rows, mem = [], []
    for name, f in fams.items():
        for (t, arm), e in f["endpoints"].items():
            base = {"analysis": analysis, "family": name, "divisor": f["divisor"], "target": t, "arm": arm,
                    "status": e["status"], "confirmatory": e["confirmatory"], "n": e["n"],
                    "n_labelled": e["n_labelled"], "n_dropped_nonfinite": e["n_dropped_nonfinite"],
                    "n_blocks": e["n_blocks"], "mdsa_arm": e["mdsa"].get(arm), "p_iu": e["p_iu"],
                    "all_favour": e["all_favour"], "per_seed_passed": bool(e["per_seed"].get("passed")),
                    "per_seed_sentence": e["per_seed"].get("sentence"),
                    "label": bool(e["label"]) if analysis == "primary" else None}
            for c in e["comparators"]:
                pr, hm = e["pairs"].get(c, {}), (e.get("holm") or {})
                h = hm.get("iu") or hm.get(c) or {}
                rows.append({**base, "comparator": c, "mdsa_comparator": e["mdsa"].get(CL.cmp_col(c)),
                             "diff": pr.get("diff"), "ci_low": pr.get("ci_low"), "ci_high": pr.get("ci_high"),
                             "p_value": pr.get("p_value"), "p_holm": h.get("p_holm"), "rejected": h.get("rejected")})
            mem += [{"analysis": analysis, "family": name, "target": t, "arm": arm, "member": k, "mdsa": v,
                     "n": e["n"]} for k, v in e["mdsa"].items() if not k.startswith(LK.COMPARATOR_PREFIX)]
    return pd.DataFrame(rows), pd.DataFrame(mem)


def point_models(tab: pd.DataFrame) -> list:
    return [c for c in tab.columns if c in (LK.V1_MODEL, *ARM_MODELS) or c.startswith(LK.COMPARATOR_PREFIX)]


def descriptive_tables(tab: pd.DataFrame, cell: str, n_boot: int) -> pd.DataFrame:
    out = []
    for t in sorted(tab["target"].unique()):
        g = CL.endpoint_rows(tab, t)
        if len(g):
            out.append(CL.descriptive(g, point_models(tab), n_boot=n_boot).assign(target=t, cell=cell))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def interval_tables(tab: pd.DataFrame, cell: str, offsets: dict | None) -> pd.DataFrame:
    """Coverage and width of the mixture intervals of v1 and v2, raw and (when the freeze names validation-fitted
    offsets) split-conformal."""
    rows = []
    for t in sorted(tab["target"].unique()):
        g = CL.endpoint_rows(tab, t)
        for model in (LK.V1_MODEL, LK.V2_MODEL):
            for lvl, (lo, hi) in CF.LEVELS.items():
                if qcol(model, lo) not in g.columns:
                    continue
                off = ((offsets or {}).get(model) or {}).get(t, {}).get(lvl)
                for kind, o in (("raw", None), ("split_conformal", off)):
                    if kind == "raw" or o is not None:
                        rows.append({"cell": cell, "target": t, "model": model, "level": lvl, "kind": kind,
                                     **CL.interval_table(g, qcol(model, lo), qcol(model, hi), o)})
    return pd.DataFrame(rows)


def censored_table(tab: pd.DataFrame, cell: str) -> pd.DataFrame:
    """Left-censored rows (§6.4): share of predictions at or below the row's detection limit."""
    rows = []
    c = tab[tab["censored"].to_numpy(bool)]
    for t, g in c.groupby("target"):
        for m in point_models(tab):
            p = g[m].to_numpy(float)
            ok = np.isfinite(p)
            if ok.any():
                rows.append({"cell": cell, "target": t, "model": m, "n_censored": int(ok.sum()),
                             "share_at_or_below_dl": float(np.mean(p[ok] <= g["y_dl"].to_numpy(float)[ok]))})
    return pd.DataFrame(rows)


def score(pred_dir: Path, freeze: dict, cells: dict, n_boot: int) -> dict:
    """Step 4, from the sealed tables only."""
    lake = pd.read_parquet(pred_dir / f"cell_{cells['lake']}.parquet")
    nonlake = pred_dir / f"cell_{cells['nonlake']}.parquet"
    sm, primary, ps = freeze["sealed_models"], freeze["test_s_primary_cell"], freeze["power_status"]
    n1, has_tn = int(sm["v1_networks"]), bool(sm.get("track_n"))
    fam = CL.all_families(CL.primary_rows(lake, primary), ps, n1, has_track_n=has_tn, n_boot=n_boot)
    sens = CL.all_families(CL.primary_rows(lake, primary, sensitivity=True), ps, n1, has_track_n=has_tn, n_boot=n_boot)
    e1, m1 = flatten(fam, "primary")
    e2, m2 = flatten(sens, "sensitivity")
    offsets = None
    if (freeze.get("artifacts") or {}).get("conformal"):
        offsets = json.loads(resolve(freeze["artifacts"]["conformal"]).read_text())
    desc = [descriptive_tables(lake, cells["lake"], n_boot)]
    ints, cen = [interval_tables(lake, cells["lake"], offsets)], [censored_table(lake, cells["lake"])]
    if nonlake.is_file():
        nl = pd.read_parquet(nonlake)
        desc.append(descriptive_tables(nl, cells["nonlake"], n_boot))
        ints.append(interval_tables(nl, cells["nonlake"], offsets))
        cen.append(censored_table(nl, cells["nonlake"]))
    labels = {name: [{"target": t, "arm": a, "label": f"{CL.label_scope(primary)}",
                      "per_seed": f["endpoints"][t, a]["per_seed"]["sentence"]}   # R-17, as scored
                     for t, a in f["labels"]] for name, f in fam.items()}
    summary = {"look": LOOK, "primary_cell": primary, "scope": CL.label_scope(primary),
               "counts_sentence": CL.counts_sentence(fam), "n_boot": int(n_boot), "labels": labels,
               "n_labels": int(sum(len(v) for v in labels.values())),
               "n_hypotheses": {k: f["n_hypotheses"] for k, f in fam.items()},
               "under_powered": {k: sorted({t for (t, _), e in f["endpoints"].items() if not e["confirmatory"]})
                                 for k, f in fam.items()}}
    return {"summary": summary,
            "tables": {"endpoints": pd.concat([e1, e2], ignore_index=True),
                       "members": pd.concat([m1, m2], ignore_index=True),
                       "descriptive": pd.concat(desc, ignore_index=True),
                       "intervals": pd.concat(ints, ignore_index=True),
                       "censored": pd.concat(cen, ignore_index=True)}}


def write_scores(res: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    for name, tab in res["tables"].items():
        tab.to_csv(out_dir / f"{name}.csv", index=False)
    PL.write_json(out_dir / "summary.json", res["summary"])


def register_scores(res: dict, freeze: dict, cells: dict, out_dir: Path) -> int:
    """Every reported number goes through the registry: endpoint scores, paired differences, labels, descriptive
    scores."""
    exp = LK.experiment(LOOK, freeze["data_version"])
    common = {"workstream": LK.WORKSTREAM, "experiment": exp, "data_manifests": [f"orbit_{freeze['data_version']}"],
              "artifacts": [str(out_dir)], "config": {"freeze_sha256": freeze["_sha256"]}}
    ci = f"paired role-block bootstrap (B={res['summary']['n_boot']}, seed {CL.SEED}; +1-corrected p, R-17a)"
    n, seen = 0, set()
    for r in res["tables"]["endpoints"].to_dict("records"):
        if r["diff"] is None or not np.isfinite(r["diff"]):
            continue
        cell = f"{LOOK} {cells['lake']} ({r['analysis']}: {res['summary']['primary_cell']}; {r['family']})"
        for model, val, base in ((r["arm"], r["mdsa_arm"], False), (CL.cmp_col(r["comparator"]), r["mdsa_comparator"], True)):
            if (cell, r["target"], model) not in seen:
                seen.add((cell, r["target"], model))
                _log_result(model=model, metric="mdsa", value=float(val), target=r["target"], split=cell,
                            n=int(r["n"]), is_baseline=base, notes=f"status {r['status']}", **common)
                n += 1
        _log_result(model=r["arm"], metric=f"mdsa_diff_vs_{r['comparator']}", value=float(r["diff"]),
                    ci_low=float(r["ci_low"]), ci_high=float(r["ci_high"]), ci_method=ci, target=r["target"],
                    split=cell, n=int(r["n"]), notes=f"status {r['status']}; negative favours the arm",
                    extra={k: r[k] for k in ("p_value", "p_holm", "rejected", "p_iu", "all_favour", "per_seed_passed",
                                             "label", "n_blocks", "divisor", "confirmatory")}, **common)
        n += 1
    for r in res["tables"]["members"].to_dict("records"):
        if r["member"] != r["arm"] and r["analysis"] == "primary":
            _log_result(model=r["member"], metric="mdsa", value=float(r["mdsa"]), target=r["target"], n=int(r["n"]),
                        split=f"{LOOK} {cells['lake']} (primary: {res['summary']['primary_cell']}; {r['family']})",
                        notes="single member (R-17); descriptive", **common)
            n += 1
    d = res["tables"]["descriptive"]
    for r in (d[d["model"].notna()].to_dict("records") if len(d) else []):
        _log_result(model=r["model"], metric="mdsa", value=float(r["mdsa"]), ci_low=float(r["mdsa_ci_low"]),
                    ci_high=float(r["mdsa_ci_high"]), ci_method="role-block cluster bootstrap (percentile)",
                    target=r["target"], split=f"{LOOK} {r['cell']} / {r['stratum']} (descriptive)",
                    n=int(r["n_common"]), is_baseline=str(r["model"]).startswith(LK.COMPARATOR_PREFIX),
                    notes="descriptive (§6.4)", **common)
        n += 1
    for r in res["tables"]["intervals"].to_dict("records"):
        if r["n"]:
            _log_result(model=r["model"], metric=f"coverage{r['level']}_{r['kind']}", value=float(r["coverage"]),
                        target=r["target"], split=f"{LOOK} {r['cell']} (descriptive)", n=int(r["n"]),
                        extra={"median_width_log10": r["median_width_log10"]}, notes="descriptive (§6.4)", **common)
            n += 1
    _log_result(model="orbit_v2_test_look", metric="n_labels", value=float(res["summary"]["n_labels"]),
                split=f"{LOOK} labels", notes=res["summary"]["counts_sentence"] + "; " + res["summary"]["scope"],
                extra={"labels": res["summary"]["labels"]}, **common)
    return n + 1


# ---------------------------------------------------------------------------------------------- dry run

def dry_freeze(candidate: Path, label: str, version: str) -> tuple[dict, list]:
    """The candidate's entries as a stand-in freeze for a dry run on validation rows. It never unlocks a
    protected row (``looks.load_freeze`` is not involved) and nothing is registered."""
    if not Path(candidate).is_file():
        raise SystemExit(f"{candidate}: no candidate file (a dry run reads the artefacts it names)")
    cand = json.loads(Path(candidate).read_text())
    for k in ("artifacts", "sealed_models", "weights_sha256", "v1_weights_sha256"):
        if not isinstance(cand.get(k), dict):
            raise SystemExit(f"{candidate}: no {k!r} entry")
    assumed = []
    if cand.get("test_s_primary_cell") not in LK.PRIMARY_CELLS:
        cand["test_s_primary_cell"] = "all_rows"
        assumed.append("test_s_primary_cell = all_rows")
    if not isinstance(cand.get("power_status"), dict):
        cand["power_status"] = {f: {t: {"status": "primary" if f == "H1-S" else "confirmatory"} for t in v["targets"]}
                                for f, v in CL.FAMILIES.items()}
        assumed.append("every endpoint confirmatory")
    return {**cand, "_sha256": f"dryrun:{label}", "_path": None, "data_version": version,
            "scoring_code_sha256": LK.scoring_code_sha256()}, assumed


def dry_cells(df: pd.DataFrame) -> dict:
    ok = ((df["role"].astype(str) == "val").to_numpy() & np.isin(df["tblock"].to_numpy(), ["train", "val"])
          & ~df["buffer_excluded"].to_numpy(bool))
    lake = df["in_lake"].to_numpy(bool)
    return {DRY_CELLS["lake"]: ok & lake, DRY_CELLS["nonlake"]: ok & ~lake}


def parity(art: dict, sealed: dict, info: dict) -> dict:
    """Dry run only: the networks' medians predicted here equal those of the stored validation mixtures, and
    the comparator columns equal ``val_rows.parquet``."""
    df = sealed["rows"][0]
    rid = df["row_id"].astype(str).to_numpy()
    out = {}
    for model, (targets, nets) in sealed["nets"].items():
        asm = U.assemble(resolve(art[model]["root"]), art[model]["arm"])
        pos = pd.Series(np.arange(len(asm["row_id"])), index=asm["row_id"])
        both = np.isin(rid, asm["row_id"])
        if not both.any() or asm["targets"] != targets:
            raise SystemExit(f"{model}: no stored validation prediction of the dry-run rows")
        ix = pos.loc[rid[both]].to_numpy()
        worst = max(float(np.nanmax(np.abs(U.medians(tuple(a[ix] for a in asm["nets"][k]))
                                           - U.medians(tuple(a[both] for a in nets[k]))))) for k in nets)
        out[model] = {"n_rows": int(both.sum()), "max_abs_diff_log10": worst}
        if not worst <= PARITY_TOL:
            raise SystemExit(f"{model}: predictions differ from the stored validation predictions by {worst:.4g}")
    val = pd.read_parquet(info["comparator_root"] / "val_rows.parquet")
    val = val[val["cell"] == DRY_CELLS["lake"]]
    tab = sealed["tables"][DRY_CELLS["lake"]].merge(val, on=["target", "row_id"], suffixes=("", "|stored"))
    worst = 0.0
    for c in S.CLAIM:
        a, b = tab[CL.cmp_col(c)].to_numpy(float), tab[CL.cmp_col(c) + "|stored"].to_numpy(float)
        if (np.isfinite(a) != np.isfinite(b)).any():
            raise SystemExit(f"comparator {c}: predicted rows differ from val_rows.parquet")
        worst = max(worst, float(np.nanmax(np.abs(a - b))) if np.isfinite(a).any() else 0.0)
    out["comparators"] = {"n_rows": int(len(tab)), "max_abs_diff_log10": worst}
    if not len(tab) or not worst <= PARITY_TOL:
        raise SystemExit(f"comparators: predictions differ from val_rows.parquet by {worst:.4g} on {len(tab)} rows")
    return out


# ------------------------------------------------------------------------------------------------- main

def protected_loader():
    """The look's own row loader (written after the independent review of the development loader, R-21). Its
    ``load(freeze)`` returns (table, pixels, features, context cache, record) in the dev_v5m format, holding
    Test-S rows only, after C-1, read under the open claim."""
    if importlib.util.find_spec(LOADER) is None:
        raise SystemExit(f"{LOADER} does not exist: the Test-S rows cannot be read; nothing was claimed")
    return importlib.import_module(LOADER)


def _terminate(signum, frame):
    raise SystemExit(f"signal {signum}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-test", action="store_true", help="required for the real look")
    ap.add_argument("--dry-run", default=None, metavar="LABEL", help="validation rows, no registry row")
    ap.add_argument("--candidate", default=str(LK.CANDIDATE_FILE), help="dry run: the candidate file")
    ap.add_argument("--n-boot", type=int, default=CL.N_BOOT, help="dry run only; the look uses B = 2,000 (R-17a)")
    ap.add_argument("--rescore-from-saved", action="store_true",
                    help="score sealed predictions of a look that stopped after sealing and before any score row")
    ap.add_argument("--gpu", action="store_true", help="lock one GPU with the shared helper (otherwise CPU)")
    a = ap.parse_args(argv)
    dry = a.dry_run is not None
    if dry and (a.allow_test or a.rescore_from_saved):
        raise SystemExit("--dry-run reads validation rows only; it takes neither --allow-test nor --rescore-from-saved")
    if not dry and a.n_boot != CL.N_BOOT:
        raise SystemExit(f"the look uses B = {CL.N_BOOT} (R-17a)")
    root = f"{LK.DRYRUN_ROOT}/{a.dry_run}" if dry else LK.ROOT
    cells = DRY_CELLS if dry else REAL_CELLS
    # ---- step 1: everything that can fail without a protected row
    if dry:
        freeze, assumed = dry_freeze(Path(a.candidate), a.dry_run, DM.VERSION)
    else:
        freeze, assumed = LK.load_freeze(), []
        if not a.allow_test:
            raise SystemExit("the real look needs --allow-test")
        loader = None if a.rescore_from_saved else protected_loader()
        LK.verify_data_pins(freeze)
    info = check_artifacts(freeze)
    if a.gpu:
        PL.setup_gpu("orbit_v2_test_s")
    import torch

    torch.set_num_threads(2 if a.gpu else 6)
    device = torch.device("cuda" if a.gpu and torch.cuda.is_available() else "cpu")
    name = f"orbit_v2_test_s/{'dryrun_' + a.dry_run if dry else 'M2S_test'}"
    with TrainingRun(LK.WORKSTREAM, name, config={"look": LOOK, "dry_run": a.dry_run, "n_boot": a.n_boot,
                                                    "freeze_sha256": freeze["_sha256"], "assumed": assumed},
                     data=[f"orbit_{freeze['data_version']}"], plan_ref=PLAN_REF, iteration="arch_v2",
                     artifacts_dir=str(LK.look_dir(LOOK, root))) as run:
        if a.rescore_from_saved:
            LK.require_no_scores(LOOK, freeze)
            sealed = None
        else:
            base, _, _ = D.load_dev(DM.BASE_VERSION)
            dev_rows = DM.load(base=base)                           # development rows only (training guard)
            dev = dev_rows[0]
            TC.reference_scenes(dev, info["fit_scheme"])            # fails here, not inside the claim
            LK.claim(LOOK, freeze, root=root, register=not dry)     # ---- step 2
            signal.signal(signal.SIGTERM, _terminate)
            try:                                                    # ---- step 3
                if dry:
                    rows, ntu_base = dev_rows[:4], base
                    cmask = dry_cells(dev)
                else:
                    df, pix, feats, ctx_cache, rec = loader.load(freeze)
                    LK.require_test_s_only(df, "Test-S rows")
                    if not rec.get("c1"):
                        raise SystemExit("the row loader did not record the C-1 exclusion")
                    cell = LK.test_s_cell(df)
                    rows, ntu_base = (df, pix, feats, ctx_cache), None
                    cmask = {c: cell == c for c in LK.CELLS}
                sealed = predict_and_seal(freeze, info, rows, dev, cmask, root=root, register=not dry, device=device,
                                          run=run, ntu_base=ntu_base)
            except BaseException as e:
                LK.mark_aborted(LOOK, freeze, repr(e), root=root, register=not dry)
                raise
        pred_dir = LK.check_sealed(LOOK, root)                      # ---- step 4: sealed files only
        res = score(pred_dir, freeze, cells, a.n_boot)
        if dry:
            res["summary"].update(dry_run=a.dry_run, assumed=assumed,
                                  parity=parity(freeze["artifacts"], sealed, info))
        out_dir = LK.look_dir(LOOK, root) / "scores"
        write_scores(res, out_dir)
        n_rows = 0
        if not dry:                                                 # ---- step 5
            LK.require_no_scores(LOOK, freeze)
            n_rows = register_scores(res, freeze, cells, out_dir)
        LK.mark_scored(LOOK, {**res["summary"], "n_registry_rows": n_rows, "scores": str(out_dir)}, root)
        run.metric("n_labels", res["summary"]["n_labels"])
        run.note(res["summary"]["counts_sentence"] + f"; scores in {out_dir}"
                 + ("; DRY RUN on validation rows: not a test result, nothing registered" if dry else ""))
    print(f"{'DRY RUN ' if dry else ''}{LOOK}: scored; {res['summary']['counts_sentence']}; tables in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
