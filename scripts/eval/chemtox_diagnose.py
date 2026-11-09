#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Why does the D-MPNN trail the tree models? Out-of-fold error anatomy, never test rows.

Compares two models' out-of-fold predictions on the official CV folds and breaks the error down by
the axes that matter for this task:

* per **taxon** (species) and per **taxon group**, with counts — the macro metrics weight these
  equally, so a model can win micro and lose macro;
* per **chemical**: error vs how often the chemical was tested, and the worst chemicals for each
  model;
* by **observation duration** (24/48/72/96 h) and by concentration type / media type — the
  conditioning the deep model is supposed to exploit;
* by **target magnitude** (quantile bins of the true log10 LC50), which exposes regression to the
  mean;
* **bias** per slice, and how much of the gap is bias rather than scatter;
* the **paired** difference with a chemical-cluster bootstrap CI, so "trails" is quantified rather
  than asserted.

    python -B scripts/eval/chemtox_diagnose.py --a chemprop --b rf
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.data.sources import adore as A
from sentinel.eval.bootstrap import paired_bootstrap
from sentinel.eval.metrics import rmse
from sentinel.models.chemtox import cv as CV
from sentinel.models.chemtox import evaluate as E
from sentinel.models.chemtox import xgb as XG
from sentinel.models.chemtox.data import ChemToxDataset, FeatureSpec

log = logging.getLogger("chemtox_diagnose")

DEFAULTS = {
    "xgboost": ({"fingerprint": "MACCS"}, {"n_estimators": 800, "eta": 0.03, "max_depth": 6,
                                            "min_child_weight": 1, "subsample": 0.8}),
    "rf": ({"fingerprint": "MACCS"}, {"n_estimators": 300, "max_depth": 20, "max_features": 0.3}),
    "chemprop": ({"fingerprint": None}, {"max_epochs": 20, "batch_size": 64, "message_hidden": 300,
                                          "depth": 3, "ffn_hidden": 300, "ffn_layers": 2,
                                          "species_dim": 16, "dropout": 0.0}),
}


def get_oof(name: str, ds: ChemToxDataset, *, accelerator: str, search_dir: Path | None) -> np.ndarray:
    """Out-of-fold predictions: reuse a saved search/CV run if present, else recompute the default."""
    if search_dir is not None:
        hits = sorted(search_dir.glob(f"oof_{name}_*.npy"))
        if hits:
            log.info("%s: using saved out-of-fold predictions %s", name, hits[-1].name)
            return np.load(hits[-1])
    features, params = DEFAULTS[name]
    spec = FeatureSpec(**features)
    if name == "chemprop":
        from sentinel.models.chemtox.chemprop_model import make_chemprop

        params = dict(params) | {"accelerator": accelerator}
        maker = make_chemprop(0)
    else:
        maker = XG.make_xgb(0) if name == "xgboost" else XG.make_rf(0)
    log.info("%s: recomputing out-of-fold predictions (%s)", name, params)
    r = CV.run_cv(ds, spec, maker, params, model_name=name)
    return r.oof


def slice_table(df: pd.DataFrame, by: str, *, a: str, b: str, min_n: int = 20) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, dropna=False):
        if len(g) < min_n:
            continue
        rows.append({
            by: key, "n": len(g),
            f"rmse_{a}": float(np.sqrt(np.mean(g["ea"] ** 2))),
            f"rmse_{b}": float(np.sqrt(np.mean(g["eb"] ** 2))),
            f"bias_{a}": float(np.mean(g["ea"])),
            f"bias_{b}": float(np.mean(g["eb"])),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["delta_rmse"] = out[f"rmse_{a}"] - out[f"rmse_{b}"]
        out = out.sort_values("n", ascending=False)
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="chemprop", choices=sorted(DEFAULTS))
    ap.add_argument("--b", default="rf", choices=sorted(DEFAULTS))
    ap.add_argument("--challenge", default="t-F2F")
    ap.add_argument("--split", default=A.PRIMARY_SPLIT)
    ap.add_argument("--accelerator", default="cpu", choices=("cpu", "gpu"))
    ap.add_argument("--use-search-oof", action="store_true",
                    help="reuse out-of-fold predictions saved by the search runs")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--oof-a", default=None, help="explicit .npy of model A's out-of-fold predictions")
    ap.add_argument("--oof-b", default=None, help="explicit .npy of model B's out-of-fold predictions")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = paths.run_dir("chemtox", f"diagnose/{args.a}_vs_{args.b}_{args.challenge}_{args.split}")

    spec_cols = ("chem_rdkit_can_smiles", "tax_group", "test_media_type", "result_conc1_type")
    ds = ChemToxDataset.load(args.challenge, args.split, specs=FeatureSpec(fingerprint="MACCS"),
                             extra_columns=spec_cols)
    ds.assert_no_chemical_leakage(log=log)
    sdir = (paths.RUNS_ROOT / "chemtox" / "search" / f"{args.a}_{args.challenge}_{args.split}"
            if args.use_search_oof else None)
    if args.oof_a:
        pa = np.load(args.oof_a)
        log.info("%s: out-of-fold predictions from %s", args.a, args.oof_a)
    else:
        pa = get_oof(args.a, ds, accelerator=args.accelerator, search_dir=sdir)
    sdir_b = (paths.RUNS_ROOT / "chemtox" / "search" / f"{args.b}_{args.challenge}_{args.split}"
              if args.use_search_oof else None)
    if args.oof_b:
        pb = np.load(args.oof_b)
        log.info("%s: out-of-fold predictions from %s", args.b, args.oof_b)
    else:
        pb = get_oof(args.b, ds, accelerator=args.accelerator, search_dir=sdir_b)
    if len(pa) != len(ds.df) or len(pb) != len(ds.df):
        raise ValueError("out-of-fold arrays must align with the challenge file's rows")

    y = ds.y
    m = np.isfinite(y) & np.isfinite(pa) & np.isfinite(pb)
    df = pd.DataFrame({
        "y": y[m], "pa": pa[m], "pb": pb[m],
        "chem": ds.chem[m], "taxon": ds.taxon[m],
        "tax_group": ds.df.loc[m, "tax_group"].astype("string").to_numpy(),
        "duration": pd.to_numeric(ds.df.loc[m, "result_obs_duration_mean"], errors="coerce").to_numpy(),
        "media": ds.df.loc[m, "test_media_type"].astype("string").to_numpy(),
        "conc_type": ds.df.loc[m, "result_conc1_type"].astype("string").to_numpy(),
    })
    df["ea"] = df["pa"] - df["y"]
    df["eb"] = df["pb"] - df["y"]
    df["y_bin"] = pd.qcut(df["y"], 5, duplicates="drop").astype(str)
    tests_per_chem = df.groupby("chem")["y"].size()
    df["chem_tests_bin"] = pd.cut(df["chem"].map(tests_per_chem),
                                  [0, 2, 5, 10, 25, 10_000],
                                  labels=["1-2", "3-5", "6-10", "11-25", ">25"]).astype(str)

    overall = {
        args.a: E.report(df["y"], df["pa"], chem=df["chem"], taxon=df["taxon"]),
        args.b: E.report(df["y"], df["pb"], chem=df["chem"], taxon=df["taxon"]),
    }
    log.info("overall out-of-fold (%d rows): %s", len(df),
             {k: {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)}
              for k, v in overall.items()})

    pb_test = paired_bootstrap(rmse, df["y"].to_numpy(), df["pa"].to_numpy(), df["pb"].to_numpy(),
                              groups=df["chem"].to_numpy(), n_boot=args.n_boot,
                              higher_is_better=False)
    log.info("paired RMSE difference (%s - %s): %.4f, CI [%.4f, %.4f], p=%.4g (chemical clusters)",
             args.a, args.b, pb_test["diff"],
             pb_test["ci_low"], pb_test["ci_high"], pb_test.get("p_value", float("nan")))

    tables = {}
    for by in ("tax_group", "duration", "media", "conc_type", "y_bin", "chem_tests_bin"):
        t = slice_table(df, by, a=args.a, b=args.b)
        tables[by] = t
        if len(t):
            with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:.3f}"):
                print(f"\n=== by {by} ===")
                print(t.to_string(index=False))

    per_taxon = slice_table(df, "taxon", a=args.a, b=args.b, min_n=30)
    per_chem = slice_table(df, "chem", a=args.a, b=args.b, min_n=10)
    tables["taxon"] = per_taxon
    tables["chem"] = per_chem
    if len(per_taxon):
        print("\n=== taxa where the gap is largest (n>=30) ===")
        print(per_taxon.sort_values("delta_rmse", ascending=False).head(10).to_string(index=False))
    if len(per_chem):
        print("\n=== chemicals where the gap is largest (n>=10) ===")
        print(per_chem.sort_values("delta_rmse", ascending=False).head(10).to_string(index=False))

    for name, t in tables.items():
        if len(t):
            t.to_csv(out_dir / f"by_{name}.csv", index=False)
    CV.write_json(out_dir / "diagnosis.json", {
        "a": args.a, "b": args.b, "challenge": args.challenge, "split": args.split,
        "n_rows": len(df), "overall": overall, "paired_rmse_test": pb_test,
        "sd_of_errors": {args.a: float(df["ea"].std()), args.b: float(df["eb"].std())},
        "bias": {args.a: float(df["ea"].mean()), args.b: float(df["eb"].mean())},
        "variance_share_of_mse": {
            args.a: float(df["ea"].var() / np.mean(df["ea"] ** 2)),
            args.b: float(df["eb"].var() / np.mean(df["eb"] ** 2)),
        },
    })
    log.info("wrote %s", out_dir / "diagnosis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
