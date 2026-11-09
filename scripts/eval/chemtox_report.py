#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Print the ChemTox results table straight from the registry (no literals anywhere).

Every row is read with ``sentinel.core.registry.registry_frame()`` and filtered to
``workstream == 'chemtox'``, so the table cannot drift from what the code actually computed.
The published reference numbers and the replicate noise floor are printed alongside: the published
values come from the baseline entries' ``extra.published`` field written by the reproduction run,
the noise floor from the trivial-baseline entries (``metric == 'replicate_pooled_sd'``).

    python -B scripts/eval/chemtox_report.py
    python -B scripts/eval/chemtox_report.py --challenge t-F2F --split split_occurrence --csv out.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core.registry import registry_frame

METRICS = ("rmse", "mae", "r2", "rmse_macro", "rmse_chemical", "rmse_taxon", "within_1_log",
           "median_abs_fold_error")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--challenge", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--kind", default=None, choices=(None, "cv", "test"))
    ap.add_argument("--csv", default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    df = registry_frame()
    if df.empty:
        print("registry is empty")
        return 0
    df = df[df["workstream"] == "chemtox"].copy()
    if df.empty:
        print("no chemtox entries in the registry")
        return 0
    parts = df["split"].str.split("/", expand=True)
    df["challenge"] = parts[0]
    # two conventions exist in the registry: the trivial-baseline run logs 'occurrence', the
    # model runs log the column name 'split_occurrence'. Normalise so one table holds both.
    df["split_col"] = parts[1].str.replace(r"^split_", "", regex=True)
    df["kind"] = parts[2] if parts.shape[1] > 2 else None
    if args.challenge:
        df = df[df["challenge"] == args.challenge]
    if args.split:
        df = df[df["split_col"] == args.split.replace("split_", "")]
    if args.kind:
        df = df[df["kind"] == args.kind]

    floor = df[df["metric"] == "replicate_pooled_sd"]
    table = df[df["metric"].isin(METRICS)]
    wide = table.pivot_table(index=["challenge", "split_col", "kind", "experiment", "model",
                                    "is_baseline"],
                             columns="metric", values="value", aggfunc="last")
    order = [m for m in METRICS if m in wide.columns]
    wide = wide[order].sort_values(["challenge", "split_col", "kind", "rmse"])
    with pd.option_context("display.width", 200, "display.max_columns", 50,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print("\n=== ChemTox metrics (registry) ===")
        print(wide)

    pub = {}
    for _, row in df[df["is_baseline"] & df["extra"].notna()].iterrows():
        p = (row["extra"] or {}).get("published")
        if isinstance(p, dict):
            for k, v in p.items():
                pub[k] = v
    if pub:
        print("\n=== published reference (Gasser et al. 2024, t-F2F occurrence test) ===")
        for k, v in pub.items():
            print(f"  {k:<22} {v['value']:.3f}  ({v['model']})")
    if len(floor):
        print("\n=== replicate noise floor (log10 units) ===")
        for _, row in floor.iterrows():
            print(f"  {row['split']:<28} {row['value']:.3f}")

    ns = df[df["metric"] == "rmse"][["model", "kind", "extra"]]
    counts = {(r["model"], r["kind"]): (r["extra"] or {}).get("n_configurations")
              for _, r in ns.iterrows()}
    if counts:
        print("\n=== configurations tried (per model, as logged) ===")
        for (m, kind), n in sorted(counts.items(), key=lambda x: str(x[0])):
            print(f"  {m:<32} {kind or '-':<5} {n}")
    if args.csv:
        wide.reset_index().to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
