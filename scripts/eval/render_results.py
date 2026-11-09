#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Render registry-backed result tables into RESULTS.md (between AUTO markers).

Every number in the rendered tables comes from results/registry.jsonl (current entries only;
superseded entries are dropped). Baselines are listed alongside models. Run after any
experiment that logs results:

    python scripts/eval/render_results.py
"""

from __future__ import annotations

import math

import pandas as pd

from sentinel.core import paths
from sentinel.core.registry import registry_frame

BEGIN, END = "<!-- BEGIN AUTO-RESULTS -->", "<!-- END AUTO-RESULTS -->"


def fmt(v, lo, hi) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    s = f"{v:.4g}"
    if lo is not None and hi is not None and not (pd.isna(lo) or pd.isna(hi)):
        s += f" [{lo:.4g}, {hi:.4g}]"
    return s


def render(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Registry is empty — no results yet._\n"
    out = [f"_Rendered from `results/registry.jsonl` ({len(df)} current entries). "
           "Values: point [95% CI]; CI unit/method per entry in the registry._\n"]
    for (ws, exp), g in df.groupby(["workstream", "experiment"], sort=True):
        out.append(f"\n### {ws} / {exp}\n")
        g = g.assign(value_ci=[fmt(v, lo, hi) for v, lo, hi in zip(g["value"], g["ci_low"], g["ci_high"])],
                     role=g["is_baseline"].map({True: "baseline", False: "SENTINEL"}),
                     n=g["n"].map(lambda x: "" if pd.isna(x) else str(int(x))),
                     target=g["target"].fillna(""))
        tab = g.sort_values(["split", "target", "metric", "role", "model"])[
            ["split", "target", "metric", "model", "role", "value_ci", "n", "id"]]
        out.append(tab.to_markdown(index=False))
        out.append("")
    return "\n".join(out) + "\n"


def index(df: pd.DataFrame) -> str:
    """Compact per-experiment index for RESULTS.md (full tables go to results/RESULTS_TABLES.md)."""
    if df.empty:
        return "_Registry is empty — no results yet._\n"
    g = df.groupby(["workstream", "experiment"]).agg(
        entries=("id", "size"), models=("model", "nunique"),
        splits=("split", lambda s: ", ".join(sorted({x.split("/")[0] for x in s}))[:80]),
        last=("timestamp", "max")).reset_index()
    return (f"_{len(df)} current registry entries; full tables in `results/RESULTS_TABLES.md`._\n\n"
            + g.to_markdown(index=False) + "\n")


def main() -> int:
    df = registry_frame()
    (paths.RESULTS_DIR / "RESULTS_TABLES.md").write_text("# Registry tables (auto-generated)\n\n" + render(df))
    block = index(df)
    res = paths.REPO_ROOT / "RESULTS.md"
    text = res.read_text()
    if BEGIN in text and END in text:
        pre, rest = text.split(BEGIN, 1)
        _, post = rest.split(END, 1)
        text = pre + BEGIN + "\n" + block + END + post
    else:
        text += f"\n## Registry tables (auto-generated)\n\n{BEGIN}\n{block}{END}\n"
    res.write_text(text)
    print(f"rendered {0 if df.empty else len(df)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
