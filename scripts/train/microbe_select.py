#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Apply the pre-registered selection rule (protocol §6) to development LODO results.

Among the DNA-encoded candidates with an OOF file, pick the highest mean pooled LODO R² over the
targets that every candidate predicts; ties within 0.005 -> fewer parameters (SeqEmb-Ridge <
SeqEmb-RF < SeqSet by parameter count from the fold diagnostics). Writes
``$SENTINEL_RUNS/microbe/phase1_selection{tag}.json`` and one registry entry per candidate
(experiment 'microbe_phase1_selection', metric 'mean_lodo_r2'), with the number of configurations
tried so far.

    python -B scripts/train/microbe_select.py --candidates seqset_dnabert seqset_kmer seqset_id ridge_seqemb rf_seqemb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.core.registry import log_result
from sentinel.eval import metrics as EM

COMPLEXITY = {"ridge_seqemb": 1, "rf_seqemb": 2}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--configs-tried", type=int, default=None, help="total DNA-candidate configurations so far")
    a = ap.parse_args(argv)
    from sentinel.models.microbe.baselines import LABELS
    from sentinel.models.microbe.design import TARGET_NAMES, load_design

    S = load_design().samples
    d = paths.RUNS_ROOT / "microbe" / f"cv{a.tag}" / "lodo"
    oofs = {c: pd.read_parquet(d / f"{c}.parquet") for c in a.candidates if (d / f"{c}.parquet").exists()}
    targets = [t for t in TARGET_NAMES if all(o[t].notna().sum() >= 10 for o in oofs.values())
               and S[t].notna().sum() >= 10]
    rec = {}
    for c, o in oofs.items():
        r2 = {t: EM.r2(S.loc[o.index, t].to_numpy(float), o[t].to_numpy(float)) for t in targets}
        n_params = None
        diag = d / f"{c}_diag.jsonl"
        if diag.exists():
            n_params = int(json.loads(diag.read_text().splitlines()[0])["n_params"])
        rec[c] = {"mean_r2": float(np.mean(list(r2.values()))), "r2": r2, "n_params": n_params,
                  "complexity": COMPLEXITY.get(c, 10 + (n_params or 0) / 1e9)}
    best = max(rec.values(), key=lambda r: r["mean_r2"])["mean_r2"]
    tied = [c for c, r in rec.items() if r["mean_r2"] >= best - 0.005]
    selected = min(tied, key=lambda c: rec[c]["complexity"])
    out = {"selected": selected, "targets_used": targets, "candidates": rec, "tied_within_0.005": tied,
           "configs_tried": a.configs_tried or len(rec), "rule": "protocol §6"}
    (paths.ensure(paths.RUNS_ROOT / "microbe") / f"phase1_selection{a.tag}.json").write_text(json.dumps(out, indent=1))
    for c, r in rec.items():
        log_result(workstream="M3-microbe", experiment="microbe_phase1_selection",
                   model=LABELS.get(c, f"SeqSet[{c}]"), metric="mean_lodo_r2", value=r["mean_r2"],
                   split="neon_dev/lodo_cv", n=len(targets), data_manifests=["microbe_wq"],
                   extra={"selected": c == selected, "targets": targets, "per_target_r2": r["r2"],
                          "configs_tried": out["configs_tried"], "tag": a.tag})
    print(json.dumps({"selected": selected, "means": {c: round(r["mean_r2"], 4) for c, r in rec.items()},
                      "targets": targets}, indent=1))


if __name__ == "__main__":
    main()
