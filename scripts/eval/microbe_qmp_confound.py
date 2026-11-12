#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Label-free confound check of the QMP input block (amendment microbe_arch_v2 §5 kill rule; arm D4).

For each QMP feature: the share of its variance explained by the qPCR laboratory and by the qPCR batch
(one-way between-group sum of squares / total sum of squares over the events with the feature and the grouping
known). Inputs are ``qmp_features.parquet`` and ``qmp_meta.parquet`` only: no target table is opened, so no
label enters. The result is reported with every +QMP result.

Kill rule: if laboratory alone explains more than 50% of the variance of the load index, +QMP results are
descriptive only. Reading (``microbe_arch_v2_deviations_01.md`` R17 with ``microbe_arch_v2_dev01_lead_rulings.md``
D5-D7): the rule fires if the share exceeds 0.5 on the development events OR on all events of the four blocks
pooled, and also if either share is not finite (missing laboratory, fewer than 2 groups, zero variance). If it
fires, G10 and every +QMP twin are reported as descriptive, G10 is not selectable as M*_GEO and still counts as a
GEO entry. The script runs ONCE: it refuses if its output exists, and the first output stands.

    PYTHONPATH=. python -B scripts/eval/microbe_qmp_confound.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.core.registry import log_result

THRESHOLD = 0.5


def eta2(x: np.ndarray, g: np.ndarray) -> dict:
    """Between-group share of the variance of ``x`` for the grouping ``g`` (rows with both known)."""
    x = np.asarray(x, float)
    g = pd.Series(g).astype("object")
    ok = np.isfinite(x) & g.notna().to_numpy()
    x, g = x[ok], g[ok].astype(str).to_numpy()
    out = {"n": int(ok.sum()), "n_groups": int(len(np.unique(g))) if ok.any() else 0, "share": float("nan")}
    if out["n"] < 3 or out["n_groups"] < 2:
        return out
    tot = float(((x - x.mean()) ** 2).sum())
    if tot <= 0:
        return out
    s = pd.Series(x).groupby(g)
    between = float((s.size() * (s.mean() - x.mean()) ** 2).sum())
    out["share"] = between / tot
    return out


def confound(features: pd.DataFrame, meta: pd.DataFrame, ids, feature_names) -> dict:
    ids = pd.Index(np.asarray(ids)).astype(str)
    f, m = features.reindex(ids), meta.reindex(ids)
    return {c: {"laboratory": eta2(f[c].to_numpy(float), m["laboratory"].to_numpy()),
                "batch": eta2(f[c].to_numpy(float), m["batch"].to_numpy())} for c in feature_names}


def rule_fired(lab_shares: dict) -> bool:
    """True if any laboratory share of the load index exceeds the threshold or is not finite (ruling D6)."""
    return bool(any((not np.isfinite(v)) or v > THRESHOLD for v in lab_shares.values()))


def main(argv=None) -> int:
    from sentinel.data import microbe_geo as G
    from sentinel.models.microbe import stamp as ST

    st = ST.require_stamp("QMP confound check (arm D4)")
    p = paths.RUNS_ROOT / "microbe" / "cv_archv2" / "qmp_confound.json"
    if p.exists():
        raise RuntimeError(f"{p} exists: the confound check runs once and its first output stands (ruling D7)")
    d = G.dataset_dir()
    features = pd.read_parquet(d / "qmp_features.parquet").set_index("sample_id")
    meta = pd.read_parquet(d / "qmp_meta.parquet").set_index("sample_id")
    features.index, meta.index = features.index.astype(str), meta.index.astype(str)
    ids = G.load_ids()
    blocks = {"development": ids["dev"], "all_blocks": [i for b in G.BLOCKS for i in ids[b]]}
    res = {k: confound(features, meta, v, G.QMP_FEATURES) for k, v in blocks.items()}
    lab = {k: res[k]["qmp_load"]["laboratory"]["share"] for k in res}
    descriptive = rule_fired(lab)
    out = {"rule": f"laboratory share of the variance of qmp_load > {THRESHOLD}, or not finite, on development "
                   "events or on all events -> +QMP results are descriptive only (amendment §5; dev01 rulings D5, D6)",
           "stamp_row": st["row_id"], "shares": res, "laboratory_share_of_load_index": lab,
           "descriptive_only": descriptive}
    paths.ensure(p.parent)
    with p.open("x") as fh:
        fh.write(json.dumps(out, indent=1, default=float))
    for blk, r in res.items():
        for c, by in r.items():
            for grp, e in by.items():
                if np.isfinite(e["share"]):
                    log_result(workstream="M3-microbe", experiment="microbe_arch_v2_qmp_confound",
                               model="QMP input block", metric=f"variance_share_{grp}", value=e["share"],
                               split=f"neon/{blk}", target=c, n=e["n"], is_baseline=False,
                               config={"n_groups": e["n_groups"], "threshold": THRESHOLD},
                               data_manifests=["microbe_wq_geo"], artifacts=[str(p)],
                               notes="label-free confound check of the QMP block (amendment §5); no target read")
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
