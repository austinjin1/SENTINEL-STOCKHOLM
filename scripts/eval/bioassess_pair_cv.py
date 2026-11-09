#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Paired development-CV comparison of two registered models (same OOF rows), logged to the registry.

Reads the current 'bioassess_phase1_cv' entries of ``--model`` and ``--reference`` on ``--split``,
computes the paired location-cluster bootstrap Δ (``sentinel.models.bioassess.evaluate.paired_vs``) for
pooled and anomaly R², and logs metric ``delta_<metric>_vs_<--slug>`` to experiment
'bioassess_phase1_selection'. ``--supersede`` maps "<target>:<metric>=<registry id>" for corrections.

    python -B scripts/eval/bioassess_pair_cv.py --model "<A>" --reference "<B>" --slug ens_rfctx_tabpfn \
        --targets log10_tn_mg_l [--supersede log10_tn_mg_l:r2=<id> ...]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core.registry import log_result, registry_frame  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--split", default="nrsa0814/loeo_cv")
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--supersede", nargs="*", default=[])
    a = ap.parse_args()
    from sentinel.models.bioassess import evaluate as E

    sup = {}
    for s in a.supersede:
        k, v = s.split("=")
        sup[tuple(k.split(":"))] = v
    df = registry_frame()
    df = df[(df["experiment"] == "bioassess_phase1_cv") & (df["split"] == a.split) & (df["metric"] == "r2")]
    df = df.sort_values("timestamp").groupby(["model", "target"], as_index=False).last()

    def oof(m, t):
        e = df[(df["model"] == m) & (df["target"] == t)].iloc[0]
        return e, next(x for x in e["artifacts"] if "/oof_" in x)

    for t in a.targets:
        (em, pm), (er, pr) = oof(a.model, t), oof(a.reference, t)
        o, r = pd.read_parquet(pm), pd.read_parquet(pr)
        if not (o["visit_id"].to_numpy() == r["visit_id"].to_numpy()).all():
            raise AssertionError("OOF rows differ")
        for metric in ("r2", "r2_anomaly"):
            pv = E.paired_vs(o[f"obs|{t}"], o[f"pred|{t}"], r[f"pred|{t}"], o["ag_eco9"], o["location_id"],
                             metric=metric, n_boot=a.n_boot)
            e = log_result(workstream="bioassess", experiment="bioassess_phase1_selection", model=a.model,
                           metric=f"delta_{metric}_vs_{a.slug}", value=pv["diff"], split=a.split,
                           n=int(o[f"obs|{t}"].notna().sum()), ci_low=pv["ci_low"], ci_high=pv["ci_high"],
                           ci_method=f"paired percentile cluster bootstrap, unit = location cluster, B={a.n_boot}",
                           target=t, data_manifests=["nars_streams"], artifacts=[pm, pr],
                           supersedes=sup.get((t, metric)),
                           notes=f"Phase-1 CV (development data): {a.model} minus {a.reference}, same OOF rows.",
                           extra={"p_value": pv["p_value"], "reference_model": a.reference,
                                  "model_entry": em["id"], "reference_entry": er["id"]})
            print(t, metric, round(pv["diff"], 4), round(pv["ci_low"], 4), round(pv["ci_high"], 4), e["id"],
                  "supersedes", sup.get((t, metric)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
