#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Paired comparisons and the selection record for SENTINEL-Microbe phase 1 (protocol §1, §5, §6).

For each scheme and each model with an OOF file, paired site-cluster bootstrap of the pooled-R²
difference vs RF[DOY+location] (tier 0) and RF[genus-CLR] on the rows where both predict; Holm
across the 8 targets; the development-CV analogue of the H-M3 endpoint ("passes" = both
comparisons have CI excluding 0 after Holm). Selection rule (§6): highest mean pooled LODO R²
among SeqSet candidates. Registry experiment 'microbe_phase1_compare'.

    python -B scripts/train/microbe_compare.py --schemes lodo --models seqset_dnabert seqset_kmer ...
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.core.registry import log_result

WORKSTREAM = "M3-microbe"
EXPERIMENT = "microbe_phase1_compare"
REFS = ("context_rf", "rf_clr")
TIER0 = ("global_mean", "context_ridge", "context_rf")


def tier0_best(d: Path, S: pd.DataFrame, targets, E) -> pd.DataFrame | None:
    """Protocol §10 amendment (2026-10-24 ~23:31): the H-M3 tier-0 reference is, per target, the best
    development-LODO scorer among GlobalMean / Ridge[DOY+location] / RF[DOY+location]. Built from the
    LODO OOF files (choice made on development LODO and fixed for every other scheme and for the test),
    saved as ``tier0_best.parquet`` + ``tier0_best_choice.json`` next to them."""
    lodo = d.parent / "lodo"
    choice_p = lodo / "tier0_best_choice.json"
    if not choice_p.exists():
        oofs = {m: pd.read_parquet(lodo / f"{m}.parquet") for m in TIER0 if (lodo / f"{m}.parquet").exists()}
        if len(oofs) < len(TIER0):
            return None
        choice = {}
        for t in targets:
            sc = {m: float(E.summary(S.loc[o.index, t], o[t], S.loc[o.index, "domain_id"])["r2"])
                  for m, o in oofs.items()}
            choice[t] = {"model": max(sc, key=sc.get), "lodo_r2": sc}
        choice_p.write_text(json.dumps(choice, indent=1))
    choice = json.loads(choice_p.read_text())
    parts = {}
    for t in targets:
        f = d / f"{choice[t]['model']}.parquet"
        if not f.exists():
            return None
        parts[t] = pd.read_parquet(f)[t]
    out = pd.DataFrame(parts)
    out.to_parquet(d / "tier0_best.parquet")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="+", default=["lodo"])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--refs", nargs="+", default=list(REFS))
    ap.add_argument("--pass-refs", nargs="+", default=None,
                    help="refs that must all be beaten for the H-M3 analogue (default: all --refs)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args(argv)
    from sentinel.eval.bootstrap import holm_bonferroni
    from sentinel.models.microbe import evaluate as E
    from sentinel.models.microbe.baselines import LABELS
    from sentinel.models.microbe.design import TARGET_NAMES, load_design

    targets = list(TARGET_NAMES)
    summary = {}
    for scheme in a.schemes:
        design = load_design(with_release_v1=scheme.startswith("release"))
        S = design.samples
        d = paths.RUNS_ROOT / "microbe" / f"cv{a.tag}" / scheme
        if "tier0_best" in a.refs:
            tier0_best(d, load_design().samples if scheme.startswith("release") else S, targets, E)
        refs = {r: pd.read_parquet(d / f"{r}.parquet") for r in a.refs if (d / f"{r}.parquet").exists()}
        for m in a.models:
            p = d / f"{m}.parquet"
            if not p.exists():
                continue
            oof = pd.read_parquet(p)
            label = LABELS.get(m, f"SeqSet[{m.replace('+context', '')}]" + ("+context" if m.endswith("+context") else ""))
            rec = {"mean_r2": float(np.nanmean([E.summary(S.loc[oof.index, t], oof[t], S.loc[oof.index, "domain_id"])["r2"]
                                                for t in targets]))}
            for r, ref in refs.items():
                if r == m:
                    continue
                idx = oof.index.intersection(ref.index)
                pv, res = {}, {}
                for t in targets:
                    y = S.loc[idx, t].to_numpy(float)
                    out = E.paired_vs(y, oof.loc[idx, t], ref.loc[idx, t], S.loc[idx, "domain_id"].astype(str),
                                      S.loc[idx, "site_id"].astype(str), n_boot=a.n_boot)
                    dom = E.paired_vs(y, oof.loc[idx, t], ref.loc[idx, t], S.loc[idx, "domain_id"].astype(str),
                                      S.loc[idx, "domain_id"].astype(str), n_boot=a.n_boot)
                    out["ci_domain"] = [dom["ci_low"], dom["ci_high"]]
                    out["p_value_domain"] = dom["p_value"]
                    if np.isfinite(out.get("p_value", np.nan)):
                        pv[t] = out["p_value"]
                    res[t] = out
                rej = holm_bonferroni(pv) if pv else {}
                pvd = {t: o["p_value_domain"] for t, o in res.items() if np.isfinite(o.get("p_value_domain", np.nan))}
                rejd = holm_bonferroni(pvd) if pvd else {}
                for t, out in res.items():
                    # H-M3 analogue (review W5): positive AND Holm-significant with BOTH resampling units
                    out["holm_reject"] = bool(rej.get(t, False)) and bool(rejd.get(t, False)) and out["diff"] > 0
                    if not a.no_log and np.isfinite(out["diff"]):
                        log_result(workstream=WORKSTREAM, experiment=EXPERIMENT, model=label,
                                   metric=f"r2_diff_vs_{r}", value=out["diff"], ci_low=out["ci_low"],
                                   ci_high=out["ci_high"], ci_method=f"paired cluster bootstrap B={a.n_boot}, unit=site",
                                   split=f"neon_dev/{scheme}_cv", target=t, n=out["n"], data_manifests=["microbe_wq"],
                                   extra={"p_value": out["p_value"], "holm_reject_and_positive": out["holm_reject"],
                                          "ci_domain": out["ci_domain"], "p_value_domain": out["p_value_domain"],
                                          "reference": LABELS.get(r, r)})
                rec[f"vs_{r}"] = {t: {k: out[k] for k in ("diff", "ci_low", "ci_high", "p_value", "ci_domain",
                                                          "p_value_domain", "holm_reject")}
                                  for t, out in res.items()}
            prefs = [r for r in (a.pass_refs or a.refs) if r != m]
            if all(f"vs_{r}" in rec for r in prefs):
                passing = [t for t in targets if all(rec[f"vs_{r}"].get(t, {}).get("holm_reject") for r in prefs)]
                rec["targets_beating_all_refs"] = passing
                rec["pass_refs"] = prefs
                rec["hm3_dev_analogue_pass"] = len(passing) >= 2
            summary.setdefault(scheme, {})[m] = rec
    out = paths.ensure(paths.RUNS_ROOT / "microbe") / f"phase1_compare{a.tag}.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    for k, v in summary.items():
        prev.setdefault(k, {}).update(v)
    out.write_text(json.dumps(prev, indent=1, default=float))
    for scheme, recs in summary.items():
        for m, r in recs.items():
            print(scheme, m, round(r["mean_r2"], 4), r.get("targets_beating_all_refs"))


if __name__ == "__main__":
    main()
