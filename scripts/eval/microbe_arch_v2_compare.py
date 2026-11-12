#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Paired development comparisons of M3 architecture round v2 (amendment §6, §7.4, §11.1; lead rulings B3, B4).

Development LODO only. It reads the out-of-fold tables of ``scripts/train/microbe_arch_v2_cv.py``
(``cv_archv2/lodo/<family>/``; family P falls back to the existing track ``cv/lodo/``). It fits nothing, reads
no holdout row and decides no gate: the D1 gate is evaluated by the lead from registry rows (ruling B6).

For each family and each candidate, against EVERY comparator of the parity list that has a table (ruling B3):

* paired cluster bootstrap (B = 2,000) of the pooled-R² difference on the rows where both predict, with the
  site and with the domain as the resampling unit; Holm across the 8 targets of the family, per comparator;
  "beats" = positive difference and Holm rejection with both units (the development analogue of §7.4 / H-M3);
* ruling B4: the point difference in each of seeds 0–2. A stochastic model is read from its per-seed tables, a
  deterministic one from its single table (it has no seed variation); the pairing is seed s against seed s;
* the strongest simple baseline per target = the comparator with the highest development-LODO R² that is not
  the candidate itself (§6), and the candidate's point difference to it.

A comparator without a table is listed as not run and enters no maximum (§6: a family whose required
comparators are not all finished at the freeze is not scored inferentially). Registry experiment
``microbe_arch_v2_compare``; one row per candidate, comparator, family and target, and one row per candidate,
family and target for the strongest simple baseline.

    python -B scripts/eval/microbe_arch_v2_compare.py --candidates niche_s xgb_v4clr --families P G
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
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
EXPERIMENT = "microbe_arch_v2_compare"
SCHEME = "lodo"
SEEDS = (0, 1, 2)
# The parity list of amendment §6 as amended by the rulings (B5: ``xgb_clr_stack`` and ``xgb_v4clr_mo`` are
# comparators and run unconditionally). Table names of the runner / of the existing track.
PARITY = ("tier0_best", "rf_clr", "rf_clr_tuned", "xgb_clr", "xgb_clr_tuned", "ridge_clr", "tabpfn_clr",
          "ens_ridge_clr+xgb_clr", "rf_clr_context", "xgb_clr_context", "xgb_v4clr", "xgb_v4clr_context",
          "ridge_v4clr", "ens_ridge_v4clr+xgb_v4clr", "rf_v4clr", "tabpfn_v4clr", "xgb_v4clr_3000", "xgb_gfo",
          "wa_tol", "wa_pls", "xgb_v4clr_mo", "xgb_clr_stack")
# Ruling B3: for D1 and D2 the comparator set also includes these, each if run.
EXTRA_IF_RUN = ("xgb_nichez_context", "xgb_v4mr", "xgb_v4mr_stack")
EXTRA_FOR = ("niche_s", "niche_s_qmp", "niche_geo_phys")


def table_path(name: str, family: str, tag: str = "", seed: int | None = None) -> Path | None:
    """The out-of-fold table of ``name`` on lodo for ``family`` (seed-mean, or one seed), or None."""
    stem = name if seed is None else f"{name}_seed{seed}"
    roots = [paths.RUNS_ROOT / "microbe" / f"cv_archv2{tag}" / SCHEME / family]
    if family == "P":
        roots.append(paths.RUNS_ROOT / "microbe" / f"cv{tag}" / SCHEME)
    for r in roots:
        if (r / f"{stem}.parquet").is_file():
            return r / f"{stem}.parquet"
    return None


def seed_tables(name: str, family: str, tag: str = "") -> tuple:
    """({seed: table}, has per-seed tables). A model without per-seed tables is deterministic here: its single
    table stands for every seed."""
    per = {s: table_path(name, family, tag, s) for s in SEEDS}
    if all(p is not None for p in per.values()):
        return {s: pd.read_parquet(p) for s, p in per.items()}, True
    one = pd.read_parquet(table_path(name, family, tag))
    return {s: one for s in SEEDS}, False


def r2_of(S: pd.DataFrame, oof: pd.DataFrame, t: str, idx=None) -> float:
    from sentinel.models.microbe import evaluate as E

    idx = oof.index if idx is None else idx
    y, p = S.loc[idx, t].to_numpy(float), oof.loc[idx, t].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < 10:
        return float("nan")
    return float(E.summary(y[ok], p[ok], S.loc[idx, "domain_id"].astype(str).to_numpy()[ok])["r2"])


def compare_pair(S, cand, comp, cand_seeds, comp_seeds, targets, n_boot: int) -> dict:
    """Per target: paired bootstrap of the seed-mean tables (both units) and the per-seed point differences."""
    from sentinel.eval.bootstrap import holm_bonferroni
    from sentinel.models.microbe import evaluate as E

    idx = cand.index.intersection(comp.index)
    dom = S.loc[idx, "domain_id"].astype(str).to_numpy()
    site = S.loc[idx, "site_id"].astype(str).to_numpy()
    res, pv_s, pv_d = {}, {}, {}
    for t in targets:
        y = S.loc[idx, t].to_numpy(float)
        a, b = cand.loc[idx, t].to_numpy(float), comp.loc[idx, t].to_numpy(float)
        if (np.isfinite(y) & np.isfinite(a) & np.isfinite(b)).sum() < 10:
            continue
        o = E.paired_vs(y, a, b, dom, site, n_boot=n_boot)
        d = E.paired_vs(y, a, b, dom, dom, n_boot=n_boot)
        per_seed = []
        for s in SEEDS:
            i2 = cand_seeds[s].index.intersection(comp_seeds[s].index)
            per_seed.append(r2_of(S, cand_seeds[s], t, i2) - r2_of(S, comp_seeds[s], t, i2))
        res[t] = {"diff": float(o["diff"]), "ci_site": [float(o["ci_low"]), float(o["ci_high"])],
                  "p_site": float(o["p_value"]), "ci_domain": [float(d["ci_low"]), float(d["ci_high"])],
                  "p_domain": float(d["p_value"]), "n": int(o["n"]), "diff_per_seed": per_seed,
                  "positive_in_every_seed": bool(np.all(np.asarray(per_seed) > 0))}
        if np.isfinite(o["p_value"]):
            pv_s[t] = float(o["p_value"])
        if np.isfinite(d["p_value"]):
            pv_d[t] = float(d["p_value"])
    rej_s = holm_bonferroni(pv_s) if pv_s else {}
    rej_d = holm_bonferroni(pv_d) if pv_d else {}
    for t, o in res.items():
        o["holm_site"], o["holm_domain"] = bool(rej_s.get(t, False)), bool(rej_d.get(t, False))
        o["beats"] = bool(o["diff"] > 0 and o["holm_site"] and o["holm_domain"])
    return res


def strongest_simple_baseline(S, tables: dict, targets) -> dict:
    """Per target the comparator with the highest development-LODO R² on its own rows (§6), with every R²."""
    out = {}
    for t in targets:
        sc = {m: r2_of(S, o, t) for m, o in tables.items()}
        sc = {m: v for m, v in sc.items() if np.isfinite(v)}
        if sc:
            out[t] = {"model": max(sc, key=sc.get), "r2": max(sc.values()), "r2_by_model": sc}
    return out


def comparators_for(cand: str, names, extra) -> list:
    out = [n for n in names if n != cand]
    if cand in EXTRA_FOR:
        out += [n for n in extra if n != cand and n not in out]
    return out


def run(candidates, families, S, *, tag: str = "", n_boot: int = 2000, names=PARITY, extra=EXTRA_IF_RUN,
        stamp_row: str | None = None, log: bool = True) -> dict:
    from sentinel.models.microbe import archv2 as AV

    summary = {}
    for fam in families:
        targets = list(AV.FAMILIES[fam])
        for cand in candidates:
            cp = table_path(cand, fam, tag)
            if cp is None:
                summary.setdefault(fam, {})[cand] = {"not_run": True}
                continue
            cmean = pd.read_parquet(cp)
            cseeds, cstoch = seed_tables(cand, fam, tag)
            comps = comparators_for(cand, names, extra)
            present = [m for m in comps if table_path(m, fam, tag) is not None]
            missing = [m for m in comps if m not in present and m in names]       # extras count only if run
            tabs = {m: pd.read_parquet(table_path(m, fam, tag)) for m in present}
            rec = {"mean_r2": float(np.nanmean([r2_of(S, cmean, t) for t in targets])),
                   "r2": {t: r2_of(S, cmean, t) for t in targets}, "has_seed_variation": cstoch,
                   "comparators_run": present, "comparators_not_run": missing, "vs": {}}
            common = dict(workstream=WORKSTREAM, experiment=EXPERIMENT, model=AV.label(cand),
                          split=f"neon_dev/{SCHEME}_cv", is_baseline=AV.is_baseline(cand),
                          data_manifests=AV.MANIFESTS,
                          config={"model": cand, "family": fam, "scheme": SCHEME, "stamp_row": stamp_row})
            for m in present:
                mseeds, mstoch = seed_tables(m, fam, tag)
                res = compare_pair(S, cmean, tabs[m], cseeds, mseeds, targets, n_boot)
                rec["vs"][m] = res
                for t, o in res.items():
                    if log:
                        log_result(metric=f"r2_diff_vs_{m}", value=o["diff"], ci_low=o["ci_site"][0],
                                   ci_high=o["ci_site"][1], target=t, n=o["n"],
                                   ci_method=f"paired cluster bootstrap B={n_boot}, unit=site",
                                   notes=f"development comparison (rulings B3, B4) | family {fam}",
                                   extra={"reference": m, "reference_label": AV.label(m),
                                          "reference_has_seed_variation": mstoch, "p_value": o["p_site"],
                                          "ci_domain": o["ci_domain"], "p_value_domain": o["p_domain"],
                                          "holm_site": o["holm_site"], "holm_domain": o["holm_domain"],
                                          "beats": o["beats"], "diff_per_seed": o["diff_per_seed"],
                                          "positive_in_every_seed": o["positive_in_every_seed"],
                                          "candidate_has_seed_variation": cstoch}, **common)
            ssb = strongest_simple_baseline(S, tabs, targets)
            rec["strongest_simple_baseline"] = {}
            for t, b in ssb.items():
                d = rec["r2"][t] - b["r2"]
                rec["strongest_simple_baseline"][t] = {"model": b["model"], "r2": b["r2"], "point_diff": d}
                if log and np.isfinite(d):
                    log_result(metric="r2_point_diff_vs_strongest_simple_baseline", value=d, target=t,
                               notes=f"point difference, no interval | family {fam}",
                               extra={"reference": b["model"], "reference_r2": b["r2"],
                                      "comparators_not_run": missing, "r2_by_model": b["r2_by_model"]}, **common)
            rec["targets_beating_every_comparator_run"] = [
                t for t in targets if present and all(rec["vs"][m].get(t, {}).get("beats") for m in present)]
            rec["targets_beating_every_comparator_run_in_every_seed"] = [
                t for t in rec["targets_beating_every_comparator_run"]
                if all(rec["vs"][m][t]["positive_in_every_seed"] for m in present)]
            summary.setdefault(fam, {})[cand] = rec
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--families", nargs="+", default=["P", "G"], choices=["P", "G"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    from sentinel.models.microbe import archv2 as AV
    from sentinel.models.microbe import stamp as ST
    from sentinel.models.microbe.design import load_design

    stamp = ST.require_stamp("the development comparisons of architecture round v2")
    S = AV.attach_master(load_design()).samples
    summary = run(a.candidates, a.families, S, tag=a.tag, n_boot=a.n_boot, stamp_row=stamp["row_id"])
    out = paths.ensure(paths.RUNS_ROOT / "microbe" / f"cv_archv2{a.tag}") / "compare.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    for fam, recs in summary.items():
        prev.setdefault(fam, {}).update(recs)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(prev, indent=1, default=float))
    os.replace(tmp, out)
    for fam, recs in summary.items():
        for m, r in recs.items():
            if r.get("not_run"):
                print(fam, m, "not run")
                continue
            print(fam, m, round(r["mean_r2"], 4), "beats every comparator run on:",
                  r["targets_beating_every_comparator_run"], "| comparators not run:", r["comparators_not_run"])


if __name__ == "__main__":
    main()
