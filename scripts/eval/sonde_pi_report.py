#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Numeric-forecast report for SENTINEL-Sonde forecasts (user requirement 2026-11-10: every model reports specific
numeric values, not only events). For each model, variable and horizon, on the frozen COMMON rows of one block:

* point forecast = the predictive median (original units); per-row values go to ``numeric_forecasts_<block>.parquet``
  (site, issue date, h, observation, q05/q10/q25/q50/q75/q90/q95) for the labels in ``--tables-for``;
* pooled MAE, RMSE and bias of the median; MAE/RMSE skill vs persistence and vs climatology (1 − metric/reference);
* prediction intervals 50 % (q25–q75), 80 % (q10–q90), 90 % (q05–q95): pooled empirical coverage and mean width;
* per-site values (sites with ≥ ``--min-site-n`` scored rows): ``numeric_site_metrics_<block>.csv`` and the median over
  sites of coverage, MAE, RMSE and MAE skill vs persistence;
* 95 % CIs for the pooled MAE, RMSE, coverage and skills from a BASIN-GROUP cluster percentile bootstrap (the split's
  ``basin_group`` = HUC4 union, as ``sonde_score.py``), B = ``--n-boot`` (2,000), seed ``--boot-seed`` (0); whole groups of
  sites are resampled with replacement, which is the frozen ``sentinel.eval.bootstrap.bootstrap_ci`` scheme computed
  on per-group sums (identical draws; tested).

Observations, scoring masks and the naive references come from the frozen ``common`` module (Approved targets only).
Protected blocks need ``--allow-test`` AND the freeze file (the frozen guard enforces it); this script only reads saved
forecasts and never fits anything.

    python -B scripts/eval/sonde_pi_report.py --config configs/sonde/common.yaml --forecast ens=<run>/forecasts \
        [--eval-blocks val] [--tables-for ens] [--ci-for ens] --out <dir>
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.models.baselines.sonde_baselines import QUANTILES
from sentinel.models.sonde import common, runner
from sentinel.models.sonde.data import from_model_space

LEVELS = {"q05": 0.05, "q10": 0.10, "q25": 0.25, "q50": 0.50, "q75": 0.75, "q90": 0.90, "q95": 0.95}
PIS = {"pi50": ("q25", "q75"), "pi80": ("q10", "q90"), "pi90": ("q05", "q95")}
REFS = ("persistence", "climatology")


def _qidx(level: float) -> int:
    i = int(np.argmin(np.abs(np.asarray(QUANTILES) - level)))
    if abs(QUANTILES[i] - level) > 1e-6:
        raise ValueError(f"quantile {level} not in the 19 forecast levels")
    return i


def group_bootstrap_counts(groups: np.ndarray, n_boot: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(inverse index of each unit into the sorted unique groups, (B, G) resample counts). Same draws as
    ``bootstrap_ci``: per replicate ``rng.integers(0, G, G)`` with ``default_rng(seed)``."""
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    g = int(inv.max()) + 1
    rs = np.random.default_rng(seed)
    counts = np.empty((n_boot, g), dtype=np.int32)
    for b in range(n_boot):
        counts[b] = np.bincount(rs.integers(0, g, g), minlength=g)
    return inv, counts


def _ci(samples: np.ndarray) -> tuple[float, float]:
    ok = samples[np.isfinite(samples)]
    if not ok.size:
        return np.nan, np.nan
    return float(np.quantile(ok, 0.025)), float(np.quantile(ok, 0.975))


def report_one(data, var, block, label, q_rows, rows, refs, *, table: bool = True, basin: dict | None = None,
               n_boot: int = 0, seed: int = 0, min_site_n: int = 30):
    """Returns (pooled records, per-row table, per-site records)."""
    out, tables, sites_out = [], [], []
    q_rows = np.sort(q_rows, axis=-1)  # same crossing guard as the frozen evaluator
    for k, h in enumerate(data.horizons):
        m = rows.score_mask[h] & np.isfinite(rows.obs[h])
        if not m.any():
            continue
        obs = from_model_space(var, rows.obs[h][m])
        qs = {n: from_model_space(var, q_rows[m, k, _qidx(lv)]) for n, lv in LEVELS.items()}
        med = qs["q50"]
        site = np.asarray(rows.site_no[m]).astype(str)
        e = med - obs
        cols = {"n": np.ones_like(e), "ae": np.abs(e), "se": e ** 2, "e": e}
        for pi, (lo, hi) in PIS.items():
            cols[f"{pi}_in"] = ((obs >= qs[lo]) & (obs <= qs[hi])).astype(float)
            cols[f"{pi}_w"] = qs[hi] - qs[lo]
        for r, qr in refs.items():
            rmed = from_model_space(var, qr[m, k, _qidx(0.5)])
            ok = np.isfinite(rmed)
            er = np.where(ok, rmed - obs, 0.0)
            cols[f"{r}_ok"] = ok.astype(float)
            cols[f"{r}_ae_m"] = np.where(ok, np.abs(e), 0.0)
            cols[f"{r}_se_m"] = np.where(ok, e ** 2, 0.0)
            cols[f"{r}_ae_r"] = np.abs(er)
            cols[f"{r}_se_r"] = er ** 2
        st = pd.DataFrame(cols).assign(site_no=site).groupby("site_no").sum()   # per-site sums

        def stats(s: pd.DataFrame | dict) -> dict:
            n = s["n"]
            d = {"mae": s["ae"] / n, "rmse": np.sqrt(s["se"] / n), "bias": s["e"] / n}
            for pi in PIS:
                d[f"{pi}_coverage"] = s[f"{pi}_in"] / n
                d[f"{pi}_mean_width"] = s[f"{pi}_w"] / n
            for r in refs:
                nr = s[f"{r}_ok"]
                with np.errstate(divide="ignore", invalid="ignore"):
                    d[f"mae_skill_vs_{r}"] = 1 - s[f"{r}_ae_m"] / s[f"{r}_ae_r"]
                    d[f"rmse_skill_vs_{r}"] = 1 - np.sqrt(s[f"{r}_se_m"] / nr) / np.sqrt(s[f"{r}_se_r"] / nr)
            return d

        tot = st.sum()
        rec = {"model": label, "var": var, "block": block, "h": h, "n": int(tot["n"]), "n_sites": int(len(st)),
               **{k_: float(v) for k_, v in stats(tot).items()}}
        # per-site values and their medians over sites with enough rows
        ps = pd.DataFrame(stats(st), index=st.index).assign(n=st["n"].astype(int))
        big = ps[ps["n"] >= min_site_n]
        rec["n_sites_min_n"] = int(len(big))
        for c in ("pi50_coverage", "pi80_coverage", "pi90_coverage", "mae", "rmse", "mae_skill_vs_persistence"):
            if c in big:
                x = big[c].to_numpy(float)
                x = x[np.isfinite(x)]
                rec[f"site_median_{c}"] = float(np.median(x)) if x.size else np.nan
        sites_out.append(ps.reset_index().assign(model=label, var=var, block=block, h=h))
        # basin-group cluster bootstrap of the pooled statistics (per-group sums x resample counts)
        if n_boot:
            grp = np.array([(basin or {}).get(s, s) for s in st.index])
            inv, counts = group_bootstrap_counts(grp, n_boot, seed)
            gs = st.groupby(inv).sum().sort_index()                              # (G, cols)
            boot = {c: counts @ gs[c].to_numpy(float) for c in gs.columns}       # (B,) per column
            bs = stats(boot)
            rec["n_basin_groups"] = int(gs.shape[0])
            for c in ("mae", "rmse", "pi50_coverage", "pi80_coverage", "pi90_coverage",
                      *[f"{s}_skill_vs_{r}" for r in refs for s in ("mae", "rmse")]):
                if c in bs:
                    rec[f"{c}_ci_low"], rec[f"{c}_ci_high"] = _ci(np.asarray(bs[c], dtype=float))
            rec["ci_method"] = f"basin-group cluster percentile bootstrap B={n_boot} seed={seed}"
        out.append(rec)
        if table:
            tables.append(pd.DataFrame({"model": label, "var": var, "h": h, "site_no": rows.site_no[m],
                                        "issue_date": rows.issue_date[m], "obs": obs.astype("float32"),
                                        **{n: v.astype("float32") for n, v in qs.items()}}))
    return (out, pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(),
            pd.concat(sites_out, ignore_index=True) if sites_out else pd.DataFrame())


def main() -> int:
    ap = runner.base_parser(__doc__)
    ap.add_argument("--forecast", action="append", required=True, help="label=[prefix@]forecast_dir (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tables-for", default="", help="comma list of labels whose per-row numeric forecasts are written "
                    "(default: all; metrics are always written for every label and for persistence/climatology)")
    ap.add_argument("--ci-for", default="", help="comma list of labels that get bootstrap CIs (default: all)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--min-site-n", type=int, default=30)
    a = ap.parse_args()
    tables_for = set(filter(None, a.tables_for.split(",")))
    ci_for = set(filter(None, a.ci_for.split(",")))
    cfg = runner.apply_overrides(runner.load_config(a.config, smoke=a.smoke), a.set)
    run_dir = Path(a.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    data = runner.prepare(a, cfg, run_dir, need_blocks=("train",), log=print)
    basin = {str(k): str(v) for k, v in data.split.basin_group.items()}
    allrecs = []
    for block in runner.eval_blocks(a):
        recs, tabs, sites = [], [], []
        for var in runner.model_variables(cfg, data):
            rows = data.rows(var, block)
            if not len(rows):
                continue
            refs = {r: np.sort(common.baseline_forecast(data, var, rows, r), axis=-1) for r in REFS}
            todo = [(f"ref_{r}", qr) for r, qr in refs.items()]   # the naive references' own numeric metrics
            for spec in a.forecast:
                label, src = spec.split("=", 1)
                prefix, d = (src.split("@", 1) if "@" in src else (label, src))
                f = Path(d) / f"{prefix}__{var}__{block}.npz"
                if f.is_file():
                    todo.append((label, common.align_forecast(common.load_forecast(f), rows, data.horizons)))
            for label, q in todo:
                is_ref = label.startswith("ref_")
                r, t, s = report_one(data, var, block, label, q, rows, refs,
                                     table=not is_ref and (not tables_for or label in tables_for), basin=basin,
                                     n_boot=a.n_boot if (not is_ref and (not ci_for or label in ci_for)) else 0,
                                     seed=a.boot_seed, min_site_n=a.min_site_n)
                recs += r
                if len(t):
                    tabs.append(t)
                if len(s):
                    sites.append(s)
        allrecs += recs
        pd.DataFrame(recs).to_csv(run_dir / f"numeric_metrics_{block}.csv", index=False)
        if sites:
            pd.concat(sites, ignore_index=True).to_csv(run_dir / f"numeric_site_metrics_{block}.csv", index=False)
        if tabs:
            pd.concat(tabs, ignore_index=True).to_parquet(run_dir / f"numeric_forecasts_{block}.parquet", index=False)
    print(pd.DataFrame(allrecs).groupby(["model", "var"])[["mae", "rmse", "pi90_coverage"]].median().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
