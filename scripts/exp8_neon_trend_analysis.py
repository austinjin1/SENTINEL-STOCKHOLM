#!/usr/bin/env python3
"""Experiment 8: NEON Aquatic Water Quality Temporal Trend Analysis.

Loads the 62.7M-row NEON DP1.20288.001 parquet and performs per-site trend
analysis using the Mann-Kendall monotonic trend test on four key water quality
parameters: pH, dissolved oxygen, turbidity, and specific conductance.

Also computes:
  - Theil-Sen slope estimator (robust linear trend in units/year)
  - Monthly anomaly rate (fraction of readings outside EPA thresholds)
  - Site ranking by degradation severity (largest negative trend in DO,
    largest positive trend in turbidity)
  - Cross-site correlation matrix of monthly DO anomaly rates

Outputs:
  - results/exp8_neon_trends/trend_results.json
  - paper/figures/fig_exp8_trend_heatmap.jpg
  - paper/figures/fig_exp8_site_trends.jpg

MIT License — Bryan Cheng, 2026
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

NEON_PARQUET = PROJECT_ROOT / "data" / "raw" / "neon_aquatic" / "neon_DP1.20288.001_consolidated.parquet"
RESULTS_DIR  = PROJECT_ROOT / "results" / "exp8_neon_trends"
FIGURES_DIR  = PROJECT_ROOT / "paper" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Parameters to analyze
PARAMS = ["pH", "dissolvedOxygen", "turbidity", "specificConductance"]
PARAM_LABELS = {"pH": "pH (s.u.)", "dissolvedOxygen": "DO (mg/L)",
                "turbidity": "Turbidity (NTU)", "specificConductance": "SpCond (µS/cm)"}

# EPA thresholds (same as NEON scan)
EPA_THRESH = {
    "pH":                  (6.0, 9.5),
    "dissolvedOxygen":     (4.0, None),
    "turbidity":           (None, 300.0),
    "specificConductance": (None, 1500.0),
}


# ---------------------------------------------------------------------------
# Mann-Kendall + Theil-Sen (pure numpy, no scipy dependency)
# ---------------------------------------------------------------------------

def mann_kendall(x: np.ndarray):
    """Mann-Kendall trend test.

    Returns (tau, s, p_approx, trend) where trend is 'increasing',
    'decreasing', or 'no trend'.
    Uses normal approximation (valid for n >= 10).
    """
    n = len(x)
    if n < 10:
        return 0.0, 0, 1.0, "insufficient_data"

    # Compute S statistic
    s = 0
    for i in range(n - 1):
        diffs = np.sign(x[i + 1:] - x[i])
        s += diffs.sum()

    # Variance of S under H0 (no ties)
    var_s = n * (n - 1) * (2 * n + 5) / 18.0

    # Z statistic
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    # Two-tailed p-value approximation
    from math import erfc, sqrt
    p = erfc(abs(z) / sqrt(2))

    tau = s / (0.5 * n * (n - 1))

    if p < 0.05:
        trend = "increasing" if s > 0 else "decreasing"
    else:
        trend = "no trend"

    return float(tau), int(s), float(p), trend


def theil_sen_slope(x: np.ndarray, t: np.ndarray) -> float:
    """Theil-Sen slope estimator (median of pairwise slopes)."""
    n = len(x)
    if n < 4:
        return 0.0
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            dt = t[j] - t[i]
            if abs(dt) > 1e-6:
                slopes.append((x[j] - x[i]) / dt)
    return float(np.median(slopes)) if slopes else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    logger.info("=" * 65)
    logger.info("EXPERIMENT 8: NEON Aquatic Temporal Trend Analysis")
    logger.info("=" * 65)
    logger.info(f"NEON parquet: {NEON_PARQUET}")

    # Load columns we need
    logger.info("Loading parquet (relevant columns only)...")
    t_read = time.time()
    pf = pq.ParquetFile(str(NEON_PARQUET))
    read_cols = ["startDateTime", "source_site"] + PARAMS
    table = pf.read(columns=read_cols)
    df = table.to_pandas()
    logger.info(f"Loaded {len(df):,} rows in {time.time()-t_read:.1f}s")

    # Parse timestamps
    df["ts"] = pd.to_datetime(df["startDateTime"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).copy()
    df["year_month"] = df["ts"].dt.to_period("M")
    df["year"] = df["ts"].dt.year

    sites = sorted(df["source_site"].dropna().unique())
    logger.info(f"Sites: {len(sites)}")

    # ------------------------------------------------------------------
    # Per-site trend analysis
    # ------------------------------------------------------------------
    site_results = {}

    for site in sites:
        df_s = df[df["source_site"] == site].copy()
        n_rows = len(df_s)

        # Monthly mean per parameter
        monthly = df_s.groupby("year_month")[PARAMS].mean()
        monthly = monthly.sort_index()

        if len(monthly) < 6:
            site_results[site] = {"status": "insufficient_months", "n_rows": n_rows}
            continue

        # Time index in decimal years for Theil-Sen
        t_idx = np.array([p.year + (p.month - 1) / 12.0 for p in monthly.index])

        param_trends = {}
        for param in PARAMS:
            vals = monthly[param].dropna().values
            t_vals = t_idx[:len(vals)]

            if len(vals) < 6:
                param_trends[param] = {"status": "insufficient_data"}
                continue

            tau, s, p, trend = mann_kendall(vals)
            slope_per_year = theil_sen_slope(vals, t_vals)

            param_trends[param] = {
                "tau": tau,
                "s": s,
                "p_value": p,
                "trend": trend,
                "slope_per_year": slope_per_year,
                "n_months": int(len(vals)),
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals)),
            }

        # Monthly EPA exceedance rate
        monthly_exc = {}
        for param, (low, high) in EPA_THRESH.items():
            col = df_s[param].astype(float)
            exc = np.zeros(len(col), dtype=bool)
            if low is not None:
                exc |= (col < low).values
            if high is not None:
                exc |= (col > high).values
            monthly_exc[param] = float(exc.mean())

        site_results[site] = {
            "status": "ok",
            "n_rows": int(n_rows),
            "n_months": int(len(monthly)),
            "date_range": [str(monthly.index.min()), str(monthly.index.max())],
            "param_trends": param_trends,
            "epa_exceedance_rate": monthly_exc,
        }

        # Log significant trends
        sig = [(p, r["trend"], r["slope_per_year"])
               for p, r in param_trends.items()
               if r.get("trend") not in ("no trend", "insufficient_data", None)
               and isinstance(r, dict) and r.get("p_value", 1) < 0.05]
        if sig:
            logger.info(f"  {site}: significant trends: " +
                        ", ".join(f"{p}={t}({s:+.3f}/yr)" for p, t, s in sig))
        else:
            logger.info(f"  {site}: no significant trends ({len(monthly)} months)")

    # ------------------------------------------------------------------
    # Cross-site correlation of monthly DO anomaly rates
    # ------------------------------------------------------------------
    logger.info("Computing cross-site DO correlation...")
    do_monthly = {}
    for site in sites:
        df_s = df[df["source_site"] == site].copy()
        if len(df_s) < 100:
            continue
        do_col = df_s["dissolvedOxygen"].astype(float)
        df_s = df_s.copy()
        df_s["do_low"] = (do_col < 4.0).astype(float)
        monthly_do = df_s.groupby("year_month")["do_low"].mean()
        do_monthly[site] = monthly_do

    # Align on common periods
    if len(do_monthly) >= 3:
        do_df = pd.DataFrame(do_monthly).dropna(thresh=3)
        corr_mat = do_df.corr()
        cross_site_corr = corr_mat.to_dict()
    else:
        cross_site_corr = {}

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    ok_sites = {s: r for s, r in site_results.items() if r.get("status") == "ok"}

    # Sites with significant degradation in DO (decreasing trend)
    do_degrading = sorted(
        [(s, r["param_trends"]["dissolvedOxygen"])
         for s, r in ok_sites.items()
         if r["param_trends"].get("dissolvedOxygen", {}).get("trend") == "decreasing"
         and r["param_trends"]["dissolvedOxygen"].get("p_value", 1) < 0.05],
        key=lambda x: x[1].get("slope_per_year", 0)
    )

    # Sites with significant turbidity increase
    turb_increasing = sorted(
        [(s, r["param_trends"]["turbidity"])
         for s, r in ok_sites.items()
         if r["param_trends"].get("turbidity", {}).get("trend") == "increasing"
         and r["param_trends"]["turbidity"].get("p_value", 1) < 0.05],
        key=lambda x: x[1].get("slope_per_year", 0), reverse=True
    )

    summary = {
        "n_sites_total": len(sites),
        "n_sites_analyzed": len(ok_sites),
        "n_sites_do_degrading": len(do_degrading),
        "n_sites_turbidity_increasing": len(turb_increasing),
        "do_degrading_sites": [(s, r.get("slope_per_year", 0)) for s, r in do_degrading],
        "turbidity_increasing_sites": [(s, r.get("slope_per_year", 0)) for s, r in turb_increasing],
        "per_site": site_results,
        "cross_site_do_correlation_sites": list(corr_mat.columns) if len(do_monthly) >= 3 else [],
        "elapsed_s": round(time.time() - t0, 1),
    }

    # Save
    out_path = RESULTS_DIR / "trend_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\nResults saved: {out_path}")

    # ------------------------------------------------------------------
    # Figure 1: Trend heatmap (tau per site × parameter)
    # ------------------------------------------------------------------
    logger.info("Generating figures...")

    param_order = PARAMS
    site_order = sorted(ok_sites.keys())

    tau_matrix = np.full((len(site_order), len(param_order)), np.nan)
    sig_matrix = np.zeros((len(site_order), len(param_order)), dtype=bool)

    for i, site in enumerate(site_order):
        for j, param in enumerate(param_order):
            pt = ok_sites[site]["param_trends"].get(param, {})
            if isinstance(pt, dict) and "tau" in pt:
                tau_matrix[i, j] = pt["tau"]
                sig_matrix[i, j] = pt.get("p_value", 1) < 0.05

    fig, ax = plt.subplots(figsize=(8, max(6, len(site_order) * 0.35)))
    param_display = [PARAM_LABELS[p] for p in param_order]
    site_display  = site_order

    # Fill NaN with 0 for display
    tau_display = np.where(np.isnan(tau_matrix), 0, tau_matrix)

    im = ax.imshow(tau_display, aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Mann-Kendall τ")

    ax.set_xticks(range(len(param_order)))
    ax.set_xticklabels(param_display, fontsize=9, rotation=15)
    ax.set_yticks(range(len(site_order)))
    ax.set_yticklabels(site_display, fontsize=7)

    # Mark significant cells with *
    for i in range(len(site_order)):
        for j in range(len(param_order)):
            if sig_matrix[i, j]:
                ax.text(j, i, "*", ha="center", va="center",
                        fontsize=11, color="black", fontweight="bold")

    ax.set_title("NEON Site Water Quality Trends (Mann-Kendall τ)\n"
                 "* = significant (p < 0.05)", fontsize=11)
    plt.tight_layout()

    path = FIGURES_DIR / "fig_exp8_trend_heatmap.jpg"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", pil_kwargs={"quality": 85})
    plt.close(fig)
    logger.info(f"Saved: {path}")

    # ------------------------------------------------------------------
    # Figure 2: Top degrading sites (DO slope)
    # ------------------------------------------------------------------
    if do_degrading or turb_increasing:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        if do_degrading:
            sites_do = [x[0] for x in do_degrading]
            slopes_do = [x[1]["slope_per_year"] for x in do_degrading]
            colors_do = ["#e74c3c" if s < 0 else "#27ae60" for s in slopes_do]
            ax.barh(sites_do, slopes_do, color=colors_do, edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("DO Trend (mg/L per year)")
        ax.set_title("Sites with Significant\nDO Decline (p < 0.05)")

        ax = axes[1]
        if turb_increasing:
            sites_t = [x[0] for x in turb_increasing]
            slopes_t = [x[1]["slope_per_year"] for x in turb_increasing]
            ax.barh(sites_t, slopes_t, color="#e67e22", edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Turbidity Trend (NTU per year)")
        ax.set_title("Sites with Significant\nTurbidity Increase (p < 0.05)")

        plt.suptitle("NEON Aquatic Site Degradation Trends", fontsize=13, fontweight="bold")
        plt.tight_layout()

        path = FIGURES_DIR / "fig_exp8_site_trends.jpg"
        fig.savefig(str(path), dpi=150, bbox_inches="tight", pil_kwargs={"quality": 85})
        plt.close(fig)
        logger.info(f"Saved: {path}")

    # Summary print
    logger.info("\n=== TREND SUMMARY ===")
    logger.info(f"Sites analyzed: {len(ok_sites)}/{len(sites)}")
    logger.info(f"Sites with significant DO decline: {len(do_degrading)}")
    logger.info(f"Sites with significant turbidity increase: {len(turb_increasing)}")
    if do_degrading:
        logger.info("DO degrading sites:")
        for s, r in do_degrading:
            logger.info(f"  {s}: {r.get('slope_per_year', 0):+.4f} mg/L/yr, p={r.get('p_value',1):.3f}")
    logger.info(f"\nTotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
