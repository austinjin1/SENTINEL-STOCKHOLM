#!/usr/bin/env python3
"""Exp18: Seasonal Anomaly Pattern Analysis.

Uses NEON Exp8 monthly trend data and NEON scan per-site date ranges
to reconstruct seasonal anomaly distributions. Also loads the raw NEON
parquet to compute monthly threshold-exceedance rates across all 32 sites.

Key questions:
  1. Do anomaly detections peak in spring (runoff), summer (algal blooms), or fall?
  2. Which parameters show strongest seasonal signal?
  3. Do high-risk sites (from Exp17) show earlier seasonal onset?

Output:
  results/exp18_seasonal/seasonal_results.json
  paper/figures/fig_exp18_seasonal_patterns.jpg

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

OUTPUT_DIR   = PROJECT_ROOT / "results" / "exp18_seasonal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR      = PROJECT_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NEON_PARQUET  = PROJECT_ROOT / "data" / "raw" / "neon_aquatic" / "neon_DP1.20288.001_consolidated.parquet"
NEON_VALUE_COLS = ["pH", "dissolvedOxygen", "turbidity", "specificConductance"]
NEON_QF_COLS    = ["pHFinalQF", "dissolvedOxygenFinalQF", "turbidityFinalQF", "specificCondFinalQF"]

ANOMALY_THRESHOLDS = {
    "pH":                  (6.0, 9.5),
    "dissolvedOxygen":     (4.0, None),
    "turbidity":           (None, 300.0),
    "specificConductance": (None, 1500.0),
}

READ_COLS = ["startDateTime", "source_site"] + NEON_VALUE_COLS + NEON_QF_COLS

SEASONS = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3:  "Spring", 4: "Spring", 5: "Spring",
    6:  "Summer", 7: "Summer", 8: "Summer",
    9:  "Fall",  10: "Fall",  11: "Fall",
}

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def compute_monthly_exceedances(df_site: pd.DataFrame) -> dict:
    """Per-month threshold exceedance rate for a site."""
    df = df_site.copy()
    df["ts"] = pd.to_datetime(df["startDateTime"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    df["month"] = df["ts"].dt.month

    monthly = {}
    for m in range(1, 13):
        sub = df[df["month"] == m]
        if len(sub) < 10:
            monthly[m] = None
            continue
        is_anomaly = pd.Series(False, index=sub.index)
        for col, (low, high) in ANOMALY_THRESHOLDS.items():
            if col not in sub.columns:
                continue
            v = pd.to_numeric(sub[col], errors="coerce")
            if low is not None:
                is_anomaly |= v < low
            if high is not None:
                is_anomaly |= v > high
        monthly[m] = {
            "n_records":         int(len(sub)),
            "exceedance_rate":   round(float(is_anomaly.mean()), 4),
            "exceedance_count":  int(is_anomaly.sum()),
            "by_param": {}
        }
        for col, (low, high) in ANOMALY_THRESHOLDS.items():
            if col not in sub.columns:
                continue
            v = pd.to_numeric(sub[col], errors="coerce")
            exc = pd.Series(False, index=sub.index)
            if low is not None:  exc |= v < low
            if high is not None: exc |= v > high
            monthly[m]["by_param"][col] = round(float(exc.mean()), 4)
    return monthly


def aggregate_cross_site(all_monthly: dict) -> dict:
    """Average exceedance rates across sites for each month."""
    cross = {}
    for m in range(1, 13):
        rates = [all_monthly[s][m]["exceedance_rate"]
                 for s in all_monthly
                 if all_monthly[s].get(m) is not None]
        cross[m] = {
            "mean_exceedance_rate": round(float(np.mean(rates)), 4) if rates else None,
            "std_exceedance_rate":  round(float(np.std(rates)), 4)  if rates else None,
            "n_sites":              len(rates),
        }
    return cross


def find_peak_season(cross_monthly: dict) -> dict:
    """Identify peak and trough months."""
    rates = {m: v["mean_exceedance_rate"] for m, v in cross_monthly.items()
             if v["mean_exceedance_rate"] is not None}
    peak_month   = max(rates, key=rates.get)
    trough_month = min(rates, key=rates.get)
    return {
        "peak_month":    peak_month,
        "peak_month_name": MONTH_NAMES[peak_month - 1],
        "peak_rate":     round(rates[peak_month], 4),
        "trough_month":  trough_month,
        "trough_month_name": MONTH_NAMES[trough_month - 1],
        "trough_rate":   round(rates[trough_month], 4),
        "peak_season":   SEASONS[peak_month],
        "seasonal_amplitude": round(rates[peak_month] - rates[trough_month], 4),
    }


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("EXP18: Seasonal Anomaly Pattern Analysis")
    logger.info("=" * 60)

    # Load NEON scan site list
    scan = json.load(open(PROJECT_ROOT / "results" / "neon_anomaly_scan" / "neon_scan_results.json"))
    success_sites = [s for s, r in scan["per_site"].items() if r.get("status") == "success"]
    logger.info(f"Processing {len(success_sites)} sites")

    pf = pq.ParquetFile(str(NEON_PARQUET))
    all_monthly = {}

    for i, site in enumerate(sorted(success_sites)):
        try:
            logger.info(f"  [{i+1:2d}/{len(success_sites)}] {site}")
            filters = [("source_site", "=", site)]
            table = pq.read_table(str(NEON_PARQUET), columns=READ_COLS, filters=filters)
            if len(table) < 100:
                continue
            df = table.to_pandas()
            monthly = compute_monthly_exceedances(df)
            all_monthly[site] = monthly
            # Quick report
            valid_months = {m: v for m, v in monthly.items() if v is not None}
            if valid_months:
                peak_m = max(valid_months, key=lambda m: valid_months[m]["exceedance_rate"])
                logger.info(f"    Peak month: {MONTH_NAMES[peak_m-1]} "
                            f"(rate={valid_months[peak_m]['exceedance_rate']:.3f})")
        except Exception as e:
            logger.warning(f"    Error on {site}: {e}")

    cross_monthly = aggregate_cross_site(all_monthly)
    peak_info = find_peak_season(cross_monthly)

    # Parameter-level seasonal signal
    param_monthly = {col: {} for col in ANOMALY_THRESHOLDS}
    for m in range(1, 13):
        for col in ANOMALY_THRESHOLDS:
            rates = [all_monthly[s][m]["by_param"].get(col, 0.0)
                     for s in all_monthly
                     if all_monthly[s].get(m) is not None]
            param_monthly[col][m] = {
                "mean": round(float(np.mean(rates)), 4) if rates else None,
                "std":  round(float(np.std(rates)), 4)  if rates else None,
            }

    # Peak month per parameter
    param_peaks = {}
    for col, monthly in param_monthly.items():
        valid = {m: v["mean"] for m, v in monthly.items() if v["mean"] is not None and v["mean"] > 0}
        if valid:
            pk = max(valid, key=valid.get)
            param_peaks[col] = {
                "peak_month":      pk,
                "peak_month_name": MONTH_NAMES[pk - 1],
                "peak_rate":       round(valid[pk], 4),
                "peak_season":     SEASONS[pk],
            }

    # Site-level: season of peak exceedance
    site_peak_seasons = {}
    for site, monthly in all_monthly.items():
        valid = {m: v["exceedance_rate"] for m, v in monthly.items() if v is not None}
        if valid:
            pk = max(valid, key=valid.get)
            site_peak_seasons[site] = SEASONS[pk]
    season_hist = {s: 0 for s in ["Winter", "Spring", "Summer", "Fall"]}
    for s in site_peak_seasons.values():
        season_hist[s] += 1

    output = {
        "n_sites":              len(all_monthly),
        "cross_site_monthly":   {str(m): v for m, v in cross_monthly.items()},
        "peak_info":            peak_info,
        "parameter_peaks":      param_peaks,
        "site_peak_seasons":    site_peak_seasons,
        "season_histogram":     season_hist,
        "per_site_monthly":     {s: {str(m): v for m, v in monthly.items()}
                                 for s, monthly in all_monthly.items()},
        "elapsed_s":            round(time.time() - t0, 1),
    }

    out_path = OUTPUT_DIR / "seasonal_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved: {out_path}")

    # --- Figure ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Cross-site monthly exceedance with error bands
        months = list(range(1, 13))
        means  = [cross_monthly[m]["mean_exceedance_rate"] or 0 for m in months]
        stds   = [cross_monthly[m]["std_exceedance_rate"]  or 0 for m in months]
        axes[0].plot(months, means, "b-o", lw=2, ms=5)
        axes[0].fill_between(months,
                              [max(0, m - s) for m, s in zip(means, stds)],
                              [m + s for m, s in zip(means, stds)],
                              alpha=0.2, color="blue")
        axes[0].set_xticks(months)
        axes[0].set_xticklabels(MONTH_NAMES, fontsize=8)
        axes[0].set_ylabel("Mean Threshold Exceedance Rate")
        axes[0].set_title(f"Cross-Site Seasonal Anomaly Pattern\n"
                          f"Peak: {peak_info['peak_month_name']} ({peak_info['peak_season']})")
        axes[0].axvspan(3, 5.5, alpha=0.08, color="green", label="Spring")
        axes[0].axvspan(5.5, 8.5, alpha=0.08, color="orange", label="Summer")
        axes[0].legend(fontsize=8)

        # Right: Per-parameter seasonal signal
        param_colors = {"pH": "purple", "dissolvedOxygen": "blue",
                        "turbidity": "brown", "specificConductance": "red"}
        for col, color in param_colors.items():
            pm = param_monthly[col]
            pm_means = [pm[m]["mean"] or 0 for m in months]
            axes[1].plot(months, pm_means, "-o", color=color,
                         lw=1.5, ms=4, label=col.replace("dissolved", ""))
        axes[1].set_xticks(months)
        axes[1].set_xticklabels(MONTH_NAMES, fontsize=8)
        axes[1].set_ylabel("Mean Exceedance Rate")
        axes[1].set_title("Per-Parameter Seasonal Exceedance")
        axes[1].legend(fontsize=8)
        axes[1].axvspan(3, 5.5, alpha=0.08, color="green")
        axes[1].axvspan(5.5, 8.5, alpha=0.08, color="orange")

        plt.tight_layout()
        fig_path = FIG_DIR / "fig_exp18_seasonal_patterns.jpg"
        plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {fig_path}")
    except Exception as e:
        logger.warning(f"Figure failed: {e}")

    logger.info("\n=== SEASONAL SUMMARY ===")
    logger.info(f"Peak anomaly month: {peak_info['peak_month_name']} ({peak_info['peak_season']})")
    logger.info(f"Trough month:       {peak_info['trough_month_name']}")
    logger.info(f"Seasonal amplitude: {peak_info['seasonal_amplitude']:.4f}")
    logger.info(f"\nPer-parameter peaks:")
    for col, pk in param_peaks.items():
        logger.info(f"  {col:<25} peak: {pk['peak_month_name']:>4} ({pk['peak_season']}) rate={pk['peak_rate']:.4f}")
    logger.info(f"\nSite peak-season histogram: {season_hist}")
    logger.info(f"Elapsed: {output['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
