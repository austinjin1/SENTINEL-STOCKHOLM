#!/usr/bin/env python3
"""Experiment 13: PRPO Site Data Audit — Critique 8 Resolution.

Critique: Exp8 found PRPO specific conductance declining at -242.3 µS/cm/year.
This is extremely large and could be a real hydrological event OR a sensor
artifact (calibration drift or QC issues in 2022–2024).

This experiment:
  1. Loads PRPO raw specific conductance time series (2017–2024)
  2. Segments the data by year and computes descriptive statistics
  3. Checks QF (quality flag) patterns: Is the trend era-specific?
  4. Cross-validates: If artifact, QF rates should spike when values change
  5. Computes rolling 90-day median to visualize the trend
  6. Compares SpCond distribution pre/post 2022 (Mann-Whitney U test)
  7. Checks DO and pH trends at PRPO for corroboration or contradiction

Verdict:
  - If QF rates are correlated with the trend period → likely sensor artifact
  - If other parameters show consistent change → likely real hydrological event
  - If pH/DO change consistent with conductance → supports real event

Outputs:
  - results/exp13_prpo_audit/prpo_audit_results.json
  - paper/figures/fig_exp13_prpo_audit.jpg

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
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results" / "exp13_prpo_audit"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

NEON_PARQUET = PROJECT_ROOT / "data" / "raw" / "neon_aquatic" / "neon_DP1.20288.001_consolidated.parquet"


def load_prpo_data():
    """Load PRPO site data from NEON parquet."""
    logger.info("Loading NEON parquet for PRPO site...")
    READ_COLS = [
        "startDateTime", "source_site",
        "specificConductance", "specificCondFinalQF",
        "pH", "pHFinalQF",
        "dissolvedOxygen", "dissolvedOxygenFinalQF",
        "turbidity", "turbidityFinalQF",
    ]
    pf = pq.ParquetFile(str(NEON_PARQUET))
    table = pf.read(columns=READ_COLS)
    df = table.to_pandas()

    prpo = df[df["source_site"] == "PRPO"].copy()
    logger.info(f"PRPO rows: {len(prpo):,}")

    prpo["ts"] = pd.to_datetime(prpo["startDateTime"], utc=True, errors="coerce")
    prpo = prpo.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    prpo["year"] = prpo["ts"].dt.year
    prpo["month"] = prpo["ts"].dt.month
    prpo["date"] = prpo["ts"].dt.date

    return prpo


def year_statistics(prpo: pd.DataFrame) -> list:
    """Per-year descriptive statistics for SpCond."""
    results = []
    for yr, g in prpo.groupby("year"):
        sc = g["specificConductance"].dropna()
        qf = g["specificCondFinalQF"].dropna()
        qf_pass_rate = (qf == 0).mean() if len(qf) > 0 else np.nan

        if len(sc) < 10:
            continue
        results.append({
            "year": int(yr),
            "n_records": int(len(g)),
            "n_valid_sc": int(len(sc)),
            "sc_mean": float(sc.mean()),
            "sc_median": float(sc.median()),
            "sc_std": float(sc.std()),
            "sc_p5": float(sc.quantile(0.05)),
            "sc_p95": float(sc.quantile(0.95)),
            "qf_pass_rate": float(qf_pass_rate),
        })
    return results


def pre_post_2022_test(prpo: pd.DataFrame) -> dict:
    """Mann-Whitney U test: SpCond pre-2022 vs. post-2022."""
    pre  = prpo[prpo["year"] < 2022]["specificConductance"].dropna().values
    post = prpo[prpo["year"] >= 2022]["specificConductance"].dropna().values

    if len(pre) < 10 or len(post) < 10:
        return {"error": "insufficient data"}

    u_stat, p_val = stats.mannwhitneyu(pre, post, alternative="two-sided")
    # Effect size: rank-biserial correlation
    n1, n2 = len(pre), len(post)
    r = 1 - 2 * u_stat / (n1 * n2)

    return {
        "n_pre_2022": int(n1),
        "n_post_2022": int(n2),
        "mean_pre_2022": float(pre.mean()),
        "mean_post_2022": float(post.mean()),
        "median_pre_2022": float(np.median(pre)),
        "median_post_2022": float(np.median(post)),
        "mannwhitney_u": float(u_stat),
        "p_value": float(p_val),
        "rank_biserial_r": float(r),
        "significant": bool(p_val < 0.001),
    }


def qf_trend_correlation(year_stats: list) -> dict:
    """Does QF pass rate correlate with SpCond? Artifact indicator."""
    yrs = [s["year"] for s in year_stats if not np.isnan(s["qf_pass_rate"])]
    sc  = [s["sc_median"] for s in year_stats if not np.isnan(s["qf_pass_rate"])]
    qf  = [s["qf_pass_rate"] for s in year_stats if not np.isnan(s["qf_pass_rate"])]

    if len(yrs) < 3:
        return {"error": "insufficient years"}

    r_sc_qf, p_sc_qf = stats.pearsonr(sc, qf) if len(sc) >= 3 else (np.nan, np.nan)
    r_yr_sc, p_yr_sc = stats.pearsonr(yrs, sc) if len(yrs) >= 3 else (np.nan, np.nan)
    r_yr_qf, p_yr_qf = stats.pearsonr(yrs, qf) if len(yrs) >= 3 else (np.nan, np.nan)

    return {
        "n_years": len(yrs),
        "sc_qf_correlation": float(r_sc_qf),
        "sc_qf_pvalue": float(p_sc_qf),
        "year_sc_correlation": float(r_yr_sc),
        "year_sc_pvalue": float(p_yr_sc),
        "year_qf_correlation": float(r_yr_qf),
        "year_qf_pvalue": float(p_yr_qf),
        "interpretation": (
            "ARTIFACT LIKELY: SpCond and QF pass rate negatively correlated "
            "(low SpCond = high QF failures)" if r_sc_qf < -0.5 and p_sc_qf < 0.05
            else "SpCond and QF pass rate not strongly correlated — artifact less likely"
        ),
    }


def corroborating_params(prpo: pd.DataFrame) -> dict:
    """Do pH and DO show consistent changes with SpCond at PRPO?"""
    pre  = prpo[prpo["year"] < 2022]
    post = prpo[prpo["year"] >= 2022]

    results = {}
    for param, qf_col in [("pH", "pHFinalQF"), ("dissolvedOxygen", "dissolvedOxygenFinalQF"),
                           ("turbidity", "turbidityFinalQF")]:
        pre_v  = pre[param].dropna().values
        post_v = post[param].dropna().values
        if len(pre_v) < 10 or len(post_v) < 10:
            results[param] = {"error": "insufficient data"}
            continue
        u, p = stats.mannwhitneyu(pre_v, post_v, alternative="two-sided")
        results[param] = {
            "mean_pre": float(pre_v.mean()),
            "mean_post": float(post_v.mean()),
            "pct_change": float((post_v.mean() - pre_v.mean()) / (abs(pre_v.mean()) + 1e-8) * 100),
            "p_value": float(p),
            "significant": bool(p < 0.001),
        }
    return results


def plot_prpo_audit(prpo, year_stats, pre_post, qf_corr):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Daily median SpCond
    prpo["date_dt"] = pd.to_datetime(prpo["date"].astype(str))
    daily = prpo.groupby("date_dt")["specificConductance"].median().reset_index()
    daily.columns = ["date", "sc_median"]
    daily = daily.dropna()

    ax = axes[0, 0]
    ax.plot(daily["date"], daily["sc_median"], alpha=0.4, linewidth=0.5, color="steelblue")
    rolling = daily.set_index("date")["sc_median"].rolling("90D").median()
    ax.plot(rolling.index, rolling.values, color="navy", linewidth=2, label="90-day rolling median")
    ax.axvline(pd.Timestamp("2022-01-01"), color="red", linestyle="--", linewidth=1.5,
               label="Pre/post 2022 split")
    ax.set_xlabel("Date")
    ax.set_ylabel("Specific Conductance (µS/cm)")
    ax.set_title("PRPO: Specific Conductance Over Time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Per-year stats
    ax = axes[0, 1]
    yrs = [s["year"] for s in year_stats]
    sc_med = [s["sc_median"] for s in year_stats]
    qf_rate = [s["qf_pass_rate"] * 100 for s in year_stats]
    ax2 = ax.twinx()
    ax.bar(yrs, sc_med, alpha=0.7, color="steelblue", label="SpCond median")
    ax2.plot(yrs, qf_rate, "r^-", linewidth=1.5, markersize=7, label="QF pass rate %")
    ax.set_xlabel("Year")
    ax.set_ylabel("SpCond median (µS/cm)", color="steelblue")
    ax2.set_ylabel("QF pass rate (%)", color="red")
    ax.set_title("Per-Year SpCond and QF Pass Rate")
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="red")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # Pre/post distributions
    ax = axes[1, 0]
    pre_v  = prpo[prpo["year"] < 2022]["specificConductance"].dropna().values
    post_v = prpo[prpo["year"] >= 2022]["specificConductance"].dropna().values
    if len(pre_v) == 0 or len(post_v) == 0:
        ax.text(0.5, 0.5, "Insufficient data for pre/post split", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("SpCond Distribution (no data)")
    else:
        lo = min(pre_v.min(), post_v.min())
        hi = max(np.percentile(pre_v, 99), np.percentile(post_v, 99))
        bins = np.linspace(lo, hi, 60)
        ax.hist(pre_v, bins=bins, alpha=0.6, color="steelblue", density=True,
                label=f"Pre-2022 (median={np.median(pre_v):.0f})")
        ax.hist(post_v, bins=bins, alpha=0.6, color="tomato", density=True,
                label=f"Post-2022 (median={np.median(post_v):.0f})")
        ax.set_xlabel("Specific Conductance (µS/cm)")
        ax.set_ylabel("Density")
        ax.set_title(f"SpCond Distribution Shift (p={pre_post.get('p_value', np.nan):.2e})")
        ax.legend(fontsize=8)

    # Corroborating params
    ax = axes[1, 1]
    # Show QF correlation scatter
    sc_vals = [s["sc_median"] for s in year_stats]
    qf_vals = [s["qf_pass_rate"] for s in year_stats]
    yr_vals = [s["year"] for s in year_stats]
    sc_arr = np.array(sc_vals)
    cm = plt.get_cmap("RdYlGn")
    colors = cm(np.linspace(0, 1, len(yr_vals)))
    for i, (sc_v, qf_v, yr_v) in enumerate(zip(sc_vals, qf_vals, yr_vals)):
        ax.scatter(sc_v, qf_v * 100, color=colors[i], s=80, zorder=5)
        ax.annotate(str(yr_v), (sc_v, qf_v * 100), fontsize=8, ha="right")
    ax.set_xlabel("SpCond median (µS/cm)")
    ax.set_ylabel("QF pass rate (%)")
    ax.set_title(f"SpCond vs QF Pass Rate\n(r={qf_corr.get('sc_qf_correlation', np.nan):.3f},"
                 f" p={qf_corr.get('sc_qf_pvalue', np.nan):.3f})")
    ax.grid(alpha=0.3)

    plt.suptitle("PRPO Site Audit: Is -242 µS/cm/year SpCond Trend Real or Artifact?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = FIGURES_DIR / "fig_exp13_prpo_audit.jpg"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", pil_kwargs={"quality": 85})
    plt.close(fig)
    logger.info(f"Saved: {path}")


def main():
    t0 = time.time()
    logger.info("=" * 65)
    logger.info("EXPERIMENT 13: PRPO Data Audit (Critique 8)")
    logger.info("=" * 65)

    prpo = load_prpo_data()
    if len(prpo) < 100:
        logger.error("Insufficient PRPO data")
        return

    logger.info("\n--- Per-year statistics ---")
    year_stats = year_statistics(prpo)
    for s in year_stats:
        logger.info(f"  {s['year']}: SpCond={s['sc_median']:.1f} µS/cm, "
                    f"QF_pass={s['qf_pass_rate']:.2%}, n={s['n_valid_sc']}")

    logger.info("\n--- Pre/post 2022 Mann-Whitney test ---")
    pre_post = pre_post_2022_test(prpo)
    logger.info(f"  Pre-2022 median: {pre_post.get('median_pre_2022', np.nan):.1f} µS/cm")
    logger.info(f"  Post-2022 median: {pre_post.get('median_post_2022', np.nan):.1f} µS/cm")
    logger.info(f"  p-value: {pre_post.get('p_value', np.nan):.4e}")

    logger.info("\n--- QF correlation analysis ---")
    qf_corr = qf_trend_correlation(year_stats)
    logger.info(f"  SpCond vs QF corr: r={qf_corr.get('sc_qf_correlation', np.nan):.3f}, "
                f"p={qf_corr.get('sc_qf_pvalue', np.nan):.3f}")
    logger.info(f"  {qf_corr.get('interpretation', '')}")

    logger.info("\n--- Corroborating parameters ---")
    corr_params = corroborating_params(prpo)
    for param, res in corr_params.items():
        if "error" not in res:
            logger.info(f"  {param}: {res['mean_pre']:.2f} → {res['mean_post']:.2f} "
                        f"({res['pct_change']:+.1f}%), p={res['p_value']:.4e}")

    # Verdict
    sc_qf_r = qf_corr.get("sc_qf_correlation", 0)
    pre_post_sig = pre_post.get("significant", False)
    verdicts = []

    if abs(sc_qf_r) > 0.5 and qf_corr.get("sc_qf_pvalue", 1) < 0.05:
        verdicts.append(
            f"POSSIBLE ARTIFACT: SpCond and QF pass rate correlated "
            f"(r={sc_qf_r:.2f}); declining SpCond coincides with declining data quality"
        )
    else:
        verdicts.append(
            "QF rates NOT strongly correlated with SpCond trend — artifact less likely"
        )

    n_corroborated = sum(
        1 for res in corr_params.values()
        if res.get("significant") and res.get("pct_change", 0) < -5
    )
    if n_corroborated >= 2:
        verdicts.append(
            f"REAL SIGNAL SUPPORTED: {n_corroborated} other parameters also show "
            "significant decline post-2022 — consistent with real hydrological change"
        )
    elif n_corroborated == 0:
        verdicts.append(
            "OTHER PARAMETERS UNCHANGED: No corroborating decline in pH/DO/turbidity "
            "— SpCond decline may be conductivity-specific (sensor artifact)"
        )

    logger.info("\n=== VERDICT ===")
    for v in verdicts:
        logger.info(f"  • {v}")

    summary = {
        "site": "PRPO",
        "year_statistics": year_stats,
        "pre_post_2022_test": pre_post,
        "qf_trend_correlation": qf_corr,
        "corroborating_parameters": corr_params,
        "verdicts": verdicts,
        "critique_addressed": "Exp8 reported PRPO SpCond = -242.3 µS/cm/year. "
            "This audit checks whether the trend is driven by sensor artifact "
            "(QF failures) or real hydrological change (corroborated by other params).",
        "elapsed_s": round(time.time() - t0, 1),
    }

    out_path = RESULTS_DIR / "prpo_audit_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nResults: {out_path}")

    try:
        plot_prpo_audit(prpo, year_stats, pre_post, qf_corr)
    except Exception as e:
        logger.warning(f"Plot failed (results saved): {e}")

    logger.info(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
