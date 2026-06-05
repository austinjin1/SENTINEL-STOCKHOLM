#!/usr/bin/env python3
"""
SENTINEL 2.0 - Publication Figure Generator
Stockholm Junior Water Prize 2026 Submission

Generates six publication-quality figures from benchmark results.
"""

import matplotlib
matplotlib.use("Agg")

import json
import os
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
BENCH = ROOT / "results" / "benchmarks"
BASELINE_DIR = ROOT / "results" / "exp2_baselines"
PROSPECTIVE_DIR = ROOT / "results" / "prospective" / "predictions"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Color-blind-friendly palette (IBM Design / Wong) ─────────────────────────
COLORS = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "cyan":   "#56B4E9",
    "yellow": "#F0E442",
    "black":  "#000000",
    "gray":   "#999999",
}
PALETTE = list(COLORS.values())

# ── Shared style ─────────────────────────────────────────────────────────────
def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1 - Model Performance Overview
# ═════════════════════════════════════════════════════════════════════════════
def fig1_model_performance():
    """Bar chart showing key metrics for every SENTINEL model."""
    # Load results
    aquassm = load_json(BENCH / "aquassm_v2_holdout.json")
    wdn = load_json(BENCH / "waterdronenet_holdout.json")
    gnn = load_json(BENCH / "stream_gnn_results.json")
    species = load_json(BENCH / "species_health_holdout.json")
    disease = load_json(BENCH / "disease_forecast_holdout.json")
    foundation = load_json(BENCH / "foundation_results.json")
    mome = load_json(BENCH / "mome_fusion_results.json")
    contrastive = load_json(BENCH / "contrastive_results.json")

    # SENTINEL 1.0 numbers from Results Summary (validated results)
    models = [
        "AquaSSM v1\n(Sensor)",
        "HydroViT\n(Satellite)",
        "MicroBiomeNet\n(Microbial)",
        "ToxiGene\n(Molecular)",
        "BioMotion\n(Behavioral)",
        "Perceiver IO\nFusion",
        "Stream GNN\nv2",
        "WaterDroneNet\nv2 (DO)",
        "Species\nHealth",
        "Contrastive\nPretrain",
    ]

    # Key metric per model (best applicable metric, see Results Summary)
    metrics = [
        0.9386,   # AquaSSM v1 AUROC
        0.8927,   # HydroViT R^2
        0.8989,   # MicroBiomeNet F1
        0.9520,   # ToxiGene F1
        0.9999,   # BioMotion AUROC
        0.9919,   # Perceiver IO Fusion AUROC
        gnn["test_metrics"]["auroc"],                            # Stream GNN AUROC ~1.0
        wdn["test_metrics"]["per_target"]["DO"]["r2"],           # WDN DO R^2
        species["test_metrics"]["occ_accuracy"],                 # Species occ accuracy
        contrastive["pair_results"]["microbial-molecular"]["test_retrieval"]["recall@1_a2b"],  # Contrastive recall@1
    ]

    metric_labels = [
        "AUROC", "R\u00b2", "F1", "F1", "AUROC",
        "AUROC", "AUROC", "R\u00b2", "Acc", "Recall@1",
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(models))
    bars = ax.bar(x, metrics, width=0.65, color=PALETTE[:len(models)],
                  edgecolor="white", linewidth=0.5)

    # Value labels on bars
    for bar, val, lab in zip(bars, metrics, metric_labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{val:.3f}\n({lab})", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0, 1.12)
    ax.set_title("SENTINEL 2.0 \u2014 Model Performance Overview")
    ax.axhline(0.9, ls="--", color=COLORS["gray"], lw=0.8, label="0.9 threshold")
    ax.legend(loc="lower right")

    out = FIG_DIR / "fig1_model_performance.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2 - Case Study Detection Timeline
# ═════════════════════════════════════════════════════════════════════════════
def fig2_case_study_timeline():
    """Timeline of 6 detected contamination events with lead times."""
    # From SENTINEL_2.0_Results_Summary.md, Section 2
    events = [
        ("Lake Erie HAB 2023",              59.3, 7199,  "DETECTED"),
        ("Gulf of Mexico\nDead Zone 2023",  87.2, 3486,  "DETECTED"),
        ("Chesapeake Bay\nHypoxia 2018",    89.8, 34831, "DETECTED"),
        ("Klamath River\nHAB 2021",         59.2, 7200,  "DETECTED"),
        ("Jordan Lake\nHAB, NC",            44.3, 5755,  "DETECTED"),
        ("Mississippi River\nSalinity 2023", 58.6, 4168, "DETECTED"),
    ]

    names      = [e[0] for e in events]
    lead_times = [e[1] for e in events]
    n_records  = [e[2] for e in events]

    fig, ax = plt.subplots(figsize=(9, 5))

    y = np.arange(len(names))
    bars = ax.barh(y, lead_times, height=0.55, color=COLORS["blue"],
                   edgecolor="white", linewidth=0.5)

    # Annotate lead times and record counts
    for i, (bar, lt, nr) in enumerate(zip(bars, lead_times, n_records)):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{lt:.1f} days  ({nr:,} records)",
                va="center", fontsize=9, color=COLORS["black"])

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Early Warning Lead Time (days)")
    ax.set_title("SENTINEL Case Study Detections \u2014 Real USGS Data")
    ax.invert_yaxis()

    # Mean lead-time line
    mean_lt = np.mean(lead_times)
    ax.axvline(mean_lt, ls="--", color=COLORS["red"], lw=1.2)
    ax.text(mean_lt + 1, len(names) - 0.3,
            f"Mean: {mean_lt:.1f} d", color=COLORS["red"], fontsize=9)

    ax.set_xlim(0, max(lead_times) + 30)

    out = FIG_DIR / "fig2_case_study_timeline.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3 - Holdout Baselines Comparison
# ═════════════════════════════════════════════════════════════════════════════
def fig3_baselines():
    """SENTINEL vs baselines on holdout set (honest AUROC comparison)."""
    bl = load_json(BASELINE_DIR / "baseline_comparison.json")

    methods = []
    aurocs = []
    # Maintain a meaningful order: baselines first, then SENTINEL variants
    order = ["Z-score", "ARIMA", "Isolation Forest", "AquaSSM-only", "SENTINEL"]
    for name in order:
        if name in bl:
            methods.append(name)
            aurocs.append(bl[name]["auroc"])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))

    bar_colors = [COLORS["gray"]] * 3 + [COLORS["cyan"], COLORS["blue"]]
    bars = ax.bar(x, aurocs, width=0.55, color=bar_colors,
                  edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 0.85)
    ax.set_title("Holdout Baseline Comparison (USGS \u2192 NEON Cross-Source)")
    ax.axhline(0.5, ls=":", color=COLORS["gray"], lw=0.8, label="Random baseline (0.5)")
    ax.legend(loc="upper left")

    # Add annotation about cross-source gap
    ax.text(0.98, 0.95,
            "Note: AquaSSM & SENTINEL trained on USGS,\nevaluated on NEON (cross-source gap expected)",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            style="italic", color=COLORS["gray"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["gray"], alpha=0.7))

    out = FIG_DIR / "fig3_baselines_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 4 - WaterDroneNet Per-Target Performance
# ═════════════════════════════════════════════════════════════════════════════
def fig4_waterdronenet():
    """R^2 for each water quality parameter predicted by WaterDroneNet."""
    wdn = load_json(BENCH / "waterdronenet_holdout.json")
    per_target = wdn["test_metrics"]["per_target"]

    targets = list(per_target.keys())
    full_names = {
        "DO": "Dissolved\nOxygen",
        "pH": "pH",
        "Turb": "Turbidity",
        "Temp": "Temperature",
        "SpCond": "Specific\nConductance",
    }
    r2_vals = [per_target[t]["r2"] for t in targets]
    mae_vals = [per_target[t]["mae"] for t in targets]
    labels = [full_names.get(t, t) for t in targets]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"wspace": 0.35})

    # Left panel: R^2
    bar_colors = [COLORS["blue"] if v > 0.9 else COLORS["orange"] if v > 0.7
                  else COLORS["red"] for v in r2_vals]
    x = np.arange(len(targets))
    bars = ax1.bar(x, r2_vals, width=0.55, color=bar_colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, r2_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("R\u00b2")
    ax1.set_ylim(0, 1.12)
    ax1.set_title("Test R\u00b2 by Target")
    ax1.axhline(0.9, ls="--", color=COLORS["gray"], lw=0.8)

    # Right panel: MAE
    bars2 = ax2.bar(x, mae_vals, width=0.55, color=PALETTE[:len(targets)],
                    edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars2, mae_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("MAE")
    ax2.set_title("Test MAE by Target")

    fig.suptitle("WaterDroneNet v2 \u2014 Per-Target Test Performance (165K samples)",
                 fontsize=13, fontweight="bold", y=1.02)

    out = FIG_DIR / "fig4_waterdronenet_targets.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 5 - Prospective Validation Timeline (Chattahoochee)
# ═════════════════════════════════════════════════════════════════════════════
def fig5_prospective():
    """Chattahoochee River anomaly score trajectory across prediction runs."""
    pred_files = sorted(PROSPECTIVE_DIR.glob("predictions_*.json"))

    run_dates = []
    mean_scores = []
    max_scores = []

    for pf in pred_files:
        data = load_json(pf)
        dt = datetime.fromisoformat(data["prediction_date"].replace("+00:00", ""))
        for pred in data["predictions"]:
            if pred["site_id"] == "02336000":
                run_dates.append(dt)
                mean_scores.append(pred["anomaly_scores"]["mean_probability"])
                max_scores.append(pred["anomaly_scores"]["max_probability"])
                break

    if not run_dates:
        print("  WARNING: No Chattahoochee data found. Skipping fig5.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    run_labels = [d.strftime("%m/%d\n%H:%M") for d in run_dates]
    x = np.arange(len(run_dates))

    ax.plot(x, max_scores, "o-", color=COLORS["red"], lw=2,
            markersize=7, label="Max window probability", zorder=5)
    ax.plot(x, mean_scores, "s-", color=COLORS["blue"], lw=2,
            markersize=6, label="Mean probability", zorder=5)

    ax.fill_between(x, mean_scores, max_scores, alpha=0.12, color=COLORS["blue"])

    # Threshold lines
    ax.axhline(0.9, ls="--", color=COLORS["red"], lw=0.8, alpha=0.6, label="Alert threshold (0.9)")
    ax.axhline(0.3, ls=":", color=COLORS["orange"], lw=0.8, alpha=0.6, label="Watch threshold (0.3)")

    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, fontsize=8)
    ax.set_xlabel("Prediction Run (UTC)")
    ax.set_ylabel("Anomaly Probability")
    ax.set_ylim(0, 1.0)
    ax.set_title("Prospective Validation \u2014 Chattahoochee River at Atlanta (02336000)")
    ax.legend(loc="upper left", fontsize=9)

    # Annotate the peak
    peak_idx = int(np.argmax(max_scores))
    ax.annotate(f"Peak: {max_scores[peak_idx]:.3f}",
                xy=(peak_idx, max_scores[peak_idx]),
                xytext=(peak_idx - 1, max_scores[peak_idx] + 0.12),
                arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.2),
                fontsize=9, color=COLORS["red"], fontweight="bold")

    out = FIG_DIR / "fig5_prospective_chattahoochee.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 6 - Predictability Audit
# ═════════════════════════════════════════════════════════════════════════════
def fig6_predictability_audit():
    """What is learnable from cheap scalar sensors (T + conductivity) alone?"""
    audit = load_json(BENCH / "predictability_audit.json")
    results = audit["results"]

    targets = [r["target"].replace("_", " ").title() for r in results]
    r2_vals = [r["r2_scalar"] for r in results]
    verdicts = [r["verdict"] for r in results]

    verdict_colors = {
        "INPUT":         COLORS["cyan"],
        "STRONG":        COLORS["green"],
        "MODERATE":      COLORS["orange"],
        "WEAK":          COLORS["red"],
        "UNRECOVERABLE": COLORS["purple"],
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(targets))
    bar_colors = [verdict_colors.get(v, COLORS["gray"]) for v in verdicts]

    bars = ax.bar(x, r2_vals, width=0.55, color=bar_colors, edgecolor="white", linewidth=0.5)

    # Value labels
    for bar, val, v in zip(bars, r2_vals, verdicts):
        y_pos = max(bar.get_height(), 0) + 0.08
        if val < 0:
            y_pos = 0.08
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"R\u00b2={val:.3f}\n{v}", ha="center", va="bottom", fontsize=8,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontsize=9)
    ax.set_ylabel("R\u00b2 (Scalar MLP from Temp + Conductivity)")
    ax.set_title("Predictability Audit \u2014 What Is Learnable from Sensor Data Alone?")

    # Reference lines
    ax.axhline(0.0, ls="-", color=COLORS["black"], lw=0.8, alpha=0.4)
    ax.axhline(0.7, ls="--", color=COLORS["green"], lw=0.8, alpha=0.5, label="STRONG (R\u00b2>0.70)")
    ax.axhline(0.3, ls="--", color=COLORS["orange"], lw=0.8, alpha=0.5, label="MODERATE (R\u00b2 0.30-0.70)")

    # Set y-lim to show negative values but cap extreme negatives
    y_min = max(min(r2_vals) - 0.5, -7.5)
    ax.set_ylim(y_min, 1.3)
    ax.legend(loc="lower right", fontsize=8)

    # Annotation
    ax.text(0.98, 0.02,
            "DO, pH, Turbidity require multimodal data\n(satellite, microbial) to predict",
            transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
            style="italic", color=COLORS["gray"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["gray"], alpha=0.7))

    out = FIG_DIR / "fig6_predictability_audit.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 7 - Digital Twin Training Convergence
# ═════════════════════════════════════════════════════════════════════════════
def fig7_twin_convergence():
    """Digital Twin v2 training convergence: Phase 1 vs Phase 2 ODE fine-tuning."""
    import re

    log_path = ROOT / "logs" / "train_twin_v2.log"
    if not log_path.exists():
        print("  WARNING: train_twin_v2.log not found. Skipping fig7.")
        return

    text = log_path.read_text()

    # Parse all epoch lines
    pattern = re.compile(
        r"(P[12]) Epoch\s+(\d+)/\d+ \| Train: ([\d.]+) \| Val: ([\d.]+) "
        r"\| 1d_mse: ([\d.]+) \| 7d_mse: ([\d.]+) \| 14d_mse: ([\d.]+)"
    )

    p1_epochs, p1_val, p1_1d, p1_7d, p1_14d = [], [], [], [], []
    p2_epochs, p2_val, p2_1d, p2_7d, p2_14d = [], [], [], [], []

    for m in pattern.finditer(text):
        phase, ep, _, val, d1, d7, d14 = m.groups()
        ep = int(ep)
        if phase == "P1":
            p1_epochs.append(ep)
            p1_val.append(float(val))
            p1_1d.append(float(d1))
            p1_7d.append(float(d7))
            p1_14d.append(float(d14))
        else:
            p2_epochs.append(ep + max(p1_epochs, default=0))
            p2_val.append(float(val))
            p2_1d.append(float(d1))
            p2_7d.append(float(d7))
            p2_14d.append(float(d14))

    if not p1_epochs:
        print("  WARNING: No training data parsed from log. Skipping fig7.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    # Plot 1-day, 7-day, 14-day MSE
    for ax, p1_data, p2_data, title in zip(
        axes,
        [p1_1d, p1_7d, p1_14d],
        [p2_1d, p2_7d, p2_14d],
        ["1-Day Forecast MSE", "7-Day Forecast MSE", "14-Day Forecast MSE"],
    ):
        ax.plot(p1_epochs, p1_data, "o-", color=COLORS["cyan"], lw=1.5,
                markersize=3, label="Phase 1: Assimilator", alpha=0.8)
        if p2_data:
            ax.plot(p2_epochs, p2_data, "s-", color=COLORS["blue"], lw=1.5,
                    markersize=3, label="Phase 2: ODE Fine-Tune", alpha=0.8)
            # Mark phase boundary
            boundary = max(p1_epochs, default=0) + 0.5
            ax.axvline(boundary, ls="--", color=COLORS["orange"], lw=1, alpha=0.6)
            ax.text(boundary + 0.5, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else max(p1_data) * 0.95,
                    "ODE\nunfrozen", fontsize=7, color=COLORS["orange"], va="top")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Digital Twin v2 \u2014 Training Convergence (Phase 1 \u2192 Phase 2 ODE Fine-Tuning)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = FIG_DIR / "fig7_twin_convergence.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 8 - Stream Network GNN Coverage Map
# ═════════════════════════════════════════════════════════════════════════════
def fig8_stream_network():
    """Map of 561 NHDPlus monitoring sites with stream connections."""
    graph_path = ROOT / "data" / "processed" / "hydrology" / "nhdplus_graph.json"
    if not graph_path.exists():
        print("  WARNING: nhdplus_graph.json not found. Skipping fig8.")
        return

    with open(graph_path) as f:
        graph = json.load(f)

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    if not nodes:
        print("  WARNING: No nodes in graph. Skipping fig8.")
        return

    # Build lookup
    site_coords = {n["site_id"]: (n["lon"], n["lat"]) for n in nodes}
    stream_orders = {n["site_id"]: n.get("stream_order", 0) for n in nodes}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                    gridspec_kw={"width_ratios": [2.2, 1]})

    # Left: Geographic scatter with edges
    order_colors = {
        1: COLORS["cyan"], 2: COLORS["cyan"],
        3: COLORS["blue"], 4: COLORS["blue"],
        5: COLORS["green"], 6: COLORS["green"],
        7: COLORS["orange"], 8: COLORS["red"],
    }

    # Draw edges first (lighter)
    for edge in edges:
        f_id, t_id = edge["from_site"], edge["to_site"]
        if f_id in site_coords and t_id in site_coords:
            x0, y0 = site_coords[f_id]
            x1, y1 = site_coords[t_id]
            ax1.plot([x0, x1], [y0, y1], "-", color=COLORS["gray"],
                     lw=0.3, alpha=0.4, zorder=1)

    # Draw nodes colored by stream order
    for n in nodes:
        so = n.get("stream_order") or 1
        c = order_colors.get(so, COLORS["gray"])
        size = 3 + so * 1.5
        ax1.scatter(n["lon"], n["lat"], c=c, s=size, alpha=0.7,
                    edgecolors="none", zorder=2)

    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_title(f"SENTINEL Stream Network — {len(nodes)} Sites, {len(edges)} Connections")

    # Legend for stream orders
    for so, label in [(1, "Order 1-2"), (3, "Order 3-4"),
                      (5, "Order 5-6"), (7, "Order 7-8")]:
        ax1.scatter([], [], c=order_colors[so], s=3 + so * 1.5,
                    label=label)
    ax1.legend(loc="lower left", fontsize=8, title="Stream Order")

    # Right: Stream order distribution
    stats_path = ROOT / "data" / "processed" / "hydrology" / "nhdplus_graph_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        dist = stats.get("stream_order", {}).get("distribution", {})
        orders = sorted(dist.keys(), key=int)
        counts = [dist[o] for o in orders]
        bar_colors = [order_colors.get(int(o), COLORS["gray"]) for o in orders]

        x = np.arange(len(orders))
        bars = ax2.bar(x, counts, width=0.6, color=bar_colors,
                       edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     str(val), ha="center", va="bottom", fontsize=9)
        ax2.set_xticks(x)
        ax2.set_xticklabels(orders)
        ax2.set_xlabel("Stream Order")
        ax2.set_ylabel("Number of Sites")
        ax2.set_title("Site Distribution by Stream Order")

        # Add travel time annotation
        tt = stats.get("travel_time_hours", {})
        ax2.text(0.95, 0.95,
                 f"Travel time:\n  Mean: {tt.get('mean', 0):.1f} h\n  Max: {tt.get('max', 0):.0f} h",
                 transform=ax2.transAxes, fontsize=8, ha="right", va="top",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["gray"], alpha=0.7))

    fig.suptitle("Stream Network GNN — Real NHDPlusV2 Topology (AUROC = 1.000)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = FIG_DIR / "fig8_stream_network.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 9 - System Architecture Diagram
# ═════════════════════════════════════════════════════════════════════════════
def fig9_architecture():
    """SENTINEL 2.0 system architecture diagram."""
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=8, alpha=0.85):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                              facecolor=color, edgecolor="white",
                              linewidth=1.5, alpha=alpha, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white", zorder=4,
                wrap=True)

    def section_bg(x, y, w, h, label, color, alpha=0.1):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                              facecolor=color, edgecolor=color,
                              linewidth=1.5, alpha=alpha, zorder=1)
        ax.add_patch(rect)
        ax.text(x + 0.5, y + h - 0.15, label, fontsize=9, fontweight="bold",
                color=color, alpha=0.8, va="top", zorder=2)

    def arrow(x1, y1, x2, y2, color=COLORS["gray"]):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                    zorder=2)

    # ── Section backgrounds ──
    # Data Sources (bottom)
    section_bg(0.2, 0.2, 13.5, 1.6, "Data Sources (390M+ records)", COLORS["cyan"])
    # Encoders (lower middle)
    section_bg(0.2, 2.2, 13.5, 1.6, "SENTINEL 1.0 Encoders", COLORS["blue"])
    # Fusion + New Models (middle)
    section_bg(0.2, 4.2, 13.5, 1.8, "SENTINEL 2.0 Fusion & Spatial Reasoning", COLORS["green"])
    # Prediction Layers (upper middle)
    section_bg(0.2, 6.4, 13.5, 1.6, "Prediction & Digital Twin", COLORS["orange"])
    # Deployment (top)
    section_bg(0.2, 8.4, 13.5, 1.3, "Deployment & Monitoring", COLORS["red"])

    # ── Data Sources row ──
    ds_y, ds_h = 0.4, 0.6
    box(0.5,  ds_y, 2.0, ds_h, "USGS NWIS\n383 stations", COLORS["cyan"], 7)
    box(2.7,  ds_y, 1.8, ds_h, "NEON\n32 sites", COLORS["cyan"], 7)
    box(4.7,  ds_y, 1.8, ds_h, "Sentinel-2\nImagery", COLORS["cyan"], 7)
    box(6.7,  ds_y, 1.8, ds_h, "16S rRNA\nMicrobial", COLORS["cyan"], 7)
    box(8.7,  ds_y, 1.8, ds_h, "RNA-seq\nMolecular", COLORS["cyan"], 7)
    box(10.7, ds_y, 1.5, ds_h, "NHDPlus\n561 sites", COLORS["cyan"], 7)
    box(12.4, ds_y, 1.2, ds_h, "BioData\n701K", COLORS["cyan"], 7)
    # Second row of data
    ds2_y = 1.1
    box(0.5,  ds2_y, 2.0, 0.5, "NOAA HABs (901K)", COLORS["cyan"], 7)
    box(2.7,  ds2_y, 1.8, 0.5, "ECOTOX", COLORS["cyan"], 7)
    box(4.7,  ds2_y, 1.8, 0.5, "Fish Video", COLORS["cyan"], 7)

    # ── Encoder row ──
    enc_y, enc_h = 2.5, 0.7
    box(0.5,  enc_y, 2.2, enc_h, "AquaSSM v2\nAUROC 0.94", COLORS["blue"], 7)
    box(2.9,  enc_y, 2.2, enc_h, "HydroViT\nR\u00b2 0.89", COLORS["blue"], 7)
    box(5.3,  enc_y, 2.2, enc_h, "MicroBiomeNet\nF1 0.90", COLORS["blue"], 7)
    box(7.7,  enc_y, 2.2, enc_h, "ToxiGene\nF1 0.95", COLORS["blue"], 7)
    box(10.1, enc_y, 2.2, enc_h, "BioMotion\nAUROC 1.00", COLORS["blue"], 7)

    # Arrows from data to encoders
    for dx, ex in [(1.5, 1.6), (3.6, 4.0), (7.6, 6.4), (9.6, 8.8), (5.6, 11.2)]:
        arrow(dx, 1.0, ex, enc_y)

    # ── Fusion + spatial row ──
    fus_y, fus_h = 4.5, 0.8
    box(0.5,  fus_y, 2.5, fus_h, "Perceiver IO Fusion\nAUROC 0.99", COLORS["green"], 7)
    box(3.3,  fus_y, 2.5, fus_h, "Foundation Model\n(Joint Pretrain)", COLORS["green"], 7)
    box(6.1,  fus_y, 2.5, fus_h, "Stream GNN v2\nAUROC 1.00", COLORS["green"], 7)
    box(9.0,  fus_y, 2.2, fus_h, "WaterDroneNet\nDO R\u00b2 0.98", COLORS["green"], 7)
    box(11.5, fus_y, 2.0, fus_h, "Contrastive\nAlign", COLORS["green"], 7)

    # Arrows from encoders to fusion
    for ex in [1.6, 4.0, 6.4, 8.8, 11.2]:
        arrow(ex, enc_y + enc_h, ex if ex < 9 else 10.1, fus_y)

    # ── Prediction row ──
    pred_y, pred_h = 6.7, 0.7
    box(0.5,  pred_y, 2.5, pred_h, "Digital Twin v2\n(Neural-ODE)", COLORS["orange"], 7)
    box(3.3,  pred_y, 2.2, pred_h, "Species Health\nR\u00b2 0.42", COLORS["orange"], 7)
    box(5.8,  pred_y, 2.2, pred_h, "Disease Forecast\n4 pathogens", COLORS["orange"], 7)
    box(8.3,  pred_y, 2.2, pred_h, "Climate Coupling\nDO MAE 1.51", COLORS["orange"], 7)
    box(10.8, pred_y, 2.8, pred_h, "Counterfactual\n& Restoration", COLORS["orange"], 7)

    # Arrows from fusion to prediction
    for fx in [1.75, 4.55, 7.35, 10.1]:
        arrow(fx, fus_y + fus_h, fx, pred_y)

    # ── Deployment row ──
    dep_y, dep_h = 8.6, 0.7
    box(0.5,  dep_y, 2.5, dep_h, "Streamlit Dashboard\n& REST API", COLORS["red"], 7)
    box(3.3,  dep_y, 2.5, dep_h, "Prospective\nValidation (18 sites)", COLORS["red"], 7)
    box(6.1,  dep_y, 2.5, dep_h, "Cascade Escalation\n(PPO Controller)", COLORS["red"], 7)
    box(8.9,  dep_y, 2.2, dep_h, "Citizen eDNA\nNetwork", COLORS["red"], 7)
    box(11.4, dep_y, 2.2, dep_h, "EJ Overlay\n& Docker/CI", COLORS["red"], 7)

    # Arrows from prediction to deployment
    for px in [1.75, 4.55, 7.35, 9.5]:
        arrow(px, pred_y + pred_h, px, dep_y)

    # Title
    ax.text(7, 9.85, "SENTINEL 2.0 — System Architecture",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=COLORS["black"])
    ax.text(7, 9.55, "The First Operational Digital Twin of Freshwater Ecosystems",
            ha="center", va="center", fontsize=10, style="italic",
            color=COLORS["gray"])

    out = FIG_DIR / "fig9_architecture.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    apply_style()
    print("SENTINEL 2.0 - Generating submission figures...")
    print(f"  Output directory: {FIG_DIR}\n")

    fig1_model_performance()
    fig2_case_study_timeline()
    fig3_baselines()
    fig4_waterdronenet()
    fig5_prospective()
    fig6_predictability_audit()
    fig7_twin_convergence()
    fig8_stream_network()
    fig9_architecture()

    print(f"\nDone. {len(list(FIG_DIR.glob('fig*.png')))} figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
