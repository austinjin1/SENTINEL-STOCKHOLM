"""
Generate publication-quality figures for SENTINEL SJWP paper (v3).
Updates for expanded training, new case studies, and baseline benchmarks.

Outputs 5 figures to paper/figures/ at 300 DPI JPG.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
FIGOUT = PROJECT / "paper" / "figures"
FIGOUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def fig_bootstrap_ci():
    """Fig 2: Updated forest plot with expanded-training CIs."""
    with open(RESULTS / "exp9_bootstrap" / "ci_results.json") as f:
        data = json.load(f)
    ci = data["ci_results"]

    rows = [
        ("AquaSSM (Sensor)",          ci["AquaSSM"]["point"],       ci["AquaSSM"]["ci_lo"],       ci["AquaSSM"]["ci_hi"],       "AUROC"),
        ("HydroViT (Satellite)",      ci["HydroViT"]["point"],      ci["HydroViT"]["ci_lo"],      ci["HydroViT"]["ci_hi"],      "R\u00b2"),
        ("MicroBiomeNet (Microbial)", ci["MicroBiomeNet"]["point"], ci["MicroBiomeNet"]["ci_lo"], ci["MicroBiomeNet"]["ci_hi"], "F1"),
        ("ToxiGene (Molecular)",      ci["ToxiGene"]["point"],      ci["ToxiGene"]["ci_lo"],      ci["ToxiGene"]["ci_hi"],      "F1"),
        ("BioMotion (Behavioral)",    ci["BioMotion"]["point"],     ci["BioMotion"]["ci_lo"],     ci["BioMotion"]["ci_hi"],     "AUROC"),
        ("SENTINEL-Fusion",           ci["Fusion"]["point"],        ci["Fusion"]["ci_lo"],        ci["Fusion"]["ci_hi"],        "AUROC"),
    ]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f4a582', '#b2182b']
    thresholds = [0.85, 0.55, 0.70, 0.80, 0.80, 0.90]

    for i, (name, point, lo, hi, metric) in enumerate(rows):
        xerr_lo = max(0, point - lo)
        xerr_hi = max(0, hi - point)
        ax.errorbar(point, i, xerr=[[xerr_lo], [xerr_hi]],
                     fmt='o', color=colors[i], markersize=8, capsize=5,
                     capthick=1.5, elinewidth=1.5, markeredgecolor='black',
                     markeredgewidth=0.5)
        ax.annotate(f'{point:.3f} [{lo:.3f}, {hi:.3f}]',
                     xy=(point, i), xytext=(12, 0),
                     textcoords='offset points', fontsize=8.5,
                     va='center', color='#333333')
        ax.plot(thresholds[i], i, marker='|', color='red', markersize=12, markeredgewidth=1.5)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel('Performance (metric-specific)')
    ax.set_title('Bootstrap 95% Confidence Intervals (2,000 iterations)', fontweight='bold')
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='Random baseline')
    ax.plot([], [], marker='|', color='red', markersize=10, markeredgewidth=1.5,
            linestyle='none', label='Threshold')
    ax.legend(loc='lower right', framealpha=0.8)
    ax.set_xlim(0.45, 1.08)
    ax.invert_yaxis()

    plt.tight_layout()
    out = FIGOUT / "fig2_bootstrap_ci.jpg"
    fig.savefig(out, format='jpeg', dpi=300)
    plt.close(fig)
    print(f"  {out.name} ({out.stat().st_size / 1024:.0f} KB)")


def fig_case_studies():
    """Fig 4: 10 unique event detection timelines — all positive lead times."""
    with open(RESULTS / "case_studies" / "summary.json") as f:
        data = json.load(f)

    # Deduplicate by event_id (NEON events are tripled in the JSON)
    seen = set()
    unique_events = []
    for e in data["per_event"]:
        eid = e["event_id"]
        if eid not in seen:
            seen.add(eid)
            unique_events.append(e)
    unique_events.sort(key=lambda e: e["lead_time_hours"])

    # Short display names
    short_names = {
        "lake_erie_hab": "Lake Erie HAB",
        "toledo_water_crisis": "Toledo Water Crisis",
        "gulf_dead_zone": "Gulf of Mexico Dead Zone",
        "chesapeake_bay_blooms": "Chesapeake Bay Blooms",
        "neon_pose_do_depletion_2025": "POSE: DO Depletion (Summer '25)",
        "neon_blde_storm_conductance_2024": "BLDE: Storm Conductance (Fall '24)",
        "neon_mart_turbidity_2025": "MART: Snowmelt Turbidity (Spring '25)",
        "neon_barc_eutrophication_2025": "BARC: Eutrophication/HAB (Aug '25)",
        "neon_leco_acid_runoff_2024": "LECO: Acid Runoff (Spring '24)",
        "neon_sugg_conductance_2024": "SUGG: Agricultural Runoff (Fall '24)",
    }

    names = []
    times = []
    is_neon = []
    for e in unique_events:
        names.append(short_names.get(e["event_id"], e["event_name"]))
        times.append(e["lead_time_hours"])
        is_neon.append(e.get("data_source") == "NEON_real_sensor_data")

    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Color: NEON events in teal, historical in green
    colors = ['#00796b' if n else '#2e7d32' for n in is_neon]
    bars = ax.barh(range(len(names)), times, color=colors,
                    edgecolor='black', linewidth=0.4, height=0.7)

    # Value labels
    for i, t in enumerate(times):
        if t > 800:
            label = f'+{t / 24:.0f}d'
        else:
            label = f'+{t:.0f}h'
        ax.text(t + max(times) * 0.01, i, label, va='center', fontsize=7.5,
                color='#1b5e20', fontweight='bold')

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel('Detection Lead Time (hours before official detection)')
    ax.set_title('SENTINEL Early Warning: 10/10 Events Detected Before Official Report',
                  fontweight='bold', fontsize=11)

    # Legend
    neon_patch = mpatches.Patch(color='#00796b', label='NEON real sensor events (6)')
    hist_patch = mpatches.Patch(color='#2e7d32', label='Historical case studies (4)')
    ax.legend(handles=[hist_patch, neon_patch], loc='lower right', framealpha=0.9, fontsize=9)

    # Mean annotation — compute from unique events
    mean_lt = np.mean(times)
    ax.axvline(x=mean_lt, color='#4575b4', linestyle='--', alpha=0.7)
    ax.text(mean_lt, -0.8, f'Mean: {mean_lt:.0f}h ({mean_lt/24:.1f}d)',
            color='#4575b4', fontsize=8, ha='center', fontweight='bold')

    plt.tight_layout()
    out = FIGOUT / "fig4_case_studies.jpg"
    fig.savefig(out, format='jpeg', dpi=300)
    plt.close(fig)
    print(f"  {out.name} ({out.stat().st_size / 1024:.0f} KB)")


def fig_risk_index():
    """Fig 8: Updated risk index ranking (32 NEON sites)."""
    with open(RESULTS / "exp17_risk_index" / "risk_index_results.json") as f:
        data = json.load(f)

    sites = data["ranked_sites"]
    names = [s["site"] for s in sites]
    scores = [s["composite_score"] for s in sites]
    tiers = [s["tier"] for s in sites]

    tier_colors = {5: '#8B0000', 4: '#CC2200', 3: '#E87722', 2: '#FFB347', 1: '#90EE90'}
    tier_names = {5: 'Critical', 4: 'High', 3: 'Elevated', 2: 'Moderate', 1: 'Low'}

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = [tier_colors[t] for t in tiers]
    ax.bar(range(len(names)), scores, color=bar_colors, edgecolor='black', linewidth=0.3, width=0.8)

    for val in [0.70, 0.55, 0.40, 0.25]:
        ax.axhline(y=val, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=55, ha='right', fontsize=7.5)
    ax.set_ylabel('Composite Risk Score')
    ax.set_xlabel('NEON Monitoring Site')
    ax.set_title('Water Quality Risk Index: 32 NEON Sites (3 Critical, 3 High, 22 Elevated)',
                  fontweight='bold')
    ax.set_ylim(0, 0.95)

    legend_patches = [mpatches.Patch(color=tier_colors[t], label=f'Tier {t}: {tier_names[t]}')
                      for t in [5, 4, 3, 2, 1]]
    ax.legend(handles=legend_patches, loc='upper right', framealpha=0.9, fontsize=9)

    plt.tight_layout()
    out = FIGOUT / "fig8_risk_ranking.jpg"
    fig.savefig(out, format='jpeg', dpi=300)
    plt.close(fig)
    print(f"  {out.name} ({out.stat().st_size / 1024:.0f} KB)")


def fig_baseline_aquassm():
    """NEW: AquaSSM vs 4 baselines — AUROC and F1."""
    with open(RESULTS / "benchmarks" / "aquassm_benchmark.json") as f:
        data = json.load(f)

    models_order = ["AquaSSM", "OneClassSVM", "LSTM", "Transformer", "IsolationForest"]
    labels = ["AquaSSM\n(Ours)", "One-Class\nSVM", "LSTM\n(2-layer)", "Transformer\n(4-head)", "Isolation\nForest"]

    aurocs = [data["models"][m]["auroc"] for m in models_order]
    f1s = [data["models"][m]["f1"] for m in models_order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)

    colors = ['#b2182b'] + ['#4393c3'] * 4  # Red for ours, blue for baselines

    # AUROC panel
    bars1 = ax1.barh(range(len(labels)), aurocs, color=colors, edgecolor='black',
                      linewidth=0.5, height=0.6)
    for i, v in enumerate(aurocs):
        ax1.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9, fontweight='bold' if i == 0 else 'normal')
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel('AUROC')
    ax1.set_title('Anomaly Detection (AUROC)', fontweight='bold')
    ax1.set_xlim(0.6, 1.0)
    ax1.invert_yaxis()

    # F1 panel
    bars2 = ax2.barh(range(len(labels)), f1s, color=colors, edgecolor='black',
                      linewidth=0.5, height=0.6)
    for i, v in enumerate(f1s):
        ax2.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9, fontweight='bold' if i == 0 else 'normal')
    ax2.set_xlabel('F1 Score')
    ax2.set_title('Classification (F1)', fontweight='bold')
    ax2.set_xlim(0.3, 0.95)
    ax2.invert_yaxis()

    fig.suptitle('AquaSSM vs. Baseline Models on Real USGS Data (n=115 test)',
                  fontweight='bold', fontsize=12, y=1.02)
    plt.tight_layout()
    out = FIGOUT / "fig_baseline_aquassm.jpg"
    fig.savefig(out, format='jpeg', dpi=300)
    plt.close(fig)
    print(f"  {out.name} ({out.stat().st_size / 1024:.0f} KB)")


def fig_baseline_hydrovit():
    """NEW: HydroViT vs 4 baselines — R² for water temperature."""
    with open(RESULTS / "benchmarks" / "hydrovit_benchmark.json") as f:
        data = json.load(f)

    models = [
        ("CNN Baseline",           data["CNN_baseline"]["water_temp_r2"]),
        ("HydroViT v7\n(Ours)",    data["SENTINEL_HydroViT_v7"]["water_temp_r2"]),
        ("ViT (no pretrain)",      data["ViT_no_pretrain"]["water_temp_r2"]),
        ("Random Forest",          data["RandomForest"]["water_temp_r2"]),
        ("Ridge Regression",       data["Ridge"]["water_temp_r2"]),
    ]
    # Sort by R² descending
    models.sort(key=lambda x: x[1], reverse=True)

    names = [m[0] for m in models]
    r2s = [m[1] for m in models]
    colors = ['#b2182b' if 'Ours' in n else '#4393c3' for n in names]

    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(range(len(names)), r2s, color=colors, edgecolor='black',
                    linewidth=0.5, height=0.6)
    for i, v in enumerate(r2s):
        ax.text(v + 0.008, i, f'{v:.3f}', va='center', fontsize=9,
                fontweight='bold' if 'Ours' in names[i] else 'normal')

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('R\u00b2 (Water Temperature)')
    ax.set_title('HydroViT vs. Baseline Models — Satellite WQ Prediction (4,202 pairs)',
                  fontweight='bold', fontsize=11)
    ax.axvline(x=0.55, color='red', linestyle='--', alpha=0.7, label='Threshold (R\u00b2>0.55)')
    ax.legend(loc='lower right', framealpha=0.8)
    ax.set_xlim(0.4, 0.82)
    ax.invert_yaxis()

    plt.tight_layout()
    out = FIGOUT / "fig_baseline_hydrovit.jpg"
    fig.savefig(out, format='jpeg', dpi=300)
    plt.close(fig)
    print(f"  {out.name} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    print("Generating SENTINEL paper figures (v3)...")
    fig_bootstrap_ci()
    fig_case_studies()
    fig_risk_index()
    fig_baseline_aquassm()
    fig_baseline_hydrovit()
    print("Done. 5 figures generated.")
