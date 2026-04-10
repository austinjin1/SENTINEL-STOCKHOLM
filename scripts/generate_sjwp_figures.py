"""
Generate publication-quality figures for SJWP competition paper.
All figures saved as JPEG at quality=85, 150 DPI.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import networkx as nx
from pathlib import Path

# ---------- Setup ----------
OUTDIR = Path(r"C:\Users\zhaoz\AppData\Local\Temp\SENTINEL-STOCKHOLM\paper\figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
})

# Consistent palette
C_GREEN = '#2ecc71'
C_GREEN_DARK = '#27ae60'
C_RED = '#e74c3c'
C_ORANGE = '#e67e22'
C_BLUE = '#3498db'
C_BLUE_DARK = '#2980b9'
C_PURPLE = '#9b59b6'
C_TEAL = '#1abc9c'
C_GRAY = '#7f8c8d'
C_GOLD = '#f39c12'

SAVE_KW = dict(format='jpg', dpi=150, bbox_inches='tight',
               pil_kwargs={'quality': 85})


def save(fig, name):
    path = OUTDIR / name
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    print(f"  Saved {path}  ({path.stat().st_size / 1024:.0f} KB)")


# =====================================================================
# Figure 1: Case Study Detection Timelines
# =====================================================================
def fig1_detection_timelines():
    print("Figure 1: Detection Timelines")

    events = [
        ("Gold King Mine Spill", -20.2),
        ("Lake Erie HAB", 324.2),
        ("Toledo Water Crisis", 79.0),
        ("Dan River Coal Ash", -22.1),
        ("Elk River MCHM", -16.0),
        ("Houston Ship Channel", -23.2),
        ("Flint Water Crisis", 12177.7),
        ("Gulf Dead Zone", 1257.5),
        ("Chesapeake Bay Blooms", 392.7),
        ("East Palestine Derailment", -13.9),
    ]

    # Convert Flint to days for display, cap hours for axis
    # Sort by lead time
    events.sort(key=lambda x: x[1])

    names = [e[0] for e in events]
    hours = [e[1] for e in events]

    # Use a broken/capped axis: show everything except Flint in main range,
    # and annotate Flint separately
    # Cap the axis at ~500 hours for readability, with Flint bar extending to cap
    CAP = 600
    display_hours = [min(h, CAP) for h in hours]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = [C_GREEN_DARK if h > 0 else C_RED for h in hours]
    bars = ax.barh(range(len(names)), display_hours, color=colors, edgecolor='white',
                   linewidth=0.5, height=0.7)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color='black', linewidth=1.0, zorder=3)
    ax.set_xlabel("Detection Lead Time (hours)", fontsize=11)
    ax.set_title("SENTINEL Early Warning: Detection Lead Time\nAcross 10 Historical Contamination Events", fontsize=12, fontweight='bold')

    # Label each bar with its actual value
    for i, (h_actual, h_disp) in enumerate(zip(hours, display_hours)):
        if h_actual == 12177.7:
            # Flint: special annotation
            ax.annotate("507 days early",
                        xy=(CAP, i), xytext=(CAP + 30, i),
                        fontsize=9, fontweight='bold', color=C_GREEN_DARK,
                        va='center',
                        arrowprops=dict(arrowstyle='->', color=C_GREEN_DARK, lw=1.5))
            # break marker on bar
            ax.plot([CAP - 15, CAP - 5], [i + 0.2, i - 0.2], color='white', lw=2, zorder=5)
            ax.plot([CAP - 10, CAP], [i + 0.2, i - 0.2], color='white', lw=2, zorder=5)
        elif h_actual > 0:
            label = f"+{h_actual:.0f}h"
            ax.text(h_disp + 8, i, label, va='center', fontsize=8, color=C_GREEN_DARK, fontweight='bold')
        else:
            label = f"{h_actual:.0f}h"
            ax.text(h_disp - 8, i, label, va='center', ha='right', fontsize=8, color=C_RED, fontweight='bold')

    # Add legend
    early_patch = mpatches.Patch(color=C_GREEN_DARK, label='Early detection')
    late_patch = mpatches.Patch(color=C_RED, label='Late detection')
    ax.legend(handles=[early_patch, late_patch], loc='lower right', fontsize=9)

    ax.set_xlim(-80, CAP + 120)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    save(fig, "fig1_detection_timelines.jpg")


# =====================================================================
# Figure 2: Missing Modality Robustness
# =====================================================================
def fig2_robustness():
    print("Figure 2: Missing Modality Robustness")

    n_available = [5, 4, 3, 2, 1]
    mean_auc = [0.992, 0.946, 0.932, 0.901, 0.680]
    ci_lower = [0.992, 0.914, 0.896, 0.860, 0.622]
    ci_upper = [0.992, 0.972, 0.965, 0.938, 0.743]

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.fill_between(n_available, ci_lower, ci_upper, alpha=0.25, color=C_BLUE)
    ax.plot(n_available, mean_auc, 'o-', color=C_BLUE_DARK, linewidth=2.5,
            markersize=8, markerfacecolor='white', markeredgewidth=2, zorder=5)

    # Threshold line
    ax.axhline(0.90, color=C_RED, linestyle='--', linewidth=1.5, alpha=0.8)
    ax.text(4.8, 0.907, "Deployment threshold (0.90)", fontsize=8.5,
            color=C_RED, ha='right', fontweight='bold')

    # Annotation
    ax.annotate("Maintains >0.90\nwith 2+ modalities",
                xy=(2, 0.901), xytext=(2.8, 0.82),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor=C_GOLD, alpha=0.9))

    # Data labels
    for x, y in zip(n_available, mean_auc):
        ax.text(x, y + 0.015, f"{y:.3f}", ha='center', fontsize=8.5, fontweight='bold', color=C_BLUE_DARK)

    ax.set_xlabel("Number of Available Modalities", fontsize=11)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_title("Graceful Degradation Under Missing Modalities", fontsize=12, fontweight='bold')
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.55, 1.05)
    ax.invert_xaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    save(fig, "fig2_robustness.jpg")


# =====================================================================
# Figure 3: Sensor Placement Cost-Effectiveness
# =====================================================================
def fig3_cost_effectiveness():
    print("Figure 3: Cost-Effectiveness")

    budgets = [50, 100, 200, 500]
    info_gain = [16.68, 21.46, 28.08, 38.70]
    modality_mix = {
        50: ["Satellite", "Sensor"],
        100: ["Satellite", "Sensor", "Behavioral"],
        200: ["Satellite", "Sensor", "Behavioral", "Microbial"],
        500: ["Satellite", "Sensor", "Behavioral", "Microbial", "Molecular"],
    }

    mod_colors = {
        "Satellite": C_BLUE,
        "Sensor": C_TEAL,
        "Behavioral": C_GOLD,
        "Microbial": C_PURPLE,
        "Molecular": C_RED,
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(budgets, info_gain, 'o-', color=C_BLUE_DARK, linewidth=2.5,
            markersize=10, markerfacecolor='white', markeredgewidth=2.5, zorder=5)

    # Fill under curve for diminishing returns visual
    ax.fill_between(budgets, info_gain, alpha=0.08, color=C_BLUE)

    # Annotations at each budget
    y_offsets = [1.5, 1.5, -3.0, 1.5]
    for i, (b, ig) in enumerate(zip(budgets, info_gain)):
        mods = modality_mix[b]
        # Find the new modality added
        if i == 0:
            new_mods = mods
        else:
            prev = modality_mix[budgets[i - 1]]
            new_mods = [m for m in mods if m not in prev]

        label = "+ " + ", ".join(new_mods) if i > 0 else ", ".join(new_mods)
        yoff = y_offsets[i]

        col = mod_colors[new_mods[0]] if new_mods else C_GRAY
        ax.annotate(label,
                    xy=(b, ig), xytext=(b, ig + yoff),
                    fontsize=8, ha='center', color=col, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=col, lw=1) if abs(yoff) > 2 else None)

        # Data label
        side = 'bottom' if yoff > 0 else 'top'
        ax.text(b, ig - 1.2 if yoff > 0 else ig + 1.2, f"{ig:.1f} bits",
                ha='center', fontsize=8.5, color=C_BLUE_DARK)

    # Diminishing returns annotation
    ax.annotate("Diminishing returns\nabove $200K",
                xy=(350, 33.4), xytext=(380, 24),
                fontsize=8.5, ha='center', style='italic', color=C_GRAY,
                arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=1))

    ax.set_xlabel("Annual Budget ($K)", fontsize=11)
    ax.set_ylabel("Mutual Information Gain (bits)", fontsize=11)
    ax.set_title("Optimal Sensor Portfolio: Cost vs. Information Gain", fontsize=12, fontweight='bold')
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"${b}K" for b in budgets])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend for modalities
    handles = [mpatches.Patch(color=c, label=m) for m, c in mod_colors.items()]
    ax.legend(handles=handles, loc='upper left', fontsize=8, title="Modality", title_fontsize=9,
              framealpha=0.9)

    fig.tight_layout()
    save(fig, "fig3_cost_effectiveness.jpg")


# =====================================================================
# Figure 4: Conformal Prediction Coverage
# =====================================================================
def fig4_conformal_coverage():
    print("Figure 4: Conformal Coverage")

    modalities = ["Satellite", "Sensor", "Microbial", "Behavioral"]
    synthetic_coverage = [0.375, 0.941, 0.000, 0.903]
    real_coverage = [0.963, 0.937, 0.917, 0.913]
    target = 0.95

    x = np.arange(len(modalities))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Color bars based on whether they meet target
    syn_colors = [C_GREEN if v >= target else C_RED for v in synthetic_coverage]
    real_colors = [C_GREEN if v >= target else C_ORANGE for v in real_coverage]

    bars1 = ax.bar(x - width / 2, synthetic_coverage, width, label='Synthetic',
                   color=syn_colors, edgecolor='white', linewidth=0.8, alpha=0.85)
    bars2 = ax.bar(x + width / 2, real_coverage, width, label='Real',
                   color=real_colors, edgecolor='white', linewidth=0.8, alpha=0.85)

    # Add hatching to synthetic bars for distinction
    for bar in bars1:
        bar.set_hatch('//')

    # Target line
    ax.axhline(target, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(len(modalities) - 0.5, target + 0.015, f"Target = {target}",
            fontsize=9, ha='right', fontweight='bold')

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                        f"{h:.2f}" if h > 0 else "0.00",
                        ha='center', fontsize=8, fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, 0.03,
                        "0.00", ha='center', fontsize=8, fontweight='bold', color=C_RED)

    ax.set_xticks(x)
    ax.set_xticklabels(modalities, fontsize=10)
    ax.set_ylabel("Conformal Coverage", fontsize=11)
    ax.set_title("Conformal Prediction Coverage:\nSynthetic vs. Real Embeddings", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.12)

    # Custom legend
    syn_patch = mpatches.Patch(facecolor=C_GRAY, hatch='//', label='Synthetic', alpha=0.85)
    real_patch = mpatches.Patch(facecolor=C_GRAY, label='Real', alpha=0.85)
    meets_patch = mpatches.Patch(facecolor=C_GREEN, label='Meets target')
    below_patch = mpatches.Patch(facecolor=C_RED, label='Below target')
    ax.legend(handles=[syn_patch, real_patch, meets_patch, below_patch],
              loc='upper left', fontsize=8, ncol=2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    save(fig, "fig4_conformal_coverage.jpg")


# =====================================================================
# Figure 5: Causal Chain Network
# =====================================================================
def fig5_causal_network():
    print("Figure 5: Causal Chain Network")

    G = nx.DiGraph()

    nodes = {
        "TP": {"color": C_BLUE, "group": "Nutrients"},
        "TN": {"color": C_TEAL, "group": "Nutrients"},
        "NH4": {"color": C_PURPLE, "group": "Nitrogen Species"},
        "Nitrate": {"color": C_PURPLE, "group": "Nitrogen Species"},
        "COD": {"color": C_RED, "group": "Oxygen Demand"},
    }

    for n, attrs in nodes.items():
        G.add_node(n, **attrs)

    edges = [
        ("TP", "COD", {"lag": "147h", "sign": "+", "color": C_GREEN_DARK}),
        ("NH4", "COD", {"lag": "81h", "sign": "−", "color": C_RED}),
        ("TN", "NH4", {"lag": "84h", "sign": "+", "color": C_GREEN_DARK}),
        ("TP", "NH4", {"lag": "57h", "sign": "−", "color": C_RED}),
        ("TP", "Nitrate", {"lag": "112h", "sign": "−", "color": C_RED}),
    ]

    for u, v, attrs in edges:
        G.add_edge(u, v, **attrs)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Manual positions for clean layout
    pos = {
        "TP": (0, 1),
        "TN": (-1.2, 0),
        "NH4": (0, -0.5),
        "Nitrate": (1.2, 0),
        "COD": (1.2, 1.2),
    }

    # Draw nodes
    node_colors = [nodes[n]["color"] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1800, node_color=node_colors,
                           edgecolors='white', linewidths=2.5, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_weight='bold',
                            font_color='white')

    # Draw edges with curvature to avoid overlap
    for u, v, d in G.edges(data=True):
        style = 'arc3,rad=0.15'
        ax.annotate("",
                     xy=pos[v], xycoords='data',
                     xytext=pos[u], textcoords='data',
                     arrowprops=dict(
                         arrowstyle='-|>',
                         color=d['color'],
                         lw=2.5,
                         connectionstyle=style,
                         mutation_scale=20,
                     ))

    # Edge labels
    edge_label_offsets = {
        ("TP", "COD"): (0.65, 1.25),
        ("NH4", "COD"): (0.75, 0.45),
        ("TN", "NH4"): (-0.7, -0.15),
        ("TP", "NH4"): (-0.25, 0.2),
        ("TP", "Nitrate"): (0.7, 0.6),
    }

    for u, v, d in G.edges(data=True):
        label = f"{d['sign']} {d['lag']}"
        lx, ly = edge_label_offsets[(u, v)]
        ax.text(lx, ly, label, fontsize=9, fontweight='bold',
                ha='center', va='center', color=d['color'],
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=d['color'], alpha=0.9))

    # Legend for node groups
    nutrient_patch = mpatches.Patch(color=C_BLUE, label='Phosphorus (TP)')
    nitrogen_patch = mpatches.Patch(color=C_TEAL, label='Total Nitrogen (TN)')
    nspecies_patch = mpatches.Patch(color=C_PURPLE, label='Nitrogen Species (NH4, Nitrate)')
    cod_patch = mpatches.Patch(color=C_RED, label='Oxygen Demand (COD)')
    pos_arrow = mpatches.FancyArrowPatch((0, 0), (1, 0), color=C_GREEN_DARK,
                                          arrowstyle='->', mutation_scale=15)
    ax.legend(handles=[nutrient_patch, nitrogen_patch, nspecies_patch, cod_patch],
              loc='lower left', fontsize=8, title="Parameters", title_fontsize=9,
              framealpha=0.9)

    # Sign legend as text
    ax.text(0.98, 0.02, "Green = positive effect\nRed = negative effect\nLabel = lag time",
            transform=ax.transAxes, fontsize=8, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor=C_GOLD, alpha=0.9))

    ax.set_title("Discovered Causal Chains in Water Quality Parameters\n(GRQA Data, PCMCI+ Analysis)",
                 fontsize=12, fontweight='bold')
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.1, 1.8)
    ax.axis('off')

    fig.tight_layout()
    save(fig, "fig5_causal_network.jpg")


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    print(f"Output directory: {OUTDIR}")
    print("=" * 50)
    fig1_detection_timelines()
    fig2_robustness()
    fig3_cost_effectiveness()
    fig4_conformal_coverage()
    fig5_causal_network()
    print("=" * 50)
    print("All figures generated successfully!")
