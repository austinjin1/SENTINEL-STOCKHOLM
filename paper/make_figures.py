"""
Regenerate paper figures from results/*.json.
Run from repo root:  python paper/make_figures.py
"""
from __future__ import annotations
import json, os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Publication style
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

PALETTE = {
    "sensor": "#1f77b4",
    "satellite": "#2ca02c",
    "microbial": "#9467bd",
    "molecular": "#d62728",
    "behavioral": "#ff7f0e",
    "fusion": "#111111",
    "baseline": "#888888",
}

# ---------------------------------------------------------------------------
def fig_per_modality_auroc():
    """Bar chart: per-modality detection AUROC vs fusion (from README headline)."""
    data = [
        ("Sensor (AquaSSM)", 0.943, PALETTE["sensor"]),
        ("Satellite (HydroViT)", 0.728, PALETTE["satellite"]),
        ("Microbial (MicroBiomeNet)", 0.609, PALETTE["microbial"]),
        ("Molecular (ToxiGene)", 0.880, PALETTE["molecular"]),
        ("Behavioral (BioMotion)", 1.000, PALETTE["behavioral"]),
        ("Fusion (Perceiver IO)", 0.992, PALETTE["fusion"]),
    ]
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    names = [d[0] for d in data]
    vals = [d[1] for d in data]
    cols = [d[2] for d in data]
    bars = ax.barh(range(len(data)), vals, color=cols, alpha=0.85)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.02)
    ax.set_xlabel("Detection AUROC")
    ax.axvline(0.5, color="gray", lw=0.5, ls=":")
    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7)
    ax.set_title("Per-modality detection performance")
    fig.savefig(OUT / "fig_per_modality_auroc.pdf")
    plt.close(fig)

# ---------------------------------------------------------------------------
def fig_ablation_heatmap():
    """31-condition modality ablation heatmap."""
    path = RES / "ablation" / "ablation_results.json"
    if not path.exists():
        print(f"[skip] {path}"); return
    rows = json.loads(path.read_text())
    # group by num_modalities
    rows = sorted(rows, key=lambda r: (r.get("num_modalities", 0), -r.get("detection_auc", 0)))
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    names = [r["condition_name"].replace("+", " + ") for r in rows]
    aucs = [r.get("detection_auc", 0) for r in rows]
    cols = ["#d62728" if a < 0.7 else "#ff7f0e" if a < 0.85 else "#2ca02c" if a < 0.95 else "#1f77b4" for a in aucs]
    ax.barh(range(len(rows)), aucs, color=cols, alpha=0.85)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.02)
    ax.set_xlabel("Detection AUROC")
    ax.axvline(0.943, color="black", ls="--", lw=0.7, label="Sensor-only baseline")
    ax.axvline(0.992, color="red", ls=":", lw=0.7, label="Full fusion")
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    ax.set_title("31-condition modality ablation ($2^5{-}1$)")
    fig.savefig(OUT / "fig_ablation.pdf")
    plt.close(fig)

# ---------------------------------------------------------------------------
def fig_case_studies():
    """Per-event scores: full fusion vs sensor-only, 10 historical events."""
    path = RES / "ablation" / "ablation_results.json"
    if not path.exists():
        print(f"[skip] {path}"); return
    rows = json.loads(path.read_text())
    sensor = next(r for r in rows if r["condition_name"] == "sensor")
    full = max(rows, key=lambda r: (r.get("num_modalities", 0), r.get("detection_auc", 0)))
    events = list(sensor["per_event_scores"].keys())
    s_vals = [sensor["per_event_scores"][e] for e in events]
    f_vals = [full["per_event_scores"][e] for e in events]
    pretty = [e.replace("_", " ").title() for e in events]
    x = np.arange(len(events))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.5, 2.7))
    ax.bar(x - w/2, s_vals, w, color=PALETTE["sensor"], alpha=0.85, label="Sensor only")
    ax.bar(x + w/2, f_vals, w, color=PALETTE["fusion"], alpha=0.85, label="Full fusion")
    ax.set_xticks(x); ax.set_xticklabels(pretty, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Per-event score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("10 historical contamination events")
    fig.savefig(OUT / "fig_case_studies.pdf")
    plt.close(fig)

# ---------------------------------------------------------------------------
def fig_robustness():
    """AUROC vs number of available modalities (graceful degradation)."""
    path = RES / "ablation" / "ablation_results.json"
    if not path.exists():
        print(f"[skip] {path}"); return
    rows = json.loads(path.read_text())
    by_n = {}
    for r in rows:
        n = r.get("num_modalities", 0)
        by_n.setdefault(n, []).append(r.get("detection_auc", 0))
    ns = sorted(by_n)
    means = [np.mean(by_n[n]) for n in ns]
    mins = [np.min(by_n[n]) for n in ns]
    maxs = [np.max(by_n[n]) for n in ns]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.fill_between(ns, mins, maxs, alpha=0.2, color=PALETTE["fusion"], label="min–max range")
    ax.plot(ns, means, "o-", color=PALETTE["fusion"], label="mean")
    ax.axhline(0.9, color="red", ls=":", lw=0.7, label="0.90 AUROC")
    ax.set_xlabel("# modalities active")
    ax.set_ylabel("Detection AUROC")
    ax.set_xticks(ns)
    ax.set_ylim(0.5, 1.02)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    ax.set_title("Graceful degradation")
    fig.savefig(OUT / "fig_robustness.pdf")
    plt.close(fig)

# ---------------------------------------------------------------------------
def fig_architecture_placeholder():
    """Placeholder architecture diagram (text)."""
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    ax.axis("off")
    boxes = [
        (0.02, 0.65, "Sensor\n(AquaSSM)", PALETTE["sensor"]),
        (0.20, 0.65, "Satellite\n(HydroViT)", PALETTE["satellite"]),
        (0.38, 0.65, "Microbial\n(MicroBiomeNet)", PALETTE["microbial"]),
        (0.56, 0.65, "Molecular\n(ToxiGene)", PALETTE["molecular"]),
        (0.74, 0.65, "Behavioral\n(BioMotion)", PALETTE["behavioral"]),
    ]
    for x, y, t, c in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.16, 0.20, fc=c, alpha=0.3, ec=c, lw=1.0))
        ax.text(x+0.08, y+0.10, t, ha="center", va="center", fontsize=7)
        ax.annotate("", xy=(0.50, 0.45), xytext=(x+0.08, y),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.6))
    ax.add_patch(plt.Rectangle((0.18, 0.30), 0.64, 0.15, fc="#cccccc", alpha=0.5, ec="black", lw=1.0))
    ax.text(0.50, 0.375, "Perceiver IO Cross-Attention Fusion (256 latents)",
            ha="center", va="center", fontsize=8, weight="bold")
    ax.add_patch(plt.Rectangle((0.18, 0.08), 0.30, 0.13, fc="#bcd", alpha=0.5, ec="black", lw=0.8))
    ax.text(0.33, 0.145, "Anomaly + Source Attribution", ha="center", va="center", fontsize=7)
    ax.add_patch(plt.Rectangle((0.52, 0.08), 0.30, 0.13, fc="#fbb", alpha=0.5, ec="black", lw=0.8))
    ax.text(0.67, 0.145, "Cascade Escalation (PPO)", ha="center", va="center", fontsize=7)
    ax.annotate("", xy=(0.33, 0.21), xytext=(0.40, 0.30), arrowprops=dict(arrowstyle="->", color="gray", lw=0.6))
    ax.annotate("", xy=(0.67, 0.21), xytext=(0.60, 0.30), arrowprops=dict(arrowstyle="->", color="gray", lw=0.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("SENTINEL architecture")
    fig.savefig(OUT / "fig_architecture.pdf")
    plt.close(fig)

if __name__ == "__main__":
    fig_per_modality_auroc()
    fig_ablation_heatmap()
    fig_case_studies()
    fig_robustness()
    fig_architecture_placeholder()
    print("Figures written to", OUT)
