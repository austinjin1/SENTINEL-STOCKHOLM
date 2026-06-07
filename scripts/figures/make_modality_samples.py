#!/usr/bin/env python
"""One real data-sample image for EACH of SENTINEL's five input modalities.

Canonical modalities (sentinel/models/fusion/embedding_registry.py):
    satellite, sensor, microbial, molecular, behavioral

All panels use real data, large axis text, and NO titles/captions (poster
will add its own labels).

Outputs (figures/):
    modality_satellite.png   2x2 Sentinel-2 algal-bloom progression (1 reservoir)
    modality_sensor.png      USGS NWIS 15-min stream in real physical units
    modality_microbial.png   EMP 16S rRNA community abundance across habitats
    modality_molecular.png   GEO transcriptomic expression heatmap
    modality_behavioral.png  EPA ECOTOX Daphnia dose-response curves
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

OUT = "figures"
plt.rcParams.update({"font.family": "DejaVu Sans"})

# ~3x the previous label sizes
AXLAB = 27
TICK = 21
CBAR = 23
LEGEND = 20


# --------------------------------------------------------------------------
# SATELLITE — 4-panel algal-bloom progression at one reservoir
# --------------------------------------------------------------------------
def stretch_rgb(img4, lo=2, hi=98, gamma=0.78, sat=1.20):
    rgb = np.stack([img4[2], img4[1], img4[0]], axis=-1).astype(np.float32)
    out = np.empty_like(rgb)
    for c in range(3):
        ch = rgb[..., c]
        p_lo, p_hi = np.percentile(ch, [lo, hi])
        out[..., c] = np.clip((ch - p_lo) / (p_hi - p_lo + 1e-6), 0, 1)
    out = np.power(out, gamma)
    hsv = rgb_to_hsv(out)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 1)
    return hsv_to_rgb(hsv)


def make_satellite():
    d = np.load("data/processed/satellite/drone_wq_pairs.npz",
                allow_pickle=True, mmap_mode="r")
    # USGS 02079500 (Kerr Reservoir / Roanoke R., VA): same water body, four
    # dates from clear water to peak green algal bloom (rising green-red index).
    seq = [10867, 8270, 5750, 2570]   # bloom index ~0.01, 0.07, 0.38, 0.42
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for ax, i in zip(axes.flat, seq):
        im = np.array(d["images"][i])
        ax.imshow(stretch_rgb(im), interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("white"); s.set_linewidth(2.0)
    plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005,
                        wspace=0.02, hspace=0.02)
    fig.savefig(f"{OUT}/modality_satellite.png", dpi=220, facecolor="white")
    plt.close(fig)
    print("wrote modality_satellite.png (4-panel bloom)")


# --------------------------------------------------------------------------
# SENSOR — real physical-unit USGS NWIS stream
# --------------------------------------------------------------------------
def make_sensor():
    import pandas as pd
    df = pd.read_parquet("data/raw/sensor/full/01376269.parquet")
    start = 4032
    win = df.iloc[start:start + 672]                 # 7 days @ 15-min
    t = (win.index - win.index[0]).total_seconds().to_numpy() / 86400.0
    rows = [
        ("DO", "DO\n(mg/L)", "#1f77b4"),
        ("Temp", "Temp\n(°C)", "#d62728"),
        ("pH", "pH", "#2ca02c"),
        ("SpCond", "SpCond\n(µS/cm)", "#9467bd"),
        ("Turb", "Turb\n(FNU)", "#8c6d31"),
    ]
    fig, axes = plt.subplots(len(rows), 1, figsize=(15, 13), sharex=True)
    for ax, (col, lab, color) in zip(axes, rows):
        ax.plot(t, win[col].to_numpy(), color=color, lw=2.0)
        ax.set_ylabel(lab, fontsize=AXLAB, rotation=0, ha="right",
                      va="center", labelpad=18)
        ax.tick_params(labelsize=TICK)
        ax.margins(x=0)
        ax.locator_params(axis="y", nbins=4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[-1].set_xlabel("Days", fontsize=AXLAB)
    plt.subplots_adjust(left=0.21, right=0.97, top=0.98, bottom=0.10, hspace=0.22)
    fig.savefig(f"{OUT}/modality_sensor.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote modality_sensor.png (real units)")


# --------------------------------------------------------------------------
# MICROBIAL — EMP 16S rRNA abundance across habitat classes
# --------------------------------------------------------------------------
HABITAT_ORDER = ["freshwater_natural", "freshwater_impacted", "freshwater_sediment",
                 "saline_water", "saline_sediment", "soil_runoff",
                 "plant_associated", "animal_fecal"]


def make_microbial(n_leaves=190):
    """Circular cladogram of the real 16S rRNA OTUs.

    The dominant OTUs are clustered by 16S sequence similarity (Hamming distance,
    average linkage) and drawn as a radial tree; the outer ring colors each
    taxon by the aquatic source habitat where it is most abundant. This shows
    both the phylogenetic breadth of the community and its habitat structure —
    far more legible than a sparse OTU heatmap."""
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist
    from matplotlib.patches import Wedge, Patch
    base = "data/processed/microbial/emp_16s"
    seqs = np.load(os.path.join(base, "selected_otu_ids.npy"), allow_pickle=True)
    # mean relative abundance per habitat
    acc = {h: [] for h in HABITAT_ORDER}
    for fn in os.listdir(base):
        if not fn.endswith(".npz"):
            continue
        dd = np.load(os.path.join(base, fn), allow_pickle=True)
        h = str(dd["source_name"])
        if h in acc:
            acc[h].append(np.asarray(dd["abundances"], np.float32))
    Mh = np.stack([np.mean(acc[h], 0) for h in HABITAT_ORDER])
    Mh = Mh / Mh.sum(1, keepdims=True)
    sel = np.argsort(Mh.mean(0))[::-1][:n_leaves]      # most prevalent OTUs
    dom = Mh[:, sel].argmax(0)                          # dominant habitat / OTU
    # Hamming distance on the 90-bp 16S reads -> UPGMA tree
    mp = {"A": 0, "C": 1, "G": 2, "T": 3}
    S = np.array([[mp[c] for c in seqs[i]] for i in sel])
    Z = linkage(pdist(S, metric="hamming"), method="average")
    dn = dendrogram(Z, no_plot=True)
    ic, dc = np.array(dn["icoord"]), np.array(dn["dcoord"])
    order = dn["leaves"]
    ymax, xmax = dc.max(), 10 * n_leaves
    GAP, R0, R1 = 0.16, 0.16, 1.0

    def th(x):
        return (np.pi / 2) - (1 - GAP) * 2 * np.pi * x / xmax

    def rr(y):
        return R1 - (y / ymax) * (R1 - R0)

    def xy(x, y):
        return rr(y) * np.cos(th(x)), rr(y) * np.sin(th(x))

    colors = plt.cm.tab10(np.linspace(0, 1, len(HABITAT_ORDER)))
    fig, ax = plt.subplots(figsize=(12, 12.5))
    ax.set_aspect("equal"); ax.axis("off")
    for xs, ys in zip(ic, dc):
        (x1, _, x3, x4), (y1, y2, _, y4) = xs, ys
        ax.plot(*zip(xy(x1, y1), xy(x1, y2)), color="0.35", lw=0.9)
        ax.plot(*zip(xy(x4, y4), xy(x4, y2)), color="0.35", lw=0.9)
        ts = np.linspace(th(x1), th(x3), 24)
        r = rr(y2)
        ax.plot(r * np.cos(ts), r * np.sin(ts), color="0.35", lw=0.9)
    leaf_x = np.array([5 + 10 * i for i in range(n_leaves)])
    ang = np.degrees(th(leaf_x))
    half = np.degrees((1 - GAP) * 2 * np.pi / xmax) * 5
    for k, i in enumerate(order):
        ax.add_patch(Wedge((0, 0), 1.15, ang[k] - half, ang[k] + half,
                           width=0.11, facecolor=colors[dom[i]], edgecolor="none"))
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.45, 1.25)
    leg = [Patch(facecolor=colors[i], label=HABITAT_ORDER[i].replace("_", " "))
           for i in range(len(HABITAT_ORDER))]
    ax.legend(handles=leg, fontsize=15, loc="lower center", frameon=False,
              ncol=4, bbox_to_anchor=(0.5, -0.02), columnspacing=1.2,
              handletextpad=0.4)
    plt.subplots_adjust(0, 0.04, 1, 1)
    fig.savefig(f"{OUT}/modality_microbial.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote modality_microbial.png (16S circular cladogram)")


# --------------------------------------------------------------------------
# MOLECULAR — GEO transcriptomic expression heatmap
# --------------------------------------------------------------------------
def make_molecular():
    d = np.load("data/processed/molecular/real/GSE73661.npz", allow_pickle=True)
    e = np.array(d["expression"])
    var = e.var(1)
    g = np.argsort(var)[::-1][:70]
    E = e[g]
    E = E[:, np.argsort(E[0])]
    E = (E - E.mean(1, keepdims=True)) / (E.std(1, keepdims=True) + 1e-6)
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(E, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                   interpolation="nearest")
    ax.set_xlabel("samples", fontsize=AXLAB)
    ax.set_ylabel("genes", fontsize=AXLAB)
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("z-scored expression", fontsize=AXLAB)
    cb.ax.tick_params(labelsize=TICK)
    plt.subplots_adjust(left=0.07, right=0.99, top=0.98, bottom=0.09)
    fig.savefig(f"{OUT}/modality_molecular.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote modality_molecular.png")


# --------------------------------------------------------------------------
# BEHAVIORAL — EPA ECOTOX Daphnia dose-response curves (per contaminant)
# --------------------------------------------------------------------------
def make_behavioral():
    import csv
    from scipy.optimize import curve_fit
    rows = list(csv.DictReader(open(
        "data/processed/behavioral/ecotox_daphnia_curves.csv")))
    data = {}
    for r in rows:
        data.setdefault(r["chemical"], []).append(
            (float(r["conc_mgL"]), float(r["effect_pct"])))
    # order by potency (left = most toxic), assign colors
    palette = {"Chlorpyrifos": "#d62728", "Copper": "#1f77b4",
               "Cadmium": "#9467bd", "Zinc": "#2ca02c",
               "Sodium chloride": "#ff7f0e"}

    def hill(logc, ec50, k):
        return 100.0 / (1.0 + 10.0 ** (k * (np.log10(ec50) - logc)))

    fig, ax = plt.subplots(figsize=(14, 10))
    order = ["Chlorpyrifos", "Copper", "Cadmium", "Zinc", "Sodium chloride"]
    for chem in order:
        pts = np.array(data[chem])
        c, e = pts[:, 0], pts[:, 1]
        color = palette[chem]
        ax.scatter(c, e, s=55, c=color, alpha=0.55, edgecolors="none")
        try:
            popt, _ = curve_fit(hill, np.log10(c), e,
                                p0=[np.median(c), 1.5], maxfev=10000,
                                bounds=([c.min() * 1e-2, 0.3],
                                        [c.max() * 1e2, 8]))
            xs = np.logspace(np.log10(c.min()), np.log10(c.max()), 200)
            ax.plot(xs, hill(np.log10(xs), *popt), color=color, lw=3.5,
                    label=f"{chem}  (EC50 {popt[0]:.2g} mg/L)")
        except Exception:
            ax.plot([], [], color=color, lw=3.5, label=chem)
    ax.set_xscale("log")
    ax.set_xlabel("Contaminant concentration (mg/L)", fontsize=AXLAB)
    ax.set_ylabel("Daphnia affected (%)", fontsize=AXLAB)
    ax.set_ylim(-5, 105)
    ax.tick_params(labelsize=TICK)
    ax.grid(True, which="both", alpha=0.2, lw=0.7)
    ax.legend(fontsize=LEGEND, loc="lower right", framealpha=0.92)
    plt.subplots_adjust(left=0.11, right=0.97, top=0.97, bottom=0.13)
    fig.savefig(f"{OUT}/modality_behavioral.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote modality_behavioral.png (dose-response curves)")


if __name__ == "__main__":
    make_satellite()
    make_sensor()
    make_microbial()
    make_molecular()
    make_behavioral()
