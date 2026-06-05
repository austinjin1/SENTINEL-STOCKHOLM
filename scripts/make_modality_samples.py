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


def make_microbial():
    base = "data/processed/microbial/emp_16s"
    files = [f for f in os.listdir(base) if f.endswith(".npz")]
    by_hab = {}
    for fn in files:
        dd = np.load(os.path.join(base, fn), allow_pickle=True)
        hab = str(dd["source_name"])
        if hab not in by_hab:
            by_hab[hab] = np.array(dd["abundances"])
    habs = [h for h in HABITAT_ORDER if h in by_hab]
    M = np.stack([by_hab[h] for h in habs], axis=0)
    top = np.argsort(M.sum(0))[::-1][:45]
    Mt = np.log10(M[:, top] + 1e-4)
    fig, ax = plt.subplots(figsize=(15, 8))
    im = ax.imshow(Mt, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(len(habs)))
    ax.set_yticklabels([h.replace("_", " ") for h in habs], fontsize=TICK)
    ax.set_xticks([])
    ax.set_xlabel("bacterial OTUs (16S rRNA)", fontsize=AXLAB)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("log₁₀ rel. abundance", fontsize=AXLAB)
    cb.ax.tick_params(labelsize=TICK)
    plt.subplots_adjust(left=0.27, right=0.99, top=0.97, bottom=0.12)
    fig.savefig(f"{OUT}/modality_microbial.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote modality_microbial.png")


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
