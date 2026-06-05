#!/usr/bin/env python
"""One real data-sample image for EACH of SENTINEL's five input modalities.

Canonical modalities (sentinel/models/fusion/embedding_registry.py):
    satellite, sensor, microbial, molecular, behavioral

Outputs (figures/):
    modality_satellite.png   Sentinel-2 true-color tile (USGS-paired)
    modality_sensor.png      USGS NWIS 15-min multi-parameter sensor stream
    modality_microbial.png   EMP 16S rRNA community abundance across habitats
    modality_molecular.png   GEO transcriptomic expression heatmap
    modality_behavioral.png  Daphnia keypoint trajectory (EPA ECOTOX)
    modality_overview.png    all five panels in one figure
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

OUT = "figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "figure.dpi": 200})

ACCENT = "#1a3a6b"


# --------------------------------------------------------------------------
# SATELLITE
# --------------------------------------------------------------------------
def stretch_rgb(img4, lo=2, hi=98, gamma=0.78, sat=1.18):
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
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


def satellite_ax(ax):
    d = np.load("data/processed/satellite/drone_wq_pairs.npz",
                allow_pickle=True, mmap_mode="r")
    meta = json.loads(str(d["metadata"]))
    # index 11650 = USGS 380548121390501 (Sacramento–San Joaquin Delta, CA):
    # vivid turbid channel through farmland, crisp, ~30% water by NDWI
    i = 11650
    im = np.array(d["images"][i])
    ax.imshow(stretch_rgb(im), interpolation="bilinear")
    m = meta[i]
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#3a6ea5"); s.set_linewidth(1.6)
    return (f"Sentinel-2 imagery", f"USGS {m['site_id']}  ·  {m['date']}  ·  "
            "4-band RGB+NIR, 10 m / 224×224 px")


# --------------------------------------------------------------------------
# SENSOR
# --------------------------------------------------------------------------
SENSOR_PARAMS = ["DO", "pH", "SpCond", "Temp", "Turb", "ORP"]
SENSOR_UNITS = ["mg/L", "", "µS/cm", "°C", "FNU", "mV"]
SENSOR_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728", "#8c6d31", "#17becf"]


def sensor_panel(fig, gs_cell):
    f = "data/processed/sensor/full/02198955_seq00318.npz"
    d = np.load(f)
    v, m = d["values"], d["mask"]
    site = os.path.basename(f).split("_")[0]
    chans = [j for j in range(6) if m[:, j].mean() > 0.9]
    t = np.arange(v.shape[0]) * 15.0 / 60.0 / 24.0   # days (15-min steps)
    inner = gridspec.GridSpecFromSubplotSpec(
        len(chans), 1, subplot_spec=gs_cell, hspace=0.18)
    axes = []
    for k, j in enumerate(chans):
        ax = fig.add_subplot(inner[k])
        y = v[:, j].copy().astype(float)
        y[~m[:, j]] = np.nan
        ax.plot(t, y, color=SENSOR_COLORS[j], lw=1.3)
        # values are z-standardized per site, so show the parameter (no units)
        # and suppress the meaningless normalized tick numbers
        ax.set_ylabel(SENSOR_PARAMS[j], fontsize=9, rotation=0, ha="right",
                      va="center", labelpad=14)
        ax.set_yticklabels([])
        ax.tick_params(labelsize=7)
        ax.margins(x=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if k < len(chans) - 1:
            ax.set_xticklabels([])
        axes.append(ax)
    axes[-1].set_xlabel("days (15-min sampling, 7-day window)", fontsize=8.5)
    return axes, (f"In-situ sensor stream",
                  f"USGS NWIS {site}  ·  {len(chans)} parameters  ·  standardized")


# --------------------------------------------------------------------------
# MICROBIAL
# --------------------------------------------------------------------------
HABITAT_ORDER = ["freshwater_natural", "freshwater_impacted", "freshwater_sediment",
                 "saline_water", "saline_sediment", "soil_runoff",
                 "plant_associated", "animal_fecal"]


def microbial_ax(ax):
    base = "data/processed/microbial/emp_16s"
    files = [f for f in os.listdir(base) if f.endswith(".npz")]
    # one representative sample per habitat class
    by_hab = {}
    for fn in files:
        d = np.load(os.path.join(base, fn), allow_pickle=True)
        hab = str(d["source_name"])
        if hab not in by_hab:
            by_hab[hab] = np.array(d["abundances"])
        if len(by_hab) == len(set(HABITAT_ORDER)):
            pass
    habs = [h for h in HABITAT_ORDER if h in by_hab]
    M = np.stack([by_hab[h] for h in habs], axis=0)           # (H, 5000)
    top = np.argsort(M.sum(0))[::-1][:45]                     # top OTUs overall
    Mt = M[:, top]
    Mt = np.log10(Mt + 1e-4)                                  # log relative abundance
    im = ax.imshow(Mt, aspect="auto", cmap="viridis",
                   interpolation="nearest")
    ax.set_yticks(range(len(habs)))
    ax.set_yticklabels([h.replace("_", " ") for h in habs], fontsize=8)
    ax.set_xticks([])
    ax.set_xlabel("top 45 bacterial OTUs (16S rRNA)", fontsize=8.5)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("log₁₀ rel. abundance", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return ("Microbial community (16S rRNA)",
            f"Earth Microbiome Project  ·  {len(habs)} habitat classes  ·  5000 OTUs")


# --------------------------------------------------------------------------
# MOLECULAR
# --------------------------------------------------------------------------
def molecular_ax(ax):
    # GSE73661: 33,252 genes x 178 samples (no NaN, largest sample set)
    d = np.load("data/processed/molecular/real/GSE73661.npz", allow_pickle=True)
    e = np.array(d["expression"])           # (genes, samples), log-scale
    var = e.var(1)
    g = np.argsort(var)[::-1][:70]          # most variable genes
    # order samples by leading expression gradient for visible structure
    E = e[g]
    order = np.argsort(E[0])
    E = E[:, order]
    # z-score per gene
    E = (E - E.mean(1, keepdims=True)) / (E.std(1, keepdims=True) + 1e-6)
    im = ax.imshow(E, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                   interpolation="nearest")
    ax.set_xlabel(f"{E.shape[1]} samples", fontsize=8.5)
    ax.set_ylabel("70 top-variable genes", fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("z-scored expression", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return ("Molecular expression (transcriptomics)",
            f"GEO GSE73661  ·  {e.shape[0]:,} genes × {e.shape[1]} samples")


# --------------------------------------------------------------------------
# BEHAVIORAL
# --------------------------------------------------------------------------
EFFECT_GROUPS = [
    ("MOR", "Mortality", "#6b6b6b"),
    ("ITX", "Immobilization", "#d62728"),
    ("AVO", "Avoidance", "#1f77b4"),
    ("REP", "Reproduction", "#2ca02c"),
    ("GRO", "Growth", "#ff7f0e"),
]


def behavioral_ax(ax):
    """Real EPA ECOTOX Daphnia dose–response: behavioral/physiological effect
    (%) vs contaminant concentration. This is the genuine signal underlying
    SENTINEL's behavioral modality (the model's keypoint tensor is a synthetic
    encoding of these ECOTOX endpoints, so we plot the real source data)."""
    import csv
    f = "data/processed/behavioral/ecotox_daphnia_conc_response.csv"
    rows = list(csv.DictReader(open(f)))
    eff = np.array([r["effect"] for r in rows])
    conc = np.array([float(r["conc_mgL"]) for r in rows])
    pct = np.array([float(r["effect_pct"]) for r in rows])
    keep = conc > 1e-4
    eff, conc, pct = eff[keep], conc[keep], pct[keep]
    n_total = len(conc)
    for code, label, color in EFFECT_GROUPS:
        m = eff == code
        if not m.any():
            continue
        ax.scatter(conc[m], pct[m], s=18, c=color, alpha=0.6,
                   edgecolors="none", label=f"{label} ({int(m.sum())})")
    ax.set_xscale("log")
    ax.set_xlabel("contaminant concentration (mg/L, log scale)", fontsize=8.5)
    ax.set_ylabel("organism effect (%)", fontsize=8.5)
    ax.set_ylim(-4, 104)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="both", alpha=0.18, lw=0.5)
    ax.legend(fontsize=7.5, loc="center left", framealpha=0.9,
              title="behavioral / physiological endpoint",
              title_fontsize=7.5)
    return ("Behavioral & physiological response (Daphnia)",
            f"EPA ECOTOX  ·  {n_total:,} dose–response records  ·  10 Daphnia species")


# --------------------------------------------------------------------------
# Standalone single-modality figures
# --------------------------------------------------------------------------
def standalone():
    # satellite
    fig, ax = plt.subplots(figsize=(5, 5))
    t, sub = satellite_ax(ax)
    ax.set_title(t, fontsize=14, fontweight="bold", color=ACCENT)
    fig.text(0.5, 0.02, sub, ha="center", fontsize=9, color="#444")
    fig.savefig(f"{OUT}/modality_satellite.png", dpi=200, bbox_inches="tight",
                facecolor="white"); plt.close(fig)

    # sensor
    fig = plt.figure(figsize=(7, 5.5))
    gs = gridspec.GridSpec(1, 1, figure=fig, top=0.86, bottom=0.10,
                           left=0.16, right=0.97)
    _, (t, sub) = sensor_panel(fig, gs[0])
    fig.suptitle(t, fontsize=14, fontweight="bold", color=ACCENT, y=0.97)
    fig.text(0.5, 0.91, sub, ha="center", fontsize=9, color="#444")
    fig.savefig(f"{OUT}/modality_sensor.png", dpi=200, bbox_inches="tight",
                facecolor="white"); plt.close(fig)

    # microbial
    fig, ax = plt.subplots(figsize=(7, 4.2))
    t, sub = microbial_ax(ax)
    ax.set_title(t, fontsize=14, fontweight="bold", color=ACCENT, pad=10)
    fig.text(0.5, 0.005, sub, ha="center", fontsize=9, color="#444")
    fig.savefig(f"{OUT}/modality_microbial.png", dpi=200, bbox_inches="tight",
                facecolor="white"); plt.close(fig)

    # molecular
    fig, ax = plt.subplots(figsize=(6, 5))
    t, sub = molecular_ax(ax)
    ax.set_title(t, fontsize=14, fontweight="bold", color=ACCENT, pad=10)
    fig.text(0.5, 0.005, sub, ha="center", fontsize=9, color="#444")
    fig.savefig(f"{OUT}/modality_molecular.png", dpi=200, bbox_inches="tight",
                facecolor="white"); plt.close(fig)

    # behavioral
    fig, ax = plt.subplots(figsize=(6.5, 5))
    t, sub = behavioral_ax(ax)
    ax.set_title(t, fontsize=14, fontweight="bold", color=ACCENT, pad=10)
    fig.text(0.5, -0.01, sub, ha="center", fontsize=9, color="#444")
    fig.savefig(f"{OUT}/modality_behavioral.png", dpi=220, bbox_inches="tight",
                facecolor="white"); plt.close(fig)
    print("wrote 5 standalone modality_*.png")


# --------------------------------------------------------------------------
# Combined overview
# --------------------------------------------------------------------------
def overview():
    fig = plt.figure(figsize=(20, 8.5))
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.42, wspace=0.55,
                           top=0.88, bottom=0.08, left=0.05, right=0.97)
    # row1: satellite, sensor(2 cols), microbial(2 cols), [molecular start]
    ax_sat = fig.add_subplot(gs[0, 0:2])
    ax_mic = fig.add_subplot(gs[0, 2:4])
    ax_mol = fig.add_subplot(gs[0, 4:6])
    ax_beh = fig.add_subplot(gs[1, 0:2])
    sensor_cell = gs[1, 2:5]

    def titled(ax, ret):
        t, sub = ret
        ax.set_title(t, fontsize=13, fontweight="bold", color=ACCENT, pad=8)
        ax.text(0.5, -0.16, sub, transform=ax.transAxes, ha="center",
                va="top", fontsize=8.5, color="#555")

    titled(ax_sat, satellite_ax(ax_sat))
    titled(ax_mic, microbial_ax(ax_mic))
    titled(ax_mol, molecular_ax(ax_mol))
    titled(ax_beh, behavioral_ax(ax_beh))
    axes, (t, sub) = sensor_panel(fig, sensor_cell)
    axes[0].set_title(t, fontsize=13, fontweight="bold", color=ACCENT, pad=8)
    axes[-1].text(0.5, -0.55, sub, transform=axes[-1].transAxes, ha="center",
                  va="top", fontsize=8.5, color="#555")

    fig.suptitle("SENTINEL's five input modalities — real data samples",
                 fontsize=22, fontweight="bold", y=0.965)
    fig.savefig(f"{OUT}/modality_overview.png", dpi=200, bbox_inches="tight",
                facecolor="white"); plt.close(fig)
    print("wrote modality_overview.png")


if __name__ == "__main__":
    standalone()
