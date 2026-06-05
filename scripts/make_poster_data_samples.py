#!/usr/bin/env python
"""Render high-quality figures of REAL Sentinel-2 imagery samples for the
SENTINEL research poster.

Data source: data/processed/satellite/drone_wq_pairs.npz
  - images:  (12000, 4, 224, 224) float32  Sentinel-2 surface reflectance
  - bands:   B02 (blue), B03 (green), B04 (red), B08 (NIR)
  - targets: (12000, 5)  DO, pH, Turb, Temp, SpCond  (paired in-situ water quality)
  - metadata: per-sample site_id / date / lat / lon (USGS stations)

Outputs (figures/):
  - poster_s2_gallery.png       12-tile true-color gallery, labeled w/ site + WQ
  - poster_s2_multimodal.png    RGB | NIR | NDWI for 3 representative sites
  - poster_s2_crisis.png        Chesapeake hypoxia 2018 event time series tiles
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

PAIRS = "data/processed/satellite/drone_wq_pairs.npz"
CRISIS_DIR = "data/processed/satellite/crisis_tiles"
OUT = "figures"

TARGET_NAMES = ["DO", "pH", "Turb", "Temp", "SpCond"]
TARGET_UNITS = ["mg/L", "", "NTU", "°C", "µS/cm"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#222222",
    "figure.dpi": 200,
})


def stretch_rgb(img4, lo=2, hi=98, gamma=0.85):
    """img4: (4,H,W) reflectance -> RGB uint-ish float in [0,1].
    bands order = B02,B03,B04,B08 -> R=B04(idx2) G=B03(idx1) B=B02(idx0)."""
    rgb = np.stack([img4[2], img4[1], img4[0]], axis=-1).astype(np.float32)
    out = np.empty_like(rgb)
    for c in range(3):
        ch = rgb[..., c]
        p_lo, p_hi = np.percentile(ch, [lo, hi])
        if p_hi <= p_lo:
            p_hi = p_lo + 1e-6
        out[..., c] = np.clip((ch - p_lo) / (p_hi - p_lo), 0, 1)
    out = np.power(out, gamma)
    return out


def ndwi(img4):
    """McFeeters NDWI = (Green - NIR)/(Green + NIR). green=B03(idx1) nir=B08(idx3)."""
    g = img4[1].astype(np.float32)
    nir = img4[3].astype(np.float32)
    return (g - nir) / (g + nir + 1e-6)


def quality_score(img4):
    """Higher = better candidate. Reward scenes with clear spatial detail AND
    visible water; penalize nodata wedges, clouds/haze, and flat tiles."""
    rgb = np.stack([img4[2], img4[1], img4[0]], axis=0)
    var = float(rgb.std())
    bright = float(np.mean(rgb))
    cloud_frac = float(np.mean(rgb > 0.55))
    # nodata wedge: pixels black across all bands (swath edge / fill)
    nodata_frac = float(np.mean(np.all(img4 < 0.02, axis=0)))
    # haze: globally bright + low local contrast
    haze = 1.0 if (bright > 0.30 and var < 0.06) else 0.0
    # water presence via NDWI; reward tiles that actually contain a water body
    nd = ndwi(img4)
    water_frac = float(np.mean(nd > 0.0))
    water_reward = 0.06 if 0.04 <= water_frac <= 0.75 else 0.0
    return (var
            + water_reward
            - 2.0 * nodata_frac
            - 1.5 * cloud_frac
            - 0.10 * haze
            - 0.4 * max(0, bright - 0.35))


def load_pairs():
    d = np.load(PAIRS, allow_pickle=True, mmap_mode="r")
    meta = json.loads(str(d["metadata"]))
    targets = np.array(d["targets"])
    return d, meta, targets


def pick_samples(d, meta, targets, n=12, seed=7):
    """Select geographically diverse, high-quality tiles with valid WQ."""
    rng = np.random.default_rng(seed)
    imgs = d["images"]
    N = imgs.shape[0]
    # candidate pool: sample a manageable subset, score, then diversify by state/lon
    cand = rng.choice(N, size=min(1500, N), replace=False)
    scored = []
    for i in cand:
        t = targets[i]
        n_valid = int(np.sum(~np.isnan(t)))
        if n_valid < 4:
            continue
        im = np.array(imgs[i])
        if not np.isfinite(im).all():
            continue
        s = quality_score(im)
        scored.append((s, int(i)))
    scored.sort(reverse=True)
    # take top quality, then greedily spread by longitude to maximize geo diversity
    top = [i for _, i in scored[:120]]
    chosen, used_lon = [], []
    for i in top:
        lon = meta[i]["lon"]
        if all(abs(lon - u) > 1.2 for u in used_lon):
            chosen.append(i)
            used_lon.append(lon)
        if len(chosen) >= n:
            break
    # backfill if geo filter was too strict
    for i in top:
        if len(chosen) >= n:
            break
        if i not in chosen:
            chosen.append(i)
    return chosen[:n]


def water_show_score(img4):
    """Rank tiles by how clearly they display a water body (for NDWI panel):
    want a meaningful, contiguous water fraction, strong contrast, no haze/cloud."""
    rgb = np.stack([img4[2], img4[1], img4[0]], axis=0)
    var = float(rgb.std())
    bright = float(np.mean(rgb))
    cloud_frac = float(np.mean(rgb > 0.55))
    nodata_frac = float(np.mean(np.all(img4 < 0.02, axis=0)))
    nd = ndwi(img4)
    water_frac = float(np.mean(nd > 0.05))
    # sweet spot: clear river/lake but also visible land for context
    water_term = 1.0 - abs(water_frac - 0.32) / 0.32  # peaks at ~32% water
    return (1.2 * max(0, water_term) + var
            - 2.5 * nodata_frac - 2.0 * cloud_frac
            - (0.6 if bright > 0.30 and var < 0.06 else 0))


def pick_water_samples(d, meta, targets, n=3, seed=11):
    rng = np.random.default_rng(seed)
    imgs = d["images"]
    N = imgs.shape[0]
    cand = rng.choice(N, size=min(2000, N), replace=False)
    scored = []
    for i in cand:
        t = targets[i]
        if int(np.sum(~np.isnan(t))) < 4:
            continue
        im = np.array(imgs[i])
        if not np.isfinite(im).all():
            continue
        scored.append((water_show_score(im), int(i)))
    scored.sort(reverse=True)
    top = [i for _, i in scored[:60]]
    chosen, used_lon = [], []
    for i in top:
        lon = meta[i]["lon"]
        if all(abs(lon - u) > 2.0 for u in used_lon):
            chosen.append(i); used_lon.append(lon)
        if len(chosen) >= n:
            break
    return chosen[:n]


def fmt_wq(t):
    parts = []
    for name, unit, v in zip(TARGET_NAMES, TARGET_UNITS, t):
        if np.isnan(v):
            continue
        if name == "SpCond":
            parts.append(f"{name} {v:,.0f}")
        elif name == "Temp":
            parts.append(f"{name} {v:.1f}°C")
        elif name == "pH":
            parts.append(f"{name} {v:.2f}")
        elif name == "DO":
            parts.append(f"{name} {v:.1f}")
        else:
            parts.append(f"{name} {v:.0f}")
    return "  •  ".join(parts[:4])


def make_gallery(d, meta, targets, idx):
    ncol, nrow = 4, 3
    fig = plt.figure(figsize=(15, 12.6))
    gs = gridspec.GridSpec(nrow, ncol, figure=fig, hspace=0.32, wspace=0.06,
                           top=0.88, bottom=0.04, left=0.02, right=0.98)
    imgs = d["images"]
    for k, i in enumerate(idx):
        r, c = divmod(k, ncol)
        ax = fig.add_subplot(gs[r, c])
        im = np.array(imgs[i])
        ax.imshow(stretch_rgb(im), interpolation="bilinear")
        m = meta[i]
        ax.set_title(f"USGS {m['site_id']}   {m['date']}\n"
                     f"{m['lat']:.3f}, {m['lon']:.3f}",
                     fontsize=10, pad=4)
        ax.text(0.5, -0.085, fmt_wq(targets[i]), transform=ax.transAxes,
                ha="center", va="top", fontsize=8.5, color="#1a3a6b")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#3a6ea5"); s.set_linewidth(1.4)
    fig.suptitle("Real Sentinel-2 imagery paired with in-situ water quality",
                 fontsize=20, fontweight="bold", y=0.965)
    fig.text(0.5, 0.925,
             "12 of 12,000 satellite–sensor pairs  ·  4-band B02/B03/B04/B08 (RGB+NIR) at 10 m  ·  379 USGS monitoring stations  ·  true-color render",
             ha="center", fontsize=11, color="#444444")
    out = f"{OUT}/poster_s2_gallery.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


def make_multimodal(d, meta, targets, sel):
    fig = plt.figure(figsize=(12, 11.5))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.18, wspace=0.06,
                           top=0.90, bottom=0.04, left=0.07, right=0.97)
    imgs = d["images"]
    col_titles = ["True color (B04/B03/B02)", "Near-infrared (B08)",
                  "NDWI water index"]
    for row, i in enumerate(sel):
        im = np.array(imgs[i])
        m = meta[i]
        # RGB
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(stretch_rgb(im), interpolation="bilinear")
        ax.set_ylabel(f"USGS {m['site_id']}\n{m['date']}", fontsize=10,
                      rotation=90, labelpad=10)
        # NIR
        ax2 = fig.add_subplot(gs[row, 1])
        nir = im[3]
        nlo, nhi = np.percentile(nir, [2, 98])
        ax2.imshow(np.clip((nir - nlo) / (nhi - nlo + 1e-6), 0, 1),
                   cmap="inferno", interpolation="bilinear")
        # NDWI
        ax3 = fig.add_subplot(gs[row, 2])
        nd = ndwi(im)
        im3 = ax3.imshow(nd, cmap="RdYlBu", vmin=-0.6, vmax=0.6,
                         interpolation="bilinear")
        for a in (ax, ax2, ax3):
            a.set_xticks([]); a.set_yticks([])
        if row == 0:
            ax.set_title(col_titles[0], fontsize=12, fontweight="bold")
            ax2.set_title(col_titles[1], fontsize=12, fontweight="bold")
            ax3.set_title(col_titles[2], fontsize=12, fontweight="bold")
        cb = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.03)
        cb.set_label("water ←  → land", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle("What WaterDroneNet sees: multi-band Sentinel-2 input",
                 fontsize=18, fontweight="bold", y=0.965)
    out = f"{OUT}/poster_s2_multimodal.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


def make_crisis():
    import glob, os
    files = sorted(glob.glob(f"{CRISIS_DIR}/chesapeake_hypoxia_2018_*.npz"))
    if not files:
        files = sorted(glob.glob(f"{CRISIS_DIR}/*.npz"))
    if not files:
        print("no crisis tiles; skipping")
        return
    # evenly sample up to 6 dates across the event
    pick = files if len(files) <= 6 else [files[int(round(x))]
            for x in np.linspace(0, len(files) - 1, 6)]
    n = len(pick)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.0))
    if n == 1:
        axes = [axes]
    for ax, f in zip(axes, pick):
        d = np.load(f, allow_pickle=True)
        im = d["image"]
        # water-dominated dark scenes: stronger stretch to reveal bloom/detail
        ax.imshow(stretch_rgb(im, lo=1, hi=99.5, gamma=0.65),
                  interpolation="bilinear")
        date = os.path.basename(f).replace(".npz", "").split("_")[-1]
        ax.set_title(date, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#9c2b2b"); s.set_linewidth(1.4)
    fig.suptitle("Crisis case study — Chesapeake Bay hypoxia event, Sentinel-2 time series",
                 fontsize=15, fontweight="bold", y=1.02)
    out = f"{OUT}/poster_s2_crisis.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


def main():
    d, meta, targets = load_pairs()
    idx = pick_samples(d, meta, targets, n=12)
    print("selected indices:", idx)
    for i in idx:
        print("  ", meta[i]["site_id"], meta[i]["date"],
              f"({meta[i]['lat']:.2f},{meta[i]['lon']:.2f})", fmt_wq(targets[i]))
    make_gallery(d, meta, targets, idx)
    water_idx = pick_water_samples(d, meta, targets, n=3)
    print("multimodal (water) indices:", water_idx)
    for i in water_idx:
        print("  ", meta[i]["site_id"], meta[i]["date"], fmt_wq(targets[i]))
    make_multimodal(d, meta, targets, water_idx)
    make_crisis()


if __name__ == "__main__":
    main()
