"""
Tier A — real biodiversity figures from USGS BioData / EPA Water Quality Portal.

Pulls real biological monitoring stations and macroinvertebrate/fish "Count"
records (2015-2025) for three HUC-2 regions (02 Mid-Atlantic/Chesapeake,
04 Great Lakes, 05 Ohio River Basin) directly from the public Water Quality
Portal REST API, then renders three publication figures:

  fig1_site_map.png      - real biomonitoring coverage (all sites, by basin)
  fig2_richness.png      - observed taxonomic richness (map + per-basin violin)
  fig3_bioindicators.png - EPT (sensitive) vs tolerant taxa occupancy

Every number is reproducible from the API call below - no synthetic data and
no model outputs are involved. This establishes that the underlying ecology is
real, independent of any downstream model claims.

Usage:  python scripts/build_tierA_figures.py
Requires: pandas, geopandas, matplotlib, requests, pyarrow (network access).
"""
from __future__ import annotations
import io, time, glob
from pathlib import Path
import requests
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT / "data" / "tierA"
FIGS = PROJECT / "figures" / "tierA"
DATA.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

WQP = "https://www.waterqualitydata.us/data"
HUCS = {"02": "Mid-Atlantic / Chesapeake", "04": "Great Lakes", "05": "Ohio River Basin"}
DATE_LO, DATE_HI = "01-01-2015", "12-31-2025"
STATES_GEOJSON = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"

# Bioindicator keyword sets (matched against SubjectTaxonomicName, any rank).
EPT = {  # Ephemeroptera/Plecoptera/Trichoptera - pollution-SENSITIVE, clean-water indicators
    "ephemeropt", "baetid", "heptagen", "ephemerell", "caenid", "leptophleb", "isonychia",
    "plecopt", "perlid", "perlodid", "capniid", "nemourid", "taeniopter",
    "trichopt", "hydropsych", "cheumatopsyche", "hydroptil", "philopotam", "leptocerid",
}
TOL = {  # pollution-TOLERANT taxa - worms & midges
    "oligochaet", "tubificid", "lumbricul", "naidid",
    "chironom", "polypedilum", "cricotopus", "tanytars", "physid", "isopod",
}


def pull_stations() -> pd.DataFrame:
    keep = ["MonitoringLocationIdentifier", "MonitoringLocationName",
            "MonitoringLocationTypeName", "LatitudeMeasure", "LongitudeMeasure",
            "HUCEightDigitCode", "StateCode"]
    frames = []
    for h, name in HUCS.items():
        p = dict(huc=h, characteristicName="Count", sampleMedia="Biological",
                 startDateLo=DATE_LO, startDateHi=DATE_HI, mimeType="csv", zip="no", sorted="no")
        r = requests.get(f"{WQP}/Station/search", params=p,
                         headers={"Accept": "text/csv"}, timeout=300)
        d = pd.read_csv(io.StringIO(r.text), low_memory=False)
        d = d[[c for c in keep if c in d.columns]].copy()
        d["region"] = name
        d["huc2"] = h
        frames.append(d)
        print(f"  stations HUC {h} ({name}): {len(d):,}")
        time.sleep(1)
    df = pd.concat(frames, ignore_index=True)
    for c in ["LatitudeMeasure", "LongitudeMeasure"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["LatitudeMeasure", "LongitudeMeasure"])
    df = df[df.LatitudeMeasure.between(24, 50) & df.LongitudeMeasure.between(-95, -66)]
    df = df.drop_duplicates(subset=["MonitoringLocationIdentifier"])
    df.to_csv(DATA / "stations_all.csv", index=False)
    print(f"  TOTAL unique biological monitoring sites: {len(df):,}")
    return df


def pull_results() -> pd.DataFrame:
    keep = ["MonitoringLocationIdentifier", "ActivityStartDate",
            "SubjectTaxonomicName", "ResultMeasureValue", "ActivityMediaName"]
    frames = []
    for h, name in HUCS.items():
        cache = DATA / f"res_{h}.parquet"
        if cache.exists():
            d = pd.read_parquet(cache)
        else:
            p = dict(huc=h, characteristicName="Count", startDateLo=DATE_LO,
                     startDateHi=DATE_HI, mimeType="csv", zip="no", sorted="no")
            r = requests.get(f"{WQP}/Result/search", params=p,
                             headers={"Accept": "text/csv"}, timeout=600)
            d = pd.read_csv(io.StringIO(r.text), low_memory=False)
            if "ActivityMediaName" in d:
                d = d[d["ActivityMediaName"] == "Biological"]
            d = d[[c for c in keep if c in d.columns]].copy()
            d["region"] = name
            d.to_parquet(cache, index=False)
        frames.append(d)
        print(f"  results HUC {h}: {len(d):,} records, "
              f"{d.MonitoringLocationIdentifier.nunique():,} sites, "
              f"{d.SubjectTaxonomicName.nunique():,} taxa")
        time.sleep(1)
    return pd.concat(frames, ignore_index=True)


def compute_site_metrics(res: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    v = res.SubjectTaxonomicName.fillna("").str.lower()

    def mask(kws):
        m = pd.Series(False, index=res.index)
        for k in kws:
            m |= v.str.contains(k, na=False)
        return m

    res = res.assign(is_ept=mask(EPT), is_tol=mask(TOL))
    g = res.groupby("MonitoringLocationIdentifier")
    site = pd.DataFrame({
        "richness": g.SubjectTaxonomicName.nunique(),
        "ept_taxa": res[res.is_ept].groupby("MonitoringLocationIdentifier").SubjectTaxonomicName.nunique(),
        "ept_present": g.is_ept.any(),
        "tol_present": g.is_tol.any(),
    }).reset_index()
    site["ept_taxa"] = site.ept_taxa.fillna(0).astype(int)
    cols = ["MonitoringLocationIdentifier", "LatitudeMeasure", "LongitudeMeasure", "region"]
    site = site.merge(st[cols].drop_duplicates("MonitoringLocationIdentifier"),
                      on="MonitoringLocationIdentifier", how="left")
    site = site.dropna(subset=["LatitudeMeasure", "LongitudeMeasure"])
    site = site[site.LatitudeMeasure.between(35, 49.5) & site.LongitudeMeasure.between(-92, -66)]
    site.to_csv(DATA / "site_metrics.csv", index=False)
    print(f"  sites with taxa + coords: {len(site):,} | median richness: {int(site.richness.median())} "
          f"| EPT present at {100*site.ept_present.mean():.0f}% of sites")
    return site


def load_states():
    import geopandas as gpd
    f = DATA / "us-states.geojson"
    if not f.exists():
        f.write_bytes(requests.get(STATES_GEOJSON, timeout=60).content)
    return gpd.read_file(f)


XL, YL = (-92, -66), (35, 49.5)
REG_COLORS = {"Ohio River Basin": "#33a02c", "Great Lakes": "#1f78b4",
              "Mid-Atlantic / Chesapeake": "#6a3d9a"}


def _base(ax, states):
    states.plot(ax=ax, color="#f7f7f7", edgecolor="#d0d0d0", linewidth=0.6, zorder=0)
    states.boundary.plot(ax=ax, color="#d0d0d0", linewidth=0.6, zorder=1)
    ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_aspect(1.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def render(st, site, states):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    regs = [r for r in REG_COLORS if r in site.region.unique()]

    # Fig 1 - site map
    fig, ax = plt.subplots(figsize=(11, 7.2), dpi=200)
    _base(ax, states)
    for r, c in REG_COLORS.items():
        s = st[st.region == r]
        ax.scatter(s.LongitudeMeasure, s.LatitudeMeasure, s=5, c=c, alpha=0.45,
                   edgecolors="none", zorder=3, rasterized=True)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Real biological monitoring coverage - USGS BioData / EPA Water Quality Portal",
                 fontsize=13, fontweight="bold", pad=12)
    ax.text(0.5, 1.005, f"{len(st):,} unique freshwater bioassessment sites - "
            "macroinvertebrate + fish community surveys - 2015-2025",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5, color="#444")
    counts = st.groupby("region").size().to_dict()
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=REG_COLORS[r],
               markersize=8, label=f"{r}  (n={counts.get(r,0):,})") for r in REG_COLORS]
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.95,
              fontsize=9.5, title="HUC-2 region", title_fontsize=9.5)
    plt.tight_layout(); plt.savefig(FIGS / "fig1_site_map.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Fig 2 - richness
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), dpi=200,
                             gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]; _base(ax, states)
    s = site.sort_values("richness")
    sc = ax.scatter(s.LongitudeMeasure, s.LatitudeMeasure, c=s.richness, s=11, cmap="viridis",
                    vmin=0, vmax=np.percentile(s.richness, 95), alpha=0.8,
                    edgecolors="none", zorder=3, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Taxonomic richness (taxa / site)")
    ax.set_title("Observed macroinvertebrate richness", fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax2 = axes[1]
    parts = ax2.violinplot([site[site.region == r].richness.values for r in regs], showmedians=True)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(REG_COLORS[regs[i]]); b.set_alpha(0.6)
    for k in ("cmedians", "cbars", "cmins", "cmaxes"):
        parts[k].set_color("#333")
    ax2.set_xticks(range(1, len(regs) + 1))
    ax2.set_xticklabels([r.replace(" / ", "/\n") for r in regs], fontsize=9)
    ax2.set_ylabel("Taxonomic richness (taxa / site)")
    ax2.set_ylim(0, site.richness.quantile(0.98))
    ax2.set_title("Richness distribution by basin", fontsize=12, fontweight="bold")
    for s_ in ("top", "right"):
        ax2.spines[s_].set_visible(False)
    fig.suptitle(f"Real biodiversity gradients - {len(site):,} sites - USGS BioData / EPA WQP",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(FIGS / "fig2_richness.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Fig 3 - bioindicators
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), dpi=200)
    axA = axes[0]; _base(axA, states)
    p = site[site.ept_present]; a = site[~site.ept_present]
    axA.scatter(a.LongitudeMeasure, a.LatitudeMeasure, s=8, c="#d9d9d9", alpha=0.6,
                edgecolors="none", zorder=2, label="EPT absent")
    sc = axA.scatter(p.LongitudeMeasure, p.LatitudeMeasure, c=p.ept_taxa, s=12, cmap="YlGnBu",
                     vmin=1, vmax=np.percentile(p.ept_taxa, 95), alpha=0.85,
                     edgecolors="none", zorder=3, rasterized=True)
    cb = fig.colorbar(sc, ax=axA, fraction=0.035, pad=0.02)
    cb.set_label("EPT taxa richness (clean-water indicators)")
    axA.set_title("EPT bioindicators (mayfly / stonefly / caddisfly)", fontsize=12, fontweight="bold")
    axA.set_xlabel("Longitude"); axA.set_ylabel("Latitude")
    axA.legend(loc="lower right", fontsize=8, frameon=True)
    axB = axes[1]
    rr = pd.DataFrame([[r, 100 * site[site.region == r].ept_present.mean(),
                        100 * site[site.region == r].tol_present.mean()] for r in regs],
                      columns=["region", "EPT", "Tolerant"])
    x = np.arange(len(rr)); w = 0.38
    axB.bar(x - w/2, rr.EPT, w, color="#2c7fb8", label="EPT (sensitive)")
    axB.bar(x + w/2, rr.Tolerant, w, color="#c79a3a", label="Tolerant (worms / midges)")
    axB.set_xticks(x); axB.set_xticklabels([r.replace(" / ", "/\n") for r in rr.region], fontsize=9)
    axB.set_ylabel("% of sites occupied"); axB.set_ylim(0, 100)
    axB.set_title("Bioindicator occupancy by basin", fontsize=12, fontweight="bold")
    axB.legend(fontsize=9, frameon=False)
    for s_ in ("top", "right"):
        axB.spines[s_].set_visible(False)
    fig.suptitle("Real keystone bioindicator occupancy - sensitive vs pollution-tolerant taxa",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(FIGS / "fig3_bioindicators.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  wrote {FIGS/'fig1_site_map.png'}, fig2_richness.png, fig3_bioindicators.png")


def main():
    print("[1/4] pulling biological stations ...")
    st = pull_stations()
    print("[2/4] pulling biological count records ...")
    res = pull_results()
    print("[3/4] computing per-site metrics ...")
    site = compute_site_metrics(res, st)
    print("[4/4] rendering figures ...")
    render(st, site, load_states())
    print("done.")


if __name__ == "__main__":
    main()
