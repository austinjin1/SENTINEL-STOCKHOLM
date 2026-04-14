"""
Generate publication-quality figures for SENTINEL SJWP paper (v3).
Updates for expanded training, 31 case studies, SOTA benchmarks, HydroViT v9.

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


def fig_sota_comparison():
    """Fig 2: SENTINEL vs Published SOTA / Best Baselines — grouped bar chart."""
    encoders = [
        {
            'name': 'AquaSSM',
            'metric': 'AUROC',
            'ours': 0.916,
            'competitor': 0.864,
            'comp_name': 'MCN-LSTM [24]',
            'comp_type': 'sota',
            'threshold': 0.85,
        },
        {
            'name': 'HydroViT',
            'metric': 'R²',
            'ours': 0.893,
            'competitor': 0.884,
            'comp_name': 'DenseNet121 [23]',
            'comp_type': 'sota',
            'threshold': 0.55,
        },
        {
            'name': 'MicroBiomeNet',
            'metric': 'F1',
            'ours': 0.913,
            'comp_name': 'SimpleMLP',
            'competitor': 0.905,
            'comp_type': 'baseline',
            'threshold': 0.70,
        },
        {
            'name': 'ToxiGene',
            'metric': 'F1',
            'ours': 0.886,
            'competitor': 0.897,
            'comp_name': 'Random Forest',
            'comp_type': 'baseline',
            'threshold': 0.80,
        },
        {
            'name': 'BioMotion',
            'metric': 'AUROC',
            'ours': 1.000,
            'competitor': 0.958,
            'comp_name': 'Deep AE [25]',
            'comp_type': 'sota',
            'threshold': 0.80,
        },
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    y = np.arange(len(encoders))
    bar_h = 0.35

    ours_vals = [e['ours'] for e in encoders]
    comp_vals = [e['competitor'] for e in encoders]

    # Colors
    ours_color = '#b2182b'
    sota_color = '#e08214'
    baseline_color = '#4393c3'
    comp_colors = [sota_color if e['comp_type'] == 'sota' else baseline_color for e in encoders]

    # Bars
    bars_ours = ax.barh(y - bar_h / 2, ours_vals, bar_h, color=ours_color,
                         edgecolor='black', linewidth=0.5, label='SENTINEL (Ours)', zorder=3)
    for i, e in enumerate(encoders):
        ax.barh(y[i] + bar_h / 2, e['competitor'], bar_h, color=comp_colors[i],
                edgecolor='black', linewidth=0.5, zorder=3)

    # Value annotations + delta
    for i, e in enumerate(encoders):
        delta = e['ours'] - e['competitor']
        delta_str = f'+{delta:.3f}' if delta >= 0 else f'{delta:.3f}'

        # Our value
        ax.text(e['ours'] + 0.004, y[i] - bar_h / 2, f"{e['ours']:.3f}",
                va='center', fontsize=8, fontweight='bold', color='#333')
        # Competitor value + name
        ax.text(e['competitor'] + 0.004, y[i] + bar_h / 2,
                f"{e['competitor']:.3f}  ({e['comp_name']})",
                va='center', fontsize=7.5, color='#555')

        # Delta badge
        badge_color = '#2e7d32' if delta >= 0 else '#888888'
        label = f'{delta_str}' if e['comp_type'] == 'sota' else 'First-in-class'
        if e['comp_type'] == 'sota':
            label = f'{delta_str} vs SOTA'
        ax.text(0.52, y[i] - bar_h / 2 - 0.08, label,
                fontsize=7, fontweight='bold', color=badge_color, va='top')

        # Threshold marker
        ax.plot(e['threshold'], y[i], marker='|', color='red', markersize=18,
                markeredgewidth=1.5, zorder=4)

    # Y-axis
    ylabels = [f"{e['name']}\n({e['metric']})" for e in encoders]
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel('Performance (metric-specific)')
    ax.set_title('SENTINEL Encoders vs. Published SOTA and Best Baselines', fontweight='bold')
    ax.set_xlim(0.50, 1.07)
    ax.invert_yaxis()

    # Legend
    sota_patch = mpatches.Patch(color=sota_color, label='Published SOTA')
    base_patch = mpatches.Patch(color=baseline_color, label='Best baseline (first-in-class)')
    ours_patch = mpatches.Patch(color=ours_color, label='SENTINEL (Ours)')
    thresh_line = plt.Line2D([], [], color='red', marker='|', markersize=10,
                              markeredgewidth=1.5, linestyle='none', label='Threshold')
    ax.legend(handles=[ours_patch, sota_patch, base_patch, thresh_line],
              loc='lower right', framealpha=0.9, fontsize=8)

    plt.tight_layout()
    out = FIGOUT / "fig2_sota_comparison.jpg"
    fig.savefig(out, format='jpeg', dpi=300)
    plt.close(fig)
    print(f"  {out.name} ({out.stat().st_size / 1024:.0f} KB)")


def fig_case_studies():
    """Fig 4: Vertical bar chart — all 31 events sorted by lead time."""
    with open(RESULTS / "case_studies_v3" / "case_studies_v3.json") as f:
        data = json.load(f)

    cat_a = data["events"]["category_a"]
    cat_b = data["events"]["category_b"]
    cat_c = data["events"]["category_c"]

    # Tag each event with its category
    for e in cat_a:
        e['_cat'] = 'a'
    for e in cat_b:
        e['_cat'] = 'b'
    for e in cat_c:
        e['_cat'] = 'c'

    all_events = cat_a + cat_b + cat_c
    all_events.sort(key=lambda e: e["lead_time_hours"])

    short_names = {
        "lake_erie_hab_2023": "Lake Erie\nHAB '23",
        "toledo_water_crisis_2014": "Toledo\nCrisis '14",
        "gulf_dead_zone_2023": "Gulf Dead\nZone '23",
        "chesapeake_bay_blooms_2023": "Ches. Bay\nBlooms '23",
        "neon_pose_do_depletion_2025": "POSE\nDO Depl.",
        "neon_blde_storm_conductance_2024": "BLDE\nStorm",
        "neon_mart_turbidity_2025": "MART\nTurbidity",
        "neon_barc_eutrophication_2025": "BARC\nEutroph.",
        "neon_leco_acid_runoff_2024": "LECO\nAcid",
        "neon_sugg_conductance_2024": "SUGG\nAgri.",
        "grand_lake_st_marys_hab_2009": "Grand Lake\nHAB '09",
        "lake_erie_hab_2015": "L. Erie\nHAB '15",
        "lake_okeechobee_hab_2016": "L. Okee.\nHAB '16",
        "lake_okeechobee_hab_2018": "L. Okee.\nHAB '18",
        "sf_bay_heterosigma_2022": "SF Bay\nFish Kill",
        "klamath_river_hab_2021": "Klamath\nHAB '21",
        "utah_lake_hab_2016": "Utah Lake\nHAB '16",
        "utah_lake_hab_2018": "Utah Lake\nHAB '18",
        "mississippi_salinity_intrusion_2023": "Mississippi\nSalt. '23",
        "delaware_river_salinity_2022": "Delaware\nSalt. '22",
        "animas_river_amd_2015": "Animas\nAMD '15",
        "neuse_river_hypoxia_2020_2022": "Neuse R.\nHypoxia",
        "jordan_lake_hab_nc": "Jordan L.\nHAB (NC)",
        "iowa_nitrate_crisis": "Iowa\nNitrate",
        "chesapeake_bay_hypoxia_2018": "Ches. Bay\nHypoxia '18",
        "green_bay_hypoxia": "Green Bay\nHypoxia",
        "saginaw_bay_hab": "Saginaw\nHAB",
        "hudson_river_hab_2025": "Hudson R.\nHAB '25",
        "clear_lake_hab_2024": "Clear L.\nHAB '24",
        "tar_creek_amd_oklahoma": "Tar Creek\nAMD (OK)",
        "lake_winnebago_hab": "L. Winneb.\nHAB",
    }

    names = []
    times = []
    cats = []
    for e in all_events:
        eid = e["event_id"]
        names.append(short_names.get(eid, eid[:12]))
        times.append(e["lead_time_hours"])
        cats.append(e['_cat'])

    fig, ax = plt.subplots(figsize=(11, 4.5))

    color_map = {'a': '#2e7d32', 'b': '#00796b', 'c': '#1565c0'}
    colors = [color_map[c] for c in cats]

    x = np.arange(len(names))
    ax.bar(x, times, color=colors, edgecolor='black', linewidth=0.3, width=0.8)

    # Value labels on top of bars (days for long, hours for short)
    for i, t in enumerate(times):
        if t >= 480:
            label = f'{t / 24:.0f}d'
        else:
            label = f'{t:.0f}h'
        ax.text(i, t + max(times) * 0.015, label, ha='center', va='bottom',
                fontsize=5.5, color='#1b5e20', fontweight='bold', rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=5.5, ha='center')
    ax.set_ylabel('Lead Time (hours)')
    ax.set_title('SENTINEL Early Warning: 31/31 Events Detected Before Official Report',
                  fontweight='bold', fontsize=12)

    # Mean and median lines
    mean_lt = data["statistics"]["all_events"]["mean_lead_time_hours"]
    median_lt = data["statistics"]["all_events"]["median_lead_time_hours"]
    ax.axhline(y=mean_lt, color='#d32f2f', linestyle='--', alpha=0.8, linewidth=1.2)
    ax.text(len(names) - 0.5, mean_lt + 30, f'Mean: {mean_lt:.0f}h ({mean_lt/24:.0f}d)',
            color='#d32f2f', fontsize=8, ha='right', fontweight='bold')

    # Legend
    hist_patch = mpatches.Patch(color='#2e7d32', label='Historical (4)')
    neon_patch = mpatches.Patch(color='#00796b', label='NEON real sensor (6)')
    new_patch = mpatches.Patch(color='#1565c0', label='Research-validated (21)')
    ax.legend(handles=[hist_patch, neon_patch, new_patch], loc='upper left',
              framealpha=0.9, fontsize=8)

    ax.set_xlim(-0.5, len(names) - 0.5)
    ax.set_ylim(0, max(times) * 1.12)

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


if __name__ == "__main__":
    print("Generating SENTINEL paper figures (v3)...")
    fig_sota_comparison()
    fig_case_studies()
    fig_risk_index()
    print("Done. 3 figures generated.")
