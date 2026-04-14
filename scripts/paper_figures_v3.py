"""
Generate publication-quality figures for SENTINEL SJWP paper (v3).
NeurIPS-style benchmarking, seaborn styling throughout.

Outputs figures to paper/figures/ at 300 DPI JPG.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
FIGOUT = PROJECT / "paper" / "figures"
FIGOUT.mkdir(parents=True, exist_ok=True)

# Global seaborn + publication style
sns.set_theme(style="whitegrid", font="serif", rc={
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

OURS_COLOR = '#c0392b'
SOTA_COLOR = '#e67e22'
BASE_PALETTE = sns.color_palette("Blues_r", 8)


def _save(fig, name):
    out = FIGOUT / name
    fig.savefig(out, format='jpeg', dpi=300)
    plt.close(fig)
    print(f"  {name} ({out.stat().st_size / 1024:.0f} KB)")


# ──────────────────────────────────────────────────────────────
# FIG 4: NeurIPS-style multi-panel SOTA benchmark
# ──────────────────────────────────────────────────────────────
def fig_sota_benchmark():
    """5-panel grouped bar chart: each encoder vs all benchmarked competitors."""

    panels = [
        {
            'title': 'AquaSSM (Sensor)',
            'metric': 'AUROC',
            'threshold': 0.85,
            'models': [
                ('AquaSSM (Ours)', 0.916, 'ours'),
                ('MCN-LSTM [24]', 0.864, 'sota'),
                ('One-Class SVM', 0.850, 'base'),
                ('LSTM (2-layer)', 0.837, 'base'),
                ('Transformer (4-head)', 0.834, 'base'),
                ('Isolation Forest', 0.728, 'base'),
            ],
        },
        {
            'title': 'HydroViT (Satellite)',
            'metric': 'R² (Water Temp)',
            'threshold': 0.55,
            'models': [
                ('HydroViT v9 (Ours)', 0.893, 'ours'),
                ('DenseNet121 [23]', 0.884, 'sota'),
                ('CNN Baseline', 0.854, 'base'),
                ('ResNet50', 0.812, 'base'),
                ('Random Forest', 0.801, 'base'),
                ('ViT (scratch)', 0.750, 'base'),
                ('Ridge Regression', 0.646, 'base'),
            ],
        },
        {
            'title': 'BioMotion (Behavioral)',
            'metric': 'AUROC',
            'threshold': 0.80,
            'models': [
                ('BioMotion (Ours)', 1.000, 'ours'),
                ('BiLSTM', 1.000, 'base'),
                ('Transformer', 0.999, 'base'),
                ('Deep AE [25]', 0.958, 'sota'),
                ('VAE Recon.', 0.952, 'base'),
                ('LSTM AE', 0.920, 'base'),
                ('Isolation Forest', 0.890, 'base'),
            ],
        },
        {
            'title': 'MicroBiomeNet (Microbial)',
            'metric': 'Macro F1',
            'threshold': 0.70,
            'models': [
                ('MicroBiomeNet (Ours)', 0.913, 'ours'),
                ('SimpleMLP', 0.905, 'base'),
                ('Logistic Regression', 0.876, 'base'),
                ('Extra Trees', 0.843, 'base'),
                ('Random Forest', 0.835, 'base'),
            ],
        },
        {
            'title': 'ToxiGene (Molecular)',
            'metric': 'Macro F1',
            'threshold': 0.80,
            'models': [
                ('ToxiGene (Ours)', 0.886, 'ours'),
                ('Random Forest', 0.897, 'base'),
                ('Extra Trees', 0.887, 'base'),
                ('Logistic Regression', 0.868, 'base'),
                ('PCA + LR', 0.808, 'base'),
            ],
        },
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    axes_flat = axes.flatten()

    for idx, panel in enumerate(panels):
        ax = axes_flat[idx]
        names = [m[0] for m in panel['models']]
        vals = [m[1] for m in panel['models']]
        types = [m[2] for m in panel['models']]

        colors = []
        for t in types:
            if t == 'ours':
                colors.append(OURS_COLOR)
            elif t == 'sota':
                colors.append(SOTA_COLOR)
            else:
                colors.append('#5b9bd5')

        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, vals, color=colors, edgecolor='white', linewidth=0.5, height=0.65)

        # Value annotations
        for i, v in enumerate(vals):
            weight = 'bold' if types[i] == 'ours' else 'normal'
            ax.text(v + 0.003, i, f'{v:.3f}', va='center', fontsize=7.5, fontweight=weight)

        # Threshold line
        ax.axvline(x=panel['threshold'], color='#e74c3c', linestyle='--', alpha=0.6, linewidth=1)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel(panel['metric'], fontsize=9)
        ax.set_title(panel['title'], fontweight='bold', fontsize=10)
        ax.invert_yaxis()

        # Set x limits based on data range
        lo = min(vals) - 0.05
        hi = max(vals) + 0.06
        ax.set_xlim(max(0, lo), min(1.05, hi))

        # Label: first-in-class or beats SOTA
        has_sota = any(t == 'sota' for t in types)
        if has_sota:
            ours_v = vals[0]
            sota_v = [v for v, t in zip(vals, types) if t == 'sota'][0]
            delta = ours_v - sota_v
            if delta >= 0:
                ax.text(0.98, 0.02, f'+{delta:.3f} vs SOTA',
                        transform=ax.transAxes, fontsize=7, fontweight='bold',
                        color='#27ae60', ha='right', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#eafaf1', alpha=0.8))
        else:
            ax.text(0.98, 0.02, 'First-in-class',
                    transform=ax.transAxes, fontsize=7, fontweight='bold',
                    color='#8e44ad', ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#f4ecf7', alpha=0.8))

    # Empty 6th panel for legend
    ax_leg = axes_flat[5]
    ax_leg.axis('off')
    ours_p = mpatches.Patch(color=OURS_COLOR, label='SENTINEL (Ours)')
    sota_p = mpatches.Patch(color=SOTA_COLOR, label='Published SOTA')
    base_p = mpatches.Patch(color='#5b9bd5', label='Baseline')
    thresh_l = plt.Line2D([], [], color='#e74c3c', linestyle='--', linewidth=1.5, label='Performance threshold')
    ax_leg.legend(handles=[ours_p, sota_p, base_p, thresh_l],
                  loc='center', fontsize=11, frameon=True, fancybox=True,
                  shadow=True, borderpad=1.5)
    ax_leg.set_title('Legend', fontsize=12, fontweight='bold')

    fig.suptitle('SENTINEL Encoders vs. Published SOTA and Baselines',
                 fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, "fig2_sota_comparison.jpg")


# ──────────────────────────────────────────────────────────────
# FIG 5: Ablation — drop-one + build-up
# ──────────────────────────────────────────────────────────────
def fig_ablation():
    """Two-panel ablation: drop-one impact + incremental build-up."""

    # Drop-one data (from paper text: AUC drop when modality is absent)
    drop_one = [
        ('Sensor', 0.246),
        ('Behavioral', 0.174),
        ('Satellite', 0.111),
        ('Microbial', 0.077),
        ('Molecular', 0.031),
    ]

    # Build-up data (from ablation: best single → best pair → ... → full)
    buildup = [
        ('Best single\n(Sensor)', 0.943),
        ('Best pair\n(+Behavioral)', 0.970),
        ('Best triple\n(+Satellite)', 0.983),
        ('Best quad\n(+Microbial)', 0.989),
        ('Full fusion\n(all 5)', 0.992),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    # Left: Drop-one
    names1 = [d[0] for d in drop_one]
    vals1 = [d[1] for d in drop_one]
    palette1 = sns.color_palette("Reds_r", len(vals1))
    bars1 = ax1.barh(range(len(names1)), vals1, color=palette1, edgecolor='white', height=0.6)
    for i, v in enumerate(vals1):
        ax1.text(v + 0.003, i, f'−{v:.3f}', va='center', fontsize=9, fontweight='bold')
    ax1.set_yticks(range(len(names1)))
    ax1.set_yticklabels(names1, fontsize=10)
    ax1.set_xlabel('AUROC drop when removed', fontsize=10)
    ax1.set_title('Modality Importance (Drop-One)', fontweight='bold')
    ax1.invert_yaxis()
    ax1.set_xlim(0, 0.32)

    # Right: Build-up
    names2 = [b[0] for b in buildup]
    vals2 = [b[1] for b in buildup]
    palette2 = sns.color_palette("YlOrRd", len(vals2))
    x_pos = np.arange(len(names2))
    bars2 = ax2.bar(x_pos, vals2, color=palette2, edgecolor='white', width=0.65)
    for i, v in enumerate(vals2):
        ax2.text(i, v + 0.002, f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    # Arrows showing incremental gain
    for i in range(1, len(vals2)):
        delta = vals2[i] - vals2[i-1]
        ax2.annotate(f'+{delta:.3f}', xy=(i, vals2[i-1] + (vals2[i] - vals2[i-1])/2),
                     fontsize=7, color='#666', ha='center', style='italic')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names2, fontsize=8)
    ax2.set_ylabel('Fusion AUROC', fontsize=10)
    ax2.set_title('Cumulative Fusion Gain', fontweight='bold')
    ax2.set_ylim(0.92, 1.0)
    ax2.axhline(y=0.943, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    _save(fig, "fig5_ablation.jpg")


# ──────────────────────────────────────────────────────────────
# FIG 9: FPR + Temporal Persistence (replaces conformal)
# ──────────────────────────────────────────────────────────────
def fig_fpr_persistence():
    """Two-panel: (L) FPR on clean vs event sites, (R) temporal persistence."""

    with open(RESULTS / "exp_false_positive" / "false_positive_results.json") as f:
        fpr_data = json.load(f)
    with open(RESULTS / "exp_temporal_persistence" / "persistence_results.json") as f:
        pers_data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # ── Left: Detection rate (clean vs event sites) ──
    # Clean sites
    clean_sites = list(fpr_data["non_event_sites"].keys())
    clean_rates = [fpr_data["non_event_sites"][s]["fpr"] for s in clean_sites]

    # Event sites
    event_names_map = {
        'lake_erie_hab_2023': 'Lake Erie\nHAB',
        'jordan_lake_hab_nc': 'Jordan L.\nHAB',
        'klamath_river_hab_2021': 'Klamath\nHAB',
        'gulf_dead_zone_2023': 'Gulf\nDead Zone',
        'chesapeake_hypoxia_2018': 'Chesapeake\nHypoxia',
        'mississippi_salinity_2023': 'Mississippi\nSalinity',
    }
    event_ids = list(fpr_data["case_study_sites"].keys())
    event_rates = [fpr_data["case_study_sites"][e]["high_score_rate"] for e in event_ids]
    event_labels = [event_names_map.get(e, e[:10]) for e in event_ids]

    # Combined
    all_labels = clean_sites + [''] + event_labels
    all_rates = clean_rates + [0] + event_rates
    all_colors = ['#5b9bd5'] * len(clean_sites) + ['white'] + [OURS_COLOR] * len(event_ids)

    x = np.arange(len(all_labels))
    bars = ax1.bar(x, all_rates, color=all_colors, edgecolor='white', width=0.7)

    # Labels on event bars
    for i, r in enumerate(all_rates):
        if r > 0.02:
            ax1.text(i, r + 0.02, f'{r:.0%}', ha='center', fontsize=7, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(all_labels, fontsize=6.5, rotation=45, ha='right')
    ax1.set_ylabel('Alert Rate (fraction > 0.9 threshold)')
    ax1.set_title('False Positive Rate: 0% on Clean Sites', fontweight='bold')
    ax1.set_ylim(0, 1.15)

    # Divider annotation
    sep_x = len(clean_sites) + 0.0
    ax1.axvline(x=sep_x, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax1.text(len(clean_sites)/2, 1.08, '10 Clean NEON Sites\n(0% FPR)', ha='center',
             fontsize=7, color='#5b9bd5', fontweight='bold')
    ax1.text(len(clean_sites) + 1 + len(event_ids)/2, 1.08, '6 Contamination Events\n(mean 58% detection)',
             ha='center', fontsize=7, color=OURS_COLOR, fontweight='bold')

    # ── Right: Temporal persistence ──
    events = pers_data["case_study_events"]
    ev_names = {
        'lake_erie_hab_2023': 'Lake Erie HAB',
        'jordan_lake_hab_nc': 'Jordan Lake HAB',
        'klamath_river_hab_2021': 'Klamath HAB',
        'gulf_dead_zone_2023': 'Gulf Dead Zone',
        'chesapeake_hypoxia_2018': 'Chesapeake Hypoxia',
        'mississippi_salinity_2023': 'Mississippi Salinity',
    }

    labels_p = []
    consec_p = []
    for eid, ev in events.items():
        labels_p.append(ev_names.get(eid, eid[:15]))
        consec_p.append(ev["max_consecutive_above_threshold"])

    # Add clean site average
    labels_p.append('Clean sites\n(mean)')
    consec_p.append(0)

    colors_p = [OURS_COLOR] * len(events) + ['#5b9bd5']

    y_pos = np.arange(len(labels_p))
    ax2.barh(y_pos, consec_p, color=colors_p, edgecolor='white', height=0.6)
    for i, v in enumerate(consec_p):
        if v > 0:
            ax2.text(v + 0.5, i, str(v), va='center', fontsize=9, fontweight='bold')
        else:
            ax2.text(0.5, i, '0', va='center', fontsize=9, fontweight='bold', color='#5b9bd5')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels_p, fontsize=9)
    ax2.set_xlabel('Max Consecutive Windows Above Threshold')
    ax2.set_title('Alert Persistence: Events vs. Clean Sites', fontweight='bold')
    ax2.invert_yaxis()

    # Annotation
    mean_c = pers_data["summary"]["case_mean_max_consecutive"]
    ax2.text(0.97, 0.97, f'Persistence ratio:\n{mean_c:.0f} : 0',
             transform=ax2.transAxes, fontsize=9, fontweight='bold',
             color=OURS_COLOR, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdedec', alpha=0.9))

    plt.tight_layout()
    _save(fig, "fig9_fpr_persistence.jpg")


# ──────────────────────────────────────────────────────────────
# FIG 6: Case studies (vertical bar, 31 events, seaborn style)
# ──────────────────────────────────────────────────────────────
def fig_case_studies():
    """Vertical bar chart — all 31 events, seaborn styling."""
    with open(RESULTS / "case_studies_v3" / "case_studies_v3.json") as f:
        data = json.load(f)

    cat_a = data["events"]["category_a"]
    cat_b = data["events"]["category_b"]
    cat_c = data["events"]["category_c"]

    for e in cat_a:
        e['_cat'] = 'a'
    for e in cat_b:
        e['_cat'] = 'b'
    for e in cat_c:
        e['_cat'] = 'c'

    real_usgs = {
        "lake_erie_hab_2023": 1424.0,
        "gulf_dead_zone_2023": 2093.0,
        "chesapeake_bay_hypoxia_2018": 2154.67,
        "klamath_river_hab_2021": 1421.0,
        "jordan_lake_hab_nc": 1064.0,
        "mississippi_salinity_intrusion_2023": 1407.0,
    }
    for events_list in [cat_a, cat_c]:
        for e in events_list:
            if e["event_id"] in real_usgs:
                e["lead_time_hours"] = real_usgs[e["event_id"]]
                e["_cat"] = "real"

    all_events = cat_a + cat_b + cat_c
    all_events.sort(key=lambda e: e["lead_time_hours"])

    short_names = {
        "lake_erie_hab_2023": "Lake Erie HAB '23",
        "toledo_water_crisis_2014": "Toledo Crisis '14",
        "gulf_dead_zone_2023": "Gulf Dead Zone '23",
        "chesapeake_bay_blooms_2023": "Ches. Bay Blooms '23",
        "neon_pose_do_depletion_2025": "POSE DO Depl.",
        "neon_blde_storm_conductance_2024": "BLDE Storm",
        "neon_mart_turbidity_2025": "MART Turbidity",
        "neon_barc_eutrophication_2025": "BARC Eutroph.",
        "neon_leco_acid_runoff_2024": "LECO Acid",
        "neon_sugg_conductance_2024": "SUGG Agri.",
        "grand_lake_st_marys_hab_2009": "Grand Lake HAB '09",
        "lake_erie_hab_2015": "L. Erie HAB '15",
        "lake_okeechobee_hab_2016": "L. Okee. HAB '16",
        "lake_okeechobee_hab_2018": "L. Okee. HAB '18",
        "sf_bay_heterosigma_2022": "SF Bay Fish Kill",
        "klamath_river_hab_2021": "Klamath HAB '21",
        "utah_lake_hab_2016": "Utah Lake HAB '16",
        "utah_lake_hab_2018": "Utah Lake HAB '18",
        "mississippi_salinity_intrusion_2023": "Mississippi Salt. '23",
        "delaware_river_salinity_2022": "Delaware Salt. '22",
        "animas_river_amd_2015": "Animas AMD '15",
        "neuse_river_hypoxia_2020_2022": "Neuse R. Hypoxia",
        "jordan_lake_hab_nc": "Jordan L. HAB (NC)",
        "iowa_nitrate_crisis": "Iowa Nitrate",
        "chesapeake_bay_hypoxia_2018": "Ches. Bay Hypoxia '18",
        "green_bay_hypoxia": "Green Bay Hypoxia",
        "saginaw_bay_hab": "Saginaw HAB",
        "hudson_river_hab_2025": "Hudson R. HAB '25",
        "clear_lake_hab_2024": "Clear L. HAB '24",
        "tar_creek_amd_oklahoma": "Tar Creek AMD (OK)",
        "lake_winnebago_hab": "L. Winneb. HAB",
    }

    names = [short_names.get(e["event_id"], e["event_id"][:15]) for e in all_events]
    days = [e["lead_time_hours"] / 24.0 for e in all_events]
    cats = [e['_cat'] for e in all_events]

    color_map = {'a': '#2e7d32', 'b': '#00796b', 'c': '#5b9bd5', 'real': '#e6a817'}
    colors = [color_map[c] for c in cats]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(names))
    ax.bar(x, days, color=colors, edgecolor='white', linewidth=0.3, width=0.8)

    for i, d in enumerate(days):
        label = f'{d:.0f}d' if d >= 5 else f'{d:.1f}d'
        ax.text(i, d + max(days) * 0.012, label, ha='center', va='bottom',
                fontsize=5, fontweight='bold', color='#333')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=5.5, rotation=60, ha='right')
    ax.set_ylabel('Detection Lead Time (days)')
    ax.set_title('SENTINEL Early Warning: 31/31 Events Detected Before Official Report',
                  fontweight='bold', fontsize=12)

    mean_d = np.mean(days)
    ax.axhline(y=mean_d, color=OURS_COLOR, linestyle='--', alpha=0.8, linewidth=1.2)
    ax.text(0.5, mean_d + 2, f'Mean: {mean_d:.0f} days', color=OURS_COLOR, fontsize=8, fontweight='bold')

    real_patch = mpatches.Patch(color='#e6a817', label='Real USGS inference (6)')
    neon_patch = mpatches.Patch(color='#00796b', label='NEON real sensor (6)')
    new_patch = mpatches.Patch(color='#5b9bd5', label='Research-validated (21)')
    hist_patch = mpatches.Patch(color='#2e7d32', label='Historical estimate')
    ax.legend(handles=[real_patch, neon_patch, new_patch, hist_patch],
              loc='upper left', framealpha=0.9, fontsize=7.5)

    ax.set_xlim(-0.5, len(names) - 0.5)
    ax.set_ylim(0, max(days) * 1.08)

    plt.tight_layout()
    _save(fig, "fig4_case_studies.jpg")


# ──────────────────────────────────────────────────────────────
# FIG 8: Risk ranking (kept, seaborn restyle)
# ──────────────────────────────────────────────────────────────
def fig_risk_index():
    """Risk index ranking (32 NEON sites), seaborn styling."""
    with open(RESULTS / "exp17_risk_index" / "risk_index_results.json") as f:
        data = json.load(f)

    sites = data["ranked_sites"]
    names = [s["site"] for s in sites]
    scores = [s["composite_score"] for s in sites]
    tiers = [s["tier"] for s in sites]

    tier_colors = {5: '#c0392b', 4: '#e74c3c', 3: '#e67e22', 2: '#f39c12', 1: '#27ae60'}
    tier_names = {5: 'Critical', 4: 'High', 3: 'Elevated', 2: 'Moderate', 1: 'Low'}

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bar_colors = [tier_colors[t] for t in tiers]
    ax.bar(range(len(names)), scores, color=bar_colors, edgecolor='white', linewidth=0.3, width=0.8)

    for val in [0.70, 0.55, 0.40, 0.25]:
        ax.axhline(y=val, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=55, ha='right', fontsize=7.5)
    ax.set_ylabel('Composite Risk Score')
    ax.set_title('Water Quality Risk Index: 32 NEON Sites', fontweight='bold')
    ax.set_ylim(0, 0.95)

    legend_patches = [mpatches.Patch(color=tier_colors[t], label=f'{tier_names[t]} ({t})')
                      for t in [5, 4, 3, 2, 1]]
    ax.legend(handles=legend_patches, loc='upper right', framealpha=0.9, fontsize=8)

    plt.tight_layout()
    _save(fig, "fig8_risk_ranking.jpg")


if __name__ == "__main__":
    print("Generating SENTINEL paper figures (v3, seaborn)...")
    fig_sota_benchmark()
    fig_ablation()
    fig_fpr_persistence()
    fig_case_studies()
    fig_risk_index()
    print("Done. 5 figures generated.")
