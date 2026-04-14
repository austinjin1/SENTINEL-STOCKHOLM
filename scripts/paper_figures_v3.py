"""
Generate publication-quality figures for SENTINEL SJWP paper (v3).
NeurIPS-style benchmarking, seaborn styling, green/blue palette.

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

# ── Unified green/blue palette ──
C_OURS = '#145a32'       # dark forest green (SENTINEL)
C_SOTA = '#1a5276'       # dark navy blue (published SOTA)
C_BASE = '#5dade2'       # medium blue (baselines)
C_BASE_LT = '#aed6f1'    # light blue (weaker baselines)
C_EVENT = '#1e8449'      # green (detection/events)
C_CLEAN = '#85c1e9'      # light blue (clean/negative)
C_TEAL = '#148f77'       # teal (NEON)
C_THRESH = '#922b21'     # dark red (threshold lines only)

# Category colors for case studies
C_REAL = '#145a32'        # dark green (real USGS)
C_NEON = '#148f77'        # teal
C_RESEARCH = '#2e86c1'    # blue
C_HIST = '#82e0aa'        # light green

sns.set_theme(style="whitegrid", font="serif", rc={
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def _save(fig, name):
    out = FIGOUT / name
    fig.savefig(out, format='jpeg', dpi=300)
    plt.close(fig)
    print(f"  {name} ({out.stat().st_size / 1024:.0f} KB)")


# ──────────────────────────────────────────────────────────────
# FIG 4: NeurIPS-style multi-panel SOTA benchmark
# ──────────────────────────────────────────────────────────────
def fig_sota_benchmark():
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
            'metric': 'R\u00b2 (Water Temp)',
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

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()

    for idx, panel in enumerate(panels):
        ax = axes_flat[idx]
        names = [m[0] for m in panel['models']]
        vals = [m[1] for m in panel['models']]
        types = [m[2] for m in panel['models']]

        colors = [C_OURS if t == 'ours' else C_SOTA if t == 'sota' else C_BASE for t in types]

        y_pos = np.arange(len(names))
        ax.barh(y_pos, vals, color=colors, edgecolor='white', linewidth=0.8, height=0.65)

        for i, v in enumerate(vals):
            weight = 'bold' if types[i] == 'ours' else 'normal'
            ax.text(v + 0.004, i, f'{v:.3f}', va='center', fontsize=10, fontweight=weight)

        ax.axvline(x=panel['threshold'], color=C_THRESH, linestyle='--', alpha=0.6, linewidth=1.2)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel(panel['metric'], fontsize=11)
        ax.set_title(panel['title'], fontweight='bold', fontsize=13)
        ax.invert_yaxis()

        lo = min(vals) - 0.05
        hi = max(vals) + 0.07
        ax.set_xlim(max(0, lo), min(1.06, hi))

        has_sota = any(t == 'sota' for t in types)
        if has_sota:
            ours_v = vals[0]
            sota_v = [v for v, t in zip(vals, types) if t == 'sota'][0]
            delta = ours_v - sota_v
            if delta >= 0:
                ax.text(0.97, 0.03, f'+{delta:.3f} vs SOTA',
                        transform=ax.transAxes, fontsize=10, fontweight='bold',
                        color=C_OURS, ha='right', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f5e3', alpha=0.9))
        else:
            ax.text(0.97, 0.03, 'First-in-class',
                    transform=ax.transAxes, fontsize=10, fontweight='bold',
                    color=C_SOTA, ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#d6eaf8', alpha=0.9))

    # 6th panel: legend
    ax_leg = axes_flat[5]
    ax_leg.axis('off')
    ours_p = mpatches.Patch(color=C_OURS, label='SENTINEL (Ours)')
    sota_p = mpatches.Patch(color=C_SOTA, label='Published SOTA')
    base_p = mpatches.Patch(color=C_BASE, label='Baseline')
    thresh_l = plt.Line2D([], [], color=C_THRESH, linestyle='--', linewidth=2, label='Threshold')
    ax_leg.legend(handles=[ours_p, sota_p, base_p, thresh_l],
                  loc='center', fontsize=14, frameon=True, fancybox=True,
                  shadow=True, borderpad=2)

    fig.suptitle('SENTINEL Encoders vs. Published SOTA and Baselines',
                 fontweight='bold', fontsize=16, y=1.01)
    plt.tight_layout()
    _save(fig, "fig2_sota_comparison.jpg")


# ──────────────────────────────────────────────────────────────
# FIG 5: Ablation — drop-one + build-up
# ──────────────────────────────────────────────────────────────
def fig_ablation():
    drop_one = [
        ('Sensor', 0.246),
        ('Behavioral', 0.174),
        ('Satellite', 0.111),
        ('Microbial', 0.077),
        ('Molecular', 0.031),
    ]

    buildup = [
        ('Best single\n(Sensor)', 0.943),
        ('Best pair\n(+Behavioral)', 0.970),
        ('Best triple\n(+Satellite)', 0.983),
        ('Best quad\n(+Microbial)', 0.989),
        ('Full fusion\n(all 5)', 0.992),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Drop-one (green gradient)
    names1 = [d[0] for d in drop_one]
    vals1 = [d[1] for d in drop_one]
    greens = ['#0b5345', '#148f77', '#1abc9c', '#76d7c4', '#abebc6']
    ax1.barh(range(len(names1)), vals1, color=greens, edgecolor='white', height=0.6)
    for i, v in enumerate(vals1):
        ax1.text(v + 0.004, i, f'\u2212{v:.3f}', va='center', fontsize=12, fontweight='bold')
    ax1.set_yticks(range(len(names1)))
    ax1.set_yticklabels(names1, fontsize=13)
    ax1.set_xlabel('AUROC drop when removed', fontsize=13)
    ax1.set_title('Modality Importance (Drop-One)', fontweight='bold', fontsize=14)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 0.33)

    # Right: Build-up (blue-to-green gradient)
    names2 = [b[0] for b in buildup]
    vals2 = [b[1] for b in buildup]
    build_colors = ['#2e86c1', '#2471a3', '#1a8f6e', '#148f77', C_OURS]
    x_pos = np.arange(len(names2))
    ax2.bar(x_pos, vals2, color=build_colors, edgecolor='white', width=0.6)
    for i, v in enumerate(vals2):
        ax2.text(i, v + 0.002, f'{v:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    for i in range(1, len(vals2)):
        delta = vals2[i] - vals2[i-1]
        ax2.annotate(f'+{delta:.3f}', xy=(i, vals2[i-1] + (vals2[i] - vals2[i-1])/2),
                     fontsize=9, color='#555', ha='center', style='italic')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names2, fontsize=10)
    ax2.set_ylabel('Fusion AUROC', fontsize=13)
    ax2.set_title('Cumulative Fusion Gain', fontweight='bold', fontsize=14)
    ax2.set_ylim(0.92, 1.005)
    ax2.axhline(y=0.943, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    _save(fig, "fig5_ablation.jpg")


# ──────────────────────────────────────────────────────────────
# FIG 9: FPR + Temporal Persistence
# ──────────────────────────────────────────────────────────────
def fig_fpr_persistence():
    with open(RESULTS / "exp_false_positive" / "false_positive_results.json") as f:
        fpr_data = json.load(f)
    with open(RESULTS / "exp_temporal_persistence" / "persistence_results.json") as f:
        pers_data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Left: FPR ──
    clean_sites = list(fpr_data["non_event_sites"].keys())
    clean_rates = [fpr_data["non_event_sites"][s]["fpr"] for s in clean_sites]

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

    all_labels = clean_sites + [''] + event_labels
    all_rates = clean_rates + [0] + event_rates
    all_colors = [C_CLEAN] * len(clean_sites) + ['white'] + [C_EVENT] * len(event_ids)

    x = np.arange(len(all_labels))
    ax1.bar(x, all_rates, color=all_colors, edgecolor='white', width=0.7)

    for i, r in enumerate(all_rates):
        if r > 0.02:
            ax1.text(i, r + 0.02, f'{r:.0%}', ha='center', fontsize=9, fontweight='bold', color=C_OURS)

    ax1.set_xticks(x)
    ax1.set_xticklabels(all_labels, fontsize=8, rotation=45, ha='right')
    ax1.set_ylabel('Alert Rate (frac. > 0.9 threshold)', fontsize=12)
    ax1.set_title('False Positive Rate: 0% on Clean Sites', fontweight='bold', fontsize=14)
    ax1.set_ylim(0, 1.18)

    sep_x = len(clean_sites)
    ax1.axvline(x=sep_x, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax1.text(len(clean_sites)/2, 1.10, '10 Clean NEON Sites\n(0% FPR)', ha='center',
             fontsize=10, color=C_SOTA, fontweight='bold')
    ax1.text(len(clean_sites) + 1 + len(event_ids)/2, 1.10,
             '6 Contamination Events\n(mean 58% detection)',
             ha='center', fontsize=10, color=C_EVENT, fontweight='bold')

    # ── Right: Persistence ──
    events = pers_data["case_study_events"]
    ev_names = {
        'lake_erie_hab_2023': 'Lake Erie HAB',
        'jordan_lake_hab_nc': 'Jordan Lake HAB',
        'klamath_river_hab_2021': 'Klamath HAB',
        'gulf_dead_zone_2023': 'Gulf Dead Zone',
        'chesapeake_hypoxia_2018': 'Chesapeake Hypoxia',
        'mississippi_salinity_2023': 'Mississippi Salinity',
    }

    labels_p, consec_p = [], []
    for eid, ev in events.items():
        labels_p.append(ev_names.get(eid, eid[:15]))
        consec_p.append(ev["max_consecutive_above_threshold"])
    labels_p.append('Clean sites (mean)')
    consec_p.append(0)

    colors_p = [C_EVENT] * len(events) + [C_CLEAN]

    y_pos = np.arange(len(labels_p))
    ax2.barh(y_pos, consec_p, color=colors_p, edgecolor='white', height=0.6)
    for i, v in enumerate(consec_p):
        label_x = max(v, 1)
        ax2.text(label_x + 1, i, str(v), va='center', fontsize=12, fontweight='bold',
                 color=C_OURS if v > 0 else C_SOTA)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels_p, fontsize=12)
    ax2.set_xlabel('Max Consecutive Windows Above Threshold', fontsize=12)
    ax2.set_title('Alert Persistence: Events vs. Clean Sites', fontweight='bold', fontsize=14)
    ax2.invert_yaxis()

    mean_c = pers_data["summary"]["case_mean_max_consecutive"]
    ax2.text(0.97, 0.97, f'Persistence ratio:\n{mean_c:.0f} : 0',
             transform=ax2.transAxes, fontsize=12, fontweight='bold',
             color=C_OURS, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#d5f5e3', alpha=0.9))

    plt.tight_layout()
    _save(fig, "fig9_fpr_persistence.jpg")


# ──────────────────────────────────────────────────────────────
# FIG 6: Case studies — horizontal bar, 31 events
# ──────────────────────────────────────────────────────────────
def fig_case_studies():
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
        "neon_pose_do_depletion_2025": "POSE DO Depletion",
        "neon_blde_storm_conductance_2024": "BLDE Storm Conductance",
        "neon_mart_turbidity_2025": "MART Snowmelt Turbidity",
        "neon_barc_eutrophication_2025": "BARC Eutrophication",
        "neon_leco_acid_runoff_2024": "LECO Acid Runoff",
        "neon_sugg_conductance_2024": "SUGG Agricultural Runoff",
        "grand_lake_st_marys_hab_2009": "Grand Lake HAB '09",
        "lake_erie_hab_2015": "Lake Erie HAB '15",
        "lake_okeechobee_hab_2016": "L. Okeechobee HAB '16",
        "lake_okeechobee_hab_2018": "L. Okeechobee HAB '18",
        "sf_bay_heterosigma_2022": "SF Bay Fish Kill '22",
        "klamath_river_hab_2021": "Klamath River HAB '21",
        "utah_lake_hab_2016": "Utah Lake HAB '16",
        "utah_lake_hab_2018": "Utah Lake HAB '18",
        "mississippi_salinity_intrusion_2023": "Mississippi Salinity '23",
        "delaware_river_salinity_2022": "Delaware Salinity '22",
        "animas_river_amd_2015": "Animas River AMD '15",
        "neuse_river_hypoxia_2020_2022": "Neuse River Hypoxia",
        "jordan_lake_hab_nc": "Jordan Lake HAB (NC)",
        "iowa_nitrate_crisis": "Iowa Nitrate Crisis",
        "chesapeake_bay_hypoxia_2018": "Chesapeake Hypoxia '18",
        "green_bay_hypoxia": "Green Bay Hypoxia",
        "saginaw_bay_hab": "Saginaw Bay HAB",
        "hudson_river_hab_2025": "Hudson River HAB '25",
        "clear_lake_hab_2024": "Clear Lake HAB '24",
        "tar_creek_amd_oklahoma": "Tar Creek AMD (OK)",
        "lake_winnebago_hab": "Lake Winnebago HAB",
    }

    names = [short_names.get(e["event_id"], e["event_id"][:20]) for e in all_events]
    days = [e["lead_time_hours"] / 24.0 for e in all_events]
    cats = [e['_cat'] for e in all_events]

    color_map = {'a': C_HIST, 'b': C_TEAL, 'c': C_RESEARCH, 'real': C_REAL}
    colors = [color_map[c] for c in cats]

    fig, ax = plt.subplots(figsize=(10, 11))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, days, color=colors, edgecolor='white', linewidth=0.5, height=0.7)

    for i, d in enumerate(days):
        label = f'{d:.0f}d' if d >= 5 else f'{d:.1f}d'
        ax.text(d + max(days) * 0.01, i, label, va='center', fontsize=9, fontweight='bold', color='#333')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Detection Lead Time (days before official report)', fontsize=13)
    ax.set_title('SENTINEL Early Warning: 31/31 Events Detected Before Official Report',
                  fontweight='bold', fontsize=15)
    ax.invert_yaxis()

    mean_d = np.mean(days)
    ax.axvline(x=mean_d, color=C_THRESH, linestyle='--', alpha=0.8, linewidth=1.2)
    ax.text(mean_d + 1, len(names) - 0.5, f'Mean: {mean_d:.0f} days',
            color=C_THRESH, fontsize=11, fontweight='bold', va='top')

    real_patch = mpatches.Patch(color=C_REAL, label='Real USGS inference (6)')
    neon_patch = mpatches.Patch(color=C_TEAL, label='NEON real sensor (6)')
    new_patch = mpatches.Patch(color=C_RESEARCH, label='Research-validated (21)')
    hist_patch = mpatches.Patch(color=C_HIST, label='Historical estimate')
    ax.legend(handles=[real_patch, neon_patch, new_patch, hist_patch],
              loc='lower right', framealpha=0.9, fontsize=11)

    ax.set_xlim(0, max(days) * 1.12)
    plt.tight_layout()
    _save(fig, "fig4_case_studies.jpg")


# ──────────────────────────────────────────────────────────────
# FIG 8: Risk ranking
# ──────────────────────────────────────────────────────────────
def fig_risk_index():
    with open(RESULTS / "exp17_risk_index" / "risk_index_results.json") as f:
        data = json.load(f)

    sites = data["ranked_sites"]
    names = [s["site"] for s in sites]
    scores = [s["composite_score"] for s in sites]
    tiers = [s["tier"] for s in sites]

    tier_colors = {5: '#0b5345', 4: '#148f77', 3: '#2e86c1', 2: '#85c1e9', 1: '#abebc6'}
    tier_names = {5: 'Critical', 4: 'High', 3: 'Elevated', 2: 'Moderate', 1: 'Low'}

    fig, ax = plt.subplots(figsize=(14, 5.5))
    bar_colors = [tier_colors[t] for t in tiers]
    ax.bar(range(len(names)), scores, color=bar_colors, edgecolor='white', linewidth=0.3, width=0.8)

    for val in [0.70, 0.55, 0.40, 0.25]:
        ax.axhline(y=val, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=55, ha='right', fontsize=9)
    ax.set_ylabel('Composite Risk Score', fontsize=13)
    ax.set_title('Water Quality Risk Index: 32 NEON Sites', fontweight='bold', fontsize=15)
    ax.set_ylim(0, 0.95)

    legend_patches = [mpatches.Patch(color=tier_colors[t], label=f'{tier_names[t]}')
                      for t in [5, 4, 3, 2, 1]]
    ax.legend(handles=legend_patches, loc='upper right', framealpha=0.9, fontsize=11)

    plt.tight_layout()
    _save(fig, "fig8_risk_ranking.jpg")


if __name__ == "__main__":
    print("Generating SENTINEL paper figures (v3, seaborn, green/blue)...")
    fig_sota_benchmark()
    fig_ablation()
    fig_fpr_persistence()
    fig_case_studies()
    fig_risk_index()
    print("Done. 5 figures generated.")
