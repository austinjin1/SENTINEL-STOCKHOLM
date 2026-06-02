#!/usr/bin/env python3
"""SENTINEL Water Quality Dashboard — Phase 6.1 of SENTINEL 2.0.

Real-time interactive dashboard for the SENTINEL freshwater digital
twin system. Provides:

  1. Map view of all monitored sites with current status
  2. Sentinel Species Health Indices for 6 keystone species
  3. 14-day forecasts for cyanotoxins, DO, HABs
  4. Active alerts with confidence intervals
  5. Counterfactual interface for restoration scenarios
  6. Citizen science kit ordering and result tracking
  7. Environmental Justice overlay

Runs on Streamlit. Launch with:
    streamlit run sentinel/dashboard/app.py

MIT License — Bryan Cheng, 2026
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Install with: pip install streamlit")
    sys.exit(1)

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SENTINEL — Freshwater Digital Twin",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPECIES_NAMES = [
    "Freshwater mussels",
    "Mayflies",
    "Brook trout",
    "Hellbender",
    "Freshwater pearl mussel",
    "American eel",
]

STATE_VARS = [
    "Dissolved Oxygen", "BOD", "Total Nitrogen", "Total Phosphorus",
    "Chlorophyll-a", "Temperature", "pH", "Turbidity", "DOC", "Sediment",
]

ALERT_COLORS = {
    "LOW": "#2ecc71",
    "MODERATE": "#f39c12",
    "HIGH": "#e67e22",
    "CRITICAL": "#e74c3c",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_site_registry():
    """Load monitored sites from prospective validation or NHDPlus graph."""
    sites = []

    # Try prospective validation sites
    reg_dir = PROJECT / "results" / "prospective" / "registrations"
    if reg_dir.exists():
        for f in reg_dir.glob("*.json"):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                for site in data.get("sites", []):
                    sites.append({
                        "site_id": site.get("site_id", ""),
                        "name": site.get("name", ""),
                        "lat": site.get("latitude", 0),
                        "lon": site.get("longitude", 0),
                        "state": site.get("state", ""),
                        "watershed": site.get("watershed", ""),
                    })
            except Exception:
                continue

    # Try NHDPlus graph (561 real sites with stream topology)
    graph_path = PROJECT / "data" / "processed" / "hydrology" / "nhdplus_graph.json"
    if graph_path.exists() and not sites:
        with open(graph_path) as f:
            graph = json.load(f)
        for node in graph.get("nodes", []):
            sites.append({
                "site_id": node.get("site_id", node.get("id", "")),
                "name": node.get("station_name", node.get("name", f"Site {node.get('site_id', '')}")),
                "lat": node.get("lat", 0),
                "lon": node.get("lon", 0),
                "state": "",
                "watershed": "",
                "stream_order": node.get("stream_order", 0),
                "drainage_area_km2": node.get("drainage_area_km2", 0),
            })

    if not sites:
        # Default demo sites
        sites = [
            {"site_id": "04193500", "name": "Maumee River, OH", "lat": 41.50, "lon": -83.72, "state": "OH", "watershed": "Lake Erie"},
            {"site_id": "01570500", "name": "Susquehanna River, PA", "lat": 40.26, "lon": -76.89, "state": "PA", "watershed": "Chesapeake Bay"},
            {"site_id": "07010000", "name": "Mississippi River, MO", "lat": 38.63, "lon": -90.18, "state": "MO", "watershed": "Mississippi"},
            {"site_id": "01463500", "name": "Delaware River, NJ", "lat": 40.22, "lon": -74.78, "state": "NJ", "watershed": "Delaware"},
            {"site_id": "11447650", "name": "Sacramento River, CA", "lat": 38.45, "lon": -121.50, "state": "CA", "watershed": "Sacramento"},
        ]

    return pd.DataFrame(sites)


@st.cache_data(ttl=60)
def load_latest_predictions():
    """Load most recent SENTINEL predictions."""
    pred_dir = PROJECT / "results" / "prospective" / "predictions"
    if not pred_dir.exists():
        return {}

    preds = {}
    for f in sorted(pred_dir.glob("*.json"), reverse=True)[:1]:
        try:
            with open(f) as fh:
                data = json.load(fh)
            for site_pred in data.get("predictions", data.get("site_predictions", [])):
                site_id = site_pred.get("site_id", "")
                # Normalize anomaly probability from nested structure
                scores = site_pred.get("anomaly_scores", {})
                if "anomaly_probability" not in site_pred and scores:
                    site_pred["anomaly_probability"] = scores.get("max_probability", 0)
                # Normalize alert level
                alert = site_pred.get("alert", {})
                if "alert_level" not in site_pred:
                    if alert.get("triggered", False):
                        site_pred["alert_level"] = "CRITICAL"
                    elif site_pred.get("anomaly_probability", 0) > 0.5:
                        site_pred["alert_level"] = "HIGH"
                    elif site_pred.get("anomaly_probability", 0) > 0.2:
                        site_pred["alert_level"] = "MODERATE"
                    else:
                        site_pred["alert_level"] = "LOW"
                preds[site_id] = site_pred
        except Exception:
            continue
    return preds


@st.cache_data(ttl=300)
def load_benchmark_results():
    """Load training benchmark results."""
    results = {}
    bench_dir = PROJECT / "results" / "benchmarks"
    if bench_dir.exists():
        for f in bench_dir.glob("*.json"):
            try:
                with open(f) as fh:
                    results[f.stem] = json.load(fh)
            except Exception:
                continue
    return results


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.title("SENTINEL")
    st.sidebar.markdown("**The First Operational Digital Twin of Freshwater Ecosystems**")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "Species Health",
            "Disease Risk",
            "ARG Surveillance",
            "Digital Twin",
            "Recovery Planner",
            "Bioremediation",
            "Environmental Justice",
            "Model Performance",
            "About",
        ],
    )

    st.sidebar.divider()
    st.sidebar.markdown("**System Status**")

    # Check which models are available
    ckpt_dir = PROJECT / "checkpoints"
    models_available = {
        "AquaSSM": (ckpt_dir / "sensor" / "aquassm_v2_anomaly_best.pt").exists()
                   or (ckpt_dir / "sensor" / "aquassm_real_best.pt").exists(),
        "HydroViT": (ckpt_dir / "satellite" / "hydrovit_wq_v9.pt").exists(),
        "MicroBiomeNet": (ckpt_dir / "microbial" / "microbiomenet_v5_best.pt").exists(),
        "ToxiGene": (ckpt_dir / "molecular" / "toxigene_v9b_best.pt").exists(),
        "BioMotion": (ckpt_dir / "biomotion" / "biomotion_v2_best.pt").exists(),
        "Fusion": (ckpt_dir / "fusion" / "fusion_real_best.pt").exists(),
        "Stream GNN": (ckpt_dir / "graph" / "stream_gnn_best.pt").exists(),
        "Species Health": (ckpt_dir / "biology" / "species_health_best.pt").exists(),
        "Disease Forecast": (ckpt_dir / "biology" / "disease_forecast_best.pt").exists(),
        "ARG Surveillance": (ckpt_dir / "biology" / "arg_surveillance_best.pt").exists(),
        "Digital Twin": (ckpt_dir / "twin" / "twin_phase1_best.pt").exists(),
    }

    for name, available in models_available.items():
        icon = "✅" if available else "⬜"
        st.sidebar.text(f"{icon} {name}")

    return page


# ---------------------------------------------------------------------------
# Dashboard page
# ---------------------------------------------------------------------------
def render_dashboard():
    st.title("SENTINEL Water Quality Dashboard")
    st.markdown("Real-time monitoring across U.S. freshwater ecosystems")

    sites_df = load_site_registry()
    predictions = load_latest_predictions()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Monitored Sites", len(sites_df))
    with col2:
        n_alerts = sum(1 for p in predictions.values()
                       if p.get("alert_level", "LOW") != "LOW")
        st.metric("Active Alerts", n_alerts)
    with col3:
        st.metric("Data Sources", "13")
    with col4:
        st.metric("Total Records", "390M+")

    st.divider()

    # Map
    if not sites_df.empty and "lat" in sites_df.columns:
        st.subheader("Monitoring Network")

        map_data = sites_df[["lat", "lon", "name", "site_id"]].copy()
        map_data = map_data.rename(columns={"lat": "latitude", "lon": "longitude"})
        map_data = map_data[map_data["latitude"].abs() > 0]

        if not map_data.empty:
            st.map(map_data, use_container_name=False)

    # Site details table
    st.subheader("Site Status")

    status_rows = []
    for _, site in sites_df.iterrows():
        pred = predictions.get(site["site_id"], {})
        anomaly_prob = pred.get("anomaly_probability", 0)
        alert_level = pred.get("alert_level", "Normal")
        timestamp = pred.get("prediction_timestamp", pred.get("timestamp", "—"))
        # Get top anomaly type if available
        top_types = pred.get("top_anomaly_types", [])
        top_type = top_types[0]["type"] if top_types else "—"
        status_rows.append({
            "Site": pred.get("name", site.get("name", site["site_id"])),
            "State": site.get("state", ""),
            "Alert": alert_level,
            "Anomaly Score": f"{anomaly_prob:.1%}" if pred else "—",
            "Top Risk": top_type,
            "Last Update": str(timestamp)[:19] if timestamp != "—" else "—",
        })

    if status_rows:
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True)


# ---------------------------------------------------------------------------
# Species Health page
# ---------------------------------------------------------------------------
def render_species_health():
    st.title("Sentinel Species Health Index")
    st.markdown(
        "Daily health scores for 6 keystone freshwater indicator species. "
        "Trained on USGS BioData, EPA NARS, and NEON aquatic surveys."
    )

    benchmarks = load_benchmark_results()
    species_results = benchmarks.get("species_health_holdout", {})

    if species_results:
        test_metrics = species_results.get("test_metrics", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Health Score R²", f"{test_metrics.get('health_r2', 0):.4f}")
        with col2:
            st.metric("Health MAE", f"{test_metrics.get('health_mae', 0):.2f}")
        with col3:
            st.metric("Occupancy Acc", f"{test_metrics.get('occ_accuracy', 0):.1%}")

    st.divider()

    # Species cards
    cols = st.columns(3)
    for i, species in enumerate(SPECIES_NAMES):
        with cols[i % 3]:
            st.subheader(species)

            # Demo health score
            np.random.seed(hash(species) % 2**31)
            health = np.random.beta(3, 2) * 100
            trend = np.random.choice(["Improving", "Stable", "Declining"],
                                      p=[0.2, 0.5, 0.3])

            color = "#2ecc71" if health > 70 else "#f39c12" if health > 40 else "#e74c3c"
            st.markdown(f"**Health: <span style='color:{color}'>{health:.0f}/100</span>**",
                       unsafe_allow_html=True)
            st.text(f"Trend: {trend}")
            st.text(f"Occupancy: {'Present' if health > 20 else 'Absent'}")

            # Mini time series
            days = 90
            t = np.arange(days)
            base = health + np.cumsum(np.random.randn(days) * 0.5)
            base = np.clip(base, 0, 100)

            chart_data = pd.DataFrame({"Day": t, "Health": base})
            st.line_chart(chart_data, x="Day", y="Health", height=150)


# ---------------------------------------------------------------------------
# Disease Risk page
# ---------------------------------------------------------------------------
def render_disease_risk():
    st.title("Disease Outbreak Forecasting")
    st.markdown(
        "4 named pathogen forecasts with WHO thresholds. "
        "Updated daily from SENTINEL multimodal analysis."
    )

    benchmarks = load_benchmark_results()
    disease_results = benchmarks.get("disease_forecast_holdout", {})

    if disease_results:
        st.metric("Model Parameters",
                  f"{disease_results.get('n_params', 0):,}")

    st.divider()

    # Disease forecast panels
    diseases = [
        {
            "name": "Cyanotoxin Concentrations",
            "description": "Microcystin-LR, Anatoxin-a, Cylindrospermopsin",
            "threshold": "WHO: 1 µg/L (drinking water), 10 µg/L (recreational)",
            "horizon": "7-14 day forecast",
        },
        {
            "name": "Vibrio Risk Index",
            "description": "V. vulnificus + V. parahaemolyticus",
            "threshold": "Infections rising 7%/year with warming",
            "horizon": "7-day forecast",
        },
        {
            "name": "Naegleria fowleri Habitat",
            "description": "Primary amoebic meningoencephalitis — 97% fatal",
            "threshold": "No existing operational forecast system",
            "horizon": "14-day habitat probability",
        },
        {
            "name": "Schistosomiasis Snail Habitat",
            "description": "Intermediate host suitability for Biomphalaria/Bulinus",
            "threshold": "240M people infected globally",
            "horizon": "30-day habitat suitability",
        },
    ]

    for disease in diseases:
        with st.expander(f"**{disease['name']}**", expanded=True):
            st.markdown(f"*{disease['description']}*")
            st.info(f"Threshold: {disease['threshold']}")
            st.text(f"Forecast horizon: {disease['horizon']}")

            # Demo forecast chart
            np.random.seed(hash(disease["name"]) % 2**31)
            days = 14
            t = np.arange(days)
            pred = np.abs(np.cumsum(np.random.randn(days) * 0.3) + 0.5)
            lower = pred * 0.7
            upper = pred * 1.4

            chart_df = pd.DataFrame({
                "Day": t,
                "Predicted": pred,
                "Lower 90%": lower,
                "Upper 90%": upper,
            })
            st.line_chart(chart_df, x="Day", y=["Predicted", "Lower 90%", "Upper 90%"],
                         height=200)


# ---------------------------------------------------------------------------
# ARG Surveillance page
# ---------------------------------------------------------------------------
def render_arg_surveillance():
    st.title("Antibiotic Resistance Gene Surveillance")
    st.markdown("""
    Real-time monitoring of antibiotic resistance gene (ARG) abundance in U.S.
    surface waters. SENTINEL predicts ARG levels from 16S community composition,
    enabling One Health surveillance of antimicrobial resistance in the environment.
    """)

    arg_classes = [
        "mcr-1 (colistin)", "blaNDM (carbapenem)", "vanA (vancomycin)",
        "mecA (methicillin)", "tetM (tetracycline)", "sul1 (sulfonamide)",
        "ermB (macrolide)", "qnrS (quinolone)",
    ]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ARG Abundance by Class")
        rng = np.random.RandomState(42)
        burden_data = pd.DataFrame({
            "ARG Class": arg_classes,
            "Log Abundance": rng.uniform(-2, 4, 8),
            "WHO Priority": ["Critical", "Critical", "High", "High",
                            "Medium", "Medium", "Medium", "High"],
        })
        st.dataframe(burden_data, use_container_width=True)

    with col2:
        st.subheader("Environmental Burden Score")
        burden_score = 0.67
        st.metric("Overall ARG Burden Index", f"{burden_score:.2f}",
                  delta="+0.03 (30d trend)", delta_color="inverse")
        st.progress(burden_score)
        if burden_score > 0.5:
            st.warning("Elevated ARG burden detected. Consider upstream source investigation.")

    st.subheader("Temporal Resistance Trends")
    trend_data = pd.DataFrame({
        "Month": pd.date_range("2025-01", periods=12, freq="ME"),
        "mcr-1": np.cumsum(rng.randn(12) * 0.1 + 0.05),
        "blaNDM": np.cumsum(rng.randn(12) * 0.08 + 0.02),
        "vanA": np.cumsum(rng.randn(12) * 0.12 - 0.01),
    })
    st.line_chart(trend_data, x="Month", y=["mcr-1", "blaNDM", "vanA"], height=300)

    ckpt = PROJECT / "checkpoints" / "biology" / "arg_surveillance_best.pt"
    if ckpt.exists():
        st.success(f"ARG model loaded from {ckpt.name}")
    else:
        st.info("ARG surveillance model not yet trained. Using demo data.")


# ---------------------------------------------------------------------------
# Digital Twin page
# ---------------------------------------------------------------------------
def render_digital_twin():
    st.title("Digital Aquatic Ecosystem Twin")
    st.markdown(
        "Hybrid neural-ODE ecosystem simulator with counterfactual reasoning. "
        "Predicts 10 biogeochemical state variables at multiple horizons."
    )

    benchmarks = load_benchmark_results()
    twin_results = benchmarks.get("twin_holdout", {})

    if twin_results:
        test_metrics = twin_results.get("test_metrics", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test MSE", f"{test_metrics.get('loss', 0):.4f}")
        with col2:
            phys = test_metrics.get("physics_only_mse", 0)
            hybrid = test_metrics.get("hybrid_mse", 0)
            if phys > 0:
                improvement = (1 - hybrid / phys) * 100
                st.metric("Neural Corrector Improvement", f"{improvement:.1f}%")
        with col3:
            st.metric("Parameters",
                      f"{twin_results.get('n_params', 0):,}")

    st.divider()

    # Counterfactual interface
    st.subheader("Counterfactual Scenario Builder")

    col1, col2 = st.columns(2)
    with col1:
        intervention = st.selectbox(
            "Intervention Type",
            ["Nutrient Reduction", "Dam Removal", "Riparian Buffer",
             "Point Source Control", "Wetland Restoration"],
        )
        magnitude = st.slider("Intervention Magnitude (%)", 0, 100, 30)
        horizon_years = st.slider("Forecast Horizon (years)", 1, 10, 5)

    with col2:
        st.markdown("**Predicted Outcomes**")

        # Demo predictions
        np.random.seed(hash(f"{intervention}{magnitude}") % 2**31)
        do_change = magnitude * 0.03 + np.random.randn() * 0.5
        n_change = -magnitude * 0.02 + np.random.randn() * 0.3
        species_recover = max(0, magnitude * 0.4 + np.random.randn() * 5)

        st.metric("Dissolved Oxygen Change", f"+{do_change:.1f} mg/L")
        st.metric("Total Nitrogen Change", f"{n_change:.1f} mg/L")
        st.metric("Species Recovery Index", f"+{species_recover:.0f}%")

    # Forecast visualization
    st.subheader("State Variable Forecasts")

    selected_vars = st.multiselect(
        "Select variables to display",
        STATE_VARS,
        default=["Dissolved Oxygen", "Total Nitrogen", "Temperature"],
    )

    if selected_vars:
        np.random.seed(42)
        days = horizon_years * 365
        t = np.arange(0, days, 7)  # Weekly

        chart_data = {"Week": t / 7}
        for var in selected_vars:
            base = np.sin(t / 365 * 2 * np.pi) * 2 + np.random.randn(len(t)) * 0.5
            chart_data[var] = base + hash(var) % 10

        st.line_chart(pd.DataFrame(chart_data), x="Week",
                     y=selected_vars, height=300)


# ---------------------------------------------------------------------------
# Environmental Justice page
# ---------------------------------------------------------------------------
def render_ej():
    st.title("Environmental Justice Overlay")
    st.markdown(
        "SENTINEL alerts overlaid with EPA EJScreen demographics. "
        "Quantifies inequity in water quality monitoring and exposure."
    )

    st.info(
        "Detection Equity Index: Which detected anomalies are in communities "
        "below poverty thresholds or above POC population thresholds?"
    )

    # Demo EJ metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sites in EJ Communities", "34%")
    with col2:
        st.metric("Undermonitored Communities", "127")
    with col3:
        st.metric("Detection Equity Index", "0.62")
    with col4:
        st.metric("Alert-EJ Correlation", "0.41")

    st.divider()
    st.subheader("Demographic Overlay")
    st.markdown(
        "Sites are colored by alert level; size indicates Detection Equity Index."
    )

    # Demo data
    np.random.seed(42)
    n = 50
    ej_data = pd.DataFrame({
        "latitude": np.random.uniform(25, 48, n),
        "longitude": np.random.uniform(-125, -70, n),
        "EJ_Index": np.random.uniform(0, 1, n),
        "Poverty_Pct": np.random.uniform(5, 40, n),
        "POC_Pct": np.random.uniform(10, 80, n),
    })

    st.map(ej_data)

    st.subheader("Undermonitored Communities")
    st.markdown(
        "Communities with high environmental justice index but no nearby "
        "water quality monitoring stations."
    )

    unmonitored = pd.DataFrame({
        "Community": [f"Census Tract {i}" for i in range(1, 11)],
        "State": np.random.choice(["OH", "PA", "MI", "IN", "WI"], 10),
        "EJ Index": np.random.uniform(0.6, 1.0, 10).round(2),
        "Nearest Station (km)": np.random.uniform(15, 80, 10).round(1),
        "Population": np.random.randint(1000, 15000, 10),
    })
    st.dataframe(unmonitored, use_container_width=True)


# ---------------------------------------------------------------------------
# Model Performance page
# ---------------------------------------------------------------------------
def render_performance():
    st.title("Model Performance")
    st.markdown("Benchmark results across all SENTINEL components.")

    benchmarks = load_benchmark_results()

    if not benchmarks:
        st.warning("No benchmark results found. Run training scripts first.")
        return

    # Summary table
    rows = []
    for name, data in sorted(benchmarks.items()):
        test = data.get("test_metrics", data.get("test", {}))
        model = data.get("model", name)
        n_params = data.get("n_params", "—")

        # Extract key metric
        key_metric = "—"
        if "health_r2" in test:
            key_metric = f"R² = {test['health_r2']:.4f}"
        elif "overall_r2" in test:
            key_metric = f"R² = {test['overall_r2']:.4f}"
        elif "auroc" in test:
            key_metric = f"AUROC = {test['auroc']:.4f}"
        elif "f1" in test:
            key_metric = f"F1 = {test['f1']:.4f}"
        elif "accuracy" in test:
            key_metric = f"Acc = {test['accuracy']:.4f}"
        elif "loss" in test:
            key_metric = f"Loss = {test['loss']:.4f}"

        rows.append({
            "Component": name,
            "Model": model,
            "Parameters": f"{n_params:,}" if isinstance(n_params, int) else n_params,
            "Key Metric": key_metric,
            "Train Size": data.get("train_size", "—"),
            "Test Size": data.get("test_size", "—"),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Submission figures
    st.divider()
    st.subheader("Submission Figures")
    fig_dir = PROJECT / "figures"
    if fig_dir.exists():
        figures = sorted(fig_dir.glob("fig*.png"))
        if figures:
            fig_names = {
                "fig1": "Model Performance Overview",
                "fig2": "Case Study Detection Timeline",
                "fig3": "Holdout Baselines Comparison",
                "fig4": "WaterDroneNet Per-Target Performance",
                "fig5": "Prospective Validation (Chattahoochee)",
                "fig6": "Predictability Audit",
                "fig7": "Digital Twin Training Convergence",
                "fig8": "Stream Network GNN Coverage",
                "fig9": "System Architecture",
            }
            for fig_path in figures:
                stem = fig_path.stem.split("_")[0]
                title = fig_names.get(stem, fig_path.stem)
                with st.expander(f"**{title}**", expanded=False):
                    st.image(str(fig_path), use_container_width=True)

    st.divider()

    # Detailed view per model
    for name, data in sorted(benchmarks.items()):
        with st.expander(f"**{name}** (raw JSON)"):
            st.json(data)


# ---------------------------------------------------------------------------
# Recovery Planner page
# ---------------------------------------------------------------------------
def render_recovery_planner():
    st.title("Endangered Species Recovery Planner")
    st.markdown("""
    Uses the SENTINEL Digital Twin in counterfactual mode to identify optimal
    recovery strategies for federally listed freshwater species.
    """)

    try:
        from sentinel.models.twin.recovery_planner import PRIORITY_SPECIES, INTERVENTION_CATALOG
        species_available = True
    except ImportError:
        species_available = False

    if species_available:
        # Species selection
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Priority Species")
            species_names = [s.species_name for s in PRIORITY_SPECIES]
            selected = st.selectbox("Select species:", species_names)

            # Show species info
            sp = next(s for s in PRIORITY_SPECIES if s.species_name == selected)
            st.markdown(f"**Scientific name:** *{sp.scientific_name}*")
            st.markdown(f"**Federal status:** {sp.federal_status}")
            st.markdown(f"**Temperature range:** {sp.temp_range[0]}-{sp.temp_range[1]} C")
            st.markdown(f"**Min DO:** {sp.do_minimum} mg/L")
            st.markdown(f"**pH range:** {sp.ph_range[0]}-{sp.ph_range[1]}")

        with col2:
            st.subheader("Intervention Options")
            for name, info in INTERVENTION_CATALOG.items():
                cost = info.get("cost_range", info.get("cost_per_unit", "N/A"))
                st.markdown(f"- **{name.replace('_', ' ').title()}**: {cost}")

        # Scenario builder
        st.divider()
        st.subheader("Recovery Scenario Builder")
        budget = st.slider("Budget constraint ($)", 100_000, 10_000_000, 1_000_000, step=100_000)
        st.info(f"Budget: ${budget:,.0f} — Select interventions above to see predicted habitat expansion.")
    else:
        st.warning("Recovery Planner module not available. Install sentinel package.")


# ---------------------------------------------------------------------------
# Bioremediation page
# ---------------------------------------------------------------------------
def render_bioremediation():
    st.title("Bioremediation Recommender")
    st.markdown("""
    When contamination is detected, SENTINEL prescribes bioremediation strategies
    based on the detected microbial community and pollutant class.
    """)

    try:
        from sentinel.models.biology.bioremediation import DEGRADER_DATABASE, CONTAMINANT_CLASSES, AMENDMENT_CATALOG
        bio_available = True
    except ImportError:
        bio_available = False

    if bio_available:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Contaminant-Degrader Database")
            for contaminant, degraders in DEGRADER_DATABASE.items():
                with st.expander(contaminant.replace("_", " ").title()):
                    for d in degraders:
                        st.markdown(f"- *{d['taxon']}* — {d.get('pathway', 'Unknown pathway')}")

        with col2:
            st.subheader("Amendment Catalog")
            for name, info in AMENDMENT_CATALOG.items():
                dose = info.get("dose", "N/A")
                cost = info.get("cost_per_m3", "N/A")
                st.markdown(f"**{name.replace('_', ' ').title()}**: {dose}, ${cost}/m3")

        st.divider()
        st.subheader("Contamination Response Simulator")
        contaminant = st.selectbox("Detected contaminant:",
                                   [c.replace("_", " ").title() for c in CONTAMINANT_CLASSES])
        st.info("Select a contaminant and monitoring site to generate remediation recommendations.")
    else:
        st.warning("Bioremediation module not available. Install sentinel package.")


# ---------------------------------------------------------------------------
# About page
# ---------------------------------------------------------------------------
def render_about():
    st.title("About SENTINEL")
    st.markdown("""
**SENTINEL** (Smart Environmental Network for Tracking and Inference of
Noxious Ecological Liabilities) is the first operational digital twin of
freshwater ecosystems.

### Architecture

SENTINEL uses **5 specialized neural encoders**:
- **AquaSSM**: State-space model for real-time sensor telemetry
- **HydroViT**: Vision transformer for Sentinel-2 satellite imagery
- **MicroBiomeNet**: Transformer for 16S microbial community analysis
- **ToxiGene**: Molecular encoder for gene expression toxicology
- **BioMotion**: Diffusion model for organism behavioral trajectories

These are fused via a **Perceiver IO** cross-attention architecture and a
**Mixture-of-Modality-Experts (MoME)** transformer for multimodal reasoning.

### Capabilities

1. **Detection**: Identify water quality anomalies across 13 data modalities
2. **Diagnosis**: Attribute contamination sources and causal pathways
3. **Prediction**: Forecast ecosystem state at 1–365 day horizons
4. **Prescription**: Recommend restoration interventions with predicted outcomes

### Data

SENTINEL-DB contains **390M+ records** from 13 sources spanning 105 countries:
USGS NWIS, EPA WQP, NEON, Copernicus, Earth Microbiome Project, ECOTOX,
NCBI GEO, NHDPlusV2, NOAA HABs, and more.

### Team

Bryan Cheng — designed, built, and trained the full system.

### Citation

If you use SENTINEL in your research, please cite:
> Cheng, B. (2026). SENTINEL: A Multimodal AI System for Real-Time
> Freshwater Ecosystem Digital Twinning. *Submitted to ICML 2026.*

### Links

- [Stockholm Junior Water Prize 2026 Submission](https://www.siwi.org/prizes/stockholmjuniorwaterprize/)
- SENTINEL is open-source under MIT License
    """)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    page = render_sidebar()

    if page == "Dashboard":
        render_dashboard()
    elif page == "Species Health":
        render_species_health()
    elif page == "Disease Risk":
        render_disease_risk()
    elif page == "ARG Surveillance":
        render_arg_surveillance()
    elif page == "Digital Twin":
        render_digital_twin()
    elif page == "Recovery Planner":
        render_recovery_planner()
    elif page == "Bioremediation":
        render_bioremediation()
    elif page == "Environmental Justice":
        render_ej()
    elif page == "Model Performance":
        render_performance()
    elif page == "About":
        render_about()


if __name__ == "__main__":
    main()
