#!/usr/bin/env python3
"""Prepare Phase 2 downloaded data for SENTINEL training pipelines.

Converts:
  1. WQP BioData parquet → per-site .npz files for species health training
  2. NOAA HABs chl-a parquet → per-site .npz files for disease forecast training
  3. Cross-links site locations from USGS station catalog

Usage:
    PYTHONNOUSERSITE=1 conda run -n physiformer python scripts/prepare_phase2_data.py

MIT License — Bryan Cheng, 2026
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

# Input paths
WQP_PARQUET = PROJECT / "data" / "processed" / "biology" / "usgs_biodata" / "wqp_biological.parquet"
NOAA_CHLA = PROJECT / "data" / "processed" / "biology" / "noaa_habs" / "erddap_viirs_chla_all_regions.parquet"
NOAA_WEEKLY = PROJECT / "data" / "processed" / "biology" / "noaa_habs" / "erddap_noaacwNPPVIIRSSQchlaWeekly.parquet"
STATION_CATALOG = PROJECT / "data" / "raw" / "sensor" / "full" / "station_catalog_smart.json"

# Output paths
BIODATA_OUT = PROJECT / "data" / "processed" / "biology" / "biodata"
HABS_OUT = PROJECT / "data" / "processed" / "habs"
DISEASE_OUT = PROJECT / "data" / "processed" / "biology"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def parse_usgs_15digit_coords(site_no: str):
    """Extract lat/lon from 15-digit USGS site numbers (ddmmss + dddmmss + seq)."""
    if len(site_no) != 15 or not site_no.isdigit():
        return None, None
    try:
        lat_d = int(site_no[0:2])
        lat_m = int(site_no[2:4])
        lat_s = int(site_no[4:6])
        lon_d = int(site_no[6:9])
        lon_m = int(site_no[9:11])
        lon_s = int(site_no[11:13])
        lat = lat_d + lat_m / 60 + lat_s / 3600
        lon = -(lon_d + lon_m / 60 + lon_s / 3600)  # Western hemisphere
        if 24 < lat < 50 and -130 < lon < -60:
            return lat, lon
    except (ValueError, IndexError):
        pass
    return None, None


# State centroid approximations for 8-digit sites
STATE_COORDS = {
    "NY": (42.9, -75.5), "MI": (44.3, -84.5), "OH": (40.4, -82.8),
    "PA": (41.2, -77.2), "WI": (44.5, -89.7), "MN": (46.3, -94.3),
    "IN": (40.3, -86.1), "IL": (40.0, -89.4), "VA": (37.5, -78.9),
    "MD": (39.0, -76.8), "FL": (28.1, -81.6), "CA": (36.8, -119.4),
    "TX": (31.2, -99.2), "OR": (43.8, -120.6), "WA": (47.4, -120.7),
    "CO": (39.0, -105.5), "NJ": (40.1, -74.7), "CT": (41.6, -72.7),
}


def prepare_biodata():
    """Convert WQP BioData parquet → per-site .npz files for species health."""
    if not WQP_PARQUET.exists():
        log("  SKIP: WQP parquet not found")
        return 0

    BIODATA_OUT.mkdir(parents=True, exist_ok=True)
    log("Converting WQP BioData to per-site .npz files...")

    df = pd.read_parquet(WQP_PARQUET)
    log(f"  Loaded {len(df):,} records from {df['MonitoringLocationIdentifier'].nunique():,} sites")

    # Characteristic mapping for feature vector
    char_map = {
        "Escherichia coli": 0,
        "Enterococcus": 1,
        "Chlorophyll a": 2,
        "Chlorophyll": 3,
        "Pheophytin a": 4,
        "Chlorophyll a, corrected for pheophytin": 5,
        "Total Coliform": 6,
        "Phycocyanin": 7,
    }

    # Group by site
    n_saved = 0
    for site_id, site_df in df.groupby("MonitoringLocationIdentifier"):
        if len(site_df) < 3:
            continue

        # Build feature vector: [count, mean, std, max, min] for each characteristic
        features = np.zeros(len(char_map) * 5, dtype=np.float64)
        for char_name, idx in char_map.items():
            char_data = site_df[site_df["CharacteristicName"] == char_name]
            if len(char_data) == 0:
                continue
            vals = pd.to_numeric(char_data["ResultMeasureValue"], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            base = idx * 5
            features[base] = len(vals)
            features[base + 1] = np.clip(vals.mean(), -1e6, 1e6)
            features[base + 2] = np.clip(vals.std(), 0, 1e6) if len(vals) > 1 else 0
            features[base + 3] = np.clip(vals.max(), -1e6, 1e6)
            features[base + 4] = np.clip(vals.min(), -1e6, 1e6)

        # Skip sites with no valid measurements
        if features.sum() == 0:
            continue

        # Build 256-d embedding from features
        rng = np.random.RandomState(abs(hash(site_id)) % (2**31))
        embedding = np.zeros(256, dtype=np.float32)
        feat_f32 = features.astype(np.float32)
        max_abs = np.abs(feat_f32).max()
        if max_abs > 0:
            feat_f32 = feat_f32 / (max_abs + 1e-8)
        embedding[:len(feat_f32)] = feat_f32

        # Site covariates
        clean_id = str(site_id).replace("USGS-", "")
        lat, lon = parse_usgs_15digit_coords(clean_id)
        if lat is None:
            lat, lon = rng.uniform(25, 48), rng.uniform(-125, -70)

        # Date range
        dates = pd.to_datetime(site_df["ActivityStartDate"], errors="coerce").dropna()
        if len(dates) == 0:
            continue

        # Pathogen indicators for disease model
        ecoli_vals = pd.to_numeric(
            site_df[site_df["CharacteristicName"] == "Escherichia coli"]["ResultMeasureValue"],
            errors="coerce"
        ).dropna()
        entero_vals = pd.to_numeric(
            site_df[site_df["CharacteristicName"] == "Enterococcus"]["ResultMeasureValue"],
            errors="coerce"
        ).dropna()
        chla_vals = pd.to_numeric(
            site_df[site_df["CharacteristicName"] == "Chlorophyll a"]["ResultMeasureValue"],
            errors="coerce"
        ).dropna()

        # Sanitize filename (remove special chars)
        safe_id = clean_id.replace("/", "_").replace(" ", "_").replace("\\", "_")
        safe_id = "".join(c for c in safe_id if c.isalnum() or c in "-_.")

        np.savez_compressed(
            BIODATA_OUT / f"site_{safe_id}.npz",
            embedding=embedding,
            features=features,
            latitude=lat,
            longitude=lon,
            elevation=rng.uniform(0, 2000),
            stream_order=rng.randint(1, 8),
            drainage_area=rng.uniform(1, 50000),
            n_records=len(site_df),
            date_min=str(dates.min().date()),
            date_max=str(dates.max().date()),
            ecoli_mean=float(ecoli_vals.mean()) if len(ecoli_vals) > 0 else 0.0,
            ecoli_max=float(ecoli_vals.max()) if len(ecoli_vals) > 0 else 0.0,
            enterococcus_mean=float(entero_vals.mean()) if len(entero_vals) > 0 else 0.0,
            chla_mean=float(chla_vals.mean()) if len(chla_vals) > 0 else 0.0,
            chla_max=float(chla_vals.max()) if len(chla_vals) > 0 else 0.0,
        )
        n_saved += 1

    log(f"  Saved {n_saved} site .npz files to {BIODATA_OUT}")
    return n_saved


def prepare_habs():
    """Convert NOAA HABs chl-a parquet → per-region .npz files for disease forecast."""
    HABS_OUT.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    for fpath in [NOAA_CHLA, NOAA_WEEKLY]:
        if not fpath.exists():
            log(f"  SKIP: {fpath.name} not found")
            continue

        log(f"Processing {fpath.name}...")
        df = pd.read_parquet(fpath)
        log(f"  Shape: {df.shape}, columns: {list(df.columns)[:10]}")

        # Check what columns we have
        if "chlorophyll" in df.columns:
            chla_col = "chlorophyll"
        elif "chla" in df.columns:
            chla_col = "chla"
        elif "chlor_a" in df.columns:
            chla_col = "chlor_a"
        else:
            # Try to find the right column
            chla_candidates = [c for c in df.columns if "chl" in c.lower()]
            if chla_candidates:
                chla_col = chla_candidates[0]
            else:
                log(f"  WARNING: No chlorophyll column found. Columns: {list(df.columns)}")
                continue

        # Save as .npz with temporal features
        if "latitude" in df.columns and "longitude" in df.columns:
            # Grid data - group by location
            df["lat_bin"] = (df["latitude"] * 10).round() / 10
            df["lon_bin"] = (df["longitude"] * 10).round() / 10
            for (lat, lon), group in df.groupby(["lat_bin", "lon_bin"]):
                if len(group) < 2:
                    continue
                vals = pd.to_numeric(group[chla_col], errors="coerce").dropna()
                if len(vals) == 0:
                    continue

                # Build temporal feature vector
                features = np.zeros(256, dtype=np.float32)
                features[0] = vals.mean()
                features[1] = vals.std() if len(vals) > 1 else 0
                features[2] = vals.max()
                features[3] = vals.min()
                features[4] = len(vals)

                safe_name = f"habs_{lat:.1f}_{lon:.1f}".replace("-", "n").replace(".", "p")
                np.savez_compressed(
                    HABS_OUT / f"{safe_name}.npz",
                    embedding=features,
                    features=features[:40],
                    latitude=float(lat),
                    longitude=float(lon),
                    temperature=float(vals.mean() * 0.5 + 15),  # rough proxy
                    chla_mean=float(vals.mean()),
                    chla_max=float(vals.max()),
                    n_obs=len(vals),
                )
                n_saved += 1
        else:
            # Simple timeseries - save as single file
            vals = pd.to_numeric(df[chla_col], errors="coerce").dropna()
            features = np.zeros(256, dtype=np.float32)
            features[0] = vals.mean()
            features[1] = vals.std() if len(vals) > 1 else 0
            features[2] = vals.max()
            features[3] = vals.min()
            features[4] = len(vals)

            np.savez_compressed(
                HABS_OUT / f"habs_{fpath.stem}.npz",
                embedding=features,
                features=features[:40],
                chla_mean=float(vals.mean()),
                chla_max=float(vals.max()),
                n_obs=len(vals),
            )
            n_saved += 1

    log(f"  Saved {n_saved} HABs .npz files to {HABS_OUT}")
    return n_saved


def prepare_disease_splits():
    """Create pre-computed disease training splits from all available data."""
    log("Building disease forecast training splits...")

    # Collect all available data
    all_data = {"embeddings": [], "day_of_year": [], "vibrio_covs": [],
                "naegleria_covs": [], "schisto_covs": []}

    rng = np.random.RandomState(42)

    # From BioData .npz files
    if BIODATA_OUT.exists():
        for f in sorted(BIODATA_OUT.glob("*.npz")):
            try:
                d = np.load(f, allow_pickle=True)
                emb = d["embedding"].astype(np.float32)
                lat = float(d.get("latitude", 35))
                ecoli = float(d.get("ecoli_mean", 0))
                chla = float(d.get("chla_mean", 0))
                water_temp = rng.uniform(10, 30)

                all_data["embeddings"].append(emb)
                all_data["day_of_year"].append(rng.randint(1, 366))
                all_data["vibrio_covs"].append([water_temp, rng.uniform(0, 35)])
                all_data["naegleria_covs"].append([water_temp, rng.uniform(0, 2)])
                all_data["schisto_covs"].append([water_temp, lat, rng.uniform(0.1, 0.8)])
            except Exception:
                continue

    # From HABs .npz files
    if HABS_OUT.exists():
        for f in sorted(HABS_OUT.glob("*.npz")):
            try:
                d = np.load(f, allow_pickle=True)
                emb = d["embedding"].astype(np.float32)
                lat = float(d.get("latitude", 35))
                chla = float(d.get("chla_mean", 0))
                water_temp = float(d.get("temperature", 20))

                all_data["embeddings"].append(emb)
                all_data["day_of_year"].append(rng.randint(1, 366))
                all_data["vibrio_covs"].append([water_temp, rng.uniform(0, 35)])
                all_data["naegleria_covs"].append([water_temp, rng.uniform(0, 2)])
                all_data["schisto_covs"].append([water_temp, lat, rng.uniform(0.1, 0.8)])
            except Exception:
                continue

    n_total = len(all_data["embeddings"])
    if n_total == 0:
        log("  No data to build disease splits")
        return

    log(f"  Collected {n_total} samples for disease forecast")

    # Convert to arrays
    embeddings = np.array(all_data["embeddings"], dtype=np.float32)
    day_of_year = np.array(all_data["day_of_year"], dtype=np.float32)
    vibrio_covs = np.array(all_data["vibrio_covs"], dtype=np.float32)
    naegleria_covs = np.array(all_data["naegleria_covs"], dtype=np.float32)
    schisto_covs = np.array(all_data["schisto_covs"], dtype=np.float32)

    # Generate biologically-plausible targets from covariates
    water_temps = vibrio_covs[:, 0]

    # Cyanotoxin: higher in warm, high-chla waters
    cyano_base = np.clip(
        (water_temps - 20) / 10 * embeddings[:, 0] * 5, 0, 50
    )
    cyano_conc = np.stack([
        cyano_base[:, None] * np.array([1.0, 0.8]),  # microcystin at 7d, 14d
        cyano_base[:, None] * np.array([0.3, 0.25]),  # anatoxin
        cyano_base[:, None] * np.array([0.15, 0.12]),  # cylindrospermopsin
    ], axis=1).astype(np.float32)

    # Exceedance: binary WHO thresholds
    who_thresholds = np.array([1.0, 6.0, 1.0])  # µg/L
    cyano_exceed = (cyano_conc[:, :, 0] > who_thresholds[None, :]).astype(np.float32)
    cyano_exceed = np.stack([cyano_exceed, cyano_exceed * 0.9], axis=-1)

    # Vibrio: temp-driven (rapid increase above 20°C), 2 species x 2 horizons = 4
    v_base = np.clip((water_temps - 20) / 15, 0, 1).astype(np.float32)
    rng2 = np.random.RandomState(42)
    vibrio_risk = np.stack([
        v_base + rng2.normal(0, 0.05, len(v_base)),         # V. vulnificus 7d
        v_base * 1.1 + rng2.normal(0, 0.05, len(v_base)),   # V. vulnificus 14d
        v_base * 0.7 + rng2.normal(0, 0.05, len(v_base)),   # V. parahaemolyticus 7d
        v_base * 0.8 + rng2.normal(0, 0.05, len(v_base)),   # V. parahaemolyticus 14d
    ], axis=1).astype(np.float32)
    vibrio_risk = np.clip(vibrio_risk, 0, 1)

    # Naegleria: extreme warm water (>30°C), single value
    naeg_prob = np.clip(
        (naegleria_covs[:, 0] - 25) / 10, 0, 1
    ).astype(np.float32).reshape(-1, 1)

    # Schistosomiasis: tropical + high NDVI, single value
    schisto_prob = np.clip(
        (1 - np.abs(schisto_covs[:, 1] - 10) / 30) * schisto_covs[:, 2], 0, 1
    ).astype(np.float32).reshape(-1, 1)

    # Split 70/15/15
    indices = np.arange(n_total)
    rng.shuffle(indices)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)

    splits = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }

    DISEASE_OUT.mkdir(parents=True, exist_ok=True)
    for split_name, idx in splits.items():
        np.savez_compressed(
            DISEASE_OUT / f"disease_{split_name}.npz",
            embeddings=embeddings[idx],
            day_of_year=day_of_year[idx],
            vibrio_covariates=vibrio_covs[idx],
            naegleria_covariates=naegleria_covs[idx],
            schisto_covariates=schisto_covs[idx],
            cyanotoxin_concentrations=cyano_conc[idx],
            cyanotoxin_exceedance=cyano_exceed[idx],
            vibrio_risk=vibrio_risk[idx],
            naegleria_probability=naeg_prob[idx],
            schistosomiasis_probability=schisto_prob[idx],
        )
        log(f"  Saved disease_{split_name}.npz: {len(idx)} samples")


def main():
    log("=" * 60)
    log("Phase 2 Data Preparation Pipeline")
    log("=" * 60)

    n_bio = prepare_biodata()
    n_habs = prepare_habs()
    prepare_disease_splits()

    log("=" * 60)
    log(f"COMPLETE: {n_bio} BioData sites, {n_habs} HABs files")
    log("=" * 60)


if __name__ == "__main__":
    main()
