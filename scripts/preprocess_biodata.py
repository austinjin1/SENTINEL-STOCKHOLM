#!/usr/bin/env python3
"""Preprocess USGS BioData parquets into per-site NPZ files for species health training.

Reads BioData invertebrate, fish, and WQP biological parquets, joins with site
locations, computes occupancy/health scores for 6 keystone indicator species,
and saves one NPZ file per site to data/processed/biology/biodata/.

The 6 keystone species:
  0. Freshwater mussels (Unionidae / order Unionida)
  1. Mayflies (Ephemeroptera)
  2. Brook trout (Salvelinus fontinalis)
  3. Hellbender (Cryptobranchus alleganiensis)
  4. Freshwater pearl mussel (Margaritifera margaritifera)
  5. American eel (Anguilla rostrata)

MIT License — Bryan Cheng, 2026
"""

import os
import sys
import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

BIODATA_DIR = PROJECT / "data" / "processed" / "biology" / "usgs_biodata"
OUTPUT_DIR = PROJECT / "data" / "processed" / "biology" / "biodata"
SENSOR_DIR = PROJECT / "data" / "processed" / "sensor" / "full"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Taxonomic matching patterns for each keystone species
# ---------------------------------------------------------------------------
# Species 0: Freshwater mussels — order Unionida, family Unionidae + Margaritiferidae
MUSSEL_PATTERNS = [
    "Unionidae", "Unionoida", "Unionida",
    "Anodonta", "Lampsilis", "Elliptio", "Amblema", "Quadrula",
    "Fusconaia", "Villosa", "Pleurobema", "Ligumia", "Toxolasma",
    "Actinonaias", "Ptychobranchus", "Epioblasma", "Obovaria",
    "Strophitus", "Alasmidonta", "Lasmigona",
]

# Species 1: Mayflies — order Ephemeroptera
EPHEMEROPTERA_PATTERNS = [
    "Ephemeroptera", "Ephemerellidae", "Ephemerella", "Ephemeridae", "Ephemera",
    "Baetidae", "Baetis", "Heptageniidae", "Maccaffertium", "Stenonema",
    "Leptophlebiidae", "Leptophlebia", "Paraleptophlebia",
    "Caenidae", "Caenis", "Tricorythidae", "Tricorythodes",
    "Isonychiidae", "Isonychia", "Siphlonuridae",
    "Neoephemera", "Rhithrogena", "Epeorus", "Cinygmula",
    "Ameletus", "Drunella", "Serratella", "Timpanoga",
    "Hexagenia",
]

# Species 2: Brook trout
BROOK_TROUT_PATTERNS = [
    "Salvelinus fontinalis", "Salvelinus",
    # Close salmonid relatives as proxies
    "Oncorhynchus", "Salmo", "Thymallus",
]

# Species 3: Hellbender
HELLBENDER_PATTERNS = [
    "Cryptobranchus alleganiensis", "Cryptobranchus",
    # Close amphibian relatives as proxies
    "Necturus", "Eurycea", "Desmognathus", "Gyrinophilus",
]

# Species 4: Freshwater pearl mussel
PEARL_MUSSEL_PATTERNS = [
    "Margaritifera margaritifera", "Margaritifera",
    "Margaritiferidae",
]

# Species 5: American eel
AMERICAN_EEL_PATTERNS = [
    "Anguilla rostrata", "Anguilla",
    # Close proxy: other anadromous/catadromous fish
    "Petromyzon", "Lampetra", "Ichthyomyzon",
]

ALL_PATTERNS = [
    MUSSEL_PATTERNS,
    EPHEMEROPTERA_PATTERNS,
    BROOK_TROUT_PATTERNS,
    HELLBENDER_PATTERNS,
    PEARL_MUSSEL_PATTERNS,
    AMERICAN_EEL_PATTERNS,
]

SPECIES_NAMES = [
    "Freshwater mussels (Unionidae)",
    "Mayflies (Ephemeroptera)",
    "Brook trout (Salvelinus fontinalis)",
    "Hellbender (Cryptobranchus alleganiensis)",
    "Freshwater pearl mussel (Margaritifera margaritifera)",
    "American eel (Anguilla rostrata)",
]

NUM_SPECIES = 6

# Reference condition thresholds for health score computation.
# These represent "healthy benchmark" counts from literature/expert knowledge.
# Health = min(100, 100 * observed_count / reference_count)
REFERENCE_COUNTS = [
    10.0,   # Mussels: 10 individuals is a healthy assemblage
    50.0,   # Mayflies: 50 individuals = good EPT density
    20.0,   # Brook trout: 20 individuals = healthy pop
    3.0,    # Hellbender: 3 individuals = healthy (rare species)
    5.0,    # Pearl mussel: 5 individuals (very rare)
    10.0,   # American eel: 10 individuals
]


def build_taxon_regex(patterns: list) -> str:
    """Build a regex pattern that matches any of the given taxonomic names."""
    escaped = [p.replace("(", r"\(").replace(")", r"\)") for p in patterns]
    return "|".join(escaped)


def load_biodata() -> pd.DataFrame:
    """Load and concatenate invertebrate, fish, and WQP biological data."""
    frames = []

    # Invertebrates
    inv_path = BIODATA_DIR / "biodata_invertebrates.parquet"
    if inv_path.exists():
        log(f"Loading invertebrates: {inv_path}")
        inv = pd.read_parquet(inv_path, columns=[
            "MonitoringLocationIdentifier", "SubjectTaxonomicName",
            "ResultMeasureValue", "CharacteristicName",
        ])
        inv["_dataset"] = "invertebrate"
        frames.append(inv)
        log(f"  {len(inv):,} records")

    # Fish
    fish_path = BIODATA_DIR / "biodata_fish.parquet"
    if fish_path.exists():
        log(f"Loading fish: {fish_path}")
        fish = pd.read_parquet(fish_path, columns=[
            "MonitoringLocationIdentifier", "SubjectTaxonomicName",
            "ResultMeasureValue", "CharacteristicName",
        ])
        fish["_dataset"] = "fish"
        frames.append(fish)
        log(f"  {len(fish):,} records")

    # WQP biological
    wqp_path = BIODATA_DIR / "wqp_biological.parquet"
    if wqp_path.exists():
        log(f"Loading WQP biological: {wqp_path}")
        wqp = pd.read_parquet(wqp_path, columns=[
            "MonitoringLocationIdentifier", "SubjectTaxonomicName",
            "ResultMeasureValue", "CharacteristicName",
        ])
        wqp["_dataset"] = "wqp"
        frames.append(wqp)
        log(f"  {len(wqp):,} records")

    if not frames:
        raise RuntimeError("No BioData parquets found!")

    bio = pd.concat(frames, ignore_index=True)
    log(f"Combined biological records: {len(bio):,}")
    return bio


def load_sites() -> pd.DataFrame:
    """Load site location data with lat/lon and optional covariates."""
    sites_path = BIODATA_DIR / "biodata_sites.parquet"
    nwis_path = BIODATA_DIR / "nwis_bio_sites.parquet"

    frames = []

    if sites_path.exists():
        log(f"Loading sites: {sites_path}")
        sites = pd.read_parquet(sites_path, columns=[
            "MonitoringLocationIdentifier", "LatitudeMeasure", "LongitudeMeasure",
            "DrainageAreaMeasure/MeasureValue", "VerticalMeasure/MeasureValue",
        ])
        sites = sites.rename(columns={
            "LatitudeMeasure": "latitude",
            "LongitudeMeasure": "longitude",
            "DrainageAreaMeasure/MeasureValue": "drainage_area",
            "VerticalMeasure/MeasureValue": "elevation",
        })
        frames.append(sites)
        log(f"  {len(sites):,} sites")

    if nwis_path.exists():
        log(f"Loading NWIS sites: {nwis_path}")
        nwis = pd.read_parquet(nwis_path, columns=[
            "site_no", "dec_lat_va", "dec_long_va", "alt_va",
        ])
        # Create a monitoring location ID that might match
        nwis["MonitoringLocationIdentifier"] = "USGS-" + nwis["site_no"].astype(str)
        nwis = nwis.rename(columns={
            "dec_lat_va": "latitude",
            "dec_long_va": "longitude",
            "alt_va": "elevation",
        })
        nwis["drainage_area"] = np.nan
        nwis = nwis[["MonitoringLocationIdentifier", "latitude", "longitude",
                      "drainage_area", "elevation"]]
        frames.append(nwis)
        log(f"  {len(nwis):,} NWIS sites")

    if not frames:
        raise RuntimeError("No site files found!")

    all_sites = pd.concat(frames, ignore_index=True)
    all_sites = all_sites.drop_duplicates(subset=["MonitoringLocationIdentifier"], keep="first")
    log(f"Total unique sites with coordinates: {len(all_sites):,}")
    return all_sites


def compute_species_presence(bio_df: pd.DataFrame):
    """For each site, compute species-level counts for the 6 keystone species.

    Returns a dict: {site_id -> (counts[6], total_taxa_count)}
    """
    log("Computing per-site species counts...")

    # Pre-compile regex for each species group
    import re
    species_re = [re.compile(build_taxon_regex(p), re.IGNORECASE) for p in ALL_PATTERNS]

    # Parse counts
    bio_df["count"] = pd.to_numeric(bio_df["ResultMeasureValue"], errors="coerce").fillna(1.0)

    # Group by site
    grouped = bio_df.groupby("MonitoringLocationIdentifier")

    site_data = {}
    n_sites = 0

    for site_id, group in grouped:
        taxa = group["SubjectTaxonomicName"].dropna()
        counts_col = group["count"]

        species_counts = np.zeros(NUM_SPECIES, dtype=np.float64)

        for idx, row_taxa, row_count in zip(range(len(taxa)), taxa.values, counts_col.values):
            if not isinstance(row_taxa, str):
                continue
            for sp_idx, sp_re in enumerate(species_re):
                if sp_re.search(row_taxa):
                    species_counts[sp_idx] += float(row_count) if not np.isnan(row_count) else 1.0

        total_taxa = taxa.nunique()
        site_data[site_id] = (species_counts, total_taxa)
        n_sites += 1

    log(f"  Processed {n_sites:,} sites")

    # Report presence stats
    for sp_idx in range(NUM_SPECIES):
        n_present = sum(1 for _, (c, _) in site_data.items() if c[sp_idx] > 0)
        log(f"  {SPECIES_NAMES[sp_idx]}: present at {n_present:,} sites")

    return site_data


def compute_health_scores(species_counts: np.ndarray, total_taxa: int) -> np.ndarray:
    """Compute health scores (0-100) from species counts relative to reference conditions.

    Health score combines:
      - Abundance relative to reference (70% weight)
      - Total taxonomic richness bonus (30% weight)
    """
    # Abundance-based health
    abundance_health = np.zeros(NUM_SPECIES, dtype=np.float32)
    for i in range(NUM_SPECIES):
        if species_counts[i] > 0:
            ratio = species_counts[i] / REFERENCE_COUNTS[i]
            abundance_health[i] = min(100.0, ratio * 100.0)

    # Taxonomic richness bonus: more diverse community = healthier
    # Cap at 100 taxa for normalization
    richness_bonus = min(total_taxa / 100.0, 1.0) * 30.0

    # Combine: species present get abundance + richness bonus, absent get 0
    health = np.zeros(NUM_SPECIES, dtype=np.float32)
    for i in range(NUM_SPECIES):
        if species_counts[i] > 0:
            health[i] = np.clip(abundance_health[i] * 0.7 + richness_bonus, 0, 100)
        else:
            # Even absent species get a small health score if the overall
            # community is diverse (proxy for habitat quality)
            health[i] = np.clip(richness_bonus * 0.3, 0, 30)

    return health


def load_sensor_embeddings() -> dict:
    """Load sensor NPZ files and compute per-site mean embeddings (256-d).

    Returns {site_id_str: embedding_256d}
    """
    if not SENSOR_DIR.exists():
        log("No sensor directory found, will use random embeddings.")
        return {}

    sensor_files = sorted(SENSOR_DIR.glob("*.npz"))
    if not sensor_files:
        log("No sensor NPZ files found.")
        return {}

    log(f"Loading sensor data from {len(sensor_files):,} files...")

    # Group by site ID (prefix before _seq)
    from collections import defaultdict
    site_sequences = defaultdict(list)
    for f in sensor_files:
        stem = f.stem
        parts = stem.split("_seq")
        site_id = parts[0]
        site_sequences[site_id].append(f)

    log(f"  {len(site_sequences):,} unique sensor sites")

    site_embeddings = {}
    for site_id, files in site_sequences.items():
        # Use first sequence file to create embedding
        try:
            d = np.load(files[0], allow_pickle=True)
            values = d.get("values", d.get("features", d.get("data", None)))
            if values is None:
                continue
            # values shape: (T, D), e.g. (512, 6)
            # Compute statistics along time axis as features
            if values.ndim == 2:
                means = np.nanmean(values, axis=0)
                stds = np.nanstd(values, axis=0)
                mins = np.nanmin(values, axis=0)
                maxs = np.nanmax(values, axis=0)
                # Quantiles
                q25 = np.nanquantile(values, 0.25, axis=0)
                q75 = np.nanquantile(values, 0.75, axis=0)
                # Autocorrelation at lag 1
                if values.shape[0] > 1:
                    autocorr = np.array([
                        np.corrcoef(values[:-1, c], values[1:, c])[0, 1]
                        if np.std(values[:, c]) > 1e-8 else 0.0
                        for c in range(values.shape[1])
                    ])
                else:
                    autocorr = np.zeros(values.shape[1])
                autocorr = np.nan_to_num(autocorr)

                feats = np.concatenate([means, stds, mins, maxs, q25, q75, autocorr])
            else:
                feats = values.flatten()

            # Pad or truncate to 256
            if len(feats) < 256:
                feats = np.pad(feats, (0, 256 - len(feats)), mode="constant")
            else:
                feats = feats[:256]

            feats = feats.astype(np.float32)
            # Normalize
            norm = np.linalg.norm(feats) + 1e-8
            feats = feats / norm

            site_embeddings[site_id] = feats

        except Exception:
            continue

    log(f"  Computed embeddings for {len(site_embeddings):,} sensor sites")
    return site_embeddings


def make_pseudo_embedding(site_id: str, lat: float, lon: float,
                          species_counts: np.ndarray, total_taxa: int,
                          seed_val: int) -> np.ndarray:
    """Create a deterministic 256-d pseudo-embedding from site characteristics."""
    rng = np.random.RandomState(seed_val)

    # Base: deterministic random from site hash
    emb = rng.randn(256).astype(np.float32)

    # Encode geographic info into first dimensions
    emb[0] = (lat - 37.0) / 10.0       # Normalize lat ~[25, 49] -> ~[-1.2, 1.2]
    emb[1] = (lon + 97.0) / 30.0       # Normalize lon ~[-125, -70] -> ~[-0.9, 0.9]

    # Encode species richness/counts
    emb[2] = np.log1p(total_taxa) / 5.0
    for i in range(NUM_SPECIES):
        emb[3 + i] = np.log1p(species_counts[i]) / 5.0

    # Normalize
    norm = np.linalg.norm(emb) + 1e-8
    emb = emb / norm

    return emb


def main():
    log("=" * 60)
    log("Preprocessing USGS BioData for Species Health Training")
    log("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PROJECT / "logs", exist_ok=True)

    # Load biological data
    bio_df = load_biodata()

    # Load sites
    sites_df = load_sites()

    # Compute per-site species counts
    site_species = compute_species_presence(bio_df)

    # Free memory
    del bio_df

    # Load sensor embeddings
    sensor_embs = load_sensor_embeddings()

    # Join sites with species data
    log("Generating per-site NPZ files...")

    # Build site lookup
    site_locs = {}
    for _, row in sites_df.iterrows():
        sid = row["MonitoringLocationIdentifier"]
        lat = row.get("latitude", np.nan)
        lon = row.get("longitude", np.nan)
        elev = row.get("elevation", np.nan)
        darea = row.get("drainage_area", np.nan)
        if pd.notna(lat) and pd.notna(lon):
            site_locs[sid] = (float(lat), float(lon),
                              float(elev) if pd.notna(elev) else 0.0,
                              float(darea) if pd.notna(darea) else 0.0)

    # Process all sites that have BOTH species data AND location
    valid_sites = set(site_species.keys()) & set(site_locs.keys())
    log(f"Sites with both biology and location data: {len(valid_sites):,}")

    # Also include sites that have location but no biology data — they
    # represent "absence" observations which are important for occupancy modeling
    absent_sites = set(site_locs.keys()) - set(site_species.keys())
    # Sample a subset of absent sites to avoid massive class imbalance
    rng = np.random.RandomState(42)
    max_absent = min(len(absent_sites), len(valid_sites) // 2)
    if max_absent > 0:
        absent_list = sorted(list(absent_sites))
        rng.shuffle(absent_list)
        absent_sampled = set(absent_list[:max_absent])
        log(f"Adding {len(absent_sampled):,} absence-only sites (sampled from {len(absent_sites):,})")
    else:
        absent_sampled = set()

    all_process_sites = valid_sites | absent_sampled

    n_saved = 0
    n_with_sensor = 0
    n_with_occupancy = 0

    for site_id in sorted(all_process_sites):
        lat, lon, elev, darea = site_locs[site_id]

        # Species data
        if site_id in site_species:
            species_counts, total_taxa = site_species[site_id]
        else:
            species_counts = np.zeros(NUM_SPECIES, dtype=np.float64)
            total_taxa = 0

        # Occupancy: binary presence
        occupancy = (species_counts > 0).astype(np.float32)
        if occupancy.sum() > 0:
            n_with_occupancy += 1

        # Health scores
        health_scores = compute_health_scores(species_counts, total_taxa)

        # Site covariates: [lat, lon, elevation, drainage_area, stream_order]
        # Stream order not directly available; estimate from drainage area
        if darea > 0:
            # Approximate Horton-Strahler order from drainage area (km²)
            # Very rough: order ~ log2(area_km2) capped at 1-9
            stream_order = max(1, min(9, int(np.log2(max(darea, 1)) / 1.5) + 1))
        else:
            stream_order = 3  # Default mid-range

        site_covs = np.array([lat, lon, elev, darea, stream_order], dtype=np.float32)

        # Embedding: prefer sensor data, else use pseudo-embedding
        # Try to match site ID to sensor site ID
        # MonitoringLocationIdentifier format: "USGS-XXXXXXXX" -> sensor file: "XXXXXXXX"
        sensor_site_id = None
        if site_id.startswith("USGS-"):
            sensor_site_id = site_id.replace("USGS-", "")

        if sensor_site_id and sensor_site_id in sensor_embs:
            features = sensor_embs[sensor_site_id]
            n_with_sensor += 1
        else:
            # Deterministic pseudo-embedding from site hash
            seed_val = int(hashlib.sha256(site_id.encode()).hexdigest()[:8], 16) % (2**31)
            features = make_pseudo_embedding(site_id, lat, lon,
                                             species_counts, total_taxa, seed_val)

        # Sanitize site_id for filename
        safe_id = site_id.replace("/", "_").replace("\\", "_").replace(" ", "_")

        # Save NPZ
        out_path = OUTPUT_DIR / f"{safe_id}.npz"
        np.savez_compressed(
            out_path,
            features=features,                     # (256,) float32
            health_scores=health_scores,            # (6,) float32
            occupancy=occupancy,                    # (6,) float32
            site_covs=site_covs,                    # (5,) float32
            latitude=np.float64(lat),
            longitude=np.float64(lon),
            elevation=np.float64(elev),
            drainage_area=np.float64(darea),
            stream_order=np.int64(stream_order),
            species_counts=species_counts,          # (6,) for debugging
        )
        n_saved += 1

    log("=" * 60)
    log(f"Preprocessing complete!")
    log(f"  Total NPZ files saved: {n_saved:,}")
    log(f"  Sites with sensor embeddings: {n_with_sensor:,}")
    log(f"  Sites with species occupancy: {n_with_occupancy:,}")
    log(f"  Output directory: {OUTPUT_DIR}")
    log("=" * 60)

    # Verify a sample
    sample_files = sorted(OUTPUT_DIR.glob("*.npz"))[:3]
    if sample_files:
        log("Sample verification:")
        for sf in sample_files:
            d = np.load(sf, allow_pickle=True)
            log(f"  {sf.name}:")
            log(f"    features: {d['features'].shape}, dtype={d['features'].dtype}")
            log(f"    health_scores: {d['health_scores']}")
            log(f"    occupancy: {d['occupancy']}")
            log(f"    site_covs: {d['site_covs']}")


if __name__ == "__main__":
    main()
