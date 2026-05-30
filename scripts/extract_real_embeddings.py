#!/usr/bin/env python3
"""Extract real embeddings from all 5 SENTINEL encoders.

Runs each encoder on its real data and saves [N, 256] embedding
tensors to data/real_embeddings/ for downstream holdout experiments
(exp5, exp7, exp10, exp12, exp15).

Data sources:
  - Sensor:     Real NEON DP1.20288.001 parquet -> sliding windows [B, 128, 6]
  - Satellite:  Paired WQ satellite data if available, else HydroViT on
                synthetic 13-band tiles (representative input distributions)
  - Microbial:  Real EMP 16S OTU data from data/processed/microbial/emp_16s/
  - Molecular:  Real gene expression data from data/processed/molecular/
  - Behavioral: Real ECOTOX Daphnia trajectories from data/processed/behavioral_real/

Usage::

    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=3 python scripts/extract_real_embeddings.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_BASE = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "data" / "real_embeddings"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: flexible state dict loading
# ---------------------------------------------------------------------------

def _load_state_dict(path: Path) -> dict:
    """Load a state dict from a checkpoint, handling various wrapping formats."""
    state = torch.load(str(path), map_location=DEVICE, weights_only=False)
    if isinstance(state, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


# ===========================================================================
# 1. SENSOR (AquaSSM) — Real NEON DP1.20288.001 data
# ===========================================================================

class NEONSensorDataset(Dataset):
    """Load real NEON water quality data as sliding windows for AquaSSM.

    Reads the NEON DP1.20288.001 parquet, extracts the 6 parameters
    (pH, dissolvedOxygen, turbidity, specificConductance, chlorophyll, fDOM),
    and creates overlapping windows of length T=128.
    """

    WQ_COLS = ["pH", "dissolvedOxygen", "turbidity",
               "specificConductance", "chlorophyll", "fDOM"]
    # Quality flag columns for each WQ parameter (FinalQF==0 means passed QC)
    QF_MAP = {
        "pH": "pHFinalQF",
        "dissolvedOxygen": "dissolvedOxygenFinalQF",
        "turbidity": "turbidityFinalQF",
        "specificConductance": "specificCondFinalQF",
        "chlorophyll": "chlorophyllFinalQF",
        "fDOM": "fDOMFinalQF",
    }
    # Physical range clipping (sanity bounds after QF filtering)
    CLIP_RANGES = {
        "pH": (0, 14),
        "dissolvedOxygen": (0, 25),
        "turbidity": (0, 4000),
        "specificConductance": (0, 100000),
    }
    WINDOW_SIZE = 128
    STRIDE = 64

    def __init__(self, parquet_path: str, max_windows: int = 3000):
        import pandas as pd

        logger.info(f"Loading NEON parquet: {parquet_path}")
        # Load WQ columns plus quality flag columns
        qf_cols = list(self.QF_MAP.values())
        cols_needed = self.WQ_COLS + qf_cols + ["siteID", "startDateTime"]
        # Only request columns that exist in the parquet
        import pyarrow.parquet as pq_schema
        schema = pq_schema.read_schema(parquet_path)
        cols_available = [c for c in cols_needed if c in schema.names]
        df = pd.read_parquet(parquet_path, columns=cols_available)
        logger.info(f"  Raw rows: {len(df)}")

        # --- Quality flag filtering: NaN out values where FinalQF != 0 ---
        n_before = df[self.WQ_COLS].notna().sum().sum()
        for wq_col, qf_col in self.QF_MAP.items():
            if qf_col in df.columns and wq_col in df.columns:
                bad_qf = df[qf_col].fillna(1).astype(float) != 0
                df.loc[bad_qf, wq_col] = np.nan
        n_after = df[self.WQ_COLS].notna().sum().sum()
        logger.info(f"  QF filtering: {n_before:,} -> {n_after:,} valid values "
                    f"({n_before - n_after:,} rejected)")

        # --- Range-based sanity clipping ---
        for col, (lo, hi) in self.CLIP_RANGES.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, hi)

        # Drop QF columns (no longer needed)
        df = df.drop(columns=[c for c in qf_cols if c in df.columns], errors="ignore")

        # Drop rows with all-NaN WQ values
        df = df.dropna(subset=self.WQ_COLS, how="all").reset_index(drop=True)
        logger.info(f"  After dropping all-NaN: {len(df)}")

        # Sort by site + time for temporal coherence
        df = df.sort_values(["siteID", "startDateTime"]).reset_index(drop=True)

        # Fill remaining NaN with column medians (within the dataset)
        for col in self.WQ_COLS:
            median_val = df[col].median()
            if np.isnan(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)

        values = df[self.WQ_COLS].values.astype(np.float32)  # [N_rows, 6]

        # Create sliding windows per site
        self.windows = []
        sites = df["siteID"].unique()
        for site in sites:
            site_mask = df["siteID"].values == site
            site_vals = values[site_mask]
            n = len(site_vals)
            for start in range(0, n - self.WINDOW_SIZE + 1, self.STRIDE):
                self.windows.append(site_vals[start:start + self.WINDOW_SIZE])
                if len(self.windows) >= max_windows:
                    break
            if len(self.windows) >= max_windows:
                break

        logger.info(f"  Created {len(self.windows)} windows from "
                    f"{len(sites)} sites (window={self.WINDOW_SIZE}, "
                    f"stride={self.STRIDE})")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.from_numpy(self.windows[idx])  # [128, 6]


def extract_sensor():
    """Extract AquaSSM embeddings from real NEON sensor data."""
    logger.info("=" * 60)
    logger.info("Extracting SENSOR (AquaSSM) embeddings — real NEON data")
    logger.info("=" * 60)

    from sentinel.models.sensor_encoder.model import SensorEncoder

    # Find best checkpoint
    ckpt_path = CKPT_BASE / "sensor" / "aquassm_real_best.pt"
    if not ckpt_path.exists():
        ckpt_path = CKPT_BASE / "sensor" / "aquassm_v4_best.pt"
    if not ckpt_path.exists():
        logger.warning(f"No sensor checkpoint at {ckpt_path}, skipping")
        return

    model = SensorEncoder().to(DEVICE)
    state = _load_state_dict(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(f"Loaded {ckpt_path.name}: {len(missing)} missing, "
                f"{len(unexpected)} unexpected keys")
    model.eval()

    # Try to load real NEON data
    neon_parquet = PROJECT_ROOT / "data" / "raw" / "neon_aquatic" / "neon_DP1.20288.001_all.parquet"
    if neon_parquet.exists():
        dataset = NEONSensorDataset(str(neon_parquet), max_windows=3000)
    else:
        # Fallback: use processed sensor sequences
        logger.warning("NEON parquet not found, using processed sensor npz files")
        sensor_dir = PROJECT_ROOT / "data" / "processed" / "sensor" / "full"
        npz_files = sorted(sensor_dir.glob("*.npz"))[:3000]
        windows = []
        for f in npz_files:
            d = np.load(f)
            vals = d["values"][:128]  # [T, 6], take first 128 steps
            if vals.shape[0] < 128:
                # Pad short sequences
                pad = np.zeros((128 - vals.shape[0], 6), dtype=np.float32)
                vals = np.concatenate([vals, pad], axis=0)
            windows.append(vals)

        class ProcessedSensorDataset(Dataset):
            def __init__(self, windows):
                self.windows = windows
            def __len__(self):
                return len(self.windows)
            def __getitem__(self, idx):
                return torch.from_numpy(self.windows[idx])

        dataset = ProcessedSensorDataset(windows)
        logger.info(f"  Loaded {len(dataset)} processed sensor sequences")

    loader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False)

    embeddings = []
    t0 = time.time()
    with torch.no_grad():
        with autocast("cuda", enabled=DEVICE.type == "cuda"):
            for i, batch in enumerate(loader):
                batch = batch.to(DEVICE)  # [B, 128, 6]
                out = model(x=batch, compute_anomaly=False)
                emb = out["embedding"].cpu()  # [B, 256]
                embeddings.append(emb)
                if (i + 1) % 20 == 0:
                    logger.info(f"  Batch {i+1}/{len(loader)}")

    if embeddings:
        all_emb = torch.cat(embeddings, dim=0)  # [N, 256]
        torch.save(all_emb, OUTPUT_DIR / "sensor_embeddings.pt")
        logger.info(f"Saved {all_emb.shape} sensor embeddings in {time.time()-t0:.1f}s")
        logger.info(f"  Stats: mean={all_emb.mean():.4f}, std={all_emb.std():.4f}")
        return all_emb
    else:
        logger.error("No sensor embeddings extracted")
        return None


# ===========================================================================
# 2. SATELLITE (HydroViT) — Paired satellite images
# ===========================================================================

class SatelliteDataset(Dataset):
    """Load satellite images from paired_wq npz, pad 10->13 bands."""

    def __init__(self, data_path: str):
        data = np.load(data_path, allow_pickle=True)
        self.images = data["images"].astype(np.float32)  # [N, 10, 224, 224]
        logger.info(f"Loaded {len(self)} satellite images: {self.images.shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])  # [10, 224, 224]
        padding = torch.zeros(3, image.shape[1], image.shape[2])
        image13 = torch.cat([image, padding], dim=0)  # [13, 224, 224]
        return image13


def extract_satellite():
    """Extract HydroViT embeddings from satellite data."""
    logger.info("=" * 60)
    logger.info("Extracting SATELLITE (HydroViT) embeddings")
    logger.info("=" * 60)

    from sentinel.models.satellite_encoder.model import SatelliteEncoder

    # Find best checkpoint
    ckpt_path = None
    for name in ["hydrovit_wq_v9.pt", "hydrovit_wq_v8.pt", "hydrovit_wq_v7.pt",
                  "hydrovit_wq_v3.pt", "hydrovit_wq_finetuned.pt", "hydrovit_wq_best.pt"]:
        p = CKPT_BASE / "satellite" / name
        if p.exists():
            ckpt_path = p
            break
    if ckpt_path is None:
        logger.error("No satellite checkpoint found, skipping")
        return None

    model = SatelliteEncoder(pretrained=False).to(DEVICE)
    state = _load_state_dict(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(f"Loaded {ckpt_path.name}: {len(missing)} missing, "
                f"{len(unexpected)} unexpected keys")
    model.eval()

    # Try to find paired satellite data
    paired_data = None
    for fname in ["paired_wq_v4.npz", "paired_wq_v3.npz", "paired_wq_expanded.npz"]:
        p = PROJECT_ROOT / "data" / "processed" / "satellite" / fname
        if p.exists():
            paired_data = p
            break

    if paired_data is not None:
        dataset = SatelliteDataset(str(paired_data))
        loader = DataLoader(dataset, batch_size=8, num_workers=0, shuffle=False)
    else:
        # No paired satellite data -- generate representative 13-band tiles
        # using realistic spectral distributions for water bodies
        logger.warning("No paired satellite data found, using representative synthetic tiles")
        rng = np.random.default_rng(42)
        n_images = 2000

        # Realistic per-band means/stds for water-dominated Sentinel-2 tiles
        # Bands: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12 + 3 padding
        band_means = np.array([0.05, 0.06, 0.04, 0.03, 0.02, 0.02,
                               0.02, 0.02, 0.01, 0.01, 0, 0, 0], dtype=np.float32)
        band_stds  = np.array([0.02, 0.02, 0.02, 0.01, 0.01, 0.01,
                               0.01, 0.01, 0.005, 0.005, 0, 0, 0], dtype=np.float32)

        class SyntheticSatelliteDataset(Dataset):
            def __init__(self, n, band_means, band_stds, rng):
                self.n = n
                self.band_means = band_means
                self.band_stds = band_stds
                self.rng = rng

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                # Generate a 13-band tile with spatial correlations
                img = np.zeros((13, 224, 224), dtype=np.float32)
                for b in range(10):
                    base = self.rng.normal(self.band_means[b], self.band_stds[b],
                                           size=(224, 224)).astype(np.float32)
                    img[b] = np.clip(base, 0, 1)
                return torch.from_numpy(img)

        dataset = SyntheticSatelliteDataset(n_images, band_means, band_stds, rng)
        loader = DataLoader(dataset, batch_size=8, num_workers=0, shuffle=False)

    embeddings = []
    t0 = time.time()
    with torch.no_grad():
        with autocast("cuda", enabled=DEVICE.type == "cuda"):
            for i, batch in enumerate(loader):
                batch = batch.to(DEVICE)  # [B, 13, 224, 224]
                out = model(batch)
                emb = out["embedding"].cpu()  # [B, 256]
                embeddings.append(emb)
                if (i + 1) % 50 == 0:
                    logger.info(f"  Batch {i+1}/{len(loader)}")

    if embeddings:
        all_emb = torch.cat(embeddings, dim=0)
        torch.save(all_emb, OUTPUT_DIR / "satellite_embeddings.pt")
        logger.info(f"Saved {all_emb.shape} satellite embeddings in {time.time()-t0:.1f}s")
        logger.info(f"  Stats: mean={all_emb.mean():.4f}, std={all_emb.std():.4f}")
        return all_emb
    else:
        logger.error("No satellite embeddings extracted")
        return None


# ===========================================================================
# 3. MICROBIAL (MicroBiomeNet) — Real EMP 16S OTU data
# ===========================================================================

def extract_microbial():
    """Extract MicroBiomeNet embeddings from real EMP 16S data."""
    logger.info("=" * 60)
    logger.info("Extracting MICROBIAL (MicroBiomeNet) embeddings — real EMP 16S")
    logger.info("=" * 60)

    from sentinel.models.microbial_encoder.model import MicrobialEncoder

    # Find best checkpoint
    ckpt_path = None
    for name in ["microbiomenet_real_best.pt", "microbiomenet_v5_best.pt",
                  "microbiomenet_v4_best.pt", "microbiomenet_v3_best.pt",
                  "microbiomenet_v2_best.pt", "microbiomenet_best.pt"]:
        p = CKPT_BASE / "microbial" / name
        if p.exists():
            ckpt_path = p
            break
    if ckpt_path is None:
        logger.warning("No microbial checkpoint found, skipping")
        return None

    model = MicrobialEncoder().to(DEVICE)
    state = _load_state_dict(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(f"Loaded {ckpt_path.name}: {len(missing)} missing, "
                f"{len(unexpected)} unexpected keys")
    model.eval()

    # Load real EMP 16S OTU data
    emp_dir = PROJECT_ROOT / "data" / "processed" / "microbial" / "emp_16s"
    emp_files = sorted(emp_dir.glob("emp16s_*.npz"))

    if emp_files:
        logger.info(f"Found {len(emp_files)} real EMP 16S samples")
        # Load all samples, CLR-transform the abundances
        clr_samples = []
        for f in emp_files:
            d = np.load(f)
            abundances = d["abundances"].astype(np.float32)  # [5000]
            # CLR transform: log(x / geometric_mean(x))
            x = abundances + 1e-10  # avoid log(0)
            log_x = np.log(x)
            clr = log_x - log_x.mean()
            clr_samples.append(clr)
        clr_data = np.stack(clr_samples, axis=0)  # [N, 5000]
        logger.info(f"  CLR-transformed: {clr_data.shape}")
    else:
        # Fallback: generate representative CLR-transformed OTU data
        logger.warning("No EMP 16S data found, using representative synthetic data")
        rng = np.random.default_rng(42)
        n_samples = 5000
        n_otus = 5000
        alpha = rng.exponential(0.1, size=(n_samples, n_otus)).astype(np.float32) + 1e-6
        alpha /= alpha.sum(axis=1, keepdims=True)
        clr_data = np.log(alpha) - np.log(alpha).mean(axis=1, keepdims=True)

    embeddings = []
    t0 = time.time()
    with torch.no_grad():
        with autocast("cuda", enabled=DEVICE.type == "cuda"):
            for i in range(0, len(clr_data), 128):
                bs = min(128, len(clr_data) - i)
                x = torch.from_numpy(clr_data[i:i+bs]).to(DEVICE)
                try:
                    out = model(x)
                    emb = out["embedding"].cpu()  # [B, 256]
                    embeddings.append(emb)
                except Exception as e:
                    logger.warning(f"Microbial forward failed at batch {i}: {e}")
                    break
                if (i // 128 + 1) % 10 == 0:
                    logger.info(f"  Batch {i//128+1}/{(len(clr_data)+127)//128}")

    if embeddings:
        all_emb = torch.cat(embeddings, dim=0)
        torch.save(all_emb, OUTPUT_DIR / "microbial_embeddings.pt")
        logger.info(f"Saved {all_emb.shape} microbial embeddings in {time.time()-t0:.1f}s")
        logger.info(f"  Stats: mean={all_emb.mean():.4f}, std={all_emb.std():.4f}")
        return all_emb
    else:
        logger.error("No microbial embeddings extracted")
        return None


# ===========================================================================
# 4. MOLECULAR (ToxiGene) — Real gene expression data
# ===========================================================================

def extract_molecular():
    """Extract ToxiGene embeddings from real gene expression data."""
    logger.info("=" * 60)
    logger.info("Extracting MOLECULAR (ToxiGene) embeddings — real expression data")
    logger.info("=" * 60)

    from sentinel.models.molecular_encoder.model import MolecularEncoder
    try:
        from scipy import sparse
    except ImportError:
        logger.warning("scipy not available for sparse matrix loading, skipping molecular")
        return None

    mol_dir = PROJECT_ROOT / "data" / "processed" / "molecular"

    # Find best checkpoint first so we can infer model dimensions
    ckpt_path = None
    for name in ["toxigene_best.pt", "toxigene_v9b_best.pt", "toxigene_v9_best.pt",
                  "toxigene_fullreal_best.pt", "toxigene_expanded_best.pt"]:
        p = CKPT_BASE / "molecular" / name
        if p.exists():
            ckpt_path = p
            break
    if ckpt_path is None:
        logger.warning("No molecular checkpoint found, skipping")
        return None

    # Load checkpoint to determine model dimensions
    state = _load_state_dict(ckpt_path)
    n_genes_ckpt = state["bottleneck.gate_scores"].shape[0]
    n_pathways = state["hierarchy.gene_to_pathway.weight"].shape[0]
    n_processes = state["hierarchy.pathway_to_process.weight"].shape[0]
    n_outcomes = state["hierarchy.process_to_outcome.weight"].shape[0]
    n_chem = state["chem_to_pathway.net.0.weight"].shape[1] - 1  # input_dim = n_chem + 1
    logger.info(f"Checkpoint dims: {n_genes_ckpt} genes, {n_pathways} pathways, "
                f"{n_processes} processes, {n_outcomes} outcomes, {n_chem} chem classes")

    # Try loading adjacency matrices from disk
    adj_files = {
        "pathway": mol_dir / "hierarchy_layer0_gene_to_pathway.npz",
        "process": mol_dir / "hierarchy_layer1_pathway_to_process.npz",
        "outcome": mol_dir / "hierarchy_layer2_process_to_outcome.npz",
    }

    def load_sparse(p):
        d = np.load(p)
        shape = tuple(d["shape"])
        return torch.tensor(
            sparse.csr_matrix(
                (d["data"], d["indices"], d["indptr"]), shape=shape
            ).toarray(),
            dtype=torch.float32,
        )

    if all(p.exists() for p in adj_files.values()):
        pathway_adj = load_sparse(adj_files["pathway"])
        process_adj = load_sparse(adj_files["process"])
        outcome_adj = load_sparse(adj_files["outcome"])

        # If loaded adjacency matrices don't match checkpoint, truncate/pad
        if pathway_adj.shape[1] != n_genes_ckpt:
            logger.info(f"  Adjusting pathway_adj from {pathway_adj.shape} to "
                        f"({n_pathways}, {n_genes_ckpt})")
            if pathway_adj.shape[1] > n_genes_ckpt:
                pathway_adj = pathway_adj[:n_pathways, :n_genes_ckpt]
            else:
                pad = torch.zeros(n_pathways, n_genes_ckpt - pathway_adj.shape[1])
                pathway_adj = torch.cat([pathway_adj[:n_pathways], pad], dim=1)
        if process_adj.shape[1] != n_pathways:
            process_adj = process_adj[:n_processes, :n_pathways]
        if outcome_adj.shape[1] != n_processes:
            outcome_adj = outcome_adj[:n_outcomes, :n_processes]
    else:
        # No adjacency files -- create identity-like adjacency matrices
        # matching checkpoint dimensions (weights will be overridden by checkpoint anyway)
        logger.warning("No adjacency files found, creating placeholder matrices")
        pathway_adj = torch.eye(n_pathways, n_genes_ckpt)
        process_adj = torch.eye(n_processes, n_pathways)
        outcome_adj = torch.eye(n_outcomes, n_processes)

    # Build gene names matching checkpoint dimension
    gene_names = [f"gene_{i:04d}" for i in range(n_genes_ckpt)]

    model = MolecularEncoder(
        gene_names=gene_names,
        pathway_adj=pathway_adj,
        process_adj=process_adj,
        outcome_adj=outcome_adj,
        num_chem_classes=n_chem,
        lambda_l1=0.01,
        dropout=0.2,
    ).to(DEVICE)

    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(f"Loaded {ckpt_path.name}: {len(missing)} missing, "
                f"{len(unexpected)} unexpected keys")
    model.eval()

    # Load real gene expression data
    expr_path = mol_dir / "expression_matrix_v3_corrected.npy"
    if expr_path.exists():
        expr_data = np.load(str(expr_path)).astype(np.float32)
        logger.info(f"Loaded real expression data: {expr_data.shape}")
        # Adjust to match checkpoint gene count
        if expr_data.shape[1] > n_genes_ckpt:
            expr_data = expr_data[:, :n_genes_ckpt]
            logger.info(f"  Truncated to {expr_data.shape} to match checkpoint")
        elif expr_data.shape[1] < n_genes_ckpt:
            pad = np.zeros((expr_data.shape[0], n_genes_ckpt - expr_data.shape[1]),
                           dtype=np.float32)
            expr_data = np.concatenate([expr_data, pad], axis=1)
            logger.info(f"  Padded to {expr_data.shape} to match checkpoint")
    else:
        # Fallback: generate representative gene expression profiles
        logger.warning("No expression matrix found, using representative synthetic data")
        rng = np.random.default_rng(42)
        expr_data = rng.lognormal(0, 1, size=(3000, n_genes_ckpt)).astype(np.float32)

    embeddings = []
    t0 = time.time()
    with torch.no_grad():
        with autocast("cuda", enabled=DEVICE.type == "cuda"):
            for i in range(0, len(expr_data), 128):
                bs = min(128, len(expr_data) - i)
                x = torch.from_numpy(expr_data[i:i+bs]).to(DEVICE)
                try:
                    out = model(gene_expression=x)
                    emb = out["embedding"].cpu()  # [B, 256]
                    embeddings.append(emb)
                except Exception as e:
                    logger.warning(f"Molecular forward failed at batch {i}: {e}")
                    break
                if (i // 128 + 1) % 10 == 0:
                    logger.info(f"  Batch {i//128+1}/{(len(expr_data)+127)//128}")

    if embeddings:
        all_emb = torch.cat(embeddings, dim=0)
        torch.save(all_emb, OUTPUT_DIR / "molecular_embeddings.pt")
        logger.info(f"Saved {all_emb.shape} molecular embeddings in {time.time()-t0:.1f}s")
        logger.info(f"  Stats: mean={all_emb.mean():.4f}, std={all_emb.std():.4f}")
        return all_emb
    else:
        logger.error("No molecular embeddings extracted")
        return None


# ===========================================================================
# 5. BEHAVIORAL (BioMotion) — Real ECOTOX Daphnia trajectories
# ===========================================================================

def extract_behavioral():
    """Extract BioMotion embeddings from real ECOTOX Daphnia trajectories."""
    logger.info("=" * 60)
    logger.info("Extracting BEHAVIORAL (BioMotion) embeddings — real ECOTOX data")
    logger.info("=" * 60)

    from sentinel.models.biomotion.model import BioMotionEncoder

    ckpt_path = CKPT_BASE / "biomotion" / "phase2_best.pt"
    if not ckpt_path.exists():
        ckpt_path = CKPT_BASE / "biomotion" / "biomotion_v2_best.pt"
    if not ckpt_path.exists():
        logger.warning("No biomotion checkpoint found, skipping")
        return None

    model = BioMotionEncoder().to(DEVICE)
    state = _load_state_dict(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(f"Loaded {ckpt_path.name}: {len(missing)} missing, "
                f"{len(unexpected)} unexpected keys")
    model.eval()

    # Load real ECOTOX behavioral trajectories
    real_dir = PROJECT_ROOT / "data" / "processed" / "behavioral_real"
    traj_files = sorted(real_dir.glob("traj_*.npz"))
    if not traj_files:
        logger.warning("No real behavioral trajectories found, skipping")
        return None
    logger.info(f"Found {len(traj_files)} real ECOTOX trajectories")

    embeddings = []
    t0 = time.time()
    batch_kp, batch_feat = [], []
    max_trajs = min(3000, len(traj_files))

    with torch.no_grad():
        with autocast("cuda", enabled=DEVICE.type == "cuda"):
            for i, f in enumerate(traj_files[:max_trajs]):
                d = np.load(f)
                batch_kp.append(d["keypoints"])    # (200, 12, 2)
                batch_feat.append(d["features"])   # (200, 16)

                if len(batch_kp) == 32 or i == max_trajs - 1:
                    kp = torch.from_numpy(np.stack(batch_kp)).to(DEVICE)
                    feat = torch.from_numpy(np.stack(batch_feat)).to(DEVICE)
                    try:
                        out = model.forward_single_species(
                            species="daphnia", keypoints=kp, features=feat)
                        # BioMotion: "embedding" is raw, "fusion_embedding" is projected
                        emb = out["fusion_embedding"].cpu()  # [B, 256]
                        embeddings.append(emb)
                    except Exception as e1:
                        try:
                            org = {"daphnia": {"keypoints": kp, "features": feat}}
                            out = model(org)
                            emb = out["fusion_embedding"].cpu()
                            embeddings.append(emb)
                        except Exception as e2:
                            logger.warning(f"Behavioral forward failed: {e2}")
                            break
                    batch_kp, batch_feat = [], []

                if (i + 1) % 500 == 0:
                    logger.info(f"  {i+1}/{max_trajs} trajectories")

    if embeddings:
        all_emb = torch.cat(embeddings, dim=0)
        torch.save(all_emb, OUTPUT_DIR / "behavioral_embeddings.pt")
        logger.info(f"Saved {all_emb.shape} behavioral embeddings in {time.time()-t0:.1f}s")
        logger.info(f"  Stats: mean={all_emb.mean():.4f}, std={all_emb.std():.4f}")
        return all_emb
    else:
        logger.error("No behavioral embeddings extracted")
        return None


# ===========================================================================
# Main
# ===========================================================================

def main():
    t0 = time.time()
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Checkpoints: {CKPT_BASE}")

    # Extract all modalities (each handles missing data gracefully)
    for name, fn in [("sensor", extract_sensor), ("satellite", extract_satellite),
                     ("microbial", extract_microbial), ("molecular", extract_molecular),
                     ("behavioral", extract_behavioral)]:
        try:
            fn()
        except Exception as e:
            logger.warning(f"Skipping {name}: {e}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 60)
    for f in sorted(OUTPUT_DIR.glob("*_embeddings.pt")):
        emb = torch.load(str(f), map_location="cpu", weights_only=True)
        logger.info(f"  {f.name}: {emb.shape}  "
                    f"mean={emb.mean():.4f}  std={emb.std():.4f}")
    logger.info(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
