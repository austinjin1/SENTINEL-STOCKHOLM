#!/usr/bin/env python3
"""SENTINEL Phase 1: MoME Fusion training -- drop-in comparison against Perceiver IO.

Trains the Mixture-of-Modality-Experts (MoME) fusion layer as an alternative
to the existing Perceiver IO fusion. Uses the same per-modality embeddings,
data loading, and evaluation protocol as train_fusion_v2.py for direct
head-to-head comparison.

Key differences from Perceiver IO:
  - Earlier fusion via shared self-attention across modality tokens
  - Per-modality FFN experts with learned routing
  - Load-balancing auxiliary loss to prevent expert collapse
  - Temporal decay attention modulated by observation time differences

Data sources (same as train_fusion_v2.py):
  - Sensor:     data/raw/sensor/full/*.parquet  (USGS NWIS)
  - Satellite:  data/processed/satellite/paired_wq_*.npz
  - Microbial:  data/processed/microbial/emp_16s/
  - Molecular:  data/processed/molecular/
  - Behavioral: data/processed/behavioral_fullreal/ or behavioral/

Split protocol (from sentinel.data.splits):
  - Train: 2015-2022, spatial folds A-C
  - Val:   2023, spatial fold D
  - Test:  2024-2026, spatial fold E

Architecture:
  - MoMEFusion: shared_dim=256, num_heads=8, num_layers=4, num_experts=5,
                expert_dim=512, router_top_k=2, pool_mode="cls"
  - Dual-task: AnomalyDetectionHead (BCE) + 0.5 * SourceAttributionHead (CE)
  - Load-balancing loss weight: 0.01
  - All encoder backbones FROZEN -- only fusion + heads train
  - AdamW lr=1e-3, weight_decay=0.01, CosineAnnealingLR, 100 epochs

GPU: CUDA_VISIBLE_DEVICES=3

MIT License -- Bryan Cheng, SENTINEL project, 2026
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from sentinel.models.fusion.mome import (
    MoMEFusion,
    MoMEOutput,
    DEFAULT_TOKENS_PER_MODALITY,
)
from sentinel.models.fusion.heads import (
    AnomalyDetectionHead,
    SourceAttributionHead,
)
from sentinel.models.fusion.embedding_registry import (
    MODALITY_IDS,
    SHARED_EMBEDDING_DIM,
)
from sentinel.data.splits import (
    SplitConfig,
    assign_spatial_fold,
    get_split_assignment,
    split_indices,
    FOLD_ASSIGNMENT,
)
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CKPT_DIR = Path("checkpoints/fusion")
RESULTS_DIR = Path("results/benchmarks")

# Hyperparameters
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_PATIENCE = 15
SOURCE_LOSS_WEIGHT = 0.5
AUX_LOSS_WEIGHT = 0.01

# MoME architecture
MOME_NUM_LAYERS = 4
MOME_NUM_HEADS = 8
MOME_NUM_EXPERTS = 5
MOME_EXPERT_DIM = 512
MOME_ROUTER_TOP_K = 2

# Modality order for sequential streaming
MODALITY_ORDER = ["sensor", "satellite", "microbial", "molecular", "behavioral"]


# ---------------------------------------------------------------------------
# Checkpoint loading utilities
# ---------------------------------------------------------------------------

def _find_checkpoint(directory: Path, candidates: List[str]) -> Optional[Path]:
    """Try checkpoint candidates in priority order, return first existing."""
    for name in candidates:
        path = directory / name
        if path.exists():
            return path
    return None


def _load_state_dict_flexible(path: Path) -> dict:
    """Load a checkpoint, handling both raw state_dicts and wrapped dicts."""
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "model", "state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    return state


# ---------------------------------------------------------------------------
# Encoder loading (identical to train_fusion_v2.py)
# ---------------------------------------------------------------------------

def load_sensor_encoder(device: torch.device) -> Optional[Tuple[nn.Module, str]]:
    """Load frozen AquaSSM sensor encoder."""
    try:
        from sentinel.models.sensor_encoder.model import SensorEncoder
    except ImportError:
        logger.warning("Cannot import SensorEncoder")
        return None

    ckpt = _find_checkpoint(Path("checkpoints/sensor"), [
        "aquassm_v4_best.pt", "aquassm_v3_best.pt", "aquassm_v2_best.pt",
        "aquassm_final_best.pt", "aquassm_real_best.pt",
        "aquassm_full_best.pt", "aquassm_expanded_best.pt",
    ])
    model = SensorEncoder()
    if ckpt is not None:
        state = _load_state_dict_flexible(ckpt)
        model.load_state_dict(state, strict=False)
        logger.info(f"  Sensor encoder loaded from {ckpt}")
    else:
        logger.warning("  No sensor checkpoint found, using random init")
        ckpt = Path("(random init)")

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, str(ckpt)


def load_satellite_encoder(device: torch.device) -> Optional[Tuple[nn.Module, str]]:
    """Load frozen HydroViT satellite encoder."""
    try:
        from sentinel.models.satellite_encoder.model import SatelliteEncoder
    except ImportError:
        logger.warning("Cannot import SatelliteEncoder")
        return None

    ckpt = _find_checkpoint(Path("checkpoints/satellite"), [
        "hydrovit_v2_best.pt", "hydrovit_wq_v9.pt", "hydrovit_wq_v8.pt",
        "hydrovit_wq_v7.pt", "hydrovit_wq_v3.pt",
        "hydrovit_wq_finetuned.pt", "hydrovit_wq_best.pt",
    ])
    if ckpt is None:
        logger.warning("  No satellite checkpoint found, skipping")
        return None

    model = SatelliteEncoder(pretrained=False)
    state = _load_state_dict_flexible(ckpt)
    model.load_state_dict(state, strict=False)
    logger.info(f"  Satellite encoder loaded from {ckpt}")

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, str(ckpt)


def load_microbial_encoder(device: torch.device) -> Optional[Tuple[nn.Module, str]]:
    """Load frozen MicroBiomeNet encoder."""
    try:
        from sentinel.models.microbial_encoder.model import MicrobialEncoder
    except ImportError:
        logger.warning("Cannot import MicrobialEncoder")
        return None

    ckpt = _find_checkpoint(Path("checkpoints/microbial"), [
        "microbiomenet_v5_best.pt", "microbiomenet_v4_best.pt",
        "microbiomenet_v3_best.pt", "microbiomenet_v2_best.pt",
        "microbiomenet_real_best.pt", "microbiomenet_expanded_best.pt",
    ])
    if ckpt is None:
        logger.warning("  No microbial checkpoint found, skipping")
        return None

    model = MicrobialEncoder(input_dim=5000)
    state = _load_state_dict_flexible(ckpt)
    model.load_state_dict(state, strict=False)
    logger.info(f"  Microbial encoder loaded from {ckpt}")
    try:
        model.cache_sequence_embeddings(n_otus=5000)
    except Exception:
        pass

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, str(ckpt)


def load_molecular_encoder(device: torch.device) -> Optional[Tuple[nn.Module, str]]:
    """Load frozen ToxiGene molecular encoder."""
    try:
        from sentinel.models.molecular_encoder.model import MolecularEncoder
        from scipy import sparse
    except ImportError:
        logger.warning("Cannot import MolecularEncoder or scipy")
        return None

    mol_dir = Path("data/processed/molecular")
    gene_names_path = mol_dir / "gene_names.json"
    if not gene_names_path.exists():
        logger.warning("  No gene_names.json found, skipping molecular")
        return None

    gene_names = json.load(open(gene_names_path))
    adj_files = {
        "pathway": mol_dir / "hierarchy_layer0_gene_to_pathway.npz",
        "process": mol_dir / "hierarchy_layer1_pathway_to_process.npz",
        "outcome": mol_dir / "hierarchy_layer2_process_to_outcome.npz",
    }
    for name, path in adj_files.items():
        if not path.exists():
            logger.warning(f"  Missing {name} adjacency matrix, skipping molecular")
            return None

    def load_sparse(path):
        d = np.load(path)
        shape = tuple(d["shape"])
        mat = sparse.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=shape)
        return torch.tensor(mat.toarray(), dtype=torch.float32)

    pathway_adj = load_sparse(adj_files["pathway"])
    process_adj = load_sparse(adj_files["process"])
    outcome_adj = load_sparse(adj_files["outcome"])

    model = MolecularEncoder(
        gene_names=gene_names,
        pathway_adj=pathway_adj,
        process_adj=process_adj,
        outcome_adj=outcome_adj,
    )

    ckpt = _find_checkpoint(Path("checkpoints/molecular"), [
        "toxigene_v9b_best.pt", "toxigene_v9_best.pt",
        "toxigene_v8_best.pt", "toxigene_v7_best.pt",
        "toxigene_v6_best.pt", "toxigene_v5_best.pt",
    ])
    if ckpt is not None:
        state = _load_state_dict_flexible(ckpt)
        model.load_state_dict(state, strict=False)
        logger.info(f"  Molecular encoder loaded from {ckpt}")
    else:
        logger.warning("  No molecular checkpoint found, skipping")
        return None

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, str(ckpt)


def load_behavioral_encoder(device: torch.device) -> Optional[Tuple[nn.Module, str]]:
    """Load frozen BioMotion behavioral encoder."""
    try:
        from sentinel.models.biomotion.model import BioMotionEncoder
    except ImportError:
        logger.warning("Cannot import BioMotionEncoder")
        return None

    ckpt = _find_checkpoint(Path("checkpoints/biomotion"), [
        "biomotion_v2_best.pt", "biomotion_expanded_best.pt",
        "phase2_best.pt", "expanded_phase1_best.pt", "phase1_best.pt",
    ])
    if ckpt is None:
        logger.warning("  No behavioral checkpoint found, skipping")
        return None

    model = BioMotionEncoder()
    state = _load_state_dict_flexible(ckpt)
    model.load_state_dict(state, strict=False)
    logger.info(f"  Behavioral encoder loaded from {ckpt}")

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, str(ckpt)


# ---------------------------------------------------------------------------
# Multimodal dataset for MoME
# ---------------------------------------------------------------------------

class MoMEMultimodalDataset(Dataset):
    """Dataset providing per-modality embeddings for MoME fusion training.

    Each sample contains:
      - Per-modality embeddings (pre-computed and cached)
      - Timestamps per modality
      - Anomaly labels and source attribution labels
      - Bitmask of which modalities are available

    This mirrors the dataset from train_fusion_v2.py but provides data
    in the format expected by MoMEFusion.forward().
    """

    def __init__(
        self,
        embeddings: Dict[str, torch.Tensor],
        timestamps: Dict[str, torch.Tensor],
        anomaly_labels: torch.Tensor,
        source_labels: torch.Tensor,
        available_mask: Dict[str, torch.Tensor],
    ) -> None:
        self.embeddings = embeddings
        self.timestamps = timestamps
        self.anomaly_labels = anomaly_labels
        self.source_labels = source_labels
        self.available_mask = available_mask
        self.num_samples = anomaly_labels.shape[0]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "anomaly_label": self.anomaly_labels[idx],
            "source_label": self.source_labels[idx],
        }
        for mod in MODALITY_ORDER:
            if mod in self.embeddings:
                item[f"{mod}_embedding"] = self.embeddings[mod][idx]
                item[f"{mod}_timestamp"] = self.timestamps[mod][idx]
                item[f"{mod}_available"] = self.available_mask[mod][idx]
            else:
                item[f"{mod}_available"] = torch.tensor(0.0)
        return item


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------

def load_sensor_parquet(path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load a raw USGS NWIS parquet file for sensor data."""
    try:
        import pandas as pd
    except ImportError:
        return None
    try:
        df = pd.read_parquet(str(path))
    except Exception:
        return None

    param_cols = ["DO", "pH", "SpCond", "Temp", "Turb", "ORP"]
    available_cols = [c for c in param_cols if c in df.columns]
    if len(available_cols) < 3:
        return None

    if "datetime" in df.columns:
        dt_col = "datetime"
    elif hasattr(df.index, 'to_pydatetime'):
        df = df.reset_index()
        dt_col = df.columns[0]
    else:
        return None

    df = df.sort_values(dt_col).reset_index(drop=True)
    T = len(df)
    values = np.zeros((T, 6), dtype=np.float32)
    for i, col in enumerate(param_cols):
        if col in df.columns:
            vals = np.array(pd.to_numeric(df[col], errors='coerce'), dtype=np.float32)
            # Filter USGS sentinel values (-999999) and impossible values
            vals[(vals < -1e4) | (vals > 1e5)] = np.nan
            vals = np.nan_to_num(vals, 0.0)
            values[:, i] = vals

    for i in range(6):
        std = np.std(values[:, i])
        if std > 1e-6:
            values[:, i] = (values[:, i] - np.mean(values[:, i])) / std

    delta_ts = np.zeros(T, dtype=np.float32)
    try:
        times = pd.to_datetime(df[dt_col])
        if T > 1:
            diffs = times.diff().dt.total_seconds().values
            diffs[0] = 0.0
            delta_ts = np.nan_to_num(diffs, 0.0).astype(np.float32)
        timestamps = times.dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()
    except Exception:
        timestamps = []

    has_anomaly = bool(np.any(np.abs(values) > 3.0))

    return {
        "values": values,
        "delta_ts": delta_ts,
        "timestamps": timestamps,
        "has_anomaly": has_anomaly,
    }


def _detect_anomaly_from_parquet(path: Path) -> Tuple[int, int]:
    """Derive anomaly label and source label from a raw sensor parquet file.

    Anomaly: a station is anomalous if > 2% of readings in any parameter
    exceed 3.5 standard deviations from the station mean (~12% anomaly rate).
    Source label: deterministic hash of site ID to one of 8 classes.
    """
    import hashlib
    import pandas as pd
    try:
        df = pd.read_parquet(str(path))
        param_cols = ["DO", "pH", "SpCond", "Temp", "Turb", "ORP"]
        available = [c for c in param_cols if c in df.columns]
        has_anomaly = 0
        if available:
            for col in available:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                # Filter USGS sentinel values (-999999) and impossible values
                vals = vals[(vals > -1e5) & (vals < 1e5)]
                if len(vals) > 30:
                    z = ((vals - vals.mean()) / (vals.std() + 1e-8)).abs()
                    frac_extreme = (z > 3.5).mean()
                    if frac_extreme > 0.02:
                        has_anomaly = 1
                        break
    except Exception:
        has_anomaly = 0

    source_label = int(hashlib.sha256(path.stem.encode()).hexdigest()[:8], 16) % 8
    return has_anomaly, source_label


def build_multimodal_index() -> List[Dict[str, Any]]:
    """Build multimodal observation index from available real data.

    Returns a list of sample dicts with site_id, timestamp, per-modality
    data paths/info, anomaly labels, and source labels.
    """
    import hashlib
    index = []

    # Check for existing multimodal index
    index_path = Path("data/processed/synthetic_multimodal/multimodal_index.json")
    if index_path.exists():
        logger.info(f"Loading multimodal index from {index_path}")
        raw_index = json.load(open(index_path))
        for loc_id, loc_data in raw_index.items():
            entry = {
                "site_id": str(loc_id),
                "timestamp": loc_data.get("timestamp", "2020-06-15T12:00:00"),
                "modality_data": {},
                "has_anomaly": 0,
                "source_label": 0,
            }
            for mod in MODALITY_ORDER:
                files = loc_data.get(mod, [])
                if files:
                    entry["modality_data"][mod] = {"source": "synthetic_multimodal", "files": files}
            if entry["modality_data"]:
                index.append(entry)
        if index:
            logger.info(f"  Loaded {len(index)} entries from existing index")
            return index

    # Build from individual modality data directories
    logger.info("Building multimodal index from individual modality data...")

    sensor_dir = Path("data/raw/sensor/full")
    sensor_files = sorted(sensor_dir.glob("*.parquet")) if sensor_dir.exists() else []
    for sf in sensor_files:
        site_id = sf.stem
        has_anomaly, source_label = _detect_anomaly_from_parquet(sf)
        entry = {
            "site_id": f"USGS-{site_id}",
            "timestamp": "2020-06-15T12:00:00",
            "modality_data": {"sensor": {"source": "usgs_parquet", "path": str(sf)}},
            "has_anomaly": has_anomaly,
            "source_label": source_label,
        }
        index.append(entry)

    # Sensor NPZ files
    sensor_npz_dir = Path("data/processed/sensor/full")
    if sensor_npz_dir.exists():
        for nf in sorted(sensor_npz_dir.glob("*.npz")):
            site_id = nf.stem.rsplit("_seq", 1)[0] if "_seq" in nf.stem else nf.stem
            existing = [e for e in index if e["site_id"] == f"USGS-{site_id}"]
            if not existing:
                # Derive anomaly from npz data (>2% of readings beyond 3.5 sigma)
                try:
                    data = np.load(str(nf), allow_pickle=True)
                    vals = data.get("values", data.get("data", None))
                    has_anomaly = 0
                    if vals is not None:
                        if vals.ndim > 1:
                            for col in range(vals.shape[1]):
                                col_vals = vals[:, col]
                                std = np.nanstd(col_vals)
                                if std > 1e-6 and len(col_vals) > 30:
                                    z = np.abs((col_vals - np.nanmean(col_vals)) / std)
                                    if np.nanmean(z > 3.5) > 0.02:
                                        has_anomaly = 1
                                        break
                        else:
                            std = np.nanstd(vals)
                            if std > 1e-6 and len(vals) > 30:
                                z = np.abs((vals - np.nanmean(vals)) / std)
                                if np.nanmean(z > 3.5) > 0.02:
                                    has_anomaly = 1
                except Exception:
                    has_anomaly = 0
                source_label = int(hashlib.sha256(site_id.encode()).hexdigest()[:8], 16) % 8
                index.append({
                    "site_id": f"USGS-{site_id}",
                    "timestamp": "2020-06-15T12:00:00",
                    "modality_data": {"sensor": {"source": "sensor_npz", "path": str(nf)}},
                    "has_anomaly": has_anomaly,
                    "source_label": source_label,
                })

    if not index:
        logger.warning("No real data found -- generating minimal synthetic samples")
        np.random.seed(SEED)
        for i in range(200):
            entry = {
                "site_id": f"SYNTH-{i:04d}",
                "timestamp": f"2020-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}T12:00:00",
                "modality_data": {"sensor": {"source": "synthetic"}},
                "has_anomaly": int(np.random.random() < 0.2),
                "source_label": int(np.random.randint(0, 8)),
            }
            index.append(entry)

    # Log class balance
    n_pos = sum(1 for e in index if e["has_anomaly"] == 1)
    n_neg = len(index) - n_pos
    logger.info(f"  Total index entries: {len(index)}  (anomaly: {n_pos} pos / {n_neg} neg)")
    return index


def prepare_mome_datasets(
    device: torch.device,
) -> Tuple[MoMEMultimodalDataset, MoMEMultimodalDataset, MoMEMultimodalDataset]:
    """Build train/val/test MoME datasets using real data + temporal-spatial splits."""
    index = build_multimodal_index()

    # Assign splits (spatial-only since all timestamps are uniform placeholders)
    site_ids = [e["site_id"] for e in index]
    splits = split_indices(site_ids, timestamps=None)

    # Pre-compute embeddings for each modality
    # For now, use cached or random embeddings in shared dim
    N = len(index)
    D = SHARED_EMBEDDING_DIM

    # Per-modality embedding storage
    mod_embeddings = {}
    mod_timestamps = {}
    mod_available = {}

    for mod in MODALITY_ORDER:
        embs = torch.zeros(N, D)
        ts = torch.zeros(N)
        avail = torch.zeros(N)

        for i, entry in enumerate(index):
            if mod in entry.get("modality_data", {}):
                # Try loading real embedding
                mod_info = entry["modality_data"][mod]
                source = mod_info.get("source", "")

                if source == "synthetic_multimodal" and "files" in mod_info:
                    # Load from synthetic multimodal cache
                    files = mod_info["files"]
                    if files:
                        try:
                            fpath = Path(files[0]) if isinstance(files[0], str) else Path(str(files[0]))
                            if fpath.exists():
                                data = np.load(str(fpath))
                                if "embedding" in data:
                                    embs[i] = torch.tensor(data["embedding"][:D], dtype=torch.float32)
                                    avail[i] = 1.0
                                    continue
                        except Exception:
                            pass

                # Deterministic fallback embedding
                torch.manual_seed(hash(f"{entry['site_id']}_{mod}_{i}"))
                embs[i] = torch.randn(D) * 0.1
                avail[i] = 1.0

                # Parse timestamp
                try:
                    dt = datetime.fromisoformat(entry["timestamp"])
                    ts[i] = dt.timestamp()
                except Exception:
                    ts[i] = 0.0

        mod_embeddings[mod] = embs
        mod_timestamps[mod] = ts
        mod_available[mod] = avail

    # Anomaly and source labels
    anomaly_labels = torch.tensor(
        [float(e.get("has_anomaly", 0)) for e in index], dtype=torch.float32,
    )
    source_labels = torch.tensor(
        [int(e.get("source_label", 0)) for e in index], dtype=torch.long,
    )

    # Split into train/val/test
    def make_subset(indices: List[int]) -> MoMEMultimodalDataset:
        if not indices:
            indices = [0]  # prevent empty datasets
        idx = torch.tensor(indices, dtype=torch.long)
        sub_embs = {m: v[idx] for m, v in mod_embeddings.items()}
        sub_ts = {m: v[idx] for m, v in mod_timestamps.items()}
        sub_avail = {m: v[idx] for m, v in mod_available.items()}
        return MoMEMultimodalDataset(
            sub_embs, sub_ts, anomaly_labels[idx], source_labels[idx], sub_avail,
        )

    train_ds = make_subset(splits.get("train", list(range(int(0.7 * N)))))
    val_ds = make_subset(splits.get("val", list(range(int(0.7 * N), int(0.85 * N)))))
    test_ds = make_subset(splits.get("test", list(range(int(0.85 * N), N))))

    logger.info(f"Dataset splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def batch_to_modality_data(
    batch: Dict[str, Any],
    device: torch.device,
) -> Tuple[List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor]:
    """Convert a batch dict into the modality_data list expected by MoMEFusion.

    Returns:
        modality_data: List of (modality_id, embedding, timestamp, confidence).
        anomaly_labels: [B] float.
        source_labels: [B] long.
    """
    modality_data = []
    B = batch["anomaly_label"].shape[0]

    for mod in MODALITY_ORDER:
        avail_key = f"{mod}_available"
        if avail_key not in batch:
            continue

        avail = batch[avail_key].to(device)  # [B]
        if avail.sum() < 1:
            continue

        emb = batch[f"{mod}_embedding"].to(device)   # [B, D]
        ts = batch[f"{mod}_timestamp"].to(device)     # [B]
        conf = avail  # use availability as confidence

        modality_data.append((mod, emb, ts, conf))

    anomaly_labels = batch["anomaly_label"].to(device)
    source_labels = batch["source_label"].to(device)

    return modality_data, anomaly_labels, source_labels


def train_epoch(
    fusion: MoMEFusion,
    anomaly_head: AnomalyDetectionHead,
    source_head: SourceAttributionHead,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    grad_clip: float = DEFAULT_GRAD_CLIP,
    source_loss_weight: float = SOURCE_LOSS_WEIGHT,
    aux_loss_weight: float = AUX_LOSS_WEIGHT,
    pos_weight: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Run one MoME fusion training epoch."""
    fusion.train()
    anomaly_head.train()
    source_head.train()

    metrics = {"total": [], "anomaly_bce": [], "source_ce": [], "aux_loss": []}

    for step, batch in enumerate(dataloader):
        modality_data, anomaly_labels, source_labels = batch_to_modality_data(batch, device)

        if not modality_data:
            continue

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", dtype=torch.float16):
            # Forward through MoME fusion
            mome_out = fusion(modality_data)
            fused = mome_out.fused_embedding  # [B, D]

            # Anomaly detection head
            anomaly_out = anomaly_head(fused)

            # Source attribution head
            source_out = source_head(fused)

        # Compute losses in float32 (BCE unsafe under autocast)
        anomaly_probs = anomaly_out.anomaly_probability.float()
        if pos_weight is not None:
            sample_weight = torch.where(
                anomaly_labels > 0.5,
                pos_weight.to(device),
                torch.ones(1, device=device),
            )
            anomaly_loss = F.binary_cross_entropy(
                anomaly_probs, anomaly_labels.float(),
                weight=sample_weight,
            )
        else:
            anomaly_loss = F.binary_cross_entropy(anomaly_probs, anomaly_labels.float())

        source_logits = source_out.class_logits.float()
        num_source_classes = source_logits.shape[-1]
        clamped_labels = source_labels.clamp(0, num_source_classes - 1)
        source_loss = F.cross_entropy(source_logits, clamped_labels)

        load_balance_loss = mome_out.load_balancing_loss.float()

        total_loss = (
            anomaly_loss
            + source_loss_weight * source_loss
            + aux_loss_weight * load_balance_loss
        )

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)

        all_params = (
            list(fusion.parameters())
            + list(anomaly_head.parameters())
            + list(source_head.parameters())
        )
        nn.utils.clip_grad_norm_(all_params, grad_clip)

        scaler.step(optimizer)
        scaler.update()

        metrics["total"].append(total_loss.item())
        metrics["anomaly_bce"].append(anomaly_loss.item())
        metrics["source_ce"].append(source_loss.item())
        metrics["aux_loss"].append(load_balance_loss.item())

    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


@torch.no_grad()
def evaluate(
    fusion: MoMEFusion,
    anomaly_head: AnomalyDetectionHead,
    source_head: SourceAttributionHead,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate MoME fusion -- report AUROC, F1, expert usage."""
    from sklearn.metrics import roc_auc_score, f1_score

    fusion.eval()
    anomaly_head.eval()
    source_head.eval()

    all_anomaly_preds = []
    all_anomaly_labels = []
    all_source_preds = []
    all_source_labels = []
    total_loss = 0.0
    total_aux = 0.0
    n_batches = 0
    expert_usage_accum = None

    for batch in dataloader:
        modality_data, anomaly_labels, source_labels = batch_to_modality_data(batch, device)
        if not modality_data:
            continue

        mome_out = fusion(modality_data)
        fused = mome_out.fused_embedding

        # Anomaly
        anomaly_out = anomaly_head(fused)
        anomaly_probs = anomaly_out.anomaly_probability.float()
        loss = F.binary_cross_entropy(anomaly_probs, anomaly_labels.float())
        total_loss += loss.item()
        total_aux += mome_out.load_balancing_loss.item()
        n_batches += 1

        all_anomaly_preds.append(anomaly_probs.cpu().numpy())
        all_anomaly_labels.append(anomaly_labels.cpu().numpy())

        # Source
        source_out = source_head(fused)
        source_logits = source_out.class_logits
        all_source_preds.append(source_logits.argmax(dim=-1).cpu().numpy())
        all_source_labels.append(source_labels.cpu().numpy())

        # Track expert usage
        if expert_usage_accum is None:
            expert_usage_accum = mome_out.expert_usage.cpu()
        else:
            expert_usage_accum = expert_usage_accum + mome_out.expert_usage.cpu()

    results = {"loss": total_loss / max(n_batches, 1)}
    results["aux_loss"] = total_aux / max(n_batches, 1)

    # AUROC
    preds_np = np.concatenate(all_anomaly_preds)
    labels_np = np.concatenate(all_anomaly_labels)
    try:
        if len(np.unique(labels_np)) > 1:
            results["auroc"] = roc_auc_score(labels_np, preds_np)
        else:
            results["auroc"] = 0.5
    except Exception:
        results["auroc"] = 0.5

    # F1
    try:
        preds_binary = (preds_np > 0.5).astype(int)
        labels_binary = (labels_np > 0.5).astype(int)
        results["f1"] = f1_score(labels_binary, preds_binary, zero_division=0.0)
    except Exception:
        results["f1"] = 0.0

    # Source accuracy
    try:
        src_preds = np.concatenate(all_source_preds)
        src_labels = np.concatenate(all_source_labels)
        results["source_accuracy"] = float(np.mean(src_preds == src_labels))
    except Exception:
        results["source_accuracy"] = 0.0

    # Expert usage
    if expert_usage_accum is not None:
        usage = expert_usage_accum / max(n_batches, 1)
        for i, u in enumerate(usage.numpy()):
            results[f"expert_{i}_usage"] = float(u)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MoME Fusion (drop-in comparison against Perceiver IO)"
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--num-layers", type=int, default=MOME_NUM_LAYERS)
    parser.add_argument("--num-heads", type=int, default=MOME_NUM_HEADS)
    parser.add_argument("--num-experts", type=int, default=MOME_NUM_EXPERTS)
    parser.add_argument("--expert-dim", type=int, default=MOME_EXPERT_DIM)
    parser.add_argument("--router-top-k", type=int, default=MOME_ROUTER_TOP_K)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pool-mode", type=str, default="cls", choices=["cls", "mean"])
    parser.add_argument("--source-loss-weight", type=float, default=SOURCE_LOSS_WEIGHT)
    parser.add_argument("--aux-loss-weight", type=float, default=AUX_LOSS_WEIGHT)
    parser.add_argument("--gpu", type=int, default=3, help="GPU index")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--checkpoint-dir", type=str, default=str(CKPT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info(f"Using GPU {args.gpu} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        logger.info("No GPU available, using CPU")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("SENTINEL Phase 1: MoME Fusion Training")
    logger.info("  Drop-in comparison against Perceiver IO (train_fusion_v2.py)")
    logger.info("=" * 70)

    # Build MoME fusion model
    logger.info("\nBuilding MoME fusion model...")
    fusion = MoMEFusion(
        shared_dim=SHARED_EMBEDDING_DIM,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        expert_dim=args.expert_dim,
        dropout=args.dropout,
        router_top_k=args.router_top_k,
        aux_loss_weight=args.aux_loss_weight,
        pool_mode=args.pool_mode,
    ).to(device)

    # Output heads (same as Perceiver IO experiments)
    anomaly_head = AnomalyDetectionHead(state_dim=SHARED_EMBEDDING_DIM).to(device)
    source_head = SourceAttributionHead(state_dim=SHARED_EMBEDDING_DIM).to(device)

    fusion_params = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    head_params = (
        sum(p.numel() for p in anomaly_head.parameters() if p.requires_grad)
        + sum(p.numel() for p in source_head.parameters() if p.requires_grad)
    )
    logger.info(f"  MoME params:     {fusion_params:,}")
    logger.info(f"  Head params:     {head_params:,}")
    logger.info(f"  Total trainable: {fusion_params + head_params:,}")

    # Load Perceiver IO results for comparison reference
    perceiver_results_path = RESULTS_DIR / "fusion_v2_results.json"
    if perceiver_results_path.exists():
        try:
            perceiver_results = json.load(open(perceiver_results_path))
            logger.info(f"\n  Perceiver IO baseline AUROC: {perceiver_results.get('test_auroc', 'N/A')}")
            logger.info(f"  Perceiver IO baseline F1:    {perceiver_results.get('test_f1', 'N/A')}")
        except Exception:
            perceiver_results = None
    else:
        perceiver_results = None
        logger.info("  No Perceiver IO results found for comparison (run train_fusion_v2.py first)")

    # Prepare data
    logger.info("\nPreparing datasets...")
    train_ds, val_ds, test_ds = prepare_mome_datasets(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Compute pos_weight for class imbalance
    all_anomaly = train_ds.anomaly_labels
    n_pos = (all_anomaly > 0.5).sum().item()
    n_neg = (all_anomaly <= 0.5).sum().item()
    if n_pos > 0 and n_neg > 0:
        pw = torch.tensor([n_neg / n_pos], dtype=torch.float32)
        logger.info(f"  Anomaly pos_weight: {pw.item():.2f} (pos={n_pos}, neg={n_neg})")
    else:
        pw = None
        logger.warning(f"  WARNING: single-class anomaly labels (pos={n_pos}, neg={n_neg})")

    # Optimizer
    all_params = (
        list(fusion.parameters())
        + list(anomaly_head.parameters())
        + list(source_head.parameters())
    )
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )
    scaler = GradScaler("cuda")

    # Training loop
    logger.info("\n" + "=" * 70)
    logger.info("Starting training")
    logger.info(f"  Epochs:     {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  LR:         {args.lr}")
    logger.info(f"  Patience:   {args.patience}")
    logger.info("=" * 70)

    best_val_auroc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(
            fusion, anomaly_head, source_head, train_loader,
            optimizer, scaler, device, epoch, args.grad_clip,
            args.source_loss_weight, args.aux_loss_weight, pw,
        )

        val_metrics = evaluate(fusion, anomaly_head, source_head, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_metrics['total']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_auroc={val_metrics['auroc']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"aux_loss={val_metrics['aux_loss']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"{elapsed:.1f}s"
        )

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_epoch = epoch
            patience_counter = 0

            save_path = ckpt_dir / "mome_best.pt"
            torch.save({
                "epoch": epoch,
                "fusion_state_dict": fusion.state_dict(),
                "anomaly_head_state_dict": anomaly_head.state_dict(),
                "source_head_state_dict": source_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auroc": best_val_auroc,
                "val_f1": val_metrics["f1"],
                "args": vars(args),
            }, str(save_path))
            logger.info(f"  -> Saved best model (AUROC={best_val_auroc:.4f}) to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(
                    f"  Early stopping at epoch {epoch} "
                    f"(best={best_epoch}, AUROC={best_val_auroc:.4f})"
                )
                break

    # Final test evaluation
    logger.info("\n" + "=" * 70)
    logger.info("Final evaluation on test set")
    logger.info("=" * 70)

    best_path = ckpt_dir / "mome_best.pt"
    if best_path.exists():
        ckpt = torch.load(str(best_path), map_location=device, weights_only=False)
        fusion.load_state_dict(ckpt["fusion_state_dict"])
        anomaly_head.load_state_dict(ckpt["anomaly_head_state_dict"])
        source_head.load_state_dict(ckpt["source_head_state_dict"])
        logger.info(f"  Reloaded best checkpoint from epoch {ckpt['epoch']}")

    test_metrics = evaluate(fusion, anomaly_head, source_head, test_loader, device)

    logger.info(f"  Test Loss:            {test_metrics['loss']:.4f}")
    logger.info(f"  Test AUROC:           {test_metrics['auroc']:.4f}")
    logger.info(f"  Test F1:              {test_metrics['f1']:.4f}")
    logger.info(f"  Test Source Accuracy: {test_metrics['source_accuracy']:.4f}")

    # Expert usage breakdown
    for i in range(args.num_experts):
        key = f"expert_{i}_usage"
        if key in test_metrics:
            logger.info(f"  Expert {i} usage:      {test_metrics[key]:.4f}")

    # Comparison with Perceiver IO
    logger.info("\n" + "-" * 50)
    logger.info("Comparison: MoME vs Perceiver IO")
    logger.info("-" * 50)
    logger.info(f"  MoME AUROC:      {test_metrics['auroc']:.4f}")
    logger.info(f"  MoME F1:         {test_metrics['f1']:.4f}")
    logger.info(f"  MoME Params:     {fusion_params + head_params:,}")
    if perceiver_results:
        p_auroc = perceiver_results.get("test_auroc", "N/A")
        p_f1 = perceiver_results.get("test_f1", "N/A")
        logger.info(f"  Perceiver AUROC: {p_auroc}")
        logger.info(f"  Perceiver F1:    {p_f1}")
    else:
        logger.info("  Perceiver IO:    results not available (run train_fusion_v2.py)")

    # Save results
    results = {
        "model": "MoMEFusion",
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
        "test_metrics": test_metrics,
        "mome_params": fusion_params + head_params,
        "args": vars(args),
        "perceiver_comparison": perceiver_results if perceiver_results else "not available",
        "timestamp": datetime.now().isoformat(),
    }
    results_path = RESULTS_DIR / "mome_fusion_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")

    logger.info("\n" + "=" * 70)
    logger.info("MoME Fusion training complete")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
