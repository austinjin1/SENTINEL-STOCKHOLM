#!/usr/bin/env python3
"""SENTINEL-Fusion training: Perceiver IO cross-modal fusion.

Trains the fusion layer on cached embeddings from all 5 modality encoders.
Phase 1: Extract embeddings from trained encoders
Phase 2: Train fusion layer with anomaly detection + source attribution

MIT License — Bryan Cheng, 2026
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import f1_score, roc_auc_score

from sentinel.models.sensor_encoder.model import SensorEncoder
from sentinel.models.satellite_encoder.model import SatelliteEncoder
from sentinel.models.microbial_encoder.model import MicrobialEncoder
from sentinel.models.microbial_encoder.aitchison_attention import clr_transform
from sentinel.models.molecular_encoder.model import MolecularEncoder
from sentinel.models.biomotion.model import BioMotionEncoder
from sentinel.models.fusion.model import PerceiverIOFusion
from sentinel.models.fusion.heads import (
    AnomalyDetectionHead,
    SourceAttributionHead,
)
from sentinel.models.fusion.embedding_registry import MODALITY_IDS
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CKPT = Path("checkpoints/fusion")
CKPT.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data/processed/synthetic_multimodal")

MODALITIES = ["sensor", "satellite", "microbial", "molecular", "behavioral"]


def load_molecular_model():
    """Load ToxiGene with hierarchy adjacency matrices."""
    from scipy import sparse
    mol_dir = Path("data/processed/molecular")
    gene_names = json.load(open(mol_dir / "gene_names.json"))

    def load_sparse(path):
        d = np.load(path)
        shape = tuple(d["shape"])
        mat = sparse.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=shape)
        return torch.tensor(mat.toarray(), dtype=torch.float32)

    pathway_adj = load_sparse(mol_dir / "hierarchy_layer0_gene_to_pathway.npz")
    process_adj = load_sparse(mol_dir / "hierarchy_layer1_pathway_to_process.npz")
    outcome_adj = load_sparse(mol_dir / "hierarchy_layer2_process_to_outcome.npz")

    model = MolecularEncoder(
        gene_names=gene_names,
        pathway_adj=pathway_adj,
        process_adj=process_adj,
        outcome_adj=outcome_adj,
    )
    ckpt_path = Path("checkpoints/molecular/toxigene_best.pt")
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        logger.info("Loaded ToxiGene checkpoint")
    return model


def extract_embeddings():
    """Extract and cache embeddings from all 5 modality encoders."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Extracting Embeddings from Trained Encoders")
    logger.info("=" * 60)

    cache_path = CKPT / "cached_embeddings.pt"
    if cache_path.exists():
        logger.info(f"Loading cached embeddings from {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    # Load index
    index = json.load(open(DATA_DIR / "multimodal_index.json"))
    n_locations = len(index)
    logger.info(f"Processing {n_locations} multimodal location-events")

    # Load encoders
    encoders = {}

    # Sensor encoder
    sensor_model = SensorEncoder().to(DEVICE)
    sensor_ckpt = Path("checkpoints/sensor/aquassm_final_best.pt")
    if sensor_ckpt.exists():
        state = torch.load(sensor_ckpt, map_location=DEVICE, weights_only=True)
        sensor_model.load_state_dict(state, strict=False)
        logger.info("Loaded AquaSSM checkpoint")
    sensor_model.eval()
    encoders["sensor"] = sensor_model

    # Satellite encoder
    sat_model = SatelliteEncoder().to(DEVICE)
    sat_ckpt = Path("checkpoints/satellite/hydrovit_mae_best.pt")
    if sat_ckpt.exists():
        state = torch.load(sat_ckpt, map_location=DEVICE, weights_only=True)
        sat_model.load_state_dict(state, strict=False)
        logger.info("Loaded HydroViT checkpoint")
    sat_model.eval()
    encoders["satellite"] = sat_model

    # Microbial encoder
    mic_model = MicrobialEncoder(input_dim=5000).to(DEVICE)
    mic_ckpt = Path("checkpoints/microbial/microbiomenet_best.pt")
    if mic_ckpt.exists():
        state = torch.load(mic_ckpt, map_location=DEVICE, weights_only=True)
        mic_model.load_state_dict(state, strict=False)
        logger.info("Loaded MicroBiomeNet checkpoint")
    mic_model.cache_sequence_embeddings(n_otus=5000)
    mic_model.eval()
    encoders["microbial"] = mic_model

    # Molecular encoder
    mol_model = load_molecular_model().to(DEVICE)
    mol_model.eval()
    encoders["molecular"] = mol_model

    # Behavioral encoder
    bio_model = BioMotionEncoder().to(DEVICE)
    bio_ckpt = Path("checkpoints/biomotion/phase2_best.pt")
    if bio_ckpt.exists():
        state = torch.load(bio_ckpt, map_location=DEVICE, weights_only=True)
        bio_model.load_state_dict(state, strict=False)
        logger.info("Loaded BioMotion checkpoint")
    bio_model.eval()
    encoders["behavioral"] = bio_model

    # Extract embeddings
    all_samples = []

    with torch.no_grad():
        for loc_id, loc_data in index.items():
            sample = {"embeddings": {}, "labels": {}}

            # Sensor
            sensor_files = loc_data.get("sensor", [])
            if sensor_files:
                f = DATA_DIR / "sensor" / f"{sensor_files[0]}.npz"
                if f.exists():
                    d = np.load(f, allow_pickle=True)
                    v = torch.tensor(d["values"].astype(np.float32)).unsqueeze(0).to(DEVICE)
                    dt = torch.tensor(d["delta_ts"].astype(np.float32)).unsqueeze(0).to(DEVICE)
                    dt[:, 0] = 0
                    v = v.clamp(-5, 5)
                    dt = dt.clamp(0, 3600)
                    out = encoders["sensor"](v, dt, compute_anomaly=False)
                    emb = out["embedding"][0].cpu()
                    if not torch.isnan(emb).any():
                        sample["embeddings"]["sensor"] = emb
                    sample["labels"]["has_anomaly"] = int(d.get("has_anomaly", 0))

            # Satellite (resize 64->224 via interpolation)
            sat_files = loc_data.get("satellite", [])
            if sat_files:
                f = DATA_DIR / "satellite" / f"{sat_files[0]}.npz"
                if f.exists():
                    d = np.load(f, allow_pickle=True)
                    img = torch.tensor(d["image"].astype(np.float32)).unsqueeze(0)
                    if img.shape[-1] != 224:
                        img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
                    img = img.to(DEVICE)
                    out = encoders["satellite"](img)
                    emb = out["embedding"][0].cpu()
                    if not torch.isnan(emb).any():
                        sample["embeddings"]["satellite"] = emb

            # Microbial
            mic_files = loc_data.get("microbial", [])
            if mic_files:
                f = DATA_DIR / "microbial" / f"{mic_files[0]}.npz"
                if f.exists():
                    d = np.load(f, allow_pickle=True)
                    abundances = torch.tensor(d["abundances"].astype(np.float32)).unsqueeze(0).to(DEVICE)
                    clr = clr_transform(abundances + 1e-10)
                    out = encoders["microbial"](x=clr)
                    emb = out["embedding"][0].cpu()
                    if not torch.isnan(emb).any():
                        sample["embeddings"]["microbial"] = emb
                    sample["labels"]["source_label"] = int(d.get("source_label", 0))
                    sample["labels"]["source_name"] = str(d.get("source_name", ""))

            # Molecular
            mol_files = loc_data.get("molecular", [])
            if mol_files:
                f = DATA_DIR / "molecular" / f"{mol_files[0]}.npz"
                if f.exists():
                    d = np.load(f, allow_pickle=True)
                    expr = torch.tensor(d["expression"].astype(np.float32)).unsqueeze(0).to(DEVICE)
                    out = encoders["molecular"](gene_expression=expr)
                    emb = out["embedding"][0].cpu()
                    if not torch.isnan(emb).any():
                        sample["embeddings"]["molecular"] = emb

            # Behavioral
            beh_files = loc_data.get("behavioral", [])
            if beh_files:
                f = DATA_DIR / "behavioral" / f"{beh_files[0]}.npz"
                if f.exists():
                    d = np.load(f, allow_pickle=True)
                    features = torch.tensor(d["features"].astype(np.float32)).unsqueeze(0).to(DEVICE)
                    out = encoders["behavioral"](features)
                    emb = out["embedding"][0].cpu()
                    if not torch.isnan(emb).any():
                        sample["embeddings"]["behavioral"] = emb
                    if "is_anomaly" not in sample["labels"]:
                        sample["labels"]["has_anomaly"] = int(d.get("is_anomaly", 0))

            if sample["embeddings"]:
                all_samples.append(sample)

            if (int(loc_id) + 1) % 20 == 0:
                logger.info(f"  Extracted {int(loc_id)+1}/{n_locations} locations")

    logger.info(f"Total samples with embeddings: {len(all_samples)}")
    mod_counts = {m: sum(1 for s in all_samples if m in s["embeddings"]) for m in MODALITIES}
    logger.info(f"Modality coverage: {mod_counts}")

    torch.save(all_samples, cache_path)
    return all_samples


class FusionDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "embeddings": s["embeddings"],
            "has_anomaly": s["labels"].get("has_anomaly", 0),
            "source_label": s["labels"].get("source_label", 0),
        }


def collate_fn(batch):
    """Custom collate for variable-modality samples."""
    has_anomaly = torch.tensor([b["has_anomaly"] for b in batch], dtype=torch.float32)
    source_label = torch.tensor([b["source_label"] for b in batch], dtype=torch.long)
    embeddings = [b["embeddings"] for b in batch]
    return {"embeddings": embeddings, "has_anomaly": has_anomaly, "source_label": source_label}


def train_fusion(samples, epochs=100, lr=1e-3):
    """Train the Perceiver IO fusion layer."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Fusion Training")
    logger.info("=" * 60)

    ds = FusionDataset(samples)
    n = len(ds)
    n_tr = int(0.7 * n)
    n_va = int(0.15 * n)
    n_te = n - n_tr - n_va
    tr, va, te = random_split(ds, [n_tr, n_va, n_te], generator=torch.Generator().manual_seed(42))

    tr_dl = DataLoader(tr, batch_size=8, shuffle=True, collate_fn=collate_fn)
    va_dl = DataLoader(va, batch_size=8, collate_fn=collate_fn)
    te_dl = DataLoader(te, batch_size=8, collate_fn=collate_fn)

    logger.info(f"Split: {n_tr}/{n_va}/{n_te}")

    # Build fusion model + heads
    fusion = PerceiverIOFusion(
        shared_dim=256,
        num_latents=64,  # Smaller for this dataset size
        num_heads=8,
        num_process_layers=2,
        dropout=0.1,
    ).to(DEVICE)

    anomaly_head = AnomalyDetectionHead(input_dim=256).to(DEVICE)
    source_head = SourceAttributionHead(input_dim=256).to(DEVICE)

    all_params = list(fusion.parameters()) + list(anomaly_head.parameters()) + list(source_head.parameters())
    n_params = sum(p.numel() for p in all_params)
    logger.info(f"Fusion + heads: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_auc = 0.0

    for epoch in range(epochs):
        fusion.train()
        anomaly_head.train()
        source_head.train()
        total_loss, nb = 0, 0
        all_preds, all_labels = [], []

        for batch in tr_dl:
            B = len(batch["embeddings"])
            ha = batch["has_anomaly"].to(DEVICE)
            sl = batch["source_label"].to(DEVICE)

            # Process each sample through fusion sequentially
            fused_states = []
            for i in range(B):
                embs = batch["embeddings"][i]
                fusion.reset_registry()
                latent = None
                t = 0.0

                for mod in MODALITIES:
                    if mod in embs:
                        out = fusion(
                            modality_id=mod,
                            raw_embedding=embs[mod].to(DEVICE),
                            timestamp=t,
                            confidence=0.9,
                            latent_state=latent,
                        )
                        latent = out.latent_state
                        t += 3600.0

                if latent is not None:
                    fused_states.append(out.fused_state)
                else:
                    fused_states.append(torch.zeros(1, 256, device=DEVICE))

            fused = torch.cat(fused_states, dim=0)

            # Compute losses
            anomaly_out = anomaly_head(fused)
            source_out = source_head(fused)

            anomaly_loss = F.binary_cross_entropy_with_logits(
                anomaly_out["severity_score"].squeeze(-1), ha
            )
            source_loss = F.cross_entropy(source_out["source_logits"], sl)
            loss = anomaly_loss + 0.5 * source_loss

            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            total_loss += loss.item()
            nb += 1

            preds = (torch.sigmoid(anomaly_out["severity_score"].squeeze(-1)) > 0.5).float().cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(ha.cpu().tolist())

        scheduler.step()

        # Validation
        fusion.eval()
        anomaly_head.eval()
        source_head.eval()
        va_preds, va_labels = [], []
        with torch.no_grad():
            for batch in va_dl:
                B = len(batch["embeddings"])
                ha = batch["has_anomaly"]

                for i in range(B):
                    embs = batch["embeddings"][i]
                    fusion.reset_registry()
                    latent = None
                    t = 0.0
                    for mod in MODALITIES:
                        if mod in embs:
                            out = fusion(
                                modality_id=mod,
                                raw_embedding=embs[mod].to(DEVICE),
                                timestamp=t,
                                confidence=0.9,
                                latent_state=latent,
                            )
                            latent = out.latent_state
                            t += 3600.0
                    if latent is not None:
                        a_out = anomaly_head(out.fused_state)
                        pred = torch.sigmoid(a_out["severity_score"]).item()
                        va_preds.append(pred)
                        va_labels.append(ha[i].item())

        try:
            va_auc = roc_auc_score(va_labels, va_preds)
        except:
            va_auc = 0.5

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Ep {epoch+1:3d}/{epochs} | Loss: {total_loss/max(nb,1):.4f} | "
                f"Val AUC: {va_auc:.4f}"
            )

        if va_auc > best_auc:
            best_auc = va_auc
            torch.save({
                "fusion": fusion.state_dict(),
                "anomaly_head": anomaly_head.state_dict(),
                "source_head": source_head.state_dict(),
            }, CKPT / "fusion_best.pt")

    return best_auc, fusion, anomaly_head, source_head, te_dl


def main():
    t0 = time.time()

    # Phase 1: Extract embeddings
    samples = extract_embeddings()

    # Phase 2: Train fusion
    best_auc, fusion, anomaly_head, source_head, te_dl = train_fusion(
        samples, epochs=100, lr=1e-3
    )

    # Reload best and test
    best_path = CKPT / "fusion_best.pt"
    if best_path.exists():
        state = torch.load(best_path, map_location=DEVICE, weights_only=True)
        fusion.load_state_dict(state["fusion"])
        anomaly_head.load_state_dict(state["anomaly_head"])
        source_head.load_state_dict(state["source_head"])

    fusion.eval()
    anomaly_head.eval()
    source_head.eval()

    te_preds, te_labels, te_source_preds, te_source_labels = [], [], [], []
    with torch.no_grad():
        for batch in te_dl:
            B = len(batch["embeddings"])
            for i in range(B):
                embs = batch["embeddings"][i]
                fusion.reset_registry()
                latent = None
                t = 0.0
                for mod in MODALITIES:
                    if mod in embs:
                        out = fusion(
                            modality_id=mod,
                            raw_embedding=embs[mod].to(DEVICE),
                            timestamp=t,
                            confidence=0.9,
                            latent_state=latent,
                        )
                        latent = out.latent_state
                        t += 3600.0
                if latent is not None:
                    a_out = anomaly_head(out.fused_state)
                    s_out = source_head(out.fused_state)
                    te_preds.append(torch.sigmoid(a_out["severity_score"]).item())
                    te_labels.append(batch["has_anomaly"][i].item())
                    te_source_preds.append(s_out["source_logits"].argmax(dim=-1).item())
                    te_source_labels.append(batch["source_label"][i].item())

    try:
        test_auc = roc_auc_score(te_labels, te_preds)
    except:
        test_auc = 0.5
    test_f1 = f1_score(te_labels, [1 if p > 0.5 else 0 for p in te_preds], zero_division=0)
    source_f1 = f1_score(te_source_labels, te_source_preds, average="macro", zero_division=0)

    logger.info("=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Anomaly AUROC: {test_auc:.4f}")
    logger.info(f"  Anomaly F1: {test_f1:.4f}")
    logger.info(f"  Source F1: {source_f1:.4f}")
    logger.info(f"  Best Val AUC: {best_auc:.4f}")

    elapsed = time.time() - t0
    results = {
        "test_anomaly_auroc": float(test_auc),
        "test_anomaly_f1": float(test_f1),
        "test_source_f1": float(source_f1),
        "best_val_auc": float(best_auc),
        "elapsed": elapsed,
        "n_samples": len(te_labels),
    }
    with open(CKPT / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Time: {elapsed/60:.1f}m")
    logger.info("DONE")


if __name__ == "__main__":
    main()
