#!/usr/bin/env python3
"""
exp_molecular_case_studies.py — SENTINEL ToxiGene Case Studies

Runs ToxiGene v7 on GEO zebrafish RNA-seq samples from 5 real contamination
exposure studies. ToxiGene predicts 7 adverse outcome pathways for each sample
and a binary toxicity flag.

Model architecture (toxigene_v7_best.pt):
  - Backbone: Linear(61479→512, BN, ReLU, dropout) → Linear(512→256, BN, ReLU, dropout)
  - Outcome head: Linear(256→7)   — 7 adverse pathway outcomes (multi-label)
  - Pathway head: Linear(256→128, ReLU) → Linear(128→200)  — 200 pathway targets

The 7 adverse outcomes:
  reproductive_impairment, growth_inhibition, immunosuppression,
  neurotoxicity, hepatotoxicity, oxidative_damage, endocrine_disruption

Positive toxicity = any outcome predicted positive (at per-class threshold).

5 Case Studies (real GEO datasets):
  1. GSE109496 — Naproxen (NSAID pharmaceutical) in zebrafish embryo
  2. GSE117260 — Atrazine (herbicide/pesticide) developmental neurotoxicity
  3. GSE3048    — Arsenic (heavy metal) transcriptome kinetics
  4. GSE50648   — Metal exposures (Cd, Zn, Cu, Pb) whole organism
  5. GSE66257   — Dichlorvos (organophosphate pesticide) energy metabolism disruption

Author: Bryan Cheng, SENTINEL project, 2026-04-14
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "results" / "case_studies_modality"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "molecular"
CKPT_PATH = PROJECT_ROOT / "checkpoints" / "molecular" / "toxigene_v7_best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTCOME_NAMES = [
    "reproductive_impairment",
    "growth_inhibition",
    "immunosuppression",
    "neurotoxicity",
    "hepatotoxicity",
    "oxidative_damage",
    "endocrine_disruption",
]

# Per-class optimal thresholds from training
DEFAULT_THRESHOLDS = [0.3, 0.2, 0.6, 0.675, 0.6, 0.7, 0.55]

# ─────────────────────────────────────────────────────────────────────────────
# Case study definitions
# ─────────────────────────────────────────────────────────────────────────────
CASE_STUDIES = [
    {
        "study_id": "GSE109496",
        "contaminant_type": "pharmaceutical_NSAID",
        "contaminant_name": "Naproxen",
        "contaminant_class": "pharmaceutical",
        "exposure_route": "waterborne, zebrafish embryo 24–72 hpf",
        "concentration_range": "0.3–309 µmol/L",
        "environmental_relevance": "NSAID detected at 1–10 µg/L in surface waters receiving wastewater effluent",
        "reference": "GEO GSE109496 — time/concentration-dependent transcriptome ZFE",
    },
    {
        "study_id": "GSE117260",
        "contaminant_type": "herbicide_pesticide",
        "contaminant_name": "Atrazine",
        "contaminant_class": "pesticide",
        "exposure_route": "developmental, embryogenesis exposure, adult neurotoxicity",
        "concentration_range": "0.1–100 µg/L",
        "environmental_relevance": "Most commonly detected herbicide in US surface water; EPA MCL 3 µg/L",
        "reference": "GEO GSE117260 — developmental origins of neurotoxicity",
    },
    {
        "study_id": "GSE3048",
        "contaminant_type": "heavy_metal_metalloid",
        "contaminant_name": "Arsenic (As(III))",
        "contaminant_class": "heavy_metal",
        "exposure_route": "waterborne, adult zebrafish liver",
        "concentration_range": "1–100 µg/L",
        "environmental_relevance": "Naturally elevated As in groundwater; WHO guideline 10 µg/L; Bangladesh/India crisis",
        "reference": "GEO GSE3048 — arsenic-induced adaptive response transcriptome kinetics",
    },
    {
        "study_id": "GSE50648",
        "contaminant_type": "multi_metal_exposure",
        "contaminant_name": "Heavy metals (Cd, Zn, Cu, Pb)",
        "contaminant_class": "heavy_metal",
        "exposure_route": "waterborne, whole adult zebrafish, chronic",
        "concentration_range": "0.1–10× LC50",
        "environmental_relevance": "Industrial effluent and mining drainage; common co-contaminants in impaired watersheds",
        "reference": "GEO GSE50648 — whole-organism metal transcriptome profiling",
    },
    {
        "study_id": "GSE66257",
        "contaminant_type": "organophosphate_pesticide",
        "contaminant_name": "Dichlorvos (DDVP)",
        "contaminant_class": "pesticide",
        "exposure_route": "waterborne, zebrafish, energy metabolism disruption",
        "concentration_range": "1–100 µg/L",
        "environmental_relevance": "Organophosphate insecticide; AChE inhibitor; detected in agricultural runoff",
        "reference": "GEO GSE66257 — large-scale energy metabolism disruption",
    },
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# ToxiGene v7 model (matches checkpoint architecture)
# ─────────────────────────────────────────────────────────────────────────────

class ToxiGeneV7(nn.Module):
    """Minimal ToxiGene v7 model matching the toxigene_v7_best.pt checkpoint."""

    def __init__(
        self,
        input_dim: int = 61479,
        hidden1: int = 512,
        hidden2: int = 256,
        n_outcomes: int = 7,
        n_pathways: int = 200,
        pathway_hidden: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.outcome_head = nn.Linear(hidden2, n_outcomes)
        self.pathway_head = nn.Sequential(
            nn.Linear(hidden2, pathway_hidden),
            nn.ReLU(),
            nn.Linear(pathway_hidden, n_pathways),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        embed = self.backbone(x)
        return {
            "outcome_logits": self.outcome_head(embed),
            "pathway_logits": self.pathway_head(embed),
            "embedding": embed,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Load expression matrix, outcomes, and GEO metadata."""
    log("Loading expression matrix (v2_expanded) ...")
    expr = np.load(DATA_DIR / "expression_matrix_v2_expanded.npy")
    labels = np.load(DATA_DIR / "outcome_labels_v2_expanded.npy")
    log(f"  Expression: {expr.shape}, Labels: {labels.shape}")

    # Normalize (z-score per gene, clipped)
    mean = expr.mean(axis=0)
    std = expr.std(axis=0)
    std[std < 1e-6] = 1.0
    expr_norm = np.clip((expr - mean) / std, -6, 6).astype(np.float32)

    # Load GEO metadata
    with open(DATA_DIR / "geo_zebrafish_metadata.json") as f:
        geo_meta = json.load(f)
    log(f"  GEO metadata: {len(geo_meta)} samples")

    return expr_norm, labels.astype(np.float32), geo_meta


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_toxigene() -> tuple[ToxiGeneV7, list[float]]:
    """Load ToxiGene v7 checkpoint."""
    log(f"Loading ToxiGene v7 from {CKPT_PATH} ...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    thresholds = ckpt.get("thresholds", DEFAULT_THRESHOLDS)
    state = ckpt.get("state_dict", ckpt)

    model = ToxiGeneV7(
        input_dim=61479,
        hidden1=512,
        hidden2=256,
        n_outcomes=7,
        n_pathways=200,
        pathway_hidden=128,
        dropout=0.3,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(DEVICE)
    log("  ToxiGene v7 loaded OK")
    return model, list(thresholds)


# ─────────────────────────────────────────────────────────────────────────────
# Case study inference
# ─────────────────────────────────────────────────────────────────────────────

def run_case_study(
    model: ToxiGeneV7,
    thresholds: list[float],
    expr_norm: np.ndarray,
    labels: np.ndarray,
    geo_meta: list[dict],
    case: dict,
) -> dict:
    """Run ToxiGene on all samples from a given GEO study."""
    gse_id = case["study_id"]

    # Match GEO metadata rows to their position in the expression matrix
    # geo_meta has 697 rows (GEO subset), expr_norm has 1697 rows (full v2_expanded)
    # We use the last len(geo_meta) rows as the GEO portion
    geo_offset = len(expr_norm) - len(geo_meta)
    matched = [
        (i, geo_meta[i])
        for i in range(len(geo_meta))
        if geo_meta[i].get("gse") == gse_id
    ]

    log(f"  {gse_id}: {len(matched)} samples in GEO metadata")

    if not matched:
        return {"study_id": gse_id, "error": "no_samples_found_in_metadata"}

    # Get expression and label arrays
    indices = [geo_offset + i for i, _ in matched]
    X = torch.tensor(expr_norm[indices], dtype=torch.float32).to(DEVICE)
    y = labels[indices]  # shape (n, 7)

    # Run ToxiGene inference in batches
    all_probs = []
    batch_size = 64
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = X[start : start + batch_size]
            out = model(batch)
            probs = torch.sigmoid(out["outcome_logits"]).cpu().numpy()
            all_probs.append(probs)
    all_probs = np.vstack(all_probs)  # (n, 7)

    # Apply per-class thresholds
    preds = np.zeros_like(all_probs)
    for j, thr in enumerate(thresholds[:all_probs.shape[1]]):
        preds[:, j] = (all_probs[:, j] >= thr).astype(float)

    # Binary toxicity = any outcome positive
    toxic_mask = preds.any(axis=1)
    toxicity_positive_rate = float(toxic_mask.mean())
    mean_toxicity_score = float(all_probs.max(axis=1).mean())

    # Per-outcome stats
    per_outcome = {}
    for j, name in enumerate(OUTCOME_NAMES):
        per_outcome[name] = {
            "positive_rate": float(preds[:, j].mean()),
            "mean_probability": float(all_probs[:, j].mean()),
            "label_positive_rate": float(y[:, j].mean()) if y.shape[1] > j else 0.0,
        }

    # Representative sample details (first 5)
    sample_details = []
    for k, (i, meta_row) in enumerate(matched[:5]):
        characteristics = meta_row.get("characteristics", [])
        concentration_str = ""
        for c in characteristics:
            if "micromol" in c.lower() or "mg" in c.lower() or "µg" in c.lower() or "ug" in c.lower():
                concentration_str = c[:80]
                break
        outcomes_positive = [
            OUTCOME_NAMES[j]
            for j in range(7)
            if j < all_probs.shape[1] and preds[k, j] > 0
        ]
        sample_details.append({
            "sample_id": meta_row.get("sample_id", f"idx_{i}"),
            "title": meta_row.get("title", ""),
            "concentration_info": concentration_str,
            "toxicity_positive": bool(toxic_mask[k]),
            "max_outcome_prob": float(all_probs[k].max()),
            "outcomes_triggered": outcomes_positive,
        })

    return {
        "study_id": gse_id,
        "contaminant_type": case["contaminant_type"],
        "contaminant_name": case["contaminant_name"],
        "contaminant_class": case["contaminant_class"],
        "environmental_relevance": case["environmental_relevance"],
        "n_samples": len(matched),
        "toxicity_positive_rate": toxicity_positive_rate,
        "mean_toxicity_score": mean_toxicity_score,
        "per_outcome_stats": per_outcome,
        "sample_details": sample_details,
        "reference": case["reference"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log("=" * 65)
    log("SENTINEL ToxiGene Case Studies — 5 GEO Contamination Studies")
    log("=" * 65)

    # 1. Load data and model
    expr_norm, labels, geo_meta = load_data()
    model, thresholds = load_toxigene()

    log(f"Thresholds: {dict(zip(OUTCOME_NAMES, thresholds))}")

    # 2. Run case studies
    results = []
    for case in CASE_STUDIES:
        log(f"\nCase study: {case['study_id']} — {case['contaminant_name']}")
        result = run_case_study(model, thresholds, expr_norm, labels, geo_meta, case)
        results.append(result)
        if "error" not in result:
            log(
                f"  n={result['n_samples']}, tox_positive={result['toxicity_positive_rate']:.1%}, "
                f"mean_score={result['mean_toxicity_score']:.3f}"
            )

    # 3. Save results
    output = {
        "model": "ToxiGene v7 (toxigene_v7_best.pt)",
        "published_test_f1_macro": 0.886,
        "outcome_names": OUTCOME_NAMES,
        "per_class_thresholds": dict(zip(OUTCOME_NAMES, thresholds)),
        "case_studies": results,
    }
    out_path = OUTPUT_DIR / "molecular_case_studies.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log(f"\nSaved to {out_path}")

    # Summary
    log("\n" + "=" * 65)
    log("SUMMARY — ToxiGene Case Studies")
    log("=" * 65)
    for r in results:
        if "error" in r:
            log(f"  {r['study_id']:15s}  ERROR: {r['error']}")
        else:
            top_outcomes = sorted(
                r["per_outcome_stats"].items(),
                key=lambda x: x[1]["positive_rate"],
                reverse=True,
            )[:2]
            top_str = ", ".join(f"{k}={v['positive_rate']:.0%}" for k, v in top_outcomes)
            log(
                f"  {r['study_id']:15s}  {r['contaminant_name']:25s}  "
                f"n={r['n_samples']:4d}  tox={r['toxicity_positive_rate']:.1%}  "
                f"top: {top_str}"
            )
    log("Done.")


if __name__ == "__main__":
    main()
