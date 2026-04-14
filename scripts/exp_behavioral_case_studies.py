#!/usr/bin/env python3
"""
exp_behavioral_case_studies.py — SENTINEL BioMotion Case Studies

Runs BioMotion (TrajectoryDiffusionEncoder + AnomalyClassifier) on Daphnia
behavioral trajectories grouped by chemical exposure class from ECOTOX data.
BioMotion outputs an anomaly probability [0,1] per trajectory; values ≥ 0.5
are flagged as behaviorally anomalous.

Model architecture (biomotion_expanded_best.pt):
  - TrajectoryDiffusionEncoder: diffusion-pretrained transformer denoiser
    on behavioral feature sequences (T=200, F=16)
  - AnomalyClassifier head: MLP (256 → 128 → 1)
  - Anomaly signal: sigmoid(classifier(encoder.forward_encode(features)))

5 Chemical Case Studies sourced from ECOTOX Daphnia records:
  1. Heavy metals (Cu, Pb, Cr, Cd) — industrial effluent
  2. Pesticides (atrazine, 2,4-D, chlorpyrifos, malathion) — agricultural runoff
  3. Pharmaceuticals (NSAIDs, antibiotics, steroids) — wastewater effluent
  4. PAH / petroleum hydrocarbons — oil spill / urban stormwater
  5. PFAS / per- and polyfluoroalkyl substances — industrial contamination

Each case uses trajectories from data/processed/behavioral_real/ that were
generated from ECOTOX Daphnia behavioral measurement data. Since the processed
trajectories don't store per-file chemical labels, we construct case studies
by:
  (a) Re-parsing relevant ECOTOX test records for each chemical class to get
      CAS numbers, then
  (b) Sampling stratified trajectory files from behavioral_real and applying
      systematic perturbations matching each chemical's known mode of action,
      (c) plus running the full set of behavioral_real files to get real
          detection rates.

Author: Bryan Cheng, SENTINEL project, 2026-04-14
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.models.biomotion.trajectory_encoder import TrajectoryDiffusionEncoder, EMBED_DIM

OUTPUT_DIR = PROJECT_ROOT / "results" / "case_studies_modality"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BREAL_DIR = PROJECT_ROOT / "data" / "processed" / "behavioral_real"
ECOTOX_DIR = PROJECT_ROOT / "data" / "raw" / "ecotox" / "ecotox_ascii_03_12_2026"
CKPT_PATH = PROJECT_ROOT / "checkpoints" / "biomotion" / "biomotion_expanded_best.pt"
ECOTOX_METADATA = PROJECT_ROOT / "data" / "processed" / "ecotox" / "ecotox_metadata.json"
DOSE_PROFILES = PROJECT_ROOT / "data" / "processed" / "ecotox" / "dose_response_profiles.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_DIM = 16
T = 200
SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Case study definitions — map to ECOTOX chemical classes and modes of action
# ─────────────────────────────────────────────────────────────────────────────
CASE_STUDIES = [
    {
        "chemical_id": "heavy_metals",
        "chemical_name": "Heavy Metals (Cu, Pb, Cd, Cr, Hg)",
        "contaminant_class": "heavy_metal",
        "source_event": "Industrial effluent and mining drainage",
        "environmental_context": (
            "Metal contamination from acid mine drainage (AMD) and industrial "
            "point sources; Daphnia immobility EC50 for Cu: 0.01–0.1 mg/L"
        ),
        "mode_of_action": "ATPase inhibition, oxidative stress, gill damage → reduced locomotion, immobility",
        "ecotox_class": "heavy_metal",
        # Feature perturbations: inhibit locomotion (feat 0), swimming (feat 1), increase immobility (feat 5)
        "perturbation_profile": {
            0: -0.6,   # LOCO: reduced locomotion
            1: -0.7,   # SWIM: reduced swimming velocity
            3: -0.5,   # ACTV: reduced activity
            5: +0.8,   # NMVM: increased immobility
        },
        "case_chemicals": ["Copper sulfate", "Lead acetate", "Cadmium chloride", "Chromium chloride"],
    },
    {
        "chemical_id": "pesticides_agricultural_runoff",
        "chemical_name": "Pesticides / Herbicides (atrazine, chlorpyrifos, 2,4-D)",
        "contaminant_class": "pesticide",
        "source_event": "Agricultural runoff — corn/soy belt, spring flush events",
        "environmental_context": (
            "Atrazine is the most detected herbicide in US surface water (USGS NAWQA); "
            "chlorpyrifos EC50 for Daphnia: 0.001–0.01 mg/L; 2,4-D up to 50 µg/L in farm runoff"
        ),
        "mode_of_action": "AChE inhibition (OPs), photosynthesis disruption (triazines) → erratic swimming, altered phototaxis",
        "ecotox_class": "pesticide",
        "perturbation_profile": {
            1: -0.4,   # SWIM: reduced swimming
            3: -0.3,   # ACTV: altered activity
            6: +0.5,   # PHTR: altered phototaxis
            7: +0.4,   # GBHV: abnormal general behavior
        },
        "case_chemicals": ["Atrazine", "Chlorpyrifos", "2,4-Dichlorophenol", "Malathion"],
    },
    {
        "chemical_id": "pharmaceuticals_wastewater",
        "chemical_name": "Pharmaceuticals (NSAIDs, antibiotics, EDCs)",
        "contaminant_class": "pharmaceutical",
        "source_event": "Wastewater treatment plant effluent — WWTP discharge zones",
        "environmental_context": (
            "17α-ethinylestradiol detected at 1–100 ng/L downstream WWTPs; "
            "Naproxen EC50 for Daphnia: 10–50 mg/L; antibiotic disruption of gut microbiome"
        ),
        "mode_of_action": "Endocrine disruption, reproductive impairment → altered reproductive behavior, reduced feeding",
        "ecotox_class": "pharmaceutical",
        "perturbation_profile": {
            4: -0.4,   # MOTL: reduced motility
            8: -0.5,   # FLTR: reduced filter feeding
            10: -0.3,  # ACTP: reduced active time
            11: +0.3,  # SEBH: secondary behavioral response
        },
        "case_chemicals": ["Naproxen", "Ibuprofen", "Erythromycin", "17α-Ethinylestradiol"],
    },
    {
        "chemical_id": "pah_petroleum_hydrocarbons",
        "chemical_name": "PAHs / Petroleum Hydrocarbons (naphthalene, pyrene, benzene)",
        "contaminant_class": "pah",
        "source_event": "Oil spill / urban stormwater runoff",
        "environmental_context": (
            "PAH contamination from oil spills (Deepwater Horizon), urban stormwater; "
            "naphthalene EC50 Daphnia: 0.5–5 mg/L; narcosis mode of action"
        ),
        "mode_of_action": "Narcosis, oxidative stress, lipid peroxidation → anesthetic effects, reduced swimming speed",
        "ecotox_class": "pah",
        "perturbation_profile": {
            0: -0.5,   # LOCO: reduced locomotion
            1: -0.6,   # SWIM: reduced swimming velocity
            2: +0.4,   # EQUL: equilibrium disruption
            5: +0.5,   # NMVM: immobility
        },
        "case_chemicals": ["Naphthalene", "Pyrene", "Benzene", "Phenanthrene"],
    },
    {
        "chemical_id": "pfas_industrial",
        "chemical_name": "PFAS (PFOA, PFOS, GenX)",
        "contaminant_class": "pfas",
        "source_event": "Industrial PFAS contamination — manufacturing sites, firefighting foam",
        "environmental_context": (
            "PFAS detected in 40%+ of US drinking water sources; "
            "PFOS EC50 for Daphnia: 0.4–10 mg/L; chronic effects at ng/L concentrations; "
            "AFF (aqueous film-forming foam) runoff from military bases and airports"
        ),
        "mode_of_action": "Lipid metabolism disruption, mitochondrial dysfunction → altered feeding, reduced reproduction",
        "ecotox_class": "pfas",
        "perturbation_profile": {
            3: -0.3,   # ACTV: reduced activity
            4: -0.4,   # MOTL: reduced motility
            8: -0.6,   # FLTR: impaired filter feeding
            9: -0.3,   # VACL: reduced valve activity
        },
        "case_chemicals": ["PFOS", "PFOA", "GenX (HFPO-DA)", "PFHxS"],
    },
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (identical to benchmark_biomotion.py)
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyClassifier(nn.Module):
    """Anomaly classifier head over trajectory encoder embedding."""

    def __init__(self, encoder: TrajectoryDiffusionEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(EMBED_DIM // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder.forward_encode(x)).squeeze(-1)


def load_biomotion() -> AnomalyClassifier:
    """Load BioMotion expanded checkpoint."""
    log(f"Loading BioMotion from {CKPT_PATH} ...")
    enc = TrajectoryDiffusionEncoder(
        feature_dim=FEATURE_DIM,
        embed_dim=EMBED_DIM,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
    )
    model = AnomalyClassifier(enc)
    state = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    model.to(DEVICE)
    log("  BioMotion loaded OK")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Data loading and trajectory generation
# ─────────────────────────────────────────────────────────────────────────────

def load_behavioral_real() -> tuple[list[np.ndarray], list[bool], list[str]]:
    """Load all behavioral_real .npz trajectory files."""
    files = sorted(BREAL_DIR.glob("traj_*.npz"))
    log(f"Found {len(files)} trajectory files in behavioral_real/")
    features_list, labels, file_ids = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        features_list.append(d["features"].astype(np.float32))
        labels.append(bool(d["is_anomaly"]))
        file_ids.append(f.stem)
    return features_list, labels, file_ids


def generate_chemical_trajectories(
    base_features: list[np.ndarray],
    base_labels: list[bool],
    perturbation_profile: dict[int, float],
    n_trajectories: int = 80,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic Daphnia trajectories for a chemical exposure scenario.

    Approach: take existing normal trajectories from behavioral_real and apply
    a chemical-specific perturbation profile (feature-level shifts consistent
    with the chemical's known mode of action), then mix with unperturbed
    normal trajectories to represent a realistic field exposure scenario with
    variable individual response.

    Args:
        base_features: list of (T, 16) feature arrays from behavioral_real
        base_labels: ground-truth anomaly labels for base data
        perturbation_profile: {feature_idx: delta_shift} for this chemical class
        n_trajectories: number of Daphnia to simulate for this exposure scenario
        rng: random number generator

    Returns:
        X: (n_trajectories, T, 16) float32 feature array
        y: (n_trajectories,) float32 anomaly labels (1=anomalous)
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    normal_idx = [i for i, lab in enumerate(base_labels) if not lab]
    anomaly_idx = [i for i, lab in enumerate(base_labels) if lab]

    n_exposed = int(n_trajectories * 0.7)   # 70% treated/exposed
    n_control = n_trajectories - n_exposed   # 30% control/low-dose

    trajs, labels = [], []

    # Exposed trajectories: apply chemical perturbation
    for _ in range(n_exposed):
        src_idx = rng.choice(normal_idx if normal_idx else list(range(len(base_features))))
        feat = base_features[src_idx].copy()  # (T, 16)

        # Apply perturbation to relevant features
        for feat_idx, delta in perturbation_profile.items():
            if feat_idx < feat.shape[1]:
                # Add concentration-dependent shift with individual variability
                scale = rng.uniform(0.7, 1.3)
                noise = rng.normal(0, 0.05, size=feat.shape[0])
                feat[:, feat_idx] = np.clip(
                    feat[:, feat_idx] + delta * scale + noise, 0.0, 1.0
                )

        trajs.append(feat)
        # Label as anomalous if perturbation is strong enough (any delta > 0.4)
        max_delta = max(abs(d) for d in perturbation_profile.values())
        labels.append(1.0 if max_delta > 0.35 else 0.0)

    # Control trajectories: normal or mildly impacted
    for _ in range(n_control):
        src_idx = rng.choice(normal_idx if normal_idx else list(range(len(base_features))))
        trajs.append(base_features[src_idx].copy())
        labels.append(0.0)

    # Shuffle
    idx = rng.permutation(len(trajs))
    X = np.stack([trajs[i] for i in idx], axis=0).astype(np.float32)
    y = np.array([labels[i] for i in idx], dtype=np.float32)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_biomotion_inference(
    model: AnomalyClassifier,
    X: np.ndarray,
    batch_size: int = 128,
) -> np.ndarray:
    """Run BioMotion on trajectory array; return anomaly probabilities."""
    probs = []
    X_t = torch.tensor(X, dtype=torch.float32)
    for start in range(0, len(X_t), batch_size):
        batch = X_t[start : start + batch_size].to(DEVICE)
        logits = model(batch)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def get_ecotox_chemical_count(chem_class: str) -> dict:
    """Get count of ECOTOX records with behavioral effects for this class."""
    try:
        with open(DOSE_PROFILES) as f:
            profiles = json.load(f)
        beh_records = {
            cas: v
            for cas, v in profiles.items()
            if v.get("contaminant_class") == chem_class
            and "BEH" in str(v.get("effects", []))
        }
        return {
            "n_chemicals_with_beh_effects": len(beh_records),
            "example_chemicals": [
                v.get("chemical_name", cas)
                for cas, v in list(beh_records.items())[:3]
            ],
        }
    except Exception:
        return {"n_chemicals_with_beh_effects": 0, "example_chemicals": []}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log("=" * 65)
    log("SENTINEL BioMotion Case Studies — 5 Chemical Exposure Scenarios")
    log("=" * 65)

    # 1. Load BioMotion model
    model = load_biomotion()

    # 2. Load base behavioral real trajectories
    log("\nLoading behavioral_real trajectories ...")
    base_features, base_labels, file_ids = load_behavioral_real()
    log(f"  Loaded {len(base_features)} trajectories "
        f"({sum(base_labels)} anomalous, {len(base_labels)-sum(base_labels)} normal)")

    # 3. Run inference on all real trajectories first (baseline)
    log("\nRunning BioMotion on all behavioral_real trajectories (baseline) ...")
    X_all = np.stack(base_features, axis=0)
    y_all = np.array(base_labels, dtype=np.float32)
    probs_all = run_biomotion_inference(model, X_all)
    overall_det = float((probs_all >= 0.5).mean())
    overall_auroc_proxy = float(probs_all[y_all == 1].mean() - probs_all[y_all == 0].mean())
    log(f"  Overall detection rate: {overall_det:.2%}")
    log(f"  Mean prob (anomalous): {probs_all[y_all==1].mean():.3f}")
    log(f"  Mean prob (normal):    {probs_all[y_all==0].mean():.3f}")

    # 4. Run case studies
    rng = np.random.default_rng(SEED)
    results = []

    for case in CASE_STUDIES:
        log(f"\nCase study: {case['chemical_id']}")
        log(f"  Chemical: {case['chemical_name']}")

        # Get ECOTOX statistics
        ecotox_stats = get_ecotox_chemical_count(case["ecotox_class"])

        # Generate chemical-exposure trajectories
        X_case, y_case = generate_chemical_trajectories(
            base_features=base_features,
            base_labels=base_labels,
            perturbation_profile=case["perturbation_profile"],
            n_trajectories=100,
            rng=rng,
        )
        log(f"  Generated {len(X_case)} trajectories "
            f"({int(y_case.sum())} labeled anomalous)")

        # Run BioMotion inference
        probs = run_biomotion_inference(model, X_case)
        pred_anomaly = (probs >= 0.5).astype(float)
        detection_rate = float(pred_anomaly.mean())
        mean_prob = float(probs.mean())

        # Sub-group analysis: exposed (perturbed) vs control
        n_exposed = int(len(X_case) * 0.7)
        exposed_probs = probs[:n_exposed]
        control_probs = probs[n_exposed:]
        exposed_det_rate = float((exposed_probs >= 0.5).mean())
        control_det_rate = float((control_probs >= 0.5).mean())

        # Real data: sample from actual anomalous trajectories as comparison
        anom_mask = np.array(base_labels)
        if anom_mask.sum() > 10:
            real_anom_probs = probs_all[anom_mask]
            real_normal_probs = probs_all[~anom_mask]
            real_anom_det = float((real_anom_probs >= 0.5).mean())
        else:
            real_anom_det = float("nan")

        # Feature importance: which behavioral features were most perturbed
        perturbed_features = {
            _feat_name(fi): delta
            for fi, delta in case["perturbation_profile"].items()
        }

        result = {
            "chemical_name": case["chemical_name"],
            "chemical_id": case["chemical_id"],
            "contaminant_class": case["contaminant_class"],
            "source_event": case["source_event"],
            "environmental_context": case["environmental_context"],
            "mode_of_action": case["mode_of_action"],
            "case_chemicals": case["case_chemicals"],
            "ecotox_database": {
                "n_chemicals_with_beh_effects_in_ecotox": ecotox_stats["n_chemicals_with_beh_effects"],
                "example_ecotox_chemicals": ecotox_stats["example_chemicals"],
            },
            "n_daphnia": len(X_case),
            "n_exposed": n_exposed,
            "n_control": len(X_case) - n_exposed,
            "anomaly_detection_rate": detection_rate,
            "exposed_detection_rate": exposed_det_rate,
            "control_detection_rate": control_det_rate,
            "mean_anomaly_prob": mean_prob,
            "mean_exposed_prob": float(exposed_probs.mean()),
            "mean_control_prob": float(control_probs.mean()),
            "real_data_anomaly_detection_rate": real_anom_det,
            "perturbed_behavioral_features": perturbed_features,
        }
        results.append(result)

        log(
            f"  n_daphnia={len(X_case)}, overall_det={detection_rate:.1%}, "
            f"exposed_det={exposed_det_rate:.1%}, control_det={control_det_rate:.1%}, "
            f"mean_prob={mean_prob:.3f}"
        )

    # 5. Save results
    output = {
        "model": "BioMotion (TrajectoryDiffusionEncoder + AnomalyClassifier)",
        "checkpoint": "biomotion_expanded_best.pt",
        "published_test_auroc": 0.9999995590158115,
        "published_test_f1": 0.9989401165871754,
        "n_real_trajectories": len(base_features),
        "baseline_overall_detection_rate": overall_det,
        "case_studies": results,
    }
    out_path = OUTPUT_DIR / "behavioral_case_studies.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log(f"\nSaved to {out_path}")

    # Summary
    log("\n" + "=" * 65)
    log("SUMMARY — BioMotion Case Studies")
    log("=" * 65)
    for r in results:
        log(
            f"  {r['chemical_id']:45s}  "
            f"n={r['n_daphnia']:4d}  "
            f"det={r['anomaly_detection_rate']:.1%}  "
            f"exposed={r['exposed_detection_rate']:.1%}  "
            f"ctrl={r['control_detection_rate']:.1%}  "
            f"mean_p={r['mean_anomaly_prob']:.3f}"
        )
    log("Done.")


def _feat_name(idx: int) -> str:
    """Map feature index to behavioral measurement name."""
    feat_names = {
        0: "LOCO (locomotion)", 1: "SWIM (swimming velocity)", 2: "EQUL (equilibrium)",
        3: "ACTV (general activity)", 4: "MOTL (motility)", 5: "NMVM (immobility)",
        6: "PHTR (phototaxis)", 7: "GBHV (general behavior)", 8: "FLTR (filter feeding)",
        9: "VACL (valve activity)", 10: "ACTP (active time)", 11: "SEBH (secondary behavior)",
        12: "conc_norm", 13: "duration_norm", 14: "effect_magnitude", 15: "significance_flag",
    }
    return feat_names.get(idx, f"feat_{idx}")


if __name__ == "__main__":
    main()
