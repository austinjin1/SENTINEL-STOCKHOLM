#!/usr/bin/env python
"""SENTINEL end-to-end training orchestrator.

Runs the full pipeline in dependency order: train the five modality encoders,
then the fusion model, then the auxiliary / deployment models. Each step is a
standalone script under ``scripts/training/``; this orchestrator just sequences
them and reports status.

Usage:
  python scripts/pipeline/run_all.py                 # run every stage
  python scripts/pipeline/run_all.py --dry-run       # print plan only
  python scripts/pipeline/run_all.py --stage encoders
  python scripts/pipeline/run_all.py --only train_aquassm

Data must be downloaded/built first — see DATABASE.md.
All training assumes PYTHONNOUSERSITE=1 and the `physiformer` conda env.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN = "scripts/training"

# stage -> ordered list of (label, script path relative to repo root)
STAGES = {
    "encoders": [
        ("AquaSSM (sensor)",          f"{TRAIN}/train_aquassm.py"),
        ("HydroViT (satellite)",      f"{TRAIN}/train_hydrovit.py"),
        ("MicroBiomeNet (microbial)", f"{TRAIN}/train_microbiomenet.py"),
        ("ToxiGene (molecular)",      f"{TRAIN}/train_toxigene.py"),
        ("BioMotion (behavioral)",    f"{TRAIN}/train_biomotion.py"),
    ],
    "fusion": [
        ("SENTINEL-Fusion (Perceiver IO)", f"{TRAIN}/train_fusion.py"),
        ("Cross-modal contrastive",        f"{TRAIN}/train_contrastive.py"),
    ],
    "auxiliary": [
        ("Stream-network GNN",            f"{TRAIN}/train_stream_gnn.py"),
        ("Digital twin",                  f"{TRAIN}/train_twin.py"),
        ("Species-health model",          f"{TRAIN}/train_species_health.py"),
        ("Waterborne-disease model",      f"{TRAIN}/train_disease_forecast.py"),
        ("HydroDenseNet (SENTINEL-Lite)", f"{TRAIN}/train_hydrodensenet.py"),
    ],
}


def run(label, script, dry):
    print(f"\n=== {label}  ({script}) ===", flush=True)
    if dry:
        print("  [dry-run] skipped")
        return True
    t0 = time.time()
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    r = subprocess.run([sys.executable, script], cwd=PROJECT_ROOT, env=env)
    ok = r.returncode == 0
    print(f"  {'OK' if ok else 'FAILED'} in {time.time() - t0:.0f}s")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stage", choices=list(STAGES), help="run a single stage")
    ap.add_argument("--only", help="run a single script by basename (no .py)")
    args = ap.parse_args()

    stages = {args.stage: STAGES[args.stage]} if args.stage else STAGES
    failed = []
    for stage, steps in stages.items():
        print(f"\n########## STAGE: {stage} ##########")
        for label, script in steps:
            if args.only and Path(script).stem != args.only:
                continue
            if not run(label, script, args.dry_run):
                failed.append(label)
    print("\n========== SUMMARY ==========")
    print("All stages complete." if not failed else f"FAILED: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
