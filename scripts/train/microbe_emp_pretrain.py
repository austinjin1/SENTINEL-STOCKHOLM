#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Cycle 6 (protocol §10.8): label-free masked-token pretraining of the SeqSet encoder on EMP freshwater
communities (V4 first 150 bp, 5-mer token vectors), for fine-tuning under LODO (`seqset_kmer_v4_150_emp`).

Exclusion (lead, 2026-11-10), decided before pretraining: every EMP sample within ``--radius-km`` (10 km)
of ANY NEON aquatic site (development, holdout and reserve sites alike), any sample without coordinates, and
any sample dated >= 2024-01-01 is excluded. The exclusion manifest
``docs/preregistration/microbe_emp_exclusion.json`` lists every excluded id with its reason and the sha256 of
the included id list. No NEON data and no labels are used here.

    python -m sentinel.core.gpu --name microbe_emp_pretrain --min-free-mb 3500 -- \
        python -B scripts/train/microbe_emp_pretrain.py --device cuda
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.core.registry import log_result
from sentinel.core.runlog import TrainingRun

REPO = Path(__file__).resolve().parents[2]


def haversine_km(lat1, lon1, lat2, lon2):
    r = np.pi / 180
    a = (np.sin((lat2 - lat1) * r / 2) ** 2
         + np.cos(lat1 * r) * np.cos(lat2 * r) * np.sin((lon2 - lon1) * r / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def exclusion(smp: pd.DataFrame, sites: pd.DataFrame, radius_km: float) -> pd.DataFrame:
    lat = pd.to_numeric(smp["latitude_deg"], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(smp["longitude_deg"], errors="coerce").to_numpy(float)
    slat, slon = sites["site_lat"].to_numpy(float), sites["site_lon"].to_numpy(float)
    dmin = np.full(len(smp), np.inf)
    near = np.array([""] * len(smp), dtype=object)
    for la, lo, s in zip(slat, slon, sites["site"]):
        d = haversine_km(lat, lon, la, lo)
        upd = d < dmin
        dmin[upd], near[upd] = d[upd], s
    date = pd.to_datetime(smp.get("collection_timestamp"), errors="coerce", utc=True)
    reason = np.where(~np.isfinite(lat) | ~np.isfinite(lon), "no_coordinates",
                      np.where(dmin < radius_km, "within_radius_of_neon_site",
                               np.where((date >= pd.Timestamp("2024-01-01", tz="UTC")).to_numpy(), "date_ge_2024", "")))
    return pd.DataFrame({"sample_id": smp["sample_id"].to_numpy(), "reason": reason, "nearest_neon_site": near,
                         "dist_km": np.round(dmin, 2)})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--radius-km", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import torch

    torch.set_num_threads(2)
    from sentinel.models.microbe import cv as CV
    from sentinel.models.microbe import dna
    from sentinel.models.microbe import seqset as SS

    e = paths.processed("emp_freshwater")
    smp = pd.read_parquet(e / "samples.parquet")
    tok = pd.read_parquet(e / "tokens.parquet")
    seqs = pd.read_parquet(e / "sequences.parquet")
    sites = pd.read_parquet(paths.PROCESSED / "neon_sites" / "sites.parquet")
    ex = exclusion(smp, sites, a.radius_km)
    keep = ex.loc[ex["reason"] == "", "sample_id"].astype(str).sort_values()
    excl = ex[ex["reason"] != ""]
    man = {"rule": f"exclude EMP samples within {a.radius_km} km of any of the {len(sites)} NEON aquatic sites "
                   "(dev, holdout and reserve sites alike), without coordinates, or dated >= 2024-01-01",
           "n_total": int(len(smp)), "n_excluded": int(len(excl)), "n_included": int(len(keep)),
           "excluded_by_reason": excl["reason"].value_counts().to_dict(),
           "included_ids_sha256": hashlib.sha256("\n".join(keep).encode()).hexdigest(),
           "excluded": excl.to_dict("records")}
    (REPO / "docs" / "preregistration" / "microbe_emp_exclusion.json").write_text(json.dumps(man, indent=1))
    tok = tok[tok["sample_id"].astype(str).isin(set(keep))]
    seqs = seqs[seqs["seq_id"].isin(tok["seq_id"].unique())].reset_index(drop=True)
    table = np.zeros((len(seqs) + 1, 4 ** 5), np.float32)
    table[1:] = dna.kmer_profile(seqs["sequence"].str[:150].tolist(), k=5)
    seq_row = {s: i + 1 for i, s in enumerate(seqs["seq_id"])}
    cfg = CV.load_config(REPO / "configs" / "microbe" / "seqset.yaml", "seqset_kmer_v4_150")
    batch = SS.encode_sets(tok, keep.to_numpy(), seq_row, None, int(cfg["max_tokens"]))
    out = paths.ensure(paths.RUNS_ROOT / "microbe" / "pretrain")
    dest = out / f"emp_v4_150_seed{a.seed}.pt"
    with TrainingRun("M3-microbe", "microbe_pretrain:emp_v4_150", config={**cfg, "radius_km": a.radius_km,
                     "seed": a.seed, "n_samples": int(len(keep))}, data=["emp_freshwater"],
                     plan_ref="microbe_protocol §10.8", iteration="cycle6", artifacts_dir=str(out)) as run:
        res = SS.pretrain_seqset(cfg, table, batch, seed=a.seed, device=a.device, log=run.note)
        torch.save(res, dest)
        sha = hashlib.sha256(dest.read_bytes()).hexdigest()
        run.metric("pretrain_best_val_mse", res["best_val"])
    log_result(workstream="M3-microbe", experiment="microbe_pretrain", model="SeqSet-encoder[EMP V4-150 masked-token]",
               metric="pretrain_val_mse", value=float(res["best_val"]), split="emp_freshwater/random5pct_val",
               n=int(len(keep)), data_manifests=["emp_freshwater"],
               extra={"epochs": len(res["history"]), "weights": str(dest), "weights_sha256": sha,
                      "exclusion_included_sha256": man["included_ids_sha256"], "n_excluded": man["n_excluded"]})
    print(json.dumps({"n_included": len(keep), "excluded": man["excluded_by_reason"], "best_val": res["best_val"],
                      "epochs": len(res["history"]), "weights": str(dest), "sha256": sha}))


if __name__ == "__main__":
    main()
