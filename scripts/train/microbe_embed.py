#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Compute frozen, label-free sequence tables for every unique sequence in processed/microbe_wq.

Writes ``$SENTINEL_DATA/interim/microbe_seqemb/{encoder}.npy`` (+ ``{encoder}_ids.parquet``) and a
manifest ``microbe_seqemb_{encoder}``. The tables see sequences only (no labels, no splits), so
computing them once for development and holdout sequences alike leaks nothing.

    python -B scripts/train/microbe_embed.py --encoder kmer5
    python -m sentinel.core.gpu --name microbe_embed --prefer 4 -- \
        python -B scripts/train/microbe_embed.py --encoder dnabert_s --device cuda
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.core.manifest import write_manifest
from sentinel.models.microbe import dna
from sentinel.models.microbe.cv import EMB_DIR_NAME


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["kmer5", "dnabert_s", "kmer5_v4", "kmer5_v4_150"], required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--fp16", action="store_true", help="fp16 inference (default fp32; see scripts/eval/microbe_dnabert_parity.py)")
    ap.add_argument("--sequences", default=None, help="parquet with seq_id, sequence (default: all microbe_wq sources)")
    a = ap.parse_args(argv)
    import torch

    torch.set_num_threads(2 if a.device != "cpu" else 6)
    srcs = [Path(a.sequences)] if a.sequences else sorted(paths.PROCESSED.glob("microbe_wq*/sequences.parquet"))
    seqs = pd.concat([pd.read_parquet(p, columns=["seq_id", "sequence"]) for p in srcs]).drop_duplicates("seq_id")
    seqs = seqs.sort_values("seq_id").reset_index(drop=True)
    out_dir = paths.interim(EMB_DIR_NAME)
    t0 = time.time()
    if a.encoder == "kmer5":
        emb = dna.kmer_profile(seqs["sequence"].tolist(), k=5)
        info = {"k": 5}
    elif a.encoder in ("kmer5_v4", "kmer5_v4_150"):   # k-mers of the harmonised V4 window (§10.3; 150 bp = EMP, §10.8)
        L = 150 if a.encoder == "kmer5_v4_150" else dna.V4_LEN
        prim = {}
        for d in sorted(paths.PROCESSED.glob("microbe_wq*/")):
            smp = pd.read_parquet(d / "samples.parquet", columns=["sample_id", "primer_set"]).set_index("sample_id")
            tok = pd.read_parquet(d / "tokens.parquet", columns=["sample_id", "seq_id"]).drop_duplicates("seq_id")
            prim.update(dict(zip(tok["seq_id"], smp["primer_set"].reindex(tok["sample_id"]).astype(str))))
        win = [dna.v4_window(q, prim.get(i, ""), L) for i, q in zip(seqs["seq_id"], seqs["sequence"])]
        has = np.array([w is not None for w in win])
        emb = np.zeros((len(seqs), 4 ** 5), np.float32)
        emb[has] = dna.kmer_profile([w for w in win if w is not None], k=5)
        info = {"k": 5, "window": f"V4 {L} bp after 515F", "frac_with_window": float(has.mean())}
    else:
        enc = dna.DNABERTS(device=a.device, fp16=a.fp16)
        emb = enc.embed(seqs["sequence"].tolist(), batch=a.batch)
        info = {"repo": dna.DNABERTS_REPO, "revision": dna.DNABERTS_REVISION, "weights_sha256": dna.DNABERTS_SHA256,
                "fp16": enc.fp16, "n_params": enc.n_params, "pooling": "mean of last layer over all tokens"}
    npy = out_dir / f"{a.encoder}.npy"
    ids = out_dir / f"{a.encoder}_ids.parquet"
    np.save(npy, emb.astype(np.float32))
    seqs[["seq_id"]].to_parquet(ids, compression="zstd")
    write_manifest(f"microbe_seqemb_{a.encoder}", files=[npy, ids], sources=[str(p) for p in srcs],
                   description=f"Frozen {a.encoder} embeddings of NEON 16S representative sequences (label-free)",
                   license="derived from NEON CC BY 4.0; DNABERT-S Apache-2.0" if a.encoder == "dnabert_s"
                   else "derived from NEON CC BY 4.0",
                   stats={"n_sequences": len(seqs), "dim": int(emb.shape[1]), "seconds": round(time.time() - t0, 1),
                          **info})
    print(f"{a.encoder}: {emb.shape} in {time.time() - t0:.0f}s -> {npy}")


if __name__ == "__main__":
    main()
