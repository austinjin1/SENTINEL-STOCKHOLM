#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Encoder checks for SENTINEL-Microbe (protocol §4.1): (1) parity of the plain-PyTorch DNABERT-S
re-implementation (sentinel/models/microbe/dna.py) with the authors' reference modules
(bert_layers.py from the pinned HF snapshot, instantiated WITHOUT transformers' meta-device init),
fp32 on CPU; (2) fp16 vs fp32 agreement of the re-implementation (cosine, raw and centred).
Real NEON sequences (first --n of interim/microbe_seqemb/all_sequences.parquet). Results -> JSON
artifact + registry (workstream M3-microbe, experiment microbe_encoder_checks).

    python -B scripts/eval/microbe_dnabert_parity.py --n 256 [--device cuda]
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.core.registry import log_result
from sentinel.models.microbe import dna


def reference_encoder(snap: Path):
    import torch

    pkg = types.ModuleType("dnab_ref")
    pkg.__path__ = [str(snap)]
    sys.modules["dnab_ref"] = pkg

    def load(name):
        spec = importlib.util.spec_from_file_location(f"dnab_ref.{name}", snap / f"{name}.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"dnab_ref.{name}"] = m
        spec.loader.exec_module(m)
        return m

    cb = load("configuration_bert")
    load("bert_padding")
    sys.modules["dnab_ref.flash_attn_triton"] = None      # force the PyTorch attention path
    bl = load("bert_layers")
    cfg = cb.BertConfig(**json.loads((snap / "config.json").read_text()))
    emb, enc = bl.BertEmbeddings(cfg), bl.BertEncoder(cfg)
    sd = torch.load(snap / dna.DNABERTS_WEIGHTS, map_location="cpu", weights_only=True)
    sd = {k[5:] if k.startswith("bert.") else k: v for k, v in sd.items()}
    emb.load_state_dict({k[11:]: v for k, v in sd.items() if k.startswith("embeddings.")}, strict=False)
    enc.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("encoder.")}, strict=False)
    return emb.eval(), enc.eval()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--device", default="cpu", help="device for the fp16 check")
    a = ap.parse_args(argv)
    import torch

    torch.set_num_threads(2 if a.device != "cpu" else 6)
    seqs = pd.read_parquet(paths.interim("microbe_seqemb") / "all_sequences.parquet")["sequence"].tolist()[: a.n]
    snap = dna._snapshot_dir()
    mine = dna.DNABERTS(device="cpu", fp16=False, snapshot_dir=snap)
    emb, enc = reference_encoder(snap)
    diffs, scale = [], []
    with torch.no_grad():
        for s in range(0, len(seqs), 32):
            ids, mask = mine.tokenize(seqs[s:s + 32])
            ids_t, m_t = torch.as_tensor(ids), torch.as_tensor(mask)
            ref = enc(emb(ids_t, token_type_ids=torch.zeros_like(ids_t)), m_t, output_all_encoded_layers=False)[-1]
            out = mine.model(ids_t, m_t)
            mk = m_t.bool()
            diffs.append(float((ref[mk] - out[mk]).abs().max()))
            scale.append(float(ref[mk].abs().mean()))
    e32 = dna.DNABERTS(device=a.device, fp16=False, snapshot_dir=snap).embed(seqs)
    e16 = dna.DNABERTS(device=a.device, fp16=True, snapshot_dir=snap).embed(seqs)

    def cos(x, y):
        return (x * y).sum(1) / np.linalg.norm(x, axis=1) / np.linalg.norm(y, axis=1)

    mu = e32.mean(0)
    pair = cos(e32[:-1], e32[1:])
    res = {"n_sequences": len(seqs), "parity_max_abs_diff_fp32": max(diffs), "hidden_mean_abs": float(np.mean(scale)),
           "fp16_vs_fp32_cos_min": float(cos(e16, e32).min()), "fp16_vs_fp32_cos_centred_min": float(cos(e16 - mu, e32 - mu).min()),
           "between_sequence_cos_median": float(np.median(pair)),
           "between_sequence_cos_centred_median": float(np.median(cos(e32[:-1] - mu, e32[1:] - mu))),
           "fp16_device": a.device, "revision": dna.DNABERTS_REVISION}
    out = paths.ensure(paths.RUNS_ROOT / "microbe" / "encoder_checks")
    (out / "dnabert_parity.json").write_text(json.dumps(res, indent=1))
    for k in ("parity_max_abs_diff_fp32", "fp16_vs_fp32_cos_min", "fp16_vs_fp32_cos_centred_min",
              "between_sequence_cos_median", "between_sequence_cos_centred_median"):
        log_result(workstream="M3-microbe", experiment="microbe_encoder_checks", model="DNABERT-S[re-impl]", metric=k,
                   value=res[k], split="neon_sequences/first_n", n=len(seqs), data_manifests=["microbe_seqemb_dnabert_s"],
                   artifacts=[str(out / "dnabert_parity.json")], extra=res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
