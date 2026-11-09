# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""DNA sequence encoders for ASV / OTU representative sequences (SENTINEL-Microbe, M3).

Two encoders, both *label-free* (they see sequences only, never chemistry), so embeddings can be
computed once for every unique sequence without any train/test leakage:

* :func:`kmer_profile` — normalised k-mer frequency vector (default k = 5, 1,024 dims; reverse
  complements are *not* merged because amplicons have a fixed orientation). Deterministic, no
  weights. Two sequences from *different* primer regions of the same organism share the k-mers of
  their overlapping region (V3V4 341F–805R vs V4V5 515F-Y–926R overlap in V4), so this encoder is
  partially primer-transferable by construction.
* :class:`DNABERTS` — a frozen **DNABERT-S** (Zhou et al. 2024, arXiv:2402.08777; MosaicBERT/
  DNABERT-2 backbone, 117M parameters, BPE vocabulary of 4,096, ALiBi, GEGLU; contrastively trained
  to place fragments of the same species together). Sample embedding = mean of the last hidden
  layer over all tokens (including [CLS]/[SEP]), as in the authors' usage example. Used in fp32
  (fp16 is avoidable error; see scripts/eval/microbe_dnabert_parity.py).

**Why a re-implementation.** The model's published ``trust_remote_code`` module fails to
instantiate under transformers 5.x (meta-device initialisation of its ALiBi buffer). This file
re-implements the forward pass in ~80 lines of plain PyTorch (padded attention + ALiBi bias, no
unpadding / Triton kernels) and loads the original checkpoint with ``torch.load(weights_only=True)``.
Numerical equivalence with the reference implementation (instantiated without meta init) is
checked by ``scripts/eval/microbe_dnabert_parity.py`` (writes a JSON artifact + registry entries);
``tests/test_microbe.py`` only checks that the encoder runs.

Pinned checkpoint: ``zhihan1996/DNABERT-S`` revision ``00e47f96cdea35e4b6f5df89e5419cbe47d490c6``,
file ``pytorch_model.bin`` (sha256 ``f3cfc3d0541859df64759e758cc4bc40fe1d5760799a7d098a9f9f09501c1e5f``),
licence Apache-2.0 (model card). Used offline from the local Hugging Face cache.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

DNABERTS_REPO = "zhihan1996/DNABERT-S"
DNABERTS_REVISION = "00e47f96cdea35e4b6f5df89e5419cbe47d490c6"
DNABERTS_WEIGHTS = "pytorch_model.bin"
DNABERTS_SHA256 = "f3cfc3d0541859df64759e758cc4bc40fe1d5760799a7d098a9f9f09501c1e5f"
DNABERTS_LICENSE = "Apache-2.0"

_BASES = "ACGT"


def seq_key(seq: str) -> str:
    """Stable id of a sequence (sha1 of the upper-case string) — used to cache embeddings."""
    return hashlib.sha1(seq.upper().encode()).hexdigest()


def clean_seq(seq: str) -> str:
    return "".join(ch for ch in str(seq).upper() if ch in "ACGTN")


# ------------------------------------------------------------------------------------------ k-mers

def kmer_profile(seqs, k: int = 5) -> np.ndarray:
    """[n, 4**k] float32 k-mer frequencies (each row sums to 1; windows containing N are skipped)."""
    lut = np.full(256, -1, np.int64)
    for i, b in enumerate(_BASES):
        lut[ord(b)] = i
    out = np.zeros((len(seqs), 4 ** k), np.float32)
    powers = 4 ** np.arange(k - 1, -1, -1, dtype=np.int64)
    for r, s in enumerate(seqs):
        a = lut[np.frombuffer(clean_seq(s).encode(), np.uint8)]
        if a.size < k:
            continue
        win = np.lib.stride_tricks.sliding_window_view(a, k)
        ok = (win >= 0).all(1)
        idx = win[ok] @ powers
        if idx.size:
            out[r] = np.bincount(idx, minlength=4 ** k).astype(np.float32) / idx.size
    return out


def kmer_names(k: int) -> list[str]:
    return ["".join(p) for p in itertools.product(_BASES, repeat=k)]


# ---------------------------------------------------------------------------------------- DNABERT-S

def _alibi_slopes(n_heads: int) -> list[float]:
    def pow2(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        return [start * start ** i for i in range(n)]

    if math.log2(n_heads).is_integer():
        return pow2(n_heads)
    c = 2 ** math.floor(math.log2(n_heads))
    return pow2(c) + _alibi_slopes(2 * c)[0::2][:n_heads - c]


def build_dnaberts(cfg: dict):
    """Plain-PyTorch MosaicBERT encoder whose parameter names match the DNABERT-S checkpoint."""
    import torch
    from torch import nn

    d, H, L, I = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_hidden_layers"], cfg["intermediate_size"]
    eps = cfg.get("layer_norm_eps", 1e-12)

    class Emb(nn.Module):
        def __init__(self):
            super().__init__()
            self.word_embeddings = nn.Embedding(cfg["vocab_size"], d)
            self.token_type_embeddings = nn.Embedding(cfg.get("type_vocab_size", 2), d)
            self.LayerNorm = nn.LayerNorm(d, eps=eps)

        def forward(self, ids):
            return self.LayerNorm(self.word_embeddings(ids) + self.token_type_embeddings(torch.zeros_like(ids)))

    class SelfAtt(nn.Module):
        def __init__(self):
            super().__init__()
            self.Wqkv = nn.Linear(d, 3 * d)

    class SelfOut(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(d, d)
            self.LayerNorm = nn.LayerNorm(d, eps=eps)

    class Att(nn.Module):
        def __init__(self):
            super().__init__()
            self.self = SelfAtt()
            self.output = SelfOut()

        def forward(self, h, bias):
            B, T, _ = h.shape
            qkv = self.self.Wqkv(h).view(B, T, 3, H, d // H)
            q, k, v = qkv.unbind(2)
            q, k, v = (x.transpose(1, 2) for x in (q, k, v))          # B H T dh
            s = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d // H) + bias
            a = torch.matmul(torch.softmax(s.float(), -1).to(v.dtype), v).transpose(1, 2).reshape(B, T, d)
            return self.output.LayerNorm(self.output.dense(a) + h)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gated_layers = nn.Linear(d, 2 * I, bias=False)
            self.wo = nn.Linear(I, d)
            self.layernorm = nn.LayerNorm(d, eps=eps)

        def forward(self, h):
            g = self.gated_layers(h)
            x = torch.nn.functional.gelu(g[..., :I]) * g[..., I:]
            return self.layernorm(self.wo(x) + h)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Att()
            self.mlp = MLP()

        def forward(self, h, bias):
            return self.mlp(self.attention(h, bias))

    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.ModuleList([Layer() for _ in range(L)])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = Emb()
            self.encoder = Enc()
            self.register_buffer("slopes", torch.tensor(_alibi_slopes(H)), persistent=False)

        def forward(self, ids, mask):
            T = ids.shape[1]
            pos = torch.arange(T, device=ids.device)
            rel = (pos[None, :] - pos[:, None]).abs().float()
            bias = -self.slopes[:, None, None] * rel[None]                          # H T T
            bias = bias[None] + (1.0 - mask[:, None, None, :].float()) * -10000.0   # B H T T
            h = self.embeddings(ids)
            bias = bias.to(h.dtype)
            for lyr in self.encoder.layer:
                h = lyr(h, bias)
            return h

    return Model()


class DNABERTS:
    """Frozen DNABERT-S sequence embedder (mean-pooled last layer, 768 dims)."""

    def __init__(self, device: str = "cpu", fp16: bool = False, snapshot_dir: str | Path | None = None,
                 verify_sha256: bool = True):
        import json

        import torch
        from tokenizers import Tokenizer

        snap = Path(snapshot_dir) if snapshot_dir else _snapshot_dir()
        cfg = json.loads((snap / "config.json").read_text())
        self.tokenizer = Tokenizer.from_file(str(snap / "tokenizer.json"))
        self.tokenizer.no_padding()      # the shipped tokenizer.json pads every input to 2,000 tokens
        self.tokenizer.no_truncation()
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        model = build_dnaberts(cfg)
        if verify_sha256:
            from sentinel.core.registry import file_sha256

            got = file_sha256(snap / DNABERTS_WEIGHTS)
            if got != DNABERTS_SHA256:
                raise RuntimeError(f"DNABERT-S weights sha256 {got} != pinned {DNABERTS_SHA256}")
        sd = torch.load(snap / DNABERTS_WEIGHTS, map_location="cpu", weights_only=True)
        sd = {k.removeprefix("bert."): v for k, v in sd.items()}
        keep = {k: v for k, v in sd.items() if k.startswith(("embeddings.", "encoder."))}
        missing, unexpected = model.load_state_dict(keep, strict=False)
        if missing:
            raise RuntimeError(f"DNABERT-S checkpoint missing keys: {missing[:5]}")
        self.fp16 = bool(fp16)       # default fp32 (fp16 blurs between-genus differences; protocol §4.1)
        self.model = model.to(device).eval()
        if self.fp16:
            self.model.half()
        self.device = device
        self.dim = cfg["hidden_size"]
        self.n_params = int(sum(p.numel() for p in model.parameters()))

    def tokenize(self, seqs):
        enc = self.tokenizer.encode_batch([clean_seq(s) for s in seqs])
        T = max(len(e.ids) for e in enc)
        ids = np.full((len(enc), T), self.pad_id, np.int64)
        mask = np.zeros((len(enc), T), np.int64)
        for i, e in enumerate(enc):
            ids[i, :len(e.ids)] = e.ids
            mask[i, :len(e.ids)] = 1
        return ids, mask

    def embed(self, seqs, batch: int = 128) -> np.ndarray:
        import torch

        out = np.zeros((len(seqs), self.dim), np.float32)
        order = np.argsort([len(s) for s in seqs])   # length-sorted batches waste less padding
        with torch.no_grad():
            for s in range(0, len(seqs), batch):
                idx = order[s:s + batch]
                ids, mask = self.tokenize([seqs[i] for i in idx])
                ids_t = torch.as_tensor(ids, device=self.device)
                m_t = torch.as_tensor(mask, device=self.device)
                h = self.model(ids_t, m_t).float()
                m = m_t.unsqueeze(-1).float()
                out[idx] = ((h * m).sum(1) / m.sum(1)).cpu().numpy()
        if not np.isfinite(out).all():
            raise FloatingPointError("non-finite DNABERT-S embeddings (fp16 overflow?)")
        return out


def _snapshot_dir() -> Path:
    from huggingface_hub import snapshot_download

    p = Path(snapshot_download(DNABERTS_REPO, revision=DNABERTS_REVISION, local_files_only=True,
                               allow_patterns=["config.json", "tokenizer.json", DNABERTS_WEIGHTS]))
    return p


# ---- primer-harmonised V4 units (improvement cycle 3, protocol §10.3) -------------------------------
# NEON .002 amplicons are V4V5 (515F-Y/926R, primers trimmed: the read starts right after 515F, E. coli
# ~534) and .001 amplicons are V3V4 (Pro341F/Pro805R: the read contains the 515F site, ends ~784).
# Their common region is the start of V4. ``v4_window`` cuts the same E. coli ~534..763 window (L=230)
# from both, so the same organism gives the same string regardless of primer set / release. Row-wise
# and label-free (depends only on the sequence and its primer set), hence no fold leakage.
import re as _re

V4_CORE_515F = _re.compile("GC[AC]GCCGCGGT[AG]A")   # 3' 13 bp of 515F-Y GTGYCAGCMGCCGCGGTAA
V4_LEN = 230


def v4_window(seq: str, primer_set: str, L: int = V4_LEN, search: tuple[int, int] = (100, 230)) -> str | None:
    if not isinstance(seq, str):
        return None
    if primer_set == "V4V5":
        return seq[:L] if len(seq) >= L else None
    if primer_set == "V3V4":
        m = V4_CORE_515F.search(seq, search[0], search[1])
        if m is None or len(seq) - m.end() < L:
            return None
        return seq[m.end():m.end() + L]
    return None


def add_v4_units(tokens: pd.DataFrame, sequences: pd.DataFrame, samples: pd.DataFrame, L: int = V4_LEN,
                 col: str = "v4") -> pd.DataFrame:
    """``tokens`` + column ``col`` = 'v4:' + sha1(V4 window)[:16] (NaN when no window: pooled into the
    unassigned/other columns by the CLR featuriser)."""
    prim = samples["primer_set"].astype(str).reindex(tokens["sample_id"]).to_numpy()
    sp = pd.DataFrame({"seq_id": tokens["seq_id"].to_numpy(), "primer": prim}).drop_duplicates("seq_id")
    seq = sequences.drop_duplicates("seq_id").set_index("seq_id")["sequence"].reindex(sp["seq_id"]).to_numpy()
    win = [v4_window(s_, p_, L) for s_, p_ in zip(seq, sp["primer"].to_numpy())]
    key = pd.Series([("v4:" + hashlib.sha1(w.encode()).hexdigest()[:16]) if w else np.nan for w in win],
                    index=sp["seq_id"].to_numpy())
    return tokens.assign(**{col: tokens["seq_id"].map(key).to_numpy()})
