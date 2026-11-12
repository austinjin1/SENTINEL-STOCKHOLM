# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SeqSet: a set transformer over DNA-encoded ASVs (SENTINEL-Microbe, M3; protocol §6).

**Input.** A sample is the set of its ``max_tokens`` most abundant sequences (ASVs/OTU
representatives). Each token = (sequence vector, abundance features):

* sequence vector — a row of a *frozen, label-free* table computed once per unique sequence by
  :mod:`sentinel.models.microbe.dna` (``dnabert``: DNABERT-S 768-d; ``kmer``: 5-mer profile,
  1,024-d). Standardised with the mean/std over the *training fold's* tokens, then projected
  (Linear -> LayerNorm) to ``d_model``. Because tokens are described by their DNA rather than an
  identifier, a sequence never seen in training (another primer region, another release's OTU
  clustering, another domain) still gets a meaningful representation — the property table models
  on OTU ids lack. ``id`` mode (ablation) replaces the DNA vector with a learned per-sequence
  embedding; sequences seen < ``min_seq_count`` times in the training fold map to UNK.
* abundance features — log10(1 + count)/5, relative abundance, sqrt(relative abundance), and the
  within-sample centred log abundance (CLR over the kept tokens) / 5.

**Encoder / heads.** As BioSet (``sentinel/models/bioassess/bioset.py``, re-implemented here, not
imported, so the two workstreams stay independent): pre-LN self-attention blocks without positional
encoding, pooling by multi-head attention (1 seed) concatenated with the abundance-weighted token
mean; sample descriptors (log reads, log richness, primer one-hot) join after pooling; shared trunk;
one output per target; masked MSE on training-fold-standardised targets.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

N_ABUND = 4
UNK = 0


@dataclass
class SetBatch:
    ids: np.ndarray          # sample ids [N]
    seq: np.ndarray          # [N, T] int64 row into the sequence table (or id vocabulary); 0 = pad/UNK
    ab: np.ndarray           # [N, T, 4] float32
    mask: np.ndarray         # [N, T] bool
    glob: np.ndarray         # [N, G] float32

    def take(self, rows) -> SetBatch:
        return SetBatch(self.ids[rows], self.seq[rows], self.ab[rows], self.mask[rows], self.glob[rows])


def encode_sets(tokens: pd.DataFrame, sample_ids, seq_row: dict, glob: np.ndarray | None,
                max_tokens: int = 256) -> SetBatch:
    """``tokens``: long table (sample_id, seq_id, count). ``seq_row``: seq_id -> table row (>= 1)."""
    sample_ids = np.asarray(sample_ids)
    pos = pd.Series(np.arange(len(sample_ids)), index=pd.Index(sample_ids))
    t = tokens[tokens["sample_id"].isin(pos.index)][["sample_id", "seq_id", "count"]].copy()
    t["_row"] = pos.reindex(t["sample_id"]).to_numpy()
    t = t.sort_values(["_row", "count", "seq_id"], ascending=[True, False, True], kind="stable")
    t["_k"] = t.groupby("_row").cumcount().to_numpy()
    tot_all = t.groupby("_row")["count"].transform("sum").to_numpy(float)
    keep = t["_k"].to_numpy() < max_tokens
    t, tot_all = t[keep], tot_all[keep]
    T = int(t["_k"].max()) + 1 if len(t) else 1
    N = len(sample_ids)
    seq = np.zeros((N, T), np.int64)
    ab = np.zeros((N, T, N_ABUND), np.float32)
    mask = np.zeros((N, T), bool)
    r, k = t["_row"].to_numpy(int), t["_k"].to_numpy(int)
    cnt = t["count"].to_numpy(float)
    seq[r, k] = t["seq_id"].map(seq_row).fillna(UNK).to_numpy(np.int64)
    rel = cnt / np.maximum(tot_all, 1e-12)
    lc = np.log(cnt + 0.5)
    lc = lc - pd.Series(lc).groupby(r).transform("mean").to_numpy()
    ab[r, k] = np.column_stack([np.log10(1 + cnt) / 5.0, rel, np.sqrt(rel), lc / 5.0]).astype(np.float32)
    mask[r, k] = True
    g = np.zeros((N, 0), np.float32) if glob is None else np.asarray(glob, np.float32)
    return SetBatch(sample_ids, seq, ab, mask, g)


def token_coverage(tokens: pd.DataFrame, max_tokens: int) -> dict:
    """Fraction of samples with more than ``max_tokens`` sequences and mean read fraction kept."""
    t = tokens.sort_values(["sample_id", "count"], ascending=[True, False])
    k = t.groupby("sample_id").cumcount()
    tot = t.groupby("sample_id")["count"].transform("sum")
    kept = t[k < max_tokens].groupby("sample_id")["count"].sum() / tot.groupby(t["sample_id"]).first()
    n = t.groupby("sample_id").size()
    return {"frac_samples_truncated": float((n > max_tokens).mean()),
            "mean_read_frac_kept": float(kept.mean()), "min_read_frac_kept": float(kept.min())}


# ---------------------------------------------------------------------------------------- model

def build_model(cfg: dict, seq_dim: int, n_vocab: int, n_glob: int, n_out: int):
    import torch
    from torch import nn

    d = int(cfg["d_model"])
    drop = float(cfg["dropout"])
    H = int(cfg["n_heads"])

    class SAB(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.att = nn.MultiheadAttention(d, H, dropout=drop, batch_first=True)
            self.ln2 = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))
            self.drop = nn.Dropout(drop)

        def forward(self, h, pad):
            x = self.ln1(h)
            a, _ = self.att(x, x, x, key_padding_mask=pad, need_weights=False)
            h = h + self.drop(a)
            return h + self.drop(self.ff(self.ln2(h)))

    class SeqSet(nn.Module):
        def __init__(self):
            super().__init__()
            self.mode = cfg.get("seq_mode", "frozen")
            if self.mode == "frozen":
                self.proj = nn.Sequential(nn.Linear(seq_dim, d), nn.LayerNorm(d))
            else:
                self.idemb = nn.Embedding(n_vocab, d, padding_idx=UNK)
            self.ab = nn.Linear(N_ABUND, d)
            self.blocks = nn.ModuleList([SAB() for _ in range(int(cfg["n_layers"]))])
            self.ln = nn.LayerNorm(d)
            self.seed = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            self.pma = nn.MultiheadAttention(d, H, dropout=drop, batch_first=True)
            self.n_glob = n_glob
            self.trunk = nn.Sequential(nn.Linear(2 * d + n_glob, d), nn.GELU(), nn.Dropout(drop),
                                       nn.Linear(d, d), nn.GELU())
            self.head = nn.Linear(d, n_out)

        def encode_tokens(self, seqvec, seq, ab, mask):
            tok = self.proj(seqvec) if self.mode == "frozen" else self.idemb(seq)
            h = tok + self.ab(ab)
            pad = ~mask
            for b in self.blocks:
                h = b(h, pad)
            return self.ln(h)

        def embed(self, seqvec, seq, ab, mask):
            h = self.encode_tokens(seqvec, seq, ab, mask)
            pad = ~mask
            w = ab[..., 1] * mask
            w = w / w.sum(1, keepdim=True).clamp_min(1e-9)
            wmean = (h * w.unsqueeze(-1)).sum(1)
            q = self.seed.expand(h.shape[0], -1, -1)
            p, _ = self.pma(q, h, h, key_padding_mask=pad, need_weights=False)
            return torch.cat([wmean, p[:, 0]], -1)

        def forward(self, seqvec, seq, ab, mask, glob):
            z = self.embed(seqvec, seq, ab, mask)
            if self.n_glob:
                z = torch.cat([z, glob], -1)
            return self.head(self.trunk(z))

    return SeqSet()


def masked_mse(pred, y):
    import torch

    ok = torch.isfinite(y)
    diff = torch.where(ok, pred - torch.nan_to_num(y), torch.zeros_like(pred))
    return (diff ** 2).sum() / ok.sum().clamp_min(1)


@dataclass
class FittedSeqSet:
    model: object
    cfg: dict
    table: object            # torch tensor [n_rows, D] standardised (frozen mode) or None
    glob_mean: np.ndarray
    glob_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    history: list
    best_epoch: int
    n_params: int

    def _fwd(self, b: SetBatch, device, embed_only=False):
        import torch

        seq = torch.as_tensor(b.seq, device=device)
        ab = torch.as_tensor(b.ab, device=device)
        m = torch.as_tensor(b.mask, device=device)
        g = torch.as_tensor(((b.glob - self.glob_mean) / self.glob_std).astype(np.float32), device=device)
        sv = self.table[seq] if self.table is not None else None
        if embed_only:
            return self.model.embed(sv, seq, ab, m)
        return self.model(sv, seq, ab, m, g)

    def predict(self, b: SetBatch, device="cpu", batch: int = 256) -> np.ndarray:
        import torch

        self.model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(b.ids), batch):
                out.append(self._fwd(b.take(slice(s, s + batch)), device).float().cpu().numpy())
        z = np.concatenate(out) if out else np.zeros((0, len(self.y_mean)))
        return z * self.y_std + self.y_mean

    def embed(self, b: SetBatch, device="cpu", batch: int = 256) -> np.ndarray:
        import torch

        self.model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(b.ids), batch):
                out.append(self._fwd(b.take(slice(s, s + batch)), device, embed_only=True).float().cpu().numpy())
        return np.concatenate(out)


def _augment(seq, ab, mask, cfg, gen):
    import torch

    td = float(cfg.get("token_dropout", 0.0))
    if td > 0:
        drop = (torch.rand(mask.shape, generator=gen, device=mask.device) < td) & mask
        keep_any = (mask & ~drop).any(1, keepdim=True)
        mask = mask & ~(drop & keep_any)
    return seq, ab, mask


def fit_seqset(cfg: dict, table: np.ndarray | None, tr: SetBatch, Ytr: np.ndarray, va: SetBatch, Yva: np.ndarray,
               *, seed: int, device: str = "cpu", n_vocab: int = 1, log=None) -> FittedSeqSet:
    """Train on (tr, Ytr) and early-stop on (va, Yva); all normalisers from ``tr`` only.

    ``table``: [n_rows, D] frozen sequence vectors (row 0 = pad/UNK, zeros) for ``seq_mode='frozen'``.
    """
    import torch

    from sentinel.core.seed import seed_everything

    seed_everything(seed)
    if not (np.isfinite(tr.glob).all() and np.isfinite(va.glob).all()):
        raise ValueError("non-finite sample descriptors")
    gm = tr.glob.mean(0) if tr.glob.shape[1] else np.zeros(0, np.float32)
    gs = tr.glob.std(0) if tr.glob.shape[1] else np.ones(0, np.float32)
    gs = np.where(gs > 0, gs, 1.0).astype(np.float32)
    ym = np.nanmean(Ytr, 0)
    ys = np.nanstd(Ytr, 0)
    ys = np.where(np.isfinite(ys) & (ys > 0), ys, 1.0)
    ym = np.where(np.isfinite(ym), ym, 0.0)
    mode = cfg.get("seq_mode", "frozen")
    tab_t = None
    init = None
    if cfg.get("init_state"):          # cycle 6: encoder pretrained label-free on EMP (pretrain_seqset)
        from pathlib import Path as _P

        from sentinel.core import paths as _paths

        ip = _P(cfg["init_state"])
        ip = ip if ip.is_absolute() else _paths.RUNS_ROOT / ip      # relative to $SENTINEL_RUNS
        init = torch.load(ip, map_location="cpu", weights_only=False)
    if mode == "frozen":
        rows = np.unique(tr.seq[tr.mask])
        mu = table[rows].mean(0)
        sd = table[rows].std(0)
        sd = np.where(sd > 1e-6, sd, 1.0)
        if init is not None and cfg.get("table_norm") == "pretrain":
            mu, sd = np.asarray(init["mu"]), np.asarray(init["sd"])
        tz = ((table - mu) / sd).astype(np.float32)
        tz[UNK] = 0.0
        tab_t = torch.as_tensor(tz, device=device)
    seq_dim = table.shape[1] if table is not None else 0
    model = build_model(cfg, seq_dim, n_vocab, tr.glob.shape[1], Ytr.shape[1]).to(device)
    if init is not None:
        missing, unexpected = model.load_state_dict(init["state"], strict=False)
        if unexpected or not any(k.startswith("blocks.") for k in init["state"]):
            raise ValueError(f"pretrained state does not match the SeqSet encoder: {unexpected[:5]}")
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    fitted = FittedSeqSet(model, cfg, tab_t, gm.astype(np.float32), gs, ym, ys, [], -1, 0)
    ytr = torch.as_tensor(((Ytr - ym) / ys).astype(np.float32), device=device)
    yva = torch.as_tensor(((Yva - ym) / ys).astype(np.float32), device=device)
    S = torch.as_tensor(tr.seq, device=device)
    A = torch.as_tensor(tr.ab, device=device)
    M = torch.as_tensor(tr.mask, device=device)
    G = torch.as_tensor(((tr.glob - gm) / gs).astype(np.float32), device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    bs, n = int(cfg["batch_size"]), len(tr.ids)
    best, best_state, best_ep, bad, hist = math.inf, None, -1, 0, []
    for ep in range(int(cfg["max_epochs"])):
        model.train()
        perm = torch.randperm(n, generator=gen, device=device)
        tot = 0.0
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            sq, ab, m = _augment(S[idx], A[idx], M[idx], cfg, gen)
            sv = tab_t[sq] if tab_t is not None else None
            loss = masked_mse(model(sv, sq, ab, m, G[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            pv = []
            for s in range(0, len(va.ids), 256):
                pv.append(fitted._fwd(va.take(slice(s, s + 256)), device))
            vl = float(masked_mse(torch.cat(pv), yva)) if pv else tot / n
        hist.append((ep, tot / n, vl))
        if math.isfinite(vl) and vl < best - 1e-4:
            best, best_state, best_ep, bad = vl, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                break
    if best_state is None:
        raise FloatingPointError("validation loss never finite/improving")
    model.load_state_dict(best_state)
    fitted.history, fitted.best_epoch = hist, best_ep
    fitted.n_params = int(sum(p.numel() for p in model.parameters()))
    if log:
        log(f"best epoch {best_ep} val {best:.4f} ({len(hist)} epochs, {fitted.n_params} params)")
    return fitted


ENCODER_PREFIXES = ("proj.", "ab.", "blocks.", "ln.")


def pretrain_seqset(cfg: dict, table: np.ndarray, data: SetBatch, *, seed: int = 0, device: str = "cpu",
                    mask_frac: float = 0.15, val_frac: float = 0.05, max_epochs: int = 200, patience: int = 15,
                    log=None) -> dict:
    """Label-free masked-token pretraining of the SeqSet token encoder (protocol §10.8, cycle 6).

    ``mask_frac`` of each sample's valid tokens get a zero (= mean, after standardisation) sequence vector; the
    encoder must reconstruct their standardised k-mer vectors from the rest of the community (MSE). Abundance
    features stay visible: the task is "which organism is this, given its abundance and the rest of the
    community". Early stopping on ``val_frac`` held-out pretraining samples. Returns the encoder state
    (``ENCODER_PREFIXES``), the table standardisation (mu, sd) and the history. No labels are used."""
    import torch
    from torch import nn

    from sentinel.core.seed import seed_everything

    seed_everything(seed)
    rows = np.unique(data.seq[data.mask])
    mu = table[rows].mean(0)
    sd = table[rows].std(0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    tz = ((table - mu) / sd).astype(np.float32)
    tz[UNK] = 0.0
    tab = torch.as_tensor(tz, device=device)
    model = build_model(cfg, table.shape[1], 1, 0, 1).to(device)
    head = nn.Linear(int(cfg["d_model"]), table.shape[1]).to(device)
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    rs = np.random.default_rng(seed)
    perm = rs.permutation(len(data.ids))
    n_val = max(1, int(round(val_frac * len(perm))))
    va, tr = perm[:n_val], perm[n_val:]
    S, A, M = (torch.as_tensor(x, device=device) for x in (data.seq, data.ab, data.mask))
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    def loss_on(idx, train: bool):
        sq, ab, m = S[idx], A[idx], M[idx]
        msk = (torch.rand(m.shape, generator=gen, device=device) < mask_frac) & m
        sv = tab[sq].clone()
        sv[msk] = 0.0
        h = model.encode_tokens(sv, sq, ab, m)
        pred = head(h[msk])
        return ((pred - tab[sq][msk]) ** 2).mean()

    bs = int(cfg["batch_size"])
    best, best_state, bad, hist = math.inf, None, 0, []
    for ep in range(max_epochs):
        model.train()
        head.train()
        order = torch.as_tensor(rs.permutation(tr), device=device)
        tot = 0.0
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            loss = loss_on(idx, True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += float(loss) * len(idx)
        model.eval()
        head.eval()
        with torch.no_grad():
            gen.manual_seed(seed + 10_000)           # fixed validation masks
            vl = float(np.mean([float(loss_on(torch.as_tensor(va[s:s + 256], device=device), False))
                                for s in range(0, len(va), 256)]))
        hist.append((ep, tot / max(len(tr), 1), vl))
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                          if k.startswith(ENCODER_PREFIXES)}
        else:
            bad += 1
            if bad >= patience:
                break
    if log:
        log(f"pretrain best val {best:.4f} after {len(hist)} epochs")
    return {"state": best_state, "mu": mu.astype(np.float32), "sd": sd.astype(np.float32), "history": hist,
            "best_val": best, "cfg": {k: cfg[k] for k in ("d_model", "n_layers", "n_heads", "dropout")}}
