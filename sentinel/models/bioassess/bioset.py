# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""BioSet: taxon-embedding set model for community -> chemistry (SENTINEL-BioAssess, M5).

**Input.** A sample is a *set* of present taxa. Each taxon token carries

* its harmonised lineage ids at six levels (phylum, class, order, family, genus, finest name;
  ``processed/taxonomy_harmonised``) — a family-level identification simply has no genus/name-below-
  family id, which is how mixed taxonomic resolution is represented (nothing is dropped);
* four abundance features: log10(1 + count)/3, relative abundance, sqrt(relative abundance),
  log10-abundance class / 5 (the last is the cross-dataset abundance scale shared with the UK EA
  data, whose abundances are published as log classes).

Sample-level descriptors (log10 total count — the *total-count offset* —, log10 number of taxa,
boatable flag) enter after pooling, standardised with training-fold statistics.

**Taxon embedding.** ``hierarchical``: e(taxon) = Σ_level E_level[id_level]; the per-level tables
are initialised with decreasing scale from phylum (std 1.0) to name (0.15), so at initialisation a
genus sits next to its family, its family next to its order, etc. — the embedding starts *from the
taxonomy* and rare taxa borrow strength from relatives. ``flat``: one table over finest names only
(the one-hot-equivalent ablation of RESEARCH_PLAN §14.5).

**Encoder.** ``settransformer``: n pre-LN self-attention blocks (SAB; Lee et al. 2019) without
positional encodings (permutation invariant), then pooling by multi-head attention (PMA, 1 seed)
concatenated with the abundance-weighted mean of token states. ``deepsets``: per-token MLP, then
abundance-weighted mean + max pooling (Zaheer et al. 2017).

**Heads.** Shared trunk MLP, one linear output per target; loss = masked MSE on training-fold
standardised targets (missing targets contribute nothing).

**Training-fold discipline.** :func:`fit_bioset` receives training and inner-validation arrays only;
names seen fewer than ``min_name_count`` times in the training fold (and not in the pretraining
pool) are encoded as UNK (index 0, a zero vector) — see :class:`TaxonVocab.allowed_from`.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LEVELS = ("phylum", "class", "order", "family", "genus", "name")
N_ABUND = 4
PAD, MASK = 0, 1           # reserved ids in every level table


# ---------------------------------------------------------------------------------------- vocab

@dataclass
class TaxonVocab:
    """Per-level name -> id maps (ids 0 = pad/UNK, 1 = MASK). Built from names only (no labels)."""

    maps: dict = field(default_factory=dict)

    @classmethod
    def build(cls, *token_tables: pd.DataFrame) -> "TaxonVocab":
        maps = {}
        for lv in LEVELS:
            names = set()
            for t in token_tables:
                names |= set(t[lv].dropna().astype(str))
            maps[lv] = {n: i + 2 for i, n in enumerate(sorted(names))}
        return cls(maps)

    def size(self, level: str) -> int:
        return len(self.maps[level]) + 2

    def ids(self, tokens: pd.DataFrame) -> np.ndarray:
        """[n_rows, n_levels] ids (0 where missing / not in vocab)."""
        out = np.zeros((len(tokens), len(LEVELS)), np.int64)
        for j, lv in enumerate(LEVELS):
            out[:, j] = tokens[lv].map(self.maps[lv]).fillna(0).to_numpy(np.int64)
        return out

    def allowed_from(self, tokens: pd.DataFrame, min_count: int = 1) -> list[np.ndarray]:
        """Boolean masks (per level) of ids seen >= min_count times in ``tokens``."""
        ids = self.ids(tokens)
        out = []
        for j, lv in enumerate(LEVELS):
            cnt = np.bincount(ids[:, j], minlength=self.size(lv))
            a = cnt >= min_count
            a[PAD] = True
            a[MASK] = True
            out.append(a)
        return out

    def to_dict(self) -> dict:
        return {"maps": self.maps}

    @classmethod
    def from_dict(cls, d: dict) -> "TaxonVocab":
        return cls({k: dict(v) for k, v in d["maps"].items()})


def abundance_features(count: np.ndarray, total: np.ndarray, log_class: np.ndarray | None = None) -> np.ndarray:
    count = np.asarray(count, float)
    rel = count / np.maximum(np.asarray(total, float), 1e-12)
    if log_class is None:
        log_class = np.clip(np.floor(np.log10(np.maximum(count, 1.0))) + 1.0, 1.0, 5.0)
    return np.column_stack([np.log10(1.0 + count) / 3.0, rel, np.sqrt(rel),
                            np.asarray(log_class, float) / 5.0]).astype(np.float32)


@dataclass
class Encoded:
    """Padded set tensors (numpy) for a list of sample ids."""

    ids: np.ndarray            # sample ids, order of rows
    lev: np.ndarray            # [N, T, L] int64
    ab: np.ndarray             # [N, T, 4] float32
    mask: np.ndarray           # [N, T] bool (True = real token)
    glob: np.ndarray           # [N, G] float32 (raw; standardised in fit/predict)

    def take(self, rows) -> "Encoded":
        return Encoded(self.ids[rows], self.lev[rows], self.ab[rows], self.mask[rows], self.glob[rows])


def encode(tokens: pd.DataFrame, sample_ids, vocab: TaxonVocab, glob: np.ndarray | None = None,
           max_tokens: int | None = None, id_col: str = "visit_id") -> Encoded:
    """Encode the token table (columns LEVELS + ``count`` [+ ``log_class``]) of ``sample_ids``."""
    sample_ids = np.asarray(sample_ids)
    pos = pd.Series(np.arange(len(sample_ids)), index=pd.Index(sample_ids))
    t = tokens[tokens[id_col].isin(pos.index)].copy()
    t["_row"] = pos.reindex(t[id_col]).to_numpy()
    t = t.sort_values(["_row", "count"], ascending=[True, False], kind="stable")
    tot = t.groupby("_row")["count"].transform("sum").to_numpy(float)
    t["_k"] = t.groupby("_row").cumcount().to_numpy()
    if max_tokens is not None:
        keep = t["_k"].to_numpy() < max_tokens
        t, tot = t[keep], tot[keep]
    T = int(t["_k"].max()) + 1 if len(t) else 1
    N = len(sample_ids)
    lev = np.zeros((N, T, len(LEVELS)), np.int64)
    ab = np.zeros((N, T, N_ABUND), np.float32)
    mask = np.zeros((N, T), bool)
    r, k = t["_row"].to_numpy(int), t["_k"].to_numpy(int)
    lev[r, k] = vocab.ids(t)
    lc = t["log_class"].to_numpy(float) if "log_class" in t else None
    ab[r, k] = abundance_features(t["count"].to_numpy(float), tot, lc)
    mask[r, k] = True
    g = np.zeros((N, 0), np.float32) if glob is None else np.asarray(glob, np.float32)
    return Encoded(sample_ids, lev, ab, mask, g)


def restrict(enc: Encoded, allowed: list[np.ndarray]) -> Encoded:
    """Map ids not allowed (unseen in the training fold) to UNK (0)."""
    lev = enc.lev.copy()
    for j, a in enumerate(allowed):
        col = lev[..., j]
        col[~a[col]] = PAD
    return Encoded(enc.ids, lev, enc.ab, enc.mask, enc.glob)


# ---------------------------------------------------------------------------------------- model

def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_model(cfg: dict, vocab_sizes: dict, n_glob: int, n_out: int):
    torch, nn = _torch()
    import torch.nn.functional as F

    d = int(cfg["d_model"])
    drop = float(cfg["dropout"])

    class TaxonEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.flat = cfg.get("embedding", "hierarchical") == "flat"
            levels = ["name"] if self.flat else list(LEVELS)
            self.levels = levels
            self.tables = nn.ModuleDict({lv: nn.Embedding(vocab_sizes[lv], d, padding_idx=PAD) for lv in levels})
            std = cfg.get("init_std", {})
            for lv, emb in self.tables.items():
                nn.init.normal_(emb.weight, 0.0, float(std.get(lv, 0.5)) if not self.flat else 1.0)
                with torch.no_grad():
                    emb.weight[PAD].zero_()

        def forward(self, lev):
            out = 0
            for lv in self.levels:
                out = out + self.tables[lv](lev[..., LEVELS.index(lv)])
            return out

    class SAB(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.att = nn.MultiheadAttention(d, int(cfg["n_heads"]), dropout=drop, batch_first=True)
            self.ln2 = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))
            self.drop = nn.Dropout(drop)

        def forward(self, h, pad):
            x = self.ln1(h)
            a, _ = self.att(x, x, x, key_padding_mask=pad, need_weights=False)
            h = h + self.drop(a)
            return h + self.drop(self.ff(self.ln2(h)))

    class Encoder(nn.Module):
        """Embeddings + set encoder + pooling -> [B, 2d] sample representation."""

        def __init__(self):
            super().__init__()
            self.arch = cfg.get("arch", "settransformer")
            self.emb = TaxonEmbedding()
            self.ab = nn.Linear(N_ABUND, d)
            if self.arch == "settransformer":
                self.blocks = nn.ModuleList([SAB() for _ in range(int(cfg["n_layers"]))])
                self.ln = nn.LayerNorm(d)
                self.seed = nn.Parameter(torch.randn(1, 1, d) * 0.02)
                self.pma = nn.MultiheadAttention(d, int(cfg["n_heads"]), dropout=drop, batch_first=True)
            else:
                self.phi = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop),
                                         nn.Linear(2 * d, d), nn.GELU())
            self.out_dim = 2 * d

        def tokens(self, lev, ab, mask):
            h = self.emb(lev) + self.ab(ab)
            pad = ~mask
            if self.arch == "settransformer":
                for b in self.blocks:
                    h = b(h, pad)
                h = self.ln(h)
            else:
                h = self.phi(h)
            return h

        def pool(self, h, ab, mask):
            w = ab[..., 1] * mask                          # relative abundance
            w = w / w.sum(1, keepdim=True).clamp_min(1e-9)
            wmean = (h * w.unsqueeze(-1)).sum(1)
            if self.arch == "settransformer":
                q = self.seed.expand(h.shape[0], -1, -1)
                p, _ = self.pma(q, h, h, key_padding_mask=~mask, need_weights=False)
                return torch.cat([wmean, p[:, 0]], -1)
            hm = h.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
            return torch.cat([wmean, hm], -1)

        def forward(self, lev, ab, mask):
            h = self.tokens(lev, ab, mask)
            return self.pool(h, ab, mask)

    class BioSet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.n_glob = n_glob
            self.trunk = nn.Sequential(nn.Linear(self.encoder.out_dim + n_glob, d), nn.GELU(), nn.Dropout(drop),
                                       nn.Linear(d, d), nn.GELU())
            self.head = nn.Linear(d, n_out)

        def forward(self, lev, ab, mask, glob):
            z = self.encoder(lev, ab, mask)
            if self.n_glob:
                z = torch.cat([z, glob], -1)
            return self.head(self.trunk(z))

    _ = F
    return BioSet()


# ------------------------------------------------------------------------------------- training

def _augment(lev, ab, mask, cfg, gen):
    """Token dropout (keep >= 1 token) and resolution dropout (drop genus + name ids)."""
    torch, _ = _torch()
    td = float(cfg.get("token_dropout", 0.0))
    if td > 0:
        drop = (torch.rand(mask.shape, generator=gen, device=mask.device) < td) & mask
        keep_any = (mask & ~drop).any(1, keepdim=True)
        drop = drop & keep_any
        mask = mask & ~drop
    rd = float(cfg.get("resolution_dropout", 0.0))
    if rd > 0:
        coarse = torch.rand(mask.shape, generator=gen, device=mask.device) < rd
        lev = lev.clone()
        for j in (LEVELS.index("genus"), LEVELS.index("name")):
            lev[..., j] = lev[..., j].masked_fill(coarse, PAD)
        # a family-level identification keeps its family id; name level for coarse tokens is removed
    return lev, ab, mask


def _to(enc: Encoded, device):
    torch, _ = _torch()
    return (torch.as_tensor(enc.lev, device=device), torch.as_tensor(enc.ab, device=device),
            torch.as_tensor(enc.mask, device=device), torch.as_tensor(enc.glob, device=device))


def region_anomaly_mse(pred, y, region, weight=None):
    """MSE of within-batch, within-region anomalies: (p − p̄_r) vs (y − ȳ_r), finite targets only.

    Regions with < 2 finite rows in the batch contribute nothing. Uses training-row regions only.
    """
    torch, _ = _torch()
    ok = torch.isfinite(y).float()
    yy = torch.nan_to_num(y)
    R = int(region.max().item()) + 1
    M = torch.nn.functional.one_hot(region, R).float()            # [B, R]
    cnt = M.t() @ ok                                              # [R, T]
    pm = (M.t() @ (pred * ok)) / cnt.clamp_min(1)
    ym = (M.t() @ (yy * ok)) / cnt.clamp_min(1)
    keep = ok * ((M @ (cnt >= 2).float()) > 0).float()
    d = ((pred - M @ pm) - (yy - M @ ym)) * keep
    if weight is not None:
        keep = keep * weight.view(1, -1)
        d = d * weight.view(1, -1).sqrt()
    return (d ** 2).sum() / keep.sum().clamp_min(1e-9)


def masked_mse(pred, y, weight=None):
    """Mean squared error over finite targets; optional per-column weights."""
    torch, _ = _torch()
    ok = torch.isfinite(y)
    diff = torch.where(ok, pred - torch.nan_to_num(y), torch.zeros_like(pred))
    if weight is None:
        return (diff ** 2).sum() / ok.sum().clamp_min(1)
    w = weight.view(1, -1).expand_as(diff) * ok
    return ((diff ** 2) * w).sum() / w.sum().clamp_min(1e-9)


@dataclass
class FittedBioSet:
    model: object
    cfg: dict
    glob_mean: np.ndarray
    glob_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    history: list
    best_epoch: int
    n_params: int

    def predict(self, enc: Encoded, device="cpu", batch: int = 512) -> np.ndarray:
        torch, _ = _torch()
        self.model.eval()
        g = (enc.glob - self.glob_mean) / self.glob_std
        e = Encoded(enc.ids, enc.lev, enc.ab, enc.mask, g.astype(np.float32))
        out = []
        with torch.no_grad():
            for s in range(0, len(e.ids), batch):
                lev, ab, m, gl = _to(e.take(slice(s, s + batch)), device)
                out.append(self.model(lev, ab, m, gl).float().cpu().numpy())
        z = np.concatenate(out) if out else np.zeros((0, len(self.y_mean)))
        return z * self.y_std + self.y_mean


def fit_bioset(cfg: dict, vocab: TaxonVocab, tr: Encoded, Ytr: np.ndarray, va: Encoded, Yva: np.ndarray, *,
               seed: int, device: str = "cpu", init_encoder: dict | None = None, log=None,
               n_primary: int | None = None, region_tr: np.ndarray | None = None) -> FittedBioSet:
    """Train on (tr, Ytr), early-stop on (va, Yva). Normalisers are fit on ``tr``/``Ytr`` only.

    Columns ``n_primary:`` of ``Ytr``/``Yva`` are auxiliary outputs: they are weighted by
    ``cfg['aux_weight']`` in the training loss and ignored by the early-stopping criterion.
    """
    torch, _ = _torch()
    from sentinel.core.seed import seed_everything

    seed_everything(seed)
    gm = tr.glob.mean(0) if tr.glob.shape[1] else np.zeros(0, np.float32)
    gs = tr.glob.std(0) if tr.glob.shape[1] else np.ones(0, np.float32)
    gs = np.where(gs > 0, gs, 1.0)
    ym = np.nanmean(Ytr, 0)
    ys = np.nanstd(Ytr, 0)
    ys = np.where(np.isfinite(ys) & (ys > 0), ys, 1.0)
    if not (np.isfinite(tr.glob).all() and np.isfinite(va.glob).all()):
        raise ValueError("non-finite sample descriptors passed to fit_bioset")
    trn = Encoded(tr.ids, tr.lev, tr.ab, tr.mask, ((tr.glob - gm) / gs).astype(np.float32))
    van = Encoded(va.ids, va.lev, va.ab, va.mask, ((va.glob - gm) / gs).astype(np.float32))
    ytr = torch.as_tensor(((Ytr - ym) / ys).astype(np.float32), device=device)
    n_primary = Ytr.shape[1] if n_primary is None else int(n_primary)
    colw = torch.ones(Ytr.shape[1], device=device)
    colw[n_primary:] = float(cfg.get("aux_weight", 0.5))
    colw_val = torch.zeros(Ytr.shape[1], device=device)
    colw_val[:n_primary] = 1.0
    lam_r = float(cfg.get("region_loss_weight", 0.0))
    if lam_r > 0 and region_tr is None:
        raise ValueError("region_loss_weight > 0 needs training-row region codes")
    reg_t = None if region_tr is None else torch.as_tensor(np.asarray(region_tr, np.int64), device=device)
    yva = torch.as_tensor(((Yva - ym) / ys).astype(np.float32), device=device)
    sizes = {lv: vocab.size(lv) for lv in LEVELS}
    model = build_model(cfg, sizes, tr.glob.shape[1], Ytr.shape[1]).to(device)
    lr = float(cfg["lr"])
    if init_encoder is not None:
        missing, unexpected = model.encoder.load_state_dict(init_encoder, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected encoder keys {unexpected[:5]}")
        lr = lr * float(cfg.get("finetune_lr_scale", 1.0))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(cfg["weight_decay"]))
    Ltr, Atr, Mtr, Gtr = _to(trn, device)
    Lva, Ava, Mva, Gva = _to(van, device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    bs = int(cfg["batch_size"])
    n = len(trn.ids)
    best, best_state, best_ep, bad, hist = math.inf, None, -1, 0, []
    for ep in range(int(cfg["max_epochs"])):
        model.train()
        perm = torch.randperm(n, generator=gen, device=device)
        tot = 0.0
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            lev, ab, m = _augment(Ltr[idx], Atr[idx], Mtr[idx], cfg, gen)
            out = model(lev, ab, m, Gtr[idx])
            loss = masked_mse(out, ytr[idx], colw)
            if lam_r > 0:
                loss = loss + lam_r * region_anomaly_mse(out, ytr[idx], reg_t[idx], colw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            vl = float(masked_mse(model(Lva, Ava, Mva, Gva), yva, colw_val)) if len(van.ids) else tot / n
        hist.append((ep, tot / n, vl))
        if vl < best - 1e-4:
            best, best_state, best_ep, bad = vl, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                break
    if best_state is None:
        raise FloatingPointError("validation loss never finite/improving (NaN inputs or targets?)")
    model.load_state_dict(best_state)
    n_params = int(sum(p.numel() for p in model.parameters()))
    if log:
        log(f"best epoch {best_ep} val {best:.4f} ({len(hist)} epochs, {n_params} params)")
    return FittedBioSet(model, cfg, gm.astype(np.float32), gs.astype(np.float32), ym, ys, hist, best_ep, n_params)
