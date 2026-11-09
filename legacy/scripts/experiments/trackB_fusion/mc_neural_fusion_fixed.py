#!/usr/bin/env python
"""Validate the CORRECTED neural Perceiver fusion on the microcystin task.

After fixing the four audited fusion bugs (nonlinear ModalityProjection,
data-driven ConfidenceGate, batched registry, retired circular label), this
script asks the ship-gate question directly:

    Does the fixed *neural* fusion (SENTINEL's own Perceiver submodules:
    ModalityProjection -> ConfidenceGate -> PerceiverCrossAttention -> head)
    beat BOTH single modalities on microcystin, matching the GBM win (~0.78)?

Inputs: satellite spectral (sf*) + in-situ pigment (CHLA, PC) from
mc_fusion_dataset.parquet. Label = microcystin > threshold (non-circular:
independent of both inputs). Station-grouped CV (GroupKFold on site).
Metric: AUROC + site-clustered bootstrap 95% CI on fusion - best-single.

This exercises the FIXED submodules end to end:
  * ModalityProjection: now Linear->GELU->LayerNorm->Linear (bug b).
  * ConfidenceGate: now data-driven per-sample FiLM gate (bug c).
  * Batched forward through PerceiverCrossAttention (stateless, so bug d's
    registry pathology is avoided by construction; every row independent).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sentinel.models.fusion.projections import ModalityProjection            # noqa: E402
from sentinel.models.fusion.confidence_gating import ConfidenceGate          # noqa: E402
from sentinel.models.fusion.latent_array import LatentArray                  # noqa: E402
from sentinel.models.fusion.perceiver_attention import PerceiverCrossAttention  # noqa: E402
from sentinel.models.fusion.embedding_registry import MODALITY_IDS, NUM_MODALITIES  # noqa: E402

DATA = ROOT / "data/processed/biology/noaa_habs/mc_fusion_dataset.parquet"
OUT = ROOT / "results/trackB_fusion/mc_neural_fusion_fixed.json"
SEED = 0
NBOOT = 2000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLDS = {"mc_gt1.6_WHO_drinking": 1.6, "mc_gt0.3_detection": 0.3, "mc_gt8_WHO_recreational": 8.0}


class NeuralFusion(nn.Module):
    """Fixed SENTINEL fusion submodules wired for batched two-modality training.

    satellite arm -> ModalityProjection(sat_dim->256)
    sensor arm     -> ModalityProjection(insitu_dim->256)   (in-situ pigment)
    -> ConfidenceGate (data-driven per-sample) -> Perceiver cross-attn -> head
    Passing only one arm (the other left as the Perceiver's learned no-data
    token) yields the single-modality ablations inside the SAME architecture.
    """

    def __init__(self, sat_dim: int, ins_dim: int, num_latents: int = 32, dropout: float = 0.3):
        super().__init__()
        self.sat_proj = ModalityProjection(sat_dim, 256)
        self.ins_proj = ModalityProjection(ins_dim, 256)
        self.gate = ConfidenceGate(256)
        self.latent_array = LatentArray(num_latents=num_latents, latent_dim=256)
        self.perceiver = PerceiverCrossAttention(256, num_heads=8, num_process_layers=2, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, sat=None, ins=None):
        ref = sat if sat is not None else ins
        B, device = ref.shape[0], ref.device
        modemb = {mid: None for mid in MODALITY_IDS}
        if sat is not None:
            modemb["satellite"] = self.sat_proj(sat)
        if ins is not None:
            # in-situ pigment carried on the "sensor" modality slot
            modemb["sensor"] = self.ins_proj(ins)
        # Data-driven confidence gating (fixed bug c): per-sample FiLM gate.
        modemb = self.gate(modemb, {})
        lat = self.latent_array.get_latents(B).to(device)
        tbias = torch.zeros(B, NUM_MODALITIES, device=device)
        _, fused, _ = self.perceiver(lat, modemb, tbias)
        return self.head(fused).squeeze(-1)


SENTINEL_FILL = 999999.0  # sentinel "no data" code used in the pigment columns


def prep(train, *arrs, add_missing=False):
    """NaN-safe standardization fit on *train* only.

    The in-situ pigment columns contain NaNs and a 999999 sentinel; the GBM
    baseline handled these natively, so for a fair neural comparison we
    median-impute (train stats) and optionally append a binary missingness
    indicator so the "no data" information is preserved rather than fabricated.
    """
    def clean(a):
        a = a.astype("float64").copy()
        a[a >= SENTINEL_FILL] = np.nan
        return a

    tr = clean(train)
    others = [clean(a) for a in arrs]
    med = np.nanmedian(tr, axis=0, keepdims=True)
    med = np.where(np.isfinite(med), med, 0.0)
    mu = np.nanmean(np.where(np.isnan(tr), med, tr), axis=0, keepdims=True)
    sd = np.nanstd(np.where(np.isnan(tr), med, tr), axis=0, keepdims=True) + 1e-6

    def finish(a):
        miss = np.isnan(a).astype("float32")
        filled = np.where(np.isnan(a), med, a)
        z = ((filled - mu) / sd).astype("float32")
        if add_missing:
            z = np.hstack([z, miss.astype("float32")])
        return z

    return [finish(a) for a in (tr, *others)]


def zscore(train, *arrs):
    """Backwards-compatible NaN-safe standardizer (no missingness channel)."""
    return prep(train, *arrs, add_missing=False)


def train_one(model, Xtr, ytr, arm, epochs=120, lr=1e-3, wd=1e-2):
    model.to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    pos = float(ytr.sum()); neg = float(len(ytr) - ytr.sum())
    pw = torch.tensor(neg / max(pos, 1.0), device=DEVICE)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=pw)
    yt = torch.tensor(ytr, dtype=torch.float32, device=DEVICE)
    sat_t = torch.tensor(Xtr[0], dtype=torch.float32, device=DEVICE) if Xtr[0] is not None else None
    ins_t = torch.tensor(Xtr[1], dtype=torch.float32, device=DEVICE) if Xtr[1] is not None else None
    n = len(ytr); bs = 256
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            s = sat_t[idx] if sat_t is not None else None
            m = ins_t[idx] if ins_t is not None else None
            logit = model(sat=s, ins=m)
            loss = lossfn(logit, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def predict(model, Xte):
    model.eval()
    s = torch.tensor(Xte[0], dtype=torch.float32, device=DEVICE) if Xte[0] is not None else None
    m = torch.tensor(Xte[1], dtype=torch.float32, device=DEVICE) if Xte[1] is not None else None
    return torch.sigmoid(model(sat=s, ins=m)).cpu().numpy()


def oof(sat, ins, y, groups, arm, sat_dim, ins_dim, ns):
    """arm in {'fusion','satellite','insitu'}. Returns OOF prob vector.

    The in-situ pigment arm carries a missingness indicator channel (bug-free
    NaN handling), so its effective input dim is ins.shape[1] * 2.
    """
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=ns).split(sat, y, groups):
        sat_tr, sat_te = prep(sat[tr], sat[te], add_missing=False)
        ins_tr, ins_te = prep(ins[tr], ins[te], add_missing=True)
        torch.manual_seed(SEED)
        model = NeuralFusion(sat_dim, ins_dim)
        if arm == "fusion":
            Xtr, Xte = (sat_tr, ins_tr), (sat_te, ins_te)
        elif arm == "satellite":
            Xtr, Xte = (sat_tr, None), (sat_te, None)
        else:
            Xtr, Xte = (None, ins_tr), (None, ins_te)
        train_one(model, Xtr, y[tr], arm)
        p[te] = predict(model, Xte)
    return p


def site_bootstrap_delta(y, pf, pb, sites, seed=SEED, B=NBOOT):
    rng = np.random.default_rng(seed)
    uniq = np.unique(sites)
    by = {s: np.where(sites == s)[0] for s in uniq}
    d = []
    for _ in range(B):
        ch = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([by[s] for s in ch])
        yi = y[idx]
        if yi.min() == yi.max():
            continue
        d.append(roc_auc_score(yi, pf[idx]) - roc_auc_score(yi, pb[idx]))
    d = np.array(d)
    return {"delta_median": round(float(np.median(d)), 3),
            "ci95": [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)],
            "P_delta_gt0": round(float((d > 0).mean()), 3)}


def main():
    df = pq.read_table(DATA).to_pandas()
    sf = [c for c in df.columns if c.startswith("sf")]
    sat = df[sf].to_numpy("float32")
    ins = df[["CHLA", "PC"]].to_numpy("float32")
    sites = df["site"].astype(str).to_numpy()
    mc = df["mc"].to_numpy("float64")
    ns = min(5, df["site"].nunique())
    # in-situ arm gets a missingness-indicator channel (NaN-safe), so 2x width.
    sat_dim, ins_dim = sat.shape[1], ins.shape[1] * 2

    results = {"dataset": DATA.name, "device": DEVICE, "n": int(len(df)),
               "n_sites": int(df["site"].nunique()), "sat_dim": sat_dim, "ins_dim": ins_dim,
               "model": "NeuralFusion (fixed ModalityProjection + data-driven ConfidenceGate + Perceiver)",
               "cv": f"GroupKFold(site) x{ns}", "labels": {}}
    for name, thr in THRESHOLDS.items():
        y = (mc > thr).astype(int)
        if y.sum() < 15 or (len(y) - y.sum()) < 15:
            results["labels"][name] = {"skipped": "too few pos/neg", "pos": int(y.sum())}
            continue
        p_sat = oof(sat, ins, y, sites, "satellite", sat_dim, ins_dim, ns)
        p_ins = oof(sat, ins, y, sites, "insitu", sat_dim, ins_dim, ns)
        p_fus = oof(sat, ins, y, sites, "fusion", sat_dim, ins_dim, ns)
        a_sat, a_ins, a_fus = (float(roc_auc_score(y, p)) for p in (p_sat, p_ins, p_fus))
        best_p, best_name, best_a = ((p_ins, "insitu", a_ins) if a_ins >= a_sat
                                     else (p_sat, "satellite", a_sat))
        boot = site_bootstrap_delta(y, p_fus, best_p, sites)
        res = {"pos": int(y.sum()), "satellite_auroc": round(a_sat, 3),
               "insitu_auroc": round(a_ins, 3), "fusion_auroc": round(a_fus, 3),
               "best_single": best_name, "fusion_minus_best": round(a_fus - best_a, 3),
               "site_bootstrap_delta": boot, "fusion_helps": bool(boot["ci95"][0] > 0)}
        results["labels"][name] = res
        print(f"[{name}] sat={a_sat:.3f} insitu={a_ins:.3f} FUSION={a_fus:.3f} "
              f"(Δ vs best {res['fusion_minus_best']:+.3f}, site-boot CI {boot['ci95']} "
              f"P={boot['P_delta_gt0']}) helps={res['fusion_helps']}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()
