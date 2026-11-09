#!/usr/bin/env python
"""Improve SENTINEL-Fusion (PerceiverIO) on the microcystin task.

DIAGNOSIS (why the shipped neural fusion ~0.75 < GBM stack ~0.82):

  1. MODALITY DEFICIT. The GBM "0.82" stack fuses THREE real modalities
     (satellite spectral sf*, in-situ PIGMENT CHLA/PC, and physicochemical
     DRIVERS temp/TP/TN). The shipped neural fusion trained on only TWO
     (satellite + pigment). On mc>1.6 the drivers alone move the GBM from
     0.776 (2-way) -> 0.823 (3-way): the neural net never saw the third arm.

  2. RELIABILITY-GATE MISMATCH. The shipped ConfidenceGate is *content*-driven
     (reads the projected embedding). But the real per-sample reliability
     signal here is OBSERVABILITY: CHLA is NaN ~34% of rows, PC ~74%, drivers
     patchy, satellite has a cloud fraction. HistGradientBoosting exploits
     missingness natively (it routes NaN); the neural net median-imputes and
     then has no idea which rows are imputed. A reliability gate must read the
     *observation mask + cloud*, not just the content.

  3. RAW-FEATURE PROJECTION. The shipped net applies a single small nonlinear
     projection to raw standardized tabular features and immediately hands them
     to Perceiver cross-attention over a shared 256-d latent. With only 26/2/3
     raw dims per arm, the shared-dim Perceiver is over-parameterized relative
     to signal, so it underfits the interaction the GBM finds by axis-aligned
     splits.

FIX (improved PerceiverIO):
  * THREE modality arms (satellite / pigment / drivers), each with a nonlinear
    per-modality projection into a shared d-space.
  * A genuinely data-driven RELIABILITY GATE that reads the per-sample
    observation mask (+ cloud fraction) for each modality and outputs a gate in
    [0,1] -> multiplicatively scales that arm's token. Missing arms are gated
    toward 0 and additionally carry a learned "no-data" token, so the model
    KNOWS which rows are imputed instead of silently trusting the median.
  * Proper cross-modal attention: a small set of learnable latents cross-attend
    to the 3 gated modality tokens (+ a mask-bias so gated-off arms are
    down-weighted in attention too), followed by latent self-attention, then a
    BCE head. This is a compact PerceiverIO.

EVAL: within-region, GroupKFold-by-site (station-clustered), site-clustered
bootstrap 95% CI. We compute on the SAME rows / SAME splits:
  - GBM 3-way stack  (HistGradientBoosting, native NaN)  == the 0.82 bar
  - current neural   (2-way sat+pigment, content gate)   == the ~0.75 bar
  - improved neural  (3-way, reliability gate, Perceiver)
beats_* only if the site-clustered bootstrap delta CI excludes 0.

All data REAL (mc_fusion_expanded.parquet). No fabrication. Bounded training
(small model, early-stop on a grouped val split inside each fold).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path("/home/bcheng/SENTINEL")
DATA = ROOT / "data/processed/biology/noaa_habs/mc_fusion_expanded.parquet"
OUT = ROOT / "results/benchmarks/architecture/fusion-perceiver.json"
SEED = 0
NBOOT = 2000
NSPLITS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SENTINEL_FILL = 999999.0

# Modality feature groups in the expanded dataset.
SAT_COLS = [f"sf{i}" for i in range(26)]      # satellite spectral / indices
PIG_COLS = ["CHLA", "PC"]                       # in-situ pigment
DRV_COLS = ["temp_C", "TP", "TN"]               # physicochemical drivers
MODALITIES = ["satellite", "pigment", "drivers"]
COLS = {"satellite": SAT_COLS, "pigment": PIG_COLS, "drivers": DRV_COLS}

# Focus thresholds: the ones with enough pos/neg AND the ones where the GBM
# "0.82" claim lives (mc>1.6 -> 0.823; mc>0.3 -> 0.80).
THRESHOLDS = {"mc_gt1.6_WHO_drinking": 1.6, "mc_gt0.3_detection": 0.3,
              "mc_gt8_WHO_recreational": 8.0}

torch.manual_seed(SEED)
np.random.seed(SEED)


# --------------------------------------------------------------------------- #
# Data prep
# --------------------------------------------------------------------------- #
def clean(a: np.ndarray) -> np.ndarray:
    a = a.astype("float64").copy()
    a[a >= SENTINEL_FILL] = np.nan
    return a


def fit_impute_stats(tr: np.ndarray):
    med = np.nanmedian(tr, axis=0, keepdims=True)
    med = np.where(np.isfinite(med), med, 0.0)
    filled = np.where(np.isnan(tr), med, tr)
    mu = filled.mean(axis=0, keepdims=True)
    sd = filled.std(axis=0, keepdims=True) + 1e-6
    return med, mu, sd


def apply_impute(a: np.ndarray, med, mu, sd):
    """Return (z-scored imputed values [N,D], mask [N,D] with 1=OBSERVED)."""
    mask = np.isfinite(a).astype("float32")
    filled = np.where(np.isnan(a), med, a)
    z = ((filled - mu) / sd).astype("float32")
    return z, mask


# --------------------------------------------------------------------------- #
# Improved PerceiverIO fusion (3-way, reliability gate, cross-modal attention)
# --------------------------------------------------------------------------- #
class ModalityArm(nn.Module):
    """Nonlinear per-modality projection: raw z-features -> d-dim token."""

    def __init__(self, in_dim: int, d: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d), nn.GELU(), nn.LayerNorm(d),
            nn.Dropout(dropout), nn.Linear(d, d), nn.LayerNorm(d),
        )
        # learned no-data token used when the arm is entirely missing.
        self.no_data = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.no_data, std=0.02)

    def forward(self, z):
        return self.net(z)


class ReliabilityGate(nn.Module):
    """Data-driven gate from the OBSERVATION mask (+cloud) per modality.

    This is the real reliability signal for this task: which features were
    actually observed vs median-imputed, and how cloudy the tile was. Outputs a
    per-sample gate in [0,1] AND an additive attention-bias logit so a gated-off
    arm is also down-weighted in cross-attention.
    """

    def __init__(self, mask_dims: dict, d: int):
        super().__init__()
        self.heads = nn.ModuleDict()
        for m, md in mask_dims.items():
            self.heads[m] = nn.Sequential(
                nn.Linear(md, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        # start near-open
        for m in mask_dims:
            nn.init.constant_(self.heads[m][-1].bias, 1.0)

    def forward(self, masks: dict):
        gates, bias = {}, {}
        for m, feat in masks.items():
            logit = self.heads[m](feat).squeeze(-1)      # [B]
            gates[m] = torch.sigmoid(logit)              # [B]
            bias[m] = logit                              # [B] attn bias
        return gates, bias


class CrossModalAttention(nn.Module):
    """Latents cross-attend to gated modality tokens, with a mask-bias."""

    def __init__(self, d: int, n_latents: int, heads: int, dropout: float):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, d) * 0.02)
        self.cross = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln_s = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2 * d, d))

    def forward(self, tokens, attn_bias):
        # tokens: [B, M, d]; attn_bias: [B, M] additive key logits.
        B = tokens.shape[0]
        q = self.ln_q(self.latents.unsqueeze(0).expand(B, -1, -1))
        kv = self.ln_kv(tokens)
        L = q.shape[1]
        # additive attention mask over keys: [B*heads, L, M] via key_padding is
        # only 0/-inf; instead fold the bias into keys through a per-key scale.
        # Down-weight low-reliability arms by masking keys where bias very low.
        key_pad = (attn_bias < -6.0)  # effectively-off arms
        out, _ = self.cross(q, kv, kv, key_padding_mask=key_pad, need_weights=False)
        z = q + out
        s, _ = self.self_attn(self.ln_s(z), self.ln_s(z), self.ln_s(z), need_weights=False)
        z = z + s
        z = z + self.ff(z)
        return z.mean(dim=1)  # pool latents -> [B, d]


class ImprovedFusion(nn.Module):
    def __init__(self, in_dims: dict, mask_dims: dict, d: int = 96,
                 n_latents: int = 8, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.arms = nn.ModuleDict({m: ModalityArm(in_dims[m], d, dropout) for m in MODALITIES})
        self.gate = ReliabilityGate(mask_dims, d)
        self.attn = CrossModalAttention(d, n_latents, heads, dropout)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, zs: dict, masks: dict, present: dict):
        B = next(iter(zs.values())).shape[0]
        gates, bias = self.gate(masks)
        toks, biases = [], []
        for m in MODALITIES:
            arm = self.arms[m]
            tok = arm(zs[m])                              # [B,d]
            # replace fully-missing rows with the learned no-data token
            pres = present[m].unsqueeze(-1)               # [B,1]
            tok = pres * tok + (1 - pres) * arm.no_data.unsqueeze(0)
            tok = tok * gates[m].unsqueeze(-1)            # reliability gate
            # if arm absent for a row, force attn bias very negative
            b = torch.where(present[m] > 0, bias[m], torch.full_like(bias[m], -1e4))
            toks.append(tok)
            biases.append(b)
        tokens = torch.stack(toks, dim=1)                 # [B,M,d]
        attn_bias = torch.stack(biases, dim=1)            # [B,M]
        pooled = self.attn(tokens, attn_bias)
        return self.head(pooled).squeeze(-1)


# --------------------------------------------------------------------------- #
# Current-style neural baseline (2-way, content gate) reproduced compactly
# --------------------------------------------------------------------------- #
class CurrentFusion(nn.Module):
    """Reproduces the shipped mc_neural_fusion_fixed design: 2 arms
    (satellite+pigment), nonlinear projection, CONTENT gate, cross-attn."""

    def __init__(self, sat_dim, pig_dim, d=256, dropout=0.3):
        super().__init__()
        def proj(i):
            return nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, d))
        self.sat = proj(sat_dim)
        self.pig = proj(pig_dim)
        self.gate_s = nn.Linear(d, 1)
        self.gate_p = nn.Linear(d, 1)
        self.attn = nn.MultiheadAttention(d, 8, dropout=dropout, batch_first=True)
        self.lat = nn.Parameter(torch.randn(4, d) * 0.02)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, sat, pig):
        ts = self.sat(sat) * torch.sigmoid(self.gate_s(self.sat(sat)))
        tp = self.pig(pig) * torch.sigmoid(self.gate_p(self.pig(pig)))
        toks = torch.stack([ts, tp], dim=1)
        B = sat.shape[0]
        q = self.lat.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.attn(q, toks, toks, need_weights=False)
        return self.head(out.mean(1)).squeeze(-1)


# --------------------------------------------------------------------------- #
# Training helpers
# --------------------------------------------------------------------------- #
def train_torch(model, batch_fn, tr_data, va_data, y_tr, y_va, epochs=200,
                lr=1.5e-3, wd=2e-2, patience=25):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    pos = float(y_tr.sum()); neg = float(len(y_tr) - pos)
    pw = torch.tensor(neg / max(pos, 1.0), device=DEVICE)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=pw)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(y_va, dtype=torch.float32, device=DEVICE)
    n = len(y_tr); bs = 256
    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            logit = model(*batch_fn(tr_data, idx))
            loss = lossfn(logit, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vlogit = model(*batch_fn(va_data, None)).cpu().numpy()
        try:
            auc = roc_auc_score(y_va, vlogit)
        except ValueError:
            auc = 0.5
        if auc > best_auc + 1e-4:
            best_auc = auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def t(a):
    return torch.tensor(a, dtype=torch.float32, device=DEVICE)


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def site_boot_delta(y, pa, pb, sites, seed=SEED, B=NBOOT):
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
        d.append(roc_auc_score(yi, pa[idx]) - roc_auc_score(yi, pb[idx]))
    d = np.array(d)
    return {"delta_median": round(float(np.median(d)), 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4),
                     round(float(np.percentile(d, 97.5)), 4)],
            "P_gt0": round(float((d > 0).mean()), 4)}


def site_boot_abs(y, p, sites, seed=SEED, B=NBOOT):
    rng = np.random.default_rng(seed)
    uniq = np.unique(sites)
    by = {s: np.where(sites == s)[0] for s in uniq}
    a = []
    for _ in range(B):
        idx = np.concatenate([by[s] for s in rng.choice(uniq, len(uniq), replace=True)])
        yi = y[idx]
        if yi.min() != yi.max():
            a.append(roc_auc_score(yi, p[idx]))
    a = np.array(a)
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_label(df, y, sites, groups):
    raw = {m: clean(df[COLS[m]].to_numpy("float32")) for m in MODALITIES}
    gkf = GroupKFold(n_splits=NSPLITS)

    p_gbm = np.zeros(len(y))
    p_cur = np.zeros(len(y))
    p_imp = np.zeros(len(y))

    for fold, (tr, te) in enumerate(gkf.split(raw["satellite"], y, groups)):
        # inner grouped val split for early stopping
        g_tr = groups[tr]
        inner = GroupKFold(n_splits=4)
        itr, iva = next(inner.split(raw["satellite"][tr], y[tr], g_tr))
        tr_core, va = tr[itr], tr[iva]

        # ---- GBM 3-way: concat raw (NaN native), no z-score needed ----
        Xall = np.hstack([raw[m] for m in MODALITIES])
        gbm = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=400, l2_regularization=1.0,
            class_weight="balanced", random_state=SEED)
        gbm.fit(Xall[tr], y[tr])
        p_gbm[te] = gbm.predict_proba(Xall[te])[:, 1]

        # ---- prepare per-modality z + mask, fit stats on tr_core only ----
        Z, MASK, PRESENT, MDIMS = {}, {}, {}, {}
        for m in MODALITIES:
            med, mu, sd = fit_impute_stats(raw[m][tr_core])
            z, mask = apply_impute(raw[m], med, mu, sd)
            Z[m] = z
            MASK[m] = mask
            PRESENT[m] = mask.max(axis=1)          # 1 if any feature observed
            MDIMS[m] = mask.shape[1]
        # cloud fraction as extra reliability feature for satellite
        cloud = clean(df[["cloud"]].to_numpy("float32"))
        cmed, cmu, csd = fit_impute_stats(cloud[tr_core])
        cz, _ = apply_impute(cloud, cmed, cmu, csd)
        MASK["satellite"] = np.hstack([MASK["satellite"], cz])  # append cloud signal
        MDIMS["satellite"] += 1

        in_dims = {m: Z[m].shape[1] for m in MODALITIES}
        mask_dims = {m: MDIMS[m] for m in MODALITIES}

        # pre-pack full tensors on device; index by absolute rows for tr/va/te
        zs_all = {m: t(Z[m]) for m in MODALITIES}
        masks_all = {m: t(MASK[m]) for m in MODALITIES}
        pres_all = {m: t(PRESENT[m]) for m in MODALITIES}

        def view(rows):
            return ({m: zs_all[m][rows] for m in MODALITIES},
                    {m: masks_all[m][rows] for m in MODALITIES},
                    {m: pres_all[m][rows] for m in MODALITIES})
        tr_view = view(tr_core); va_view = view(va)

        def batch_imp(data, idx):
            zs, masks, pres = data
            if idx is None:
                return zs, masks, pres
            return ({m: zs[m][idx] for m in MODALITIES},
                    {m: masks[m][idx] for m in MODALITIES},
                    {m: pres[m][idx] for m in MODALITIES})
        torch.manual_seed(SEED + fold)
        imp = ImprovedFusion(in_dims, mask_dims)
        imp = train_torch(imp, batch_imp, tr_view, va_view, y[tr_core], y[va])
        imp.eval()
        with torch.no_grad():
            te_view = view(te)
            p_imp[te] = torch.sigmoid(imp(*te_view)).cpu().numpy()

        # ---- current-style neural (2-way sat+pigment, content gate) ----
        sat_all = t(Z["satellite"]); pig_all = t(Z["pigment"])
        def batch_cur(data, idx):
            s, p = data
            if idx is None:
                return s, p
            return s[idx], p[idx]
        torch.manual_seed(SEED + fold)
        cur = CurrentFusion(Z["satellite"].shape[1], Z["pigment"].shape[1])
        cur = train_torch(cur, batch_cur, (sat_all[tr_core], pig_all[tr_core]),
                          (sat_all[va], pig_all[va]), y[tr_core], y[va])
        cur.eval()
        with torch.no_grad():
            p_cur[te] = torch.sigmoid(cur(sat_all[te], pig_all[te])).cpu().numpy()

    a_gbm = float(roc_auc_score(y, p_gbm))
    a_cur = float(roc_auc_score(y, p_cur))
    a_imp = float(roc_auc_score(y, p_imp))

    res = {
        "pos": int(y.sum()), "neg": int(len(y) - y.sum()),
        "gbm_auroc": round(a_gbm, 4), "gbm_ci95": site_boot_abs(y, p_gbm, sites),
        "current_neural_auroc": round(a_cur, 4), "current_ci95": site_boot_abs(y, p_cur, sites),
        "improved_neural_auroc": round(a_imp, 4), "improved_ci95": site_boot_abs(y, p_imp, sites),
        "improved_vs_gbm": site_boot_delta(y, p_imp, p_gbm, sites),
        "improved_vs_current": site_boot_delta(y, p_imp, p_cur, sites),
        "gbm_vs_current": site_boot_delta(y, p_gbm, p_cur, sites),
    }
    res["beats_gbm"] = bool(res["improved_vs_gbm"]["ci95"][0] > 0)
    res["beats_current"] = bool(res["improved_vs_current"]["ci95"][0] > 0)
    return res


def main():
    df = pd.read_parquet(DATA)
    df = df.reset_index(drop=True)
    sites = df["site"].astype(str).to_numpy()
    groups = sites
    mc = df["mc"].to_numpy("float64")

    out = {
        "dataset": DATA.name, "n": int(len(df)), "n_sites": int(df["site"].nunique()),
        "device": DEVICE, "cv": f"GroupKFold(site) x{NSPLITS}",
        "modalities": {"satellite": SAT_COLS, "pigment": PIG_COLS, "drivers": DRV_COLS},
        "diagnosis": (
            "Shipped neural fusion (~0.75) trailed the GBM stack (~0.82) for three "
            "reasons: (1) it fused only 2 of the 3 real modalities (missing the "
            "physicochemical drivers that lift the GBM 0.776->0.823 on mc>1.6); "
            "(2) its ConfidenceGate was content-driven, not observability-driven, so "
            "it could not exploit the missingness/cloud signal the HGB routes "
            "natively; (3) raw few-dim tabular features fed a 256-d Perceiver that "
            "underfit the axis-aligned interaction. Fix: 3-way arms + a data-driven "
            "reliability gate on the observation mask+cloud + a compact cross-modal "
            "Perceiver with learned no-data tokens."),
        "coverage": {m: round(float(np.isfinite(clean(df[COLS[m]].to_numpy("float32"))).any(1).mean()), 3)
                     for m in MODALITIES},
        "labels": {},
    }
    for name, thr in THRESHOLDS.items():
        y = (mc > thr).astype(int)
        if y.sum() < 20 or (len(y) - y.sum()) < 20:
            out["labels"][name] = {"skipped": "too few pos/neg", "pos": int(y.sum())}
            continue
        print(f"\n=== {name} (pos={int(y.sum())}) ===", flush=True)
        r = run_label(df, y, sites, groups)
        out["labels"][name] = r
        print(f"  GBM={r['gbm_auroc']:.3f}  current={r['current_neural_auroc']:.3f}  "
              f"IMPROVED={r['improved_neural_auroc']:.3f}", flush=True)
        print(f"  improved vs GBM     Δ={r['improved_vs_gbm']['delta_median']:+.3f} "
              f"CI{r['improved_vs_gbm']['ci95']} P={r['improved_vs_gbm']['P_gt0']} "
              f"beats_gbm={r['beats_gbm']}", flush=True)
        print(f"  improved vs current Δ={r['improved_vs_current']['delta_median']:+.3f} "
              f"CI{r['improved_vs_current']['ci95']} P={r['improved_vs_current']['P_gt0']} "
              f"beats_current={r['beats_current']}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()
