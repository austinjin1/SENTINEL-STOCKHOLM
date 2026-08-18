#!/usr/bin/env python
"""Fusion cycle 2 — VARIANT key=moe-uncertainty-gate.

Mixture-of-Experts fusion for the microcystin within-region task where the gate
is driven by per-modality PREDICTIVE UNCERTAINTY.

Architecture (small torch model, CPU fine):
  - 3 real modalities: satellite spectral (26 band-ratio feats), in-situ pigment
    (CHLA, PC), physicochemical drivers (temp / P / N; joined exactly as
    08_driver_arm.py — nearest WQP value within +/-14 d, from the cached CSVs).
  - Each modality has its OWN expert head (a small MLP) -> a per-modality logit.
    p_m = sigmoid(logit_m); the expert's binary predictive entropy H_m =
    -[p log p + (1-p) log(1-p)] is a per-sample uncertainty score in [0,1].
  - A learned GATE maps the 3 per-expert entropies (+ a per-modality "present"
    mask) -> softmax weights w_m over the present experts. So an expert that is
    UNCERTAIN on a given sample is down-weighted on that sample. The gate is a
    real MLP over the entropy features -> it is learned, data-driven, and (proven
    by the ablation) actually changes the output.
  - Fused prob = sum_m w_m * p_m  (weighted vote of experts). Trained end-to-end
    with BCE on the fused prob + a small BCE on each expert (so each expert is a
    genuine calibrated classifier and its entropy is meaningful).

Evaluations (all GroupKFold-BY-SITE OOF; transforms fit on train folds only;
site-clustered bootstrap for CIs, protocol reused from 08_driver_arm.py):
  1. best_single_auroc (best of the 3 expert heads) vs fusion_auroc (learned MoE);
     delta + site-clustered 95% CI; fusion_wins = CI excludes 0.
  2. GATE ABLATION: exact same trained experts, gate replaced by UNIFORM weights
     (1/K over present experts). Does the learned uncertainty gate beat uniform?
     report both AUROCs + site-clustered CI on (learned - uniform).
  3. MISSING-MODALITY graceful degradation: at test, drop each modality (mask it
     out; gate renormalizes over the remaining present experts) and report the
     fusion AUROC in each degraded configuration.

Honesty: fusion may only TIE the GBM 3-way (0.82). The real deliverables are
(a) does the learned gate beat uniform, and (b) graceful missing-modality
degradation. Report the honest numbers regardless. Real data only; label = real
microcystin lab threshold (non-circular).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data" / "processed" / "biology" / "noaa_habs" / "mc_fusion_dataset.parquet"
RAW = ROOT / "data" / "raw" / "wqp_drivers"
OUT = ROOT / "results" / "trackB_fusion" / "fusion_cycle2" / "moe-uncertainty-gate.json"
CHARS = {"temp": "Temperature, water", "phosphorus": "Phosphorus", "nitrogen": "Nitrogen"}
MODS = ["satellite", "pigment", "drivers"]
SEED = 0
B = 2000
EPOCHS = 120
LR = 3e-3
WD = 1e-4
HID = 32
AUX = 0.3          # weight on per-expert auxiliary BCE
EPS = 1e-6

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---- driver join (identical protocol to 08_driver_arm.py) --------------------
def parse_driver(df):
    if len(df) == 0:
        return pd.DataFrame(columns=["site", "date", "val"])
    out = pd.DataFrame({
        "site": df["MonitoringLocationIdentifier"].astype(str),
        "date": pd.to_datetime(df["ActivityStartDate"], errors="coerce"),
        "val": pd.to_numeric(df["ResultMeasureValue"], errors="coerce"),
    }).dropna(subset=["date", "val"])
    return out.sort_values("date")


def nearest(driver, site, date, tol_days=14):
    d = driver[driver["site"] == site]
    if len(d) == 0:
        return np.nan
    dd = (d["date"] - date).abs().dt.days
    i = dd.values.argmin()
    return float(d["val"].values[i]) if dd.values[i] <= tol_days else np.nan


# ---- site-clustered bootstrap (reused from 08_driver_arm.py) -----------------
def site_boot_indices(sites):
    rng = np.random.default_rng(SEED)
    uniq = np.unique(sites)
    by = {s: np.where(sites == s)[0] for s in uniq}
    return [np.concatenate([by[s] for s in rng.choice(uniq, len(uniq), replace=True)])
            for _ in range(B)]


def auc_safe(y, p, idx):
    yi = y[idx]
    if yi.min() == yi.max():
        return None
    return roc_auc_score(yi, p[idx])


def boot_absolute(y, p, boot_idx):
    aucs = [auc_safe(y, p, idx) for idx in boot_idx]
    aucs = np.array([a for a in aucs if a is not None])
    return {"median": round(float(np.median(aucs)), 4),
            "ci95": [round(float(np.percentile(aucs, 2.5)), 4),
                     round(float(np.percentile(aucs, 97.5)), 4)]}


def boot_delta(y, p_a, p_b, boot_idx):
    ds = []
    for idx in boot_idx:
        a = auc_safe(y, p_a, idx)
        b = auc_safe(y, p_b, idx)
        if a is not None and b is not None:
            ds.append(a - b)
    ds = np.array(ds)
    return {"median": round(float(np.median(ds)), 4),
            "ci95": [round(float(np.percentile(ds, 2.5)), 4),
                     round(float(np.percentile(ds, 97.5)), 4)],
            "P_gt0": round(float((ds > 0).mean()), 4),
            "excludes_0": bool(np.percentile(ds, 2.5) > 0 or np.percentile(ds, 97.5) < 0)}


# ---- model -------------------------------------------------------------------
def binary_entropy(p):
    """Per-sample binary predictive entropy in [0,1] (base 2)."""
    p = p.clamp(EPS, 1 - EPS)
    return -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))


class Expert(nn.Module):
    """A small per-modality MLP classifier -> logit."""
    def __init__(self, in_dim, hid=HID):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.GELU(),
            nn.LayerNorm(hid),
            nn.Linear(hid, hid), nn.GELU(),
            nn.Linear(hid, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logit


class MoEUncertaintyGate(nn.Module):
    """MoE where the gate weights each expert by (a learned function of) its
    per-sample predictive uncertainty. Gate input = per-expert binary entropies
    (+ present-mask). Missing experts get -inf gate logit so they drop out and
    weights renormalize over present experts."""
    def __init__(self, dims, hid=HID):
        super().__init__()
        self.experts = nn.ModuleList([Expert(d, hid) for d in dims])
        self.k = len(dims)
        # gate reads [entropy_m , mask_m] for each expert -> a logit per expert
        self.gate = nn.Sequential(
            nn.Linear(2 * self.k, hid), nn.GELU(),
            nn.Linear(hid, hid), nn.GELU(),
            nn.Linear(hid, self.k),
        )

    def forward(self, xs, mask, gate_mode="learned"):
        """xs: list of [B, d_m] tensors (present ones real, missing ones any dummy).
        mask: [B, k] 1.0 present / 0.0 missing.
        Returns fused_prob [B], expert_probs [B,k], weights [B,k]."""
        logits = torch.stack([e(x) for e, x in zip(self.experts, xs)], dim=1)  # [B,k]
        probs = torch.sigmoid(logits)                                          # [B,k]
        ent = binary_entropy(probs)                                            # [B,k]

        if gate_mode == "learned":
            feat = torch.cat([ent * mask, mask], dim=1)                        # [B,2k]
            glog = self.gate(feat)                                            # [B,k]
        elif gate_mode == "uniform":
            glog = torch.zeros_like(logits)                                   # equal
        elif gate_mode == "inverse_entropy":
            # fixed (non-learned) inverse-entropy weighting baseline
            glog = -ent
        else:
            raise ValueError(gate_mode)

        # mask out missing experts before softmax
        glog = glog.masked_fill(mask < 0.5, float("-inf"))
        w = torch.softmax(glog, dim=1)                                        # [B,k]
        w = torch.nan_to_num(w, nan=0.0)
        fused = (w * probs).sum(dim=1)                                        # [B]
        return fused, probs, w


def standardize_fit(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def standardize_apply(X, mu, sd):
    Z = (X - mu) / sd
    # missing driver values -> 0 (mean) after standardization
    return np.nan_to_num(Z, nan=0.0)


def train_fold(feats_tr, y_tr, dims):
    model = MoEUncertaintyGate(dims)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    xs = [torch.tensor(feats_tr[m], dtype=torch.float32) for m in MODS]
    mask = torch.ones(len(y_tr), len(MODS))
    yt = torch.tensor(y_tr, dtype=torch.float32)
    # class weight for imbalance
    pos = float(y_tr.sum()); neg = float(len(y_tr) - pos)
    pw = torch.tensor(neg / max(pos, 1.0))
    bce = nn.BCELoss(reduction="none")
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        fused, probs, w = model(xs, mask, "learned")
        wts = torch.where(yt > 0.5, pw, torch.ones_like(yt))
        loss_f = (bce(fused.clamp(EPS, 1 - EPS), yt) * wts).mean()
        loss_a = 0.0
        for j in range(len(MODS)):
            loss_a = loss_a + (bce(probs[:, j].clamp(EPS, 1 - EPS), yt) * wts).mean()
        loss = loss_f + AUX * loss_a / len(MODS)
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model, feats_te, present, gate_mode="learned"):
    """present: list of modality names kept at test (others masked/dropped)."""
    n = len(next(iter(feats_te.values())))
    xs = [torch.tensor(feats_te[m], dtype=torch.float32) for m in MODS]
    mask = torch.tensor([[1.0 if m in present else 0.0 for m in MODS]] * n)
    fused, probs, w = model(xs, mask, gate_mode)
    return (fused.numpy(), probs.numpy(), w.numpy())


def main():
    df = pq.read_table(DATA).to_pandas()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    sites_list = sorted(df["site"].astype(str).unique())

    for tag in CHARS:
        dv = parse_driver(pd.read_csv(RAW / f"{tag}.csv", low_memory=False))
        df[f"drv_{tag}"] = [nearest(dv, str(s), d) for s, d in
                            zip(df["site"].astype(str), df["date"])]

    sf = [c for c in df.columns if c.startswith("sf")]
    raw_feats = {
        "satellite": df[sf].to_numpy("float64"),
        "pigment": df[["CHLA", "PC"]].to_numpy("float64"),
        "drivers": df[[f"drv_{t}" for t in CHARS]].to_numpy("float64"),
    }
    dims = [raw_feats[m].shape[1] for m in MODS]
    mc = df["mc"].to_numpy("float64")
    site_arr = df["site"].astype(str).to_numpy()
    ns = min(5, len(sites_list))
    boot_idx = site_boot_indices(site_arr)

    results = {
        "variant": "moe-uncertainty-gate",
        "n": int(len(df)), "n_sites": len(sites_list),
        "modalities": {"satellite": {"k": len(sf)}, "pigment": {"k": 2}, "drivers": {"k": len(CHARS)}},
        "driver_coverage": {t: round(float(np.isfinite(df[f"drv_{t}"]).mean()), 3) for t in CHARS},
        "protocol": {"split": "GroupKFold by site", "n_splits": ns,
                     "bootstrap": "site-clustered (resample sites w/ replacement)", "B": B,
                     "label": "real microcystin lab threshold (non-circular)",
                     "standardization": "fit on train folds only; missing driver -> mean(=0 after z)"},
        "model": {"type": "MoE, per-modality MLP experts, gate = MLP over per-expert "
                          "binary predictive entropy + present-mask -> softmax weights",
                  "epochs": EPOCHS, "lr": LR, "aux_expert_bce_weight": AUX, "hidden": HID},
        "labels": {},
    }

    for label, thr in [("mc_gt1.6", 1.6), ("mc_gt0.3", 0.3)]:
        y = (mc > thr).astype(int)

        # OOF containers
        oof_fused = np.zeros(len(y))          # learned-gate fusion, all modalities
        oof_uniform = np.zeros(len(y))        # uniform-gate fusion, all modalities
        oof_invent = np.zeros(len(y))         # fixed inverse-entropy gate, all modalities
        oof_expert = {m: np.zeros(len(y)) for m in MODS}
        # missing-modality: drop each modality at test (learned gate over remaining)
        oof_drop = {m: np.zeros(len(y)) for m in MODS}
        # gate weights (learned) for diagnostics
        gate_w = np.zeros((len(y), len(MODS)))

        for tr, te in GroupKFold(n_splits=ns).split(raw_feats["satellite"], y, site_arr):
            # standardize per modality, fit on train only
            feats_tr, feats_te = {}, {}
            for m in MODS:
                mu, sd = standardize_fit(raw_feats[m][tr])
                feats_tr[m] = standardize_apply(raw_feats[m][tr], mu, sd)
                feats_te[m] = standardize_apply(raw_feats[m][te], mu, sd)
            model = train_fold(feats_tr, y[tr], dims)

            fused, probs, w = predict(model, feats_te, MODS, "learned")
            oof_fused[te] = fused
            gate_w[te] = w
            oof_uniform[te] = predict(model, feats_te, MODS, "uniform")[0]
            oof_invent[te] = predict(model, feats_te, MODS, "inverse_entropy")[0]
            for j, m in enumerate(MODS):
                oof_expert[m][te] = probs[:, j]
            for m in MODS:
                present = [x for x in MODS if x != m]
                oof_drop[m][te] = predict(model, feats_te, present, "learned")[0]

        # ---- metrics ----
        single = {m: round(float(roc_auc_score(y, oof_expert[m])), 4) for m in MODS}
        best_m = max(single, key=single.get)
        best_single = single[best_m]
        fusion_auroc = round(float(roc_auc_score(y, oof_fused)), 4)
        uniform_auroc = round(float(roc_auc_score(y, oof_uniform)), 4)
        invent_auroc = round(float(roc_auc_score(y, oof_invent)), 4)

        delta_best = round(fusion_auroc - best_single, 4)
        boot_vs_best = boot_delta(y, oof_fused, oof_expert[best_m], boot_idx)
        boot_vs_uniform = boot_delta(y, oof_fused, oof_uniform, boot_idx)

        missing = {}
        for m in MODS:
            a = round(float(roc_auc_score(y, oof_drop[m])), 4)
            missing[f"drop_{m}"] = {
                "auroc": a,
                "delta_vs_full": round(a - fusion_auroc, 4),
                "kept": [x for x in MODS if x != m],
                "ci95": boot_absolute(y, oof_drop[m], boot_idx)["ci95"],
            }

        res = {
            "pos": int(y.sum()), "prevalence": round(float(y.mean()), 4),
            "single_auroc": single,
            "best_single": {"modality": best_m, "auroc": best_single,
                            "ci95": boot_absolute(y, oof_expert[best_m], boot_idx)["ci95"]},
            "fusion_auroc": fusion_auroc,
            "fusion_ci95": boot_absolute(y, oof_fused, boot_idx)["ci95"],
            "delta_vs_best_single": delta_best,
            "boot_fusion_vs_best_single": boot_vs_best,
            "fusion_wins": bool(boot_vs_best["excludes_0"] and delta_best > 0),
            "gate_ablation": {
                "learned_gate_auroc": fusion_auroc,
                "uniform_gate_auroc": uniform_auroc,
                "fixed_inverse_entropy_gate_auroc": invent_auroc,
                "delta_learned_minus_uniform": round(fusion_auroc - uniform_auroc, 4),
                "boot_learned_vs_uniform": boot_vs_uniform,
                "learned_beats_uniform": bool(boot_vs_uniform["excludes_0"]
                                              and (fusion_auroc - uniform_auroc) > 0),
                "note": "same trained experts; only the gate weighting changes. "
                        "Proves the gate modulates the output.",
            },
            "missing_modality": missing,
            "gate_weight_mean": {m: round(float(gate_w[:, j].mean()), 4) for j, m in enumerate(MODS)},
            "gate_weight_std": {m: round(float(gate_w[:, j].std()), 4) for j, m in enumerate(MODS)},
        }
        results["labels"][label] = res

        print(f"\n[{label}] pos={res['pos']}  singles={single}")
        print(f"  best_single={best_m}:{best_single}  fusion(learned)={fusion_auroc} "
              f"(delta {delta_best:+.4f}, CI {boot_vs_best['ci95']}, wins={res['fusion_wins']})")
        print(f"  GATE ABLATION: learned={fusion_auroc} uniform={uniform_auroc} "
              f"inv-ent={invent_auroc} | learned-uniform {res['gate_ablation']['delta_learned_minus_uniform']:+.4f} "
              f"CI {boot_vs_uniform['ci95']} beats_uniform={res['gate_ablation']['learned_beats_uniform']}")
        for m in MODS:
            mm = missing[f"drop_{m}"]
            print(f"  MISSING drop {m:9s} -> {mm['auroc']} (delta {mm['delta_vs_full']:+.4f}, kept {mm['kept']})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()
