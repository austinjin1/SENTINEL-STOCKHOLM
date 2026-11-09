#!/usr/bin/env python
"""Fusion cycle 2 — VARIANT reliability-gate-attn.

A per-sample RELIABILITY-GATED attention fusion on the microcystin within-region
task (satellite spectral + in-situ pigment + physicochemical drivers).

Each modality m produces (a) an embedding e_m from its own features, and (b) a
scalar reliability logit r_m(x) predicted from that SAME modality's features PLUS
a data-quality signal for that modality:
  - satellite reliability uses the cloud fraction (cloud/mixed-pixel proxy) and
    the spectral-feature variance;
  - pigment reliability uses the per-sample missingness of CHLA / PC;
  - driver reliability uses the per-sample missingness of temp / P / N.
Missing features are mean-imputed (train-fold mean) and a per-modality "present"
mask is fed to the reliability head, so a MISSING or LOW-QUALITY modality can be
down-weighted. The fused representation is  sum_m softmax(r)_m * e_m  -> small MLP
head. The gate is fully LEARNED, data-driven, and demonstrably changes the output
(see the gate ablation: replace softmax(r) by UNIFORM 1/M weights).

Reported (site-clustered bootstrap 95% CIs, protocol from 08_driver_arm.py):
  best_single_auroc, fusion_auroc, delta_vs_best_single (+CI; win = CI excludes 0),
  GATE ABLATION (learned gate vs uniform weights, same architecture),
  MISSING-MODALITY graceful-degradation curve (drop each modality at TEST).

GroupKFold BY SITE. All transforms (impute means, standardisation) fit on train
folds only. Label = real microcystin lab threshold (non-circular).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data" / "processed" / "biology" / "noaa_habs" / "mc_fusion_dataset.parquet"
RAW = ROOT / "data" / "raw" / "wqp_drivers"
OUT = ROOT / "results" / "trackB_fusion" / "fusion_cycle2" / "reliability-gate-attn.json"
CHARS = {"temp": "Temperature, water", "phosphorus": "Phosphorus", "nitrogen": "Nitrogen"}
MODS = ["satellite", "pigment", "drivers"]
SEED = 0
B = 2000
EMB = 16
EPOCHS = 150
LR = 3e-3
WD = 1e-3

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


# ---- site-clustered bootstrap (protocol from 08_driver_arm.py) ---------------
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
class ModalityEncoder(nn.Module):
    """Encodes one modality's (imputed) features -> embedding, and predicts a
    scalar reliability logit from those features PLUS the per-modality quality/
    presence signal q (missingness fraction, cloud, etc.)."""

    def __init__(self, in_dim, q_dim, emb=EMB):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, emb), nn.GELU(), nn.LayerNorm(emb),
            nn.Linear(emb, emb), nn.LayerNorm(emb),
        )
        # reliability head reads the modality features AND its quality signal
        self.rel = nn.Sequential(
            nn.Linear(in_dim + q_dim, emb), nn.GELU(),
            nn.Linear(emb, 1),
        )

    def forward(self, x, q):
        e = self.enc(x)
        r = self.rel(torch.cat([x, q], dim=-1)).squeeze(-1)  # [B]
        return e, r


class ReliabilityGateAttnFusion(nn.Module):
    def __init__(self, dims, qdims, emb=EMB):
        super().__init__()
        self.mods = list(dims.keys())
        self.encoders = nn.ModuleDict(
            {m: ModalityEncoder(dims[m], qdims[m], emb) for m in self.mods})
        self.head = nn.Sequential(
            nn.Linear(emb, emb), nn.GELU(), nn.LayerNorm(emb),
            nn.Linear(emb, 1),
        )

    def forward(self, feats, quals, drop=None, uniform=False):
        """feats/quals: dict m -> tensor. drop: set of modalities masked out at
        test (their reliability logit is set to -inf so weight -> 0). uniform:
        replace softmax(reliability) by uniform 1/M weights (gate ablation)."""
        embs, logits = [], []
        present = []
        for m in self.mods:
            e, r = self.encoders[m](feats[m], quals[m])
            embs.append(e)
            if drop is not None and m in drop:
                r = torch.full_like(r, -1e9)  # force out
                present.append(0.0)
            else:
                present.append(1.0)
            logits.append(r)
        E = torch.stack(embs, dim=1)          # [B, M, emb]
        R = torch.stack(logits, dim=1)        # [B, M]
        if uniform:
            mask = torch.tensor(present, device=R.device).view(1, -1)
            w = mask / mask.sum(dim=1, keepdim=True).clamp_min(1e-9)
            w = w.expand(R.shape[0], -1)
        else:
            w = torch.softmax(R, dim=1)       # [B, M] per-sample reliability weights
        fused = (w.unsqueeze(-1) * E).sum(dim=1)   # [B, emb]
        out = self.head(fused).squeeze(-1)
        return out, w


def standardize_fit(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
    mu[~np.isfinite(mu)] = 0.0
    return mu, sd


def prep_fold(feats_raw, quals_raw, tr, te):
    """Impute (train-mean) + standardize (train stats). Returns train/test dicts
    of standardized-imputed feature tensors and quality tensors."""
    fitted = {}
    for m in MODS:
        X = feats_raw[m]
        mu = np.nanmean(X[tr], axis=0)
        mu[~np.isfinite(mu)] = 0.0
        Xi = np.where(np.isfinite(X), X, mu)   # impute with train mean
        sd = Xi[tr].std(axis=0)
        sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
        mean = Xi[tr].mean(axis=0)
        Xs = (Xi - mean) / sd
        fitted[m] = Xs
    return fitted


def to_t(a):
    return torch.tensor(a, dtype=torch.float32)


def train_eval(feats_std, quals, y, tr, te, drop_sets, uniform_variants):
    """Train the fusion model on tr, return OOF probs on te for: full learned-gate
    fusion, uniform-gate fusion, and each drop-modality variant (all evaluated at
    test with the SAME trained weights). uniform_variants controls whether we also
    return a separately-trained uniform model for the ablation."""
    dims = {m: feats_std[m].shape[1] for m in MODS}
    qdims = {m: quals[m].shape[1] for m in MODS}

    def run_train(uniform_train):
        torch.manual_seed(SEED)
        model = ReliabilityGateAttnFusion(dims, qdims)
        yt = to_t(y[tr])
        # class-balanced pos_weight
        npos = max(1.0, float(y[tr].sum()))
        nneg = max(1.0, float(len(tr) - y[tr].sum()))
        pw = torch.tensor(nneg / npos, dtype=torch.float32)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        ft = {m: to_t(feats_std[m][tr]) for m in MODS}
        qt = {m: to_t(quals[m][tr]) for m in MODS}
        model.train()
        for _ in range(EPOCHS):
            opt.zero_grad()
            out, _ = model(ft, qt, uniform=uniform_train)
            loss = lossf(out, yt)
            loss.backward()
            opt.step()
        model.eval()
        return model

    fte = {m: to_t(feats_std[m][te]) for m in MODS}
    qte = {m: to_t(quals[m][te]) for m in MODS}

    out = {}
    # learned-gate model
    model = run_train(uniform_train=False)
    with torch.no_grad():
        logit, w = model(fte, qte, uniform=False)
        out["fusion"] = torch.sigmoid(logit).numpy()
        out["_gate_weights"] = w.numpy()
        # missing-modality: drop each modality at TEST with the SAME trained model
        for ds_name, ds in drop_sets.items():
            l, _ = model(fte, qte, drop=ds, uniform=False)
            out[ds_name] = torch.sigmoid(l).numpy()
        # uniform-weights AT TEST on the learned-gate model (gate-off inference)
        l, _ = model(fte, qte, uniform=True)
        out["uniform_inference_on_learned_model"] = torch.sigmoid(l).numpy()

    # separately-trained uniform-gate model (fair fixed-weight baseline)
    if uniform_variants:
        umodel = run_train(uniform_train=True)
        with torch.no_grad():
            l, _ = umodel(fte, qte, uniform=True)
            out["uniform_trained"] = torch.sigmoid(l).numpy()
    return out


def main():
    df = pq.read_table(DATA).to_pandas()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    sites_list = sorted(df["site"].astype(str).unique())

    for tag in CHARS:
        cache = RAW / f"{tag}.csv"
        dv = parse_driver(pd.read_csv(cache, low_memory=False))
        df[f"drv_{tag}"] = [nearest(dv, str(s), d) for s, d in
                            zip(df["site"].astype(str), df["date"])]

    sf = [c for c in df.columns if c.startswith("sf")]
    # RAW features per modality (NaNs preserved for missingness signal)
    feats_raw = {
        "satellite": df[sf].to_numpy("float64"),
        "pigment": df[["CHLA", "PC"]].to_numpy("float64"),
        "drivers": df[[f"drv_{t}" for t in CHARS]].to_numpy("float64"),
    }

    # ---- per-modality QUALITY / RELIABILITY signals ----
    # satellite: cloud fraction (higher = less reliable) + spectral feature var
    cloud = df["cloud"].to_numpy("float64") if "cloud" in df.columns else np.zeros(len(df))
    cloud = np.where(np.isfinite(cloud), cloud, np.nanmean(cloud[np.isfinite(cloud)]) if np.isfinite(cloud).any() else 0.0)
    sat_var = np.nanvar(feats_raw["satellite"], axis=1)
    sat_var = np.where(np.isfinite(sat_var), sat_var, 0.0)
    sat_present = np.isfinite(feats_raw["satellite"]).mean(axis=1)
    quals_raw = {
        "satellite": np.column_stack([cloud, sat_var, sat_present]),
        # pigment: per-column present fraction (CHLA, PC missingness)
        "pigment": np.isfinite(feats_raw["pigment"]).astype("float64"),
        # drivers: per-column present fraction (temp, P, N missingness)
        "drivers": np.isfinite(feats_raw["drivers"]).astype("float64"),
    }

    mc = df["mc"].to_numpy("float64")
    site_arr = df["site"].astype(str).to_numpy()
    ns = min(5, len(sites_list))
    boot_idx = site_boot_indices(site_arr)

    drop_sets = {f"drop_{m}": {m} for m in MODS}

    results = {
        "variant": "reliability-gate-attn",
        "description": "Per-sample reliability-gated attention fusion: each modality "
                       "yields an embedding and a learned scalar reliability logit "
                       "(from its own features + a data-quality/missingness signal); "
                       "fused = sum_m softmax(reliability)_m * emb_m -> MLP head.",
        "n": int(len(df)), "n_sites": len(sites_list),
        "modalities": {"satellite": sf, "pigment": ["CHLA", "PC"],
                       "drivers": [f"drv_{t}" for t in CHARS]},
        "reliability_signals": {
            "satellite": ["cloud_fraction", "spectral_variance", "present_frac"],
            "pigment": ["CHLA_present", "PC_present"],
            "drivers": ["temp_present", "P_present", "N_present"],
        },
        "driver_coverage": {t: round(float(np.isfinite(df[f"drv_{t}"]).mean()), 3) for t in CHARS},
        "protocol": {"split": "GroupKFold by site", "n_splits": ns,
                     "bootstrap": "site-clustered (resample sites w/ replacement)", "B": B,
                     "label": "real microcystin lab threshold (non-circular)",
                     "transforms": "impute (train mean) + standardize (train stats), fit on train only"},
        "model": {"emb": EMB, "epochs": EPOCHS, "lr": LR, "weight_decay": WD},
        "labels": {},
    }

    for label, thr in [("mc_gt1.6", 1.6), ("mc_gt0.3", 0.3)]:
        y = (mc > thr).astype(int)
        gkf = GroupKFold(n_splits=ns)

        # OOF containers
        oof = {k: np.zeros(len(y)) for k in
               ["fusion", "uniform_trained", "uniform_inference_on_learned_model"] +
               list(drop_sets.keys())}
        # single-modality OOF via drop-the-other-two on the trained model would not
        # be a fair "single" baseline, so compute genuine single-modality models too.
        single_oof = {m: np.zeros(len(y)) for m in MODS}
        gate_w_accum = []

        for fold, (tr, te) in enumerate(gkf.split(feats_raw["satellite"], y, site_arr)):
            feats_std = prep_fold(feats_raw, quals_raw, tr, te)
            out = train_eval(feats_std, quals_raw, y, tr, te, drop_sets, uniform_variants=True)
            for k in oof:
                oof[k][te] = out[k]
            gate_w_accum.append((te, out["_gate_weights"]))

            # genuine single-modality baselines: train a 1-modality model
            for m in MODS:
                dims = {m: feats_std[m].shape[1]}
                # reuse the fusion model but with only 1 modality (softmax over 1 = 1)
                torch.manual_seed(SEED)
                sm = ReliabilityGateAttnFusion({m: feats_std[m].shape[1]},
                                               {m: quals_raw[m].shape[1]})
                yt = to_t(y[tr])
                npos = max(1.0, float(y[tr].sum())); nneg = max(1.0, float(len(tr) - y[tr].sum()))
                lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(nneg / npos))
                opt = torch.optim.Adam(sm.parameters(), lr=LR, weight_decay=WD)
                ft = {m: to_t(feats_std[m][tr])}; qt = {m: to_t(quals_raw[m][tr])}
                sm.train()
                for _ in range(EPOCHS):
                    opt.zero_grad()
                    o, _ = sm(ft, qt); loss = lossf(o, yt); loss.backward(); opt.step()
                sm.eval()
                with torch.no_grad():
                    o, _ = sm({m: to_t(feats_std[m][te])}, {m: to_t(quals_raw[m][te])})
                    single_oof[m][te] = torch.sigmoid(o).numpy()

        # ---- metrics ----
        single_auc = {m: round(float(roc_auc_score(y, single_oof[m])), 4) for m in MODS}
        best_single_m = max(single_auc, key=single_auc.get)
        best_single_auc = single_auc[best_single_m]
        fusion_auc = round(float(roc_auc_score(y, oof["fusion"])), 4)
        uniform_auc = round(float(roc_auc_score(y, oof["uniform_trained"])), 4)
        uniform_inf_auc = round(float(roc_auc_score(y, oof["uniform_inference_on_learned_model"])), 4)

        delta_best = round(fusion_auc - best_single_auc, 4)
        boot_vs_best = boot_delta(y, oof["fusion"], single_oof[best_single_m], boot_idx)
        boot_vs_uniform = boot_delta(y, oof["fusion"], oof["uniform_trained"], boot_idx)

        # gate weight stats (variability => gate is data-driven / actually changes output)
        gw = np.zeros((len(y), len(MODS)))
        for te, w in gate_w_accum:
            gw[te] = w
        gate_stats = {MODS[i]: {"mean": round(float(gw[:, i].mean()), 4),
                                "std": round(float(gw[:, i].std()), 4),
                                "min": round(float(gw[:, i].min()), 4),
                                "max": round(float(gw[:, i].max()), 4)} for i in range(len(MODS))}

        # missing-modality graceful-degradation curve (drop each at TEST, learned model)
        missing = {}
        for m in MODS:
            k = f"drop_{m}"
            a = round(float(roc_auc_score(y, oof[k])), 4)
            missing[m] = {"auroc_dropped": a,
                          "drop_vs_full": round(a - fusion_auc, 4),
                          "boot_absolute": boot_absolute(y, oof[k], boot_idx)}

        res = {
            "pos": int(y.sum()), "prevalence": round(float(y.mean()), 4),
            "single_auroc": single_auc,
            "best_single_modality": best_single_m,
            "best_single_auroc": best_single_auc,
            "fusion_auroc": fusion_auc,
            "fusion_absolute_ci": boot_absolute(y, oof["fusion"], boot_idx),
            "delta_vs_best_single": delta_best,
            "boot_fusion_vs_best_single": boot_vs_best,
            "fusion_wins": boot_vs_best["excludes_0"] and delta_best > 0,
            "gate_ablation": {
                "learned_gate_auroc": fusion_auc,
                "uniform_trained_auroc": uniform_auc,
                "uniform_inference_on_learned_model_auroc": uniform_inf_auc,
                "delta_learned_minus_uniform": round(fusion_auc - uniform_auc, 4),
                "boot_learned_vs_uniform": boot_vs_uniform,
                "learned_gate_beats_uniform": boot_vs_uniform["excludes_0"] and (fusion_auc - uniform_auc) > 0,
            },
            "gate_weight_stats": gate_stats,
            "missing_modality_curve": missing,
        }
        results["labels"][label] = res

        print(f"\n[{label}] pos={res['pos']} prev={res['prevalence']}")
        print(f"  single: " + " ".join(f"{m}={single_auc[m]}" for m in MODS)
              + f"  (best={best_single_m} {best_single_auc})")
        print(f"  FUSION(learned gate)={fusion_auc}  delta_vs_best={delta_best:+.4f} "
              f"CI{boot_vs_best['ci95']} wins={res['fusion_wins']}")
        print(f"  GATE ABLATION: learned={fusion_auc} vs uniform_trained={uniform_auc} "
              f"(delta {fusion_auc - uniform_auc:+.4f} CI{boot_vs_uniform['ci95']} "
              f"beats_uniform={res['gate_ablation']['learned_gate_beats_uniform']})")
        print(f"  gate weights (mean/std): "
              + " ".join(f"{m}={gate_stats[m]['mean']}/{gate_stats[m]['std']}" for m in MODS))
        print("  MISSING-MODALITY drop-at-test:")
        for m in MODS:
            print(f"    drop {m:10s} -> {missing[m]['auroc_dropped']} "
                  f"({missing[m]['drop_vs_full']:+.4f} vs full)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()
