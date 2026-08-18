#!/usr/bin/env python3
"""Retrain the ACTUAL Species Health network LEAKAGE-FREE.

WHY THIS EXISTS
---------------
The shipped Species Health model (``scripts/training/train_species_health.py``)
is CIRCULAR.  For the ~99.8% of sites without a matched USGS sensor station it
builds the 256-d ``sentinel_embedding`` as::

    bio_feats = [lat/50, lon/-100, richness/100, log1p(total_count)/10,
                 log1p(count_species_0)/10,   # <- positions 4..9 ARE the
                 ...,                           #    per-species log-counts that
                 log1p(count_species_5)/10]     #    DEFINE the occupancy labels

The occupancy label for species s is exactly ``count_species_s > 0``.  So the
network reads the label straight off its own input -> AUROC 1.000 is leakage,
not skill.

WHAT THIS SCRIPT DOES
---------------------
Retrains the SAME ``SentinelSpeciesHealthIndex`` network on the SAME real USGS
BioData with the SAME seed=42 spatial (hash-of-site-id) holdout split, but the
input embedding is rebuilt from LEAKAGE-FREE features ONLY:

  * geographic : lat, lon, |lat|, elevation
  * watershed  : log drainage area, HUC2 region one-hot
  * community  : log1p(community richness EXCLUDING the 6 target taxa),
                 log1p(community total count EXCLUDING the 6 target taxa)
  * effort     : log1p(n distinct sampling activities at the site)
  * sensor     : real matched USGS water-quality summary stats
                 (mean/std/min/max/median) for the 54 sites that have a sensor
                 station; median-imputed otherwise.

The per-species count features (positions 4..9) are REMOVED entirely.  The
community richness/count are computed AFTER dropping the 6 target taxa so they
cannot trivially encode "is a target species present".

The 2 always-absent species (Hellbender idx 3, Pearl Mussel idx 4) never occur
in this matched dataset; they are dropped from the occupancy/abundance loss and
from all reported metrics (they are trivially / undefinedly "predicted").

The full network is trained (occupancy + abundance + health + stressor heads).
Health/abundance labels are derived from the REAL counts (as in the original),
but the INPUT no longer contains those counts.  We report on the held-out TEST
split:
  * per-species occupancy AUROC + F1 (threshold 0.5)
  * macro-AUROC / macro-F1 (excluding Mayfly high-prevalence + dropped species,
    matching the honest-probe convention; macro-incl-mayfly also reported)
  * present-only abundance R^2 (network's predicted abundance vs log1p(count),
    over sites where the species is actually present)

Output: results/benchmarks/species_health_leakagefree_retrained.json

Run:
    PYTHONNOUSERSITE=1 ~/.conda/envs/physiformer/bin/python \
        scripts/training/train_species_health_leakagefree.py --gpu 2

MIT License -- Bryan Cheng, 2026
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

BIODATA_DIR = PROJECT / "data" / "processed" / "biology" / "usgs_biodata"
SENSOR_DIR = PROJECT / "data" / "raw" / "sensor" / "full"
CKPT_DIR = PROJECT / "checkpoints" / "biology"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = PROJECT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = RESULTS_DIR / "species_health_leakagefree_retrained.json"

SEED = 42
EMBED_DIM = 256

SPECIES_NAMES = [
    "Freshwater mussels (Unionidae)",
    "Mayflies (Ephemeroptera)",
    "Brook trout (Salvelinus fontinalis)",
    "Hellbender (Cryptobranchus alleganiensis)",
    "Freshwater pearl mussel (Margaritifera margaritifera)",
    "American eel (Anguilla rostrata)",
]
NUM_SPECIES = 6
ALWAYS_ABSENT_IDX = {3, 4}  # Hellbender, Pearl Mussel (verified empirically; dropped)
MAYFLY_IDX = 1

TAXON_PATTERNS = {
    0: ["unionidae", "unionid", "lampsilis", "amblema", "elliptio",
        "quadrula", "fusconaia", "pleurobema", "villosa", "corbicula",
        "anodonta", "pyganodon", "utterbackia", "lasmigona", "strophitus"],
    1: ["ephemeroptera", "ephemerell", "hexagenia", "baetis", "heptageni",
        "leptophlebi", "caenis", "tricorythodes", "isonychia", "stenonema",
        "epeorus", "drunella", "paraleptophlebia", "ameletus", "cinygmula"],
    2: ["salvelinus fontinalis", "brook trout", "salvelinus"],
    3: ["cryptobranchus", "hellbender"],
    4: ["margaritifera", "pearl mussel"],
    5: ["anguilla rostrata", "american eel", "anguilla"],
}


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def split_of(site_id: str, seed: int = SEED) -> str:
    """Identical hash split to the original training script (seed:site_id, mod 10)."""
    h = hashlib.sha256(f"{seed}:{site_id}".encode()).hexdigest()
    fold = int(h[:8], 16) % 10
    if fold < 7:
        return "train"
    if fold < 9:
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# Build per-site table: LEAKAGE-FREE features + occupancy/abundance/health labels
# ---------------------------------------------------------------------------
def build_site_table():
    import pandas as pd

    inv_path = BIODATA_DIR / "biodata_invertebrates.parquet"
    fish_path = BIODATA_DIR / "biodata_fish.parquet"
    sites_path = BIODATA_DIR / "biodata_sites.parquet"

    log("Loading USGS BioData (real biological surveys)...")
    df_inv = pd.read_parquet(inv_path)
    df_fish = pd.read_parquet(fish_path) if fish_path.exists() else pd.DataFrame()
    df_sites = pd.read_parquet(sites_path)
    bio = pd.concat([df_inv, df_fish], ignore_index=True) if len(df_fish) else df_inv

    bio["count"] = pd.to_numeric(bio["ResultMeasureValue"], errors="coerce").fillna(0)
    bio = bio[bio["count"] > 0].copy()

    taxon_lower = bio["SubjectTaxonomicName"].astype(str).str.lower()
    bio["species_group"] = -1
    for grp, pats in TAXON_PATTERNS.items():
        mask = taxon_lower.str.contains("|".join(pats), na=False)
        bio.loc[mask, "species_group"] = grp

    site_col = "MonitoringLocationIdentifier"
    matched = bio[bio["species_group"] >= 0]
    if len(matched) == 0:
        raise RuntimeError("No records matched target species groups.")
    log(f"  {len(matched)} records matched target species groups "
        f"out of {len(bio)} positive-count records.")

    # Per-site x species_group total count -> defines occupancy + abundance labels.
    ss = matched.groupby([site_col, "species_group"])["count"].sum().reset_index()
    pivot = ss.pivot(index=site_col, columns="species_group", values="count").fillna(0.0)
    for g in range(NUM_SPECIES):
        if g not in pivot.columns:
            pivot[g] = 0.0
    pivot = pivot[[g for g in range(NUM_SPECIES)]]

    # ---- LEAKAGE-FREE community features: NON-target taxa only ----
    bio_nontarget = bio[bio["species_group"] < 0]
    comm_rich = (bio_nontarget.groupby(site_col)["SubjectTaxonomicName"]
                 .nunique().rename("comm_richness_nontarget"))
    comm_count = (bio_nontarget.groupby(site_col)["count"]
                  .sum().rename("comm_count_nontarget"))
    if "ActivityIdentifier" in bio.columns:
        n_events = bio.groupby(site_col)["ActivityIdentifier"].nunique().rename("n_events")
    else:
        n_events = bio.groupby(site_col)["ActivityStartDate"].nunique().rename("n_events")

    # ---- Site geographic / watershed metadata (leakage-free) ----
    s = df_sites.drop_duplicates(subset=[site_col]).set_index(site_col)
    site_meta = pd.DataFrame(index=s.index)
    site_meta["lat"] = pd.to_numeric(s["LatitudeMeasure"], errors="coerce")
    site_meta["lon"] = pd.to_numeric(s["LongitudeMeasure"], errors="coerce")
    site_meta["elev"] = pd.to_numeric(s["VerticalMeasure/MeasureValue"], errors="coerce")
    site_meta["drainage"] = pd.to_numeric(s["DrainageAreaMeasure/MeasureValue"], errors="coerce")
    site_meta["huc2"] = s["_huc2"].astype(str) if "_huc2" in s.columns else "NA"

    tab = pivot.join(site_meta, how="inner")
    tab = tab.join(comm_rich, how="left").join(comm_count, how="left").join(n_events, how="left")
    tab = tab.dropna(subset=["lat", "lon"])
    tab[["comm_richness_nontarget", "comm_count_nontarget", "n_events"]] = tab[
        ["comm_richness_nontarget", "comm_count_nontarget", "n_events"]].fillna(0.0)

    tab["split"] = [split_of(sid) for sid in tab.index]
    log(f"  {len(tab)} sites with coordinates. "
        f"Split: train={int((tab.split=='train').sum())}, "
        f"val={int((tab.split=='val').sum())}, test={int((tab.split=='test').sum())}")
    return tab, len(df_inv), len(df_fish)


def attach_sensor_features(tab):
    """Attach REAL water-quality summary stats for sites with a sensor match."""
    import pandas as pd
    sensor_ids = {f.stem: f for f in SENSOR_DIR.glob("*.parquet")} if SENSOR_DIR.exists() else {}
    stat_names = ["wq_mean", "wq_std", "wq_min", "wq_max", "wq_median"]
    feat_rows = {}
    n_match = 0
    for sid in tab.index:
        station = str(sid).replace("USGS-", "")
        if station not in sensor_ids:
            continue
        try:
            dfx = pd.read_parquet(sensor_ids[station])
        except Exception:
            continue
        vals = []
        for col in dfx.columns:
            if col == "datetime":
                continue
            v = pd.to_numeric(dfx[col], errors="coerce").dropna()
            if len(v):
                vals.append([v.mean(), v.std(), v.min(), v.max(), v.median()])
        if not vals:
            continue
        agg = np.nanmean(np.array(vals, dtype=float), axis=0)
        feat_rows[sid] = agg
        n_match += 1
    sens = pd.DataFrame.from_dict(feat_rows, orient="index", columns=stat_names)
    tab = tab.join(sens, how="left")
    log(f"  Attached real sensor water-quality stats for {n_match} sites "
        f"(median-imputed elsewhere).")
    return tab, stat_names, n_match


# ---------------------------------------------------------------------------
# Build leakage-free feature matrix + 256-d embedding
# ---------------------------------------------------------------------------
def build_feature_matrix(tab, sensor_stats):
    """Leakage-free feature matrix. EXCLUDES per-species log-counts entirely."""
    import pandas as pd
    df = pd.DataFrame(index=tab.index)
    df["lat"] = tab["lat"]
    df["lon"] = tab["lon"]
    df["abslat"] = tab["lat"].abs()
    df["elev"] = tab["elev"]
    df["log_drainage"] = np.log1p(tab["drainage"].clip(lower=0).fillna(0))
    df["log_comm_richness_nontarget"] = np.log1p(tab["comm_richness_nontarget"])
    df["log_comm_count_nontarget"] = np.log1p(tab["comm_count_nontarget"])
    df["log_n_events"] = np.log1p(tab["n_events"])
    huc_dummies = pd.get_dummies(tab["huc2"].fillna("NA"), prefix="huc2")
    df = pd.concat([df, huc_dummies], axis=1)
    for c in sensor_stats:
        df[c] = tab[c] if c in tab.columns else np.nan
    return df


def standardize_and_embed(X_df, train_mask):
    """Median-impute + standardize on TRAIN stats, then pad to 256-d embedding.

    The leakage-free standardized features are placed at the FRONT of the
    256-d embedding (the rest is zero-padded), then L2-normalized -- mirroring
    the original embedding construction, but WITHOUT the per-species counts.
    """
    X = X_df.values.astype(np.float64)
    # Median impute using TRAIN medians.
    med = np.nanmedian(X[train_mask], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])
    # Standardize using TRAIN mean/std.
    mu = X[train_mask].mean(axis=0)
    sd = X[train_mask].std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xs = (X - mu) / sd
    n, d = Xs.shape
    emb = np.zeros((n, EMBED_DIM), dtype=np.float32)
    emb[:, :d] = Xs.astype(np.float32)
    # L2-normalize each row (matches original embedding normalization).
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    emb = emb / norms
    return emb, list(X_df.columns)


def build_site_covariates(tab, train_mask):
    """5-d site covariates [lat, lon, elev, stream_order(=0, N/A), log_drainage].

    These are also leakage-free (no per-species counts). Standardized on train.
    """
    lat = tab["lat"].values.astype(np.float64)
    lon = tab["lon"].values.astype(np.float64)
    elev = tab["elev"].fillna(tab["elev"].median()).fillna(0.0).values.astype(np.float64)
    log_drainage = np.log1p(tab["drainage"].clip(lower=0).fillna(0).values.astype(np.float64))
    stream_order = np.zeros_like(lat)  # not available in BioData; kept as 0 (leakage-free)
    covs = np.stack([lat, lon, elev, stream_order, log_drainage], axis=1)
    # Standardize on train.
    mu = covs[train_mask].mean(axis=0)
    sd = covs[train_mask].std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    covs = ((covs - mu) / sd).astype(np.float32)
    return covs


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def main():
    import pandas as pd
    globals()["pd"] = pd  # for build_site_covariates' pandas use

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log("=" * 70)
    log("Species Health -- LEAKAGE-FREE RETRAIN of the actual network")
    log("=" * 70)
    log(f"Device: {device}")

    # --- Build data ---
    tab, n_inv, n_fish = build_site_table()
    tab, sensor_stats, n_sensor_match = attach_sensor_features(tab)

    prevalence = {SPECIES_NAMES[g]: float((tab[g] > 0).mean()) for g in range(NUM_SPECIES)}
    log("\nEmpirical occupancy prevalence (all sites):")
    for g in range(NUM_SPECIES):
        log(f"  [{g}] {SPECIES_NAMES[g]}: {prevalence[SPECIES_NAMES[g]]:.4f} "
            f"(present at {int((tab[g] > 0).sum())} sites)")

    train_mask = (tab["split"] == "train").values
    val_mask = (tab["split"] == "val").values
    test_mask = (tab["split"] == "test").values

    # Leakage-free embedding (NO per-species counts).
    X_df = build_feature_matrix(tab, sensor_stats)
    emb_all, feat_names = standardize_and_embed(X_df, train_mask)
    covs_all = build_site_covariates(tab, train_mask)
    log(f"\nLeakage-free embedding: {len(feat_names)} real features "
        f"(per-species counts EXCLUDED) -> padded/normalized to {EMBED_DIM}-d.")
    log(f"Feature names: {feat_names}")

    # Labels from REAL counts (input no longer contains them).
    counts = tab[[g for g in range(NUM_SPECIES)]].values.astype(np.float32)  # [N, 6]
    occ = (counts > 0).astype(np.float32)                                     # [N, 6]
    # Health label: same definition as original (relative log-abundance scaled).
    # Use site total count (all taxa) to mirror original health label scaling.
    site_total = (counts.sum(axis=1) + tab["comm_count_nontarget"].values)  # all taxa
    site_total = np.maximum(site_total, 1.0)
    health = np.zeros_like(counts)
    for g in range(NUM_SPECIES):
        c = counts[:, g]
        rel = c / site_total
        h = np.log1p(c) / np.log1p(site_total) * 100.0 * (1.0 + rel)
        health[:, g] = np.clip(np.where(c > 0, h, 0.0), 0, 100)
    abund_log = np.log1p(counts)  # abundance target (log scale)

    # Species mask: 1 for species we keep, 0 for always-absent dropped species.
    keep_species = np.array([0 if g in ALWAYS_ABSENT_IDX else 1
                             for g in range(NUM_SPECIES)], dtype=np.float32)
    log(f"\nDropping always-absent species: "
        f"{[SPECIES_NAMES[i] for i in sorted(ALWAYS_ABSENT_IDX)]}")

    # Tensors
    def to_t(a):
        return torch.from_numpy(a).float()
    emb_t = to_t(emb_all)
    covs_t = to_t(covs_all)
    occ_t = to_t(occ)
    health_t = to_t(health)
    abund_t = to_t(abund_log)
    keep_t = torch.from_numpy(keep_species).float().to(device)

    idx_train = np.where(train_mask)[0]
    idx_val = np.where(val_mask)[0]
    idx_test = np.where(test_mask)[0]

    # --- Model ---
    from sentinel.models.biology.species_health import SentinelSpeciesHealthIndex
    model = SentinelSpeciesHealthIndex().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"\nModel parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Pos weights per species for occupancy BCE (handle imbalance), train-only.
    train_occ = occ[idx_train]
    pos = train_occ.sum(axis=0)
    neg = (1 - train_occ).sum(axis=0)
    pos_weight = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0).astype(np.float32)
    pos_weight_t = torch.from_numpy(pos_weight).to(device)

    bs = args.batch_size

    def run_epoch(indices, train: bool):
        if train:
            model.train()
        else:
            model.eval()
        np.random.shuffle(indices) if train else None
        total_loss = 0.0
        n_seen = 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for i in range(0, len(indices), bs):
                bidx = indices[i:i + bs]
                if train and len(bidx) < 2:
                    continue
                e = emb_t[bidx].to(device)
                c = covs_t[bidx].to(device)
                o = occ_t[bidx].to(device)
                h = health_t[bidx].to(device)
                ab = abund_t[bidx].to(device)

                if train:
                    optimizer.zero_grad(set_to_none=True)
                out = model(e, site_covariates=c)

                # Occupancy: weighted BCE, masked to kept species.
                occ_logit = torch.logit(out.occupancy_probs.clamp(1e-6, 1 - 1e-6))
                bce = nn.functional.binary_cross_entropy_with_logits(
                    occ_logit, o, pos_weight=pos_weight_t, reduction="none")
                bce = (bce * keep_t).sum() / (keep_t.sum() * e.size(0))
                # Health MSE (kept species).
                hl = ((out.health_scores - h) ** 2 * keep_t).sum() / (keep_t.sum() * e.size(0))
                # Abundance MSE on present sites only (kept species).
                present = (o > 0).float() * keep_t
                denom = present.sum().clamp(min=1.0)
                abl = (((out.abundance - ab) ** 2) * present).sum() / denom
                loss = 1.0 * bce + 0.01 * hl + 0.5 * abl

                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                total_loss += loss.item() * e.size(0)
                n_seen += e.size(0)
        return total_loss / max(n_seen, 1)

    @torch.no_grad()
    def collect(indices):
        model.eval()
        op, hp, ap = [], [], []
        for i in range(0, len(indices), bs):
            bidx = indices[i:i + bs]
            e = emb_t[bidx].to(device)
            c = covs_t[bidx].to(device)
            out = model(e, site_covariates=c)
            op.append(out.occupancy_probs.cpu().numpy())
            hp.append(out.health_scores.cpu().numpy())
            ap.append(out.abundance.cpu().numpy())
        return (np.concatenate(op), np.concatenate(hp), np.concatenate(ap))

    from sklearn.metrics import roc_auc_score, f1_score, r2_score

    def macro_auroc_val(indices):
        op, _, _ = collect(indices)
        y = occ[indices]
        aurocs = []
        for g in range(NUM_SPECIES):
            if g in ALWAYS_ABSENT_IDX or g == MAYFLY_IDX:
                continue
            if len(np.unique(y[:, g])) < 2:
                continue
            aurocs.append(roc_auc_score(y[:, g], op[:, g]))
        return float(np.mean(aurocs)) if aurocs else 0.0

    # --- Train loop (early stop on val macro-AUROC) ---
    best_val = -1.0
    best_state = None
    patience_ctr = 0
    log("\nTraining...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = run_epoch(idx_train.copy(), train=True)
        scheduler.step()
        val_auroc = macro_auroc_val(idx_val)
        dt = time.time() - t0
        if epoch % 10 == 0 or epoch == 1:
            log(f"Epoch {epoch:3d}/{args.epochs} | train_loss {tr_loss:.4f} | "
                f"val macro-AUROC {val_auroc:.4f} | {dt:.1f}s")
        if val_auroc > best_val:
            best_val = val_auroc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                log(f"  Early stopping at epoch {epoch} (best val macro-AUROC {best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save({"model_state_dict": best_state,
                    "feature_names": feat_names,
                    "best_val_macro_auroc": best_val},
                   CKPT_DIR / "species_health_leakagefree_best.pt")

    # --- Final TEST evaluation ---
    log("\n" + "=" * 70)
    log("TEST evaluation (held-out spatial split)")
    log("=" * 70)
    op, hp, ap = collect(idx_test)
    y_test = occ[idx_test]
    counts_test = counts[idx_test]

    per_species = {}
    evaluated_idx = []
    for g in range(NUM_SPECIES):
        name = SPECIES_NAMES[g]
        present_total = int((tab[g] > 0).sum())
        if g in ALWAYS_ABSENT_IDX or present_total == 0:
            per_species[name] = {"status": "dropped_always_absent",
                                 "present_sites_total": present_total}
            log(f"  [{g}] {name}: DROPPED (always-absent)")
            continue
        yt = y_test[:, g].astype(int)
        if len(np.unique(yt)) < 2:
            per_species[name] = {"status": "degenerate_test_single_class",
                                 "test_prevalence": float(yt.mean())}
            log(f"  [{g}] {name}: degenerate test (single class)")
            continue
        p = op[:, g]
        auroc = float(roc_auc_score(yt, p))
        f1 = float(f1_score(yt, (p >= 0.5).astype(int), zero_division=0))
        # Present-only abundance R^2: network predicted abundance vs log1p(count).
        mask_pre = counts_test[:, g] > 0
        if mask_pre.sum() >= 10:
            r2 = float(r2_score(np.log1p(counts_test[mask_pre, g]), ap[mask_pre, g]))
            n_pre = int(mask_pre.sum())
        else:
            r2, n_pre = None, int(mask_pre.sum())
        per_species[name] = {
            "status": "ok",
            "test_prevalence": float(yt.mean()),
            "n_test": int(len(yt)),
            "auroc": auroc,
            "f1": f1,
            "present_only_r2": r2,
            "n_present_test": n_pre,
        }
        evaluated_idx.append(g)
        log(f"  [{g}] {name}: prev={yt.mean():.3f} AUROC={auroc:.4f} "
            f"F1={f1:.4f} presentR2={r2}")

    macro_idx = [g for g in evaluated_idx if g != MAYFLY_IDX]

    def _macro(idxs, key):
        vals = [per_species[SPECIES_NAMES[g]][key] for g in idxs
                if per_species[SPECIES_NAMES[g]].get(key) is not None]
        return float(np.mean(vals)) if vals else None

    macro_excl = {
        "species_in_macro": [SPECIES_NAMES[g] for g in macro_idx],
        "macro_auroc": _macro(macro_idx, "auroc"),
        "macro_f1": _macro(macro_idx, "f1"),
    }
    macro_incl = {
        "macro_auroc_incl_mayfly": _macro(evaluated_idx, "auroc"),
        "macro_f1_incl_mayfly": _macro(evaluated_idx, "f1"),
    }
    mayfly = per_species.get(SPECIES_NAMES[MAYFLY_IDX])

    log("\n" + "=" * 70)
    log("LEAKAGE-FREE RETRAINED NETWORK -- TEST SUMMARY")
    log("=" * 70)
    log(f"Macro-AUROC (excl Mayfly + dropped): {macro_excl['macro_auroc']:.4f}")
    log(f"Macro-F1   (excl Mayfly + dropped): {macro_excl['macro_f1']:.4f}")
    log(f"Macro-AUROC (incl Mayfly):          {macro_incl['macro_auroc_incl_mayfly']:.4f}")
    if mayfly and mayfly.get("status") == "ok":
        log(f"Mayflies (high prevalence): AUROC={mayfly['auroc']:.4f} F1={mayfly['f1']:.4f}")
    log("vs leaked/circular baseline: AUROC=1.000, occ_acc=0.999, health_r2=0.9996")
    log("vs honest GBM probe (Variant B): macro-AUROC 0.944 / (Variant C no-geo) 0.853")

    results = {
        "_PROVENANCE": (
            "LEAKAGE-FREE RETRAIN of the actual SentinelSpeciesHealthIndex network. "
            "The shipped training script built the 256-d input embedding with "
            "positions 4-9 = log1p(per-species counts), which DEFINE the occupancy "
            "labels (count>0 == present) -> the previously reported AUROC=1.000 / "
            "occ_acc=0.999 / health_r2=0.9996 are CIRCULAR (label read off the input). "
            "Here the input embedding is rebuilt from genuinely independent predictors "
            "ONLY (geographic lat/lon/|lat|/elev, watershed log-drainage + HUC2 "
            "one-hot, NON-target community richness/count computed AFTER dropping the "
            "6 target taxa, sampling effort, and real matched USGS sensor water-quality "
            "stats for 54 sites). Per-species counts are REMOVED entirely. Same real "
            "USGS BioData, same seed=42 spatial hash holdout split. The 2 always-absent "
            "species (Hellbender, Pearl Mussel) are dropped. This is the full network "
            "retrained end-to-end, NOT a probe -- the reported numbers are this honest "
            "network's held-out test skill."
        ),
        "model": "SentinelSpeciesHealthIndex (full network, retrained leakage-free)",
        "data_source": "USGS BioData (real surveys) + matched USGS sensor stations",
        "synthetic_data": False,
        "seed": args.seed,
        "split": "spatial hash holdout (sha256 of seed:site_id, mod 10)",
        "leaked_features_removed": "embedding positions 4-9: per-species log1p(count) (define the labels)",
        "leakage_free_features": feat_names,
        "n_leakage_free_features": len(feat_names),
        "embedding_dim": EMBED_DIM,
        "n_params": int(n_params),
        "n_sites_total": int(len(tab)),
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_sites_with_real_sensor_match": int(n_sensor_match),
        "sensor_note": (
            "54 USGS NWIS sensor stations match site-IDs in the FULL sites table, "
            "but NONE of them overlap the 5462 sites that have any of the 6 target "
            "taxa, so within this model's site universe there are 0 real sensor "
            "matches. The wq_* feature columns are therefore entirely median-imputed "
            "(constant) and contribute no real signal here -- they are kept for "
            "schema parity but are effectively inert. The reported skill comes from "
            "geography, watershed, NON-target community context, and sampling effort."
        ),
        "n_invertebrate_records": int(n_inv),
        "n_fish_records": int(n_fish),
        "occupancy_prevalence_all_sites": prevalence,
        "species_dropped_always_absent": [SPECIES_NAMES[i] for i in sorted(ALWAYS_ABSENT_IDX)],
        "best_val_macro_auroc": float(best_val),
        "test_metrics": {
            "macro_excl_mayfly_and_dropped": macro_excl,
            "macro_reference_incl_mayfly": macro_incl,
            "mayfly_high_prevalence_separate": mayfly,
            "per_species": per_species,
        },
        "baseline_comparison": {
            "leaked_circular_auroc": 1.000,
            "leaked_circular_occ_accuracy": 0.9988738894462585,
            "leaked_circular_health_r2": 0.9995507135053653,
            "honest_gbm_probe_variantB_macro_auroc": 0.943669227579283,
            "honest_gbm_probe_variantC_nogeo_macro_auroc": 0.8528712389833438,
            "note": ("Leaked numbers are circular. The honest GBM probe was an "
                     "unbiased ESTIMATE on the leakage-free features; this entry is "
                     "the ACTUAL full network retrained on those features."),
        },
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nWrote {OUT_PATH}")
    log("DONE")


if __name__ == "__main__":
    main()
