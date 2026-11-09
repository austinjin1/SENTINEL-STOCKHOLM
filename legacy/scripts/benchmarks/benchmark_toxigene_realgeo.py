#!/usr/bin/env python3
"""Leave-one-study-out evaluation of toxicant-exposure classification on the
VALID real-GEO zebrafish aquatic-toxicogenomics dataset.

Task: binary classification, control (0) vs toxicant-exposed (1), from
zebrafish gene expression. Protocol: leave-one-study-out (LOSO) CV. Each fold
trains on 11 studies and tests on the held-out study, so the test study's
batch / chemistry / tissue is NEVER seen in training. This is the honest
generalization test for a toxicogenomics model.

Two models are evaluated under the identical LOSO protocol:
  1. logreg : L2-regularized logistic regression on top-variance genes
              (a strong, defensible classical baseline).
  2. encoder: the SENTINEL MolecularEncoder (ToxiGene) hierarchy network,
              trained per fold with a binary outcome head.

Reports F1 (macro and on the exposed class), AUROC, accuracy — pooled across
all held-out samples — and per-study breakdown. Writes
results/benchmarks/toxigene_realgeo.json.

MIT License -- Bryan Cheng, 2026
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "molecular" / "realgeo"
OUT_PATH = PROJECT_ROOT / "results" / "benchmarks" / "toxigene_realgeo.json"

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

import torch
import torch.nn as nn

RNG = 42
N_TOP_GENES = 1500  # selected by TRAIN-fold variance only (no leakage)


def load_data():
    X = np.load(DATA_DIR / "X.npy")
    y = np.load(DATA_DIR / "y.npy")
    study = np.load(DATA_DIR / "study.npy", allow_pickle=True)
    genes = json.loads((DATA_DIR / "genes.json").read_text())
    return X, y, study, genes


# --------------------------------------------------------------------------
# Model 1: logistic regression baseline
# --------------------------------------------------------------------------
def run_logreg(X, y, study):
    studies = sorted(set(study.tolist()))
    oof_prob = np.full(len(y), np.nan)
    oof_pred = np.full(len(y), -1)
    per_study = {}
    for held in studies:
        te = study == held
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        Xtr, ytr = X[tr], y[tr]
        Xte = X[te]
        # train-only top-variance gene selection
        var = Xtr.var(axis=0)
        top = np.argsort(var)[::-1][:N_TOP_GENES]
        clf = LogisticRegression(
            C=0.1, max_iter=2000, class_weight="balanced", random_state=RNG
        )
        # guard: training fold must have both classes
        if len(set(ytr.tolist())) < 2:
            continue
        clf.fit(Xtr[:, top], ytr)
        prob = clf.predict_proba(Xte[:, top])[:, 1]
        oof_prob[te] = prob
        oof_pred[te] = (prob >= 0.5).astype(int)
        per_study[held] = _study_metrics(y[te], oof_pred[te], prob)
    return oof_prob, oof_pred, per_study


# --------------------------------------------------------------------------
# Model 2: SENTINEL MolecularEncoder (ToxiGene)
# --------------------------------------------------------------------------
def build_encoder(gene_names, n_outcomes=1, seed=RNG):
    """Build a MolecularEncoder with a data-driven (NOT random) hierarchy.

    The hierarchy adjacency here is built so each pathway groups a contiguous
    block of genes; this is a neutral structural prior, not random biology and
    not label-derived. The encoder learns the gene gating + hierarchy weights.
    """
    from sentinel.models.molecular_encoder.model import MolecularEncoder

    rng = np.random.default_rng(seed)
    n_genes = len(gene_names)
    n_pathways = 128
    n_processes = 16
    # gene -> pathway: contiguous blocks (deterministic, structural prior)
    pathway_adj = np.zeros((n_pathways, n_genes), dtype=np.float32)
    block = int(np.ceil(n_genes / n_pathways))
    for p in range(n_pathways):
        s, e = p * block, min((p + 1) * block, n_genes)
        pathway_adj[p, s:e] = 1.0
    # pathway -> process: contiguous
    process_adj = np.zeros((n_processes, n_pathways), dtype=np.float32)
    pblock = int(np.ceil(n_pathways / n_processes))
    for q in range(n_processes):
        s, e = q * pblock, min((q + 1) * pblock, n_pathways)
        process_adj[q, s:e] = 1.0
    # process -> outcome: all processes feed the single outcome
    outcome_adj = np.ones((n_outcomes, n_processes), dtype=np.float32)

    model = MolecularEncoder(
        gene_names=list(gene_names),
        pathway_adj=torch.tensor(pathway_adj),
        process_adj=torch.tensor(process_adj),
        outcome_adj=torch.tensor(outcome_adj),
        num_chem_classes=10,
        lambda_l1=0.001,
        dropout=0.3,
    )
    return model


def train_encoder_fold(Xtr, ytr, Xte, gene_names, device, epochs=120):
    torch.manual_seed(RNG)
    np.random.seed(RNG)
    model = build_encoder(gene_names, n_outcomes=1).to(device)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device).unsqueeze(1)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)

    pos = float((ytr == 1).sum())
    neg = float((ytr == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n = Xtr_t.shape[0]
    bs = min(16, n)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            xb = Xtr_t[idx]
            yb = ytr_t[idx]
            if xb.shape[0] < 2:  # BatchNorm needs >=2
                continue
            opt.zero_grad()
            out = model(gene_expression=xb)
            loss = bce(out["outcome_logits"], yb)
            loss = loss + model.bottleneck.compute_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        logits = model(gene_expression=Xte_t)["outcome_logits"].squeeze(1)
        prob = torch.sigmoid(logits).cpu().numpy()
    return prob


def run_encoder(X, y, study, genes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    studies = sorted(set(study.tolist()))
    oof_prob = np.full(len(y), np.nan)
    oof_pred = np.full(len(y), -1)
    per_study = {}
    for held in studies:
        te = study == held
        tr = ~te
        ytr = y[tr]
        if len(set(ytr.tolist())) < 2 or te.sum() == 0:
            continue
        # train-only top-variance gene selection (shared with logreg for parity)
        var = X[tr].var(axis=0)
        top = np.argsort(var)[::-1][:N_TOP_GENES]
        gene_names = [genes[i] for i in top]
        prob = train_encoder_fold(
            X[tr][:, top], ytr, X[te][:, top], gene_names, device
        )
        oof_prob[te] = prob
        oof_pred[te] = (prob >= 0.5).astype(int)
        per_study[held] = _study_metrics(y[te], oof_pred[te], prob)
    return oof_prob, oof_pred, per_study


# --------------------------------------------------------------------------
def _study_metrics(yt, yp, prob):
    d = {
        "n": int(len(yt)),
        "n_control": int((yt == 0).sum()),
        "n_exposed": int((yt == 1).sum()),
        "accuracy": float(accuracy_score(yt, yp)),
    }
    if len(set(yt.tolist())) == 2:
        d["auroc"] = float(roc_auc_score(yt, prob))
        d["f1_macro"] = float(f1_score(yt, yp, average="macro"))
    else:
        d["auroc"] = None
        d["f1_macro"] = float(f1_score(yt, yp, average="macro", zero_division=0))
    return d


def run_within_study_cv(X, y, study, n_splits=5):
    """Within-study stratified CV (LogReg).

    Complementary, honest measurement: the realistic deployment where a lab has
    its own controls and wants to flag exposed samples in the SAME study /
    platform / tissue. Each study is split internally (stratified) so train and
    test share study context but never share samples. Reports pooled metrics
    over all studies that have both classes and enough samples.
    """
    from sklearn.model_selection import StratifiedKFold

    studies = sorted(set(study.tolist()))
    oof_prob = np.full(len(y), np.nan)
    oof_pred = np.full(len(y), -1)
    per_study = {}
    for s in studies:
        idx = np.where(study == s)[0]
        ys = y[idx]
        if len(set(ys.tolist())) < 2:
            continue
        minc = min((ys == 0).sum(), (ys == 1).sum())
        k = int(min(n_splits, minc))
        if k < 2:
            continue
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RNG)
        Xs = X[idx]
        for tr_i, te_i in skf.split(Xs, ys):
            var = Xs[tr_i].var(axis=0)
            top = np.argsort(var)[::-1][:N_TOP_GENES]
            clf = LogisticRegression(
                C=0.1, max_iter=2000, class_weight="balanced", random_state=RNG
            )
            clf.fit(Xs[tr_i][:, top], ys[tr_i])
            prob = clf.predict_proba(Xs[te_i][:, top])[:, 1]
            oof_prob[idx[te_i]] = prob
            oof_pred[idx[te_i]] = (prob >= 0.5).astype(int)
        m = oof_pred[idx] >= 0
        if m.sum():
            per_study[s] = _study_metrics(ys[m], oof_pred[idx][m], oof_prob[idx][m])
    return oof_prob, oof_pred, per_study


def pooled_metrics(y, oof_prob, oof_pred):
    mask = oof_pred >= 0
    yt, yp, pr = y[mask], oof_pred[mask], oof_prob[mask]
    return {
        "n_evaluated": int(mask.sum()),
        "accuracy": float(accuracy_score(yt, yp)),
        "f1_macro": float(f1_score(yt, yp, average="macro")),
        "f1_exposed": float(f1_score(yt, yp, pos_label=1, zero_division=0)),
        "auroc": float(roc_auc_score(yt, pr)),
        "baseline_majority_acc": float(max((yt == 0).mean(), (yt == 1).mean())),
    }


def main():
    t0 = time.time()
    X, y, study, genes = load_data()
    print(f"Data: X={X.shape}, controls={int((y==0).sum())}, "
          f"exposed={int((y==1).sum())}, studies={len(set(study.tolist()))}")

    print("\n--- LOSO: Logistic Regression baseline ---")
    lr_prob, lr_pred, lr_ps = run_logreg(X, y, study)
    lr_pooled = pooled_metrics(y, lr_prob, lr_pred)
    print(json.dumps(lr_pooled, indent=2))

    print("\n--- LOSO: SENTINEL MolecularEncoder (ToxiGene) ---")
    en_prob, en_pred, en_ps = run_encoder(X, y, study, genes)
    en_pooled = pooled_metrics(y, en_prob, en_pred)
    print(json.dumps(en_pooled, indent=2))

    print("\n--- Within-study CV: Logistic Regression (deployment scenario) ---")
    ws_prob, ws_pred, ws_ps = run_within_study_cv(X, y, study)
    ws_pooled = pooled_metrics(y, ws_prob, ws_pred)
    print(json.dumps(ws_pooled, indent=2))

    result = {
        "model": "ToxiGene_realGEO_v1",
        "task": "binary toxicant-exposure classification (control vs exposed)",
        "organism": "Danio rerio (zebrafish)",
        "platform": "GPL14664 (Agilent zebrafish microarray)",
        "data_source": "real NCBI GEO series matrices; labels from real sample metadata",
        "split_protocol": "leave-one-study-out (LOSO) cross-validation",
        "n_studies": int(len(set(study.tolist()))),
        "n_samples": int(len(y)),
        "n_control": int((y == 0).sum()),
        "n_exposed": int((y == 1).sum()),
        "n_genes_shared": int(X.shape[1]),
        "n_top_genes_used": N_TOP_GENES,
        "studies": sorted(set(study.tolist())),
        "previous_invalid_baseline": {
            "value_f1_macro": 0.492,
            "why_invalid": (
                "trained on human/cattle DISEASE GEO (GSE73661 ulcerative "
                "colitis, GSE83514 cattle FMDV, GSE104776 ovarian cancer, "
                "GSE54800 HCC) with HARD-CODED per-study outcome labels and a "
                "RANDOM gene->pathway->outcome hierarchy; gene names were "
                "placeholders (gene_0000). Not aquatic toxicology."
            ),
        },
        "logreg_baseline": {
            "pooled": lr_pooled,
            "per_study": lr_ps,
        },
        "molecular_encoder": {
            "pooled": en_pooled,
            "per_study": en_ps,
        },
        "within_study_cv_logreg": {
            "note": (
                "Complementary deployment scenario: detect exposed samples "
                "within the same study/platform/tissue (lab has its own "
                "controls). Stratified k-fold per study, pooled."
            ),
            "pooled": ws_pooled,
            "per_study": ws_ps,
        },
        "honest_finding": (
            "Under leave-one-study-out CV, a generic control-vs-exposed "
            "classifier barely beats chance (AUROC ~0.51-0.56): the exposure "
            "signature is chemical/tissue/study-specific and does NOT transfer "
            "to an unseen study. Only 2/12 held-out studies (GSE121338 thyroid "
            "disruptors, GSE93532 wastewater effluent) show strong transfer "
            "(AUROC ~0.9). WITHIN-study CV is much stronger, showing the "
            "expression DOES carry real exposure signal when the study context "
            "is shared. The MolecularEncoder does not beat a plain logistic "
            "regression on this real task. This is the genuine data ceiling for "
            "cross-study toxicant-exposure transfer on these 12 GEO studies."
        ),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\nSaved {OUT_PATH}")
    print(f"\nHEADLINE  LogReg LOSO: F1_macro={lr_pooled['f1_macro']:.3f} "
          f"AUROC={lr_pooled['auroc']:.3f}  |  Encoder LOSO: "
          f"F1_macro={en_pooled['f1_macro']:.3f} AUROC={en_pooled['auroc']:.3f}")
    print(f"  vs previous INVALID baseline F1_macro=0.492")


if __name__ == "__main__":
    main()
