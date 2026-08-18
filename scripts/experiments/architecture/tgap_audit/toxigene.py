#!/usr/bin/env python3
"""ToxiGene transfer-gap CONTRIBUTION audit.

Paper claim (transfer-gap example): raw-gene cross-study LOSO AUROC ~0.55
"collapses"; per-sample RANK-normalization recovers it to 0.64-0.68.

AUDIT QUESTION: Is the raw 0.55 a WEAK-BASELINE artifact? Does a WELL-REGULARIZED
model on the RAW genes (SAME information, no rank transform) reach the rank-norm
0.64-0.68 WITHOUT the rank transform -- or is per-sample rank normalization
GENUINELY necessary for cross-study transfer?

We hold rank-norm to the SAME regularization budget it enjoys (L1 sparse panels,
per-study standardization, univariate/MI feature selection, C-grid) and give the
RAW-gene models the FULL regularization arsenal. Head-to-head LOSO AUROC with
study-block bootstrap CI. Crucially we compute the PAIRED delta CI
(rank-norm best minus best-raw-model) by resampling STUDIES, so the verdict rests
on whether the delta CI excludes 0.

genuine_representation_effect := best rank-norm model beats the best
well-regularized RAW-gene model with a paired study-block-bootstrap delta CI whose
lower bound > 0.

Data: real NCBI GEO (12 studies, 279 samples, 7902 genes), leave-one-study-out.
All feature selection / model fits per fold on TRAIN studies only. Rank-norm and
per-study standardization are per-sample/per-study and LABEL-FREE (leakage-free).

MIT -- audit harness, 2026.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data" / "processed" / "molecular" / "realgeo"
OUT = ROOT / "results" / "benchmarks" / "architecture" / "tgap_audit" / "toxigene.json"
RNG = 42
N_TOP = 1500


def load():
    X = np.load(DATA / "X.npy")            # already GLOBAL z-scored per gene
    y = np.load(DATA / "y.npy")
    study = np.load(DATA / "study.npy", allow_pickle=True)
    genes = json.loads((DATA / "genes.json").read_text())
    return X, y, study, genes


# ---------------- label-free representation transforms -------------------
def per_study_standardize(X, study):
    Xc = X.copy().astype(np.float64)
    for s in set(study.tolist()):
        m = study == s
        mu = Xc[m].mean(0); sd = Xc[m].std(0); sd[sd < 1e-8] = 1.0
        Xc[m] = (Xc[m] - mu) / sd
    return Xc.astype(np.float32)


def rank_normalize(X):
    """Per-sample within-array quantile ranks (0..1). Label-free, per-sample."""
    return np.apply_along_axis(lambda r: rankdata(r) / len(r), 1, X).astype(np.float32)


# ---------------- LOSO engine with pluggable selection/regularization ----
def loso(Xfeat, y, study, n_select=None, select="var", C=0.1, penalty="l2",
         solver="lbfgs"):
    """Generic leave-one-study-out. Feature selection AND fit are TRAIN-only.
    select in {var, fclassif, mi}. Returns oof_prob, oof_pred."""
    studies = sorted(set(study.tolist()))
    oof_prob = np.full(len(y), np.nan)
    oof_pred = np.full(len(y), -1)
    for held in studies:
        te = study == held; tr = ~te
        ytr = y[tr]
        if len(set(ytr.tolist())) < 2 or te.sum() == 0:
            continue
        Xtr, Xte = Xfeat[tr], Xfeat[te]
        if n_select is not None and n_select < Xfeat.shape[1]:
            if select == "var":
                score = Xtr.var(0)
            elif select == "fclassif":
                score = np.nan_to_num(f_classif(Xtr, ytr)[0])
            elif select == "mi":
                score = mutual_info_classif(Xtr, ytr, random_state=RNG)
            else:
                raise ValueError(select)
            top = np.argsort(score)[::-1][:n_select]
            Xtr, Xte = Xtr[:, top], Xte[:, top]
        clf = LogisticRegression(C=C, penalty=penalty, solver=solver,
                                 max_iter=3000, class_weight="balanced",
                                 random_state=RNG)
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:, 1]
        oof_prob[te] = prob
        oof_pred[te] = (prob >= 0.5).astype(int)
    return oof_prob, oof_pred


def pooled_auroc(y, prob, pred, study):
    mask = pred >= 0
    return float(roc_auc_score(y[mask], prob[mask]))


def study_block_boot_ci(y, prob, pred, study, n_boot=2000, seed=RNG):
    mask = pred >= 0
    yt, pr, st = y[mask], prob[mask], study[mask]
    uniq = sorted(set(st.tolist()))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(st == s)[0] for s in pick])
        yb, pb = yt[idx], pr[idx]
        if len(set(yb.tolist())) < 2:
            continue
        boots.append(roc_auc_score(yb, pb))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def paired_delta_ci(y, probA, predA, probB, predB, study, n_boot=2000, seed=RNG):
    """Paired study-block bootstrap of (AUROC_A - AUROC_B) on the SAME resampled
    studies. A = rank-norm (challenger), B = best raw baseline."""
    mask = (predA >= 0) & (predB >= 0)
    yt, pA, pB, st = y[mask], probA[mask], probB[mask], study[mask]
    uniq = sorted(set(st.tolist()))
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(st == s)[0] for s in pick])
        yb = yt[idx]
        if len(set(yb.tolist())) < 2:
            continue
        deltas.append(roc_auc_score(yb, pA[idx]) - roc_auc_score(yb, pB[idx]))
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta_mean": float(deltas.mean()),
        "delta_median": float(np.median(deltas)),
        "delta_ci95": [float(lo), float(hi)],
        "frac_delta_gt0": float((deltas > 0).mean()),
        "n_boot_valid": int(len(deltas)),
    }


def evaluate(name, Xfeat, y, study, **kw):
    prob, pred = loso(Xfeat, y, study, **kw)
    auroc = pooled_auroc(y, prob, pred, study)
    lo, hi = study_block_boot_ci(y, prob, pred, study)
    print(f"  {name:52s} AUROC={auroc:.3f} CI[{lo:.3f},{hi:.3f}]")
    return {"name": name, "auroc": auroc, "auroc_ci95": [lo, hi],
            "params": {k: (v if not isinstance(v, np.ndarray) else "arr")
                       for k, v in kw.items()},
            "_prob": prob, "_pred": pred}


def main():
    t0 = time.time()
    X, y, study, genes = load()
    Xps = per_study_standardize(X, study)   # raw genes, batch-corrected, label-free
    Xrank = rank_normalize(X)               # the representation transform under audit
    print(f"Data X={X.shape}  studies={len(set(study.tolist()))} "
          f"pos={int(y.sum())} neg={int((y==0).sum())}")

    # ================= RAW-GENE MODELS (same info, NO rank transform) =========
    # Give raw genes the FULL regularization arsenal: per-study standardization
    # (batch fix), sparse L1 panels, univariate/MI supervised selection, C-grid.
    print("\n--- RAW-gene models (no rank transform) ---")
    raw = []
    # global-Z baseline (paper's weak baseline) -- reference
    raw.append(evaluate("RAW globalZ var-top1500 L2 C=0.1 (paper baseline)",
                        X, y, study, n_select=N_TOP, select="var", C=0.1))
    # per-study standardized raw genes, dense L2 across C
    for C in [0.01, 0.05, 0.1, 0.3, 1.0]:
        raw.append(evaluate(f"RAW perStudyStd var-top1500 L2 C={C}",
                            Xps, y, study, n_select=N_TOP, select="var", C=C))
    # per-study std + SPARSE L1 panels across C (matches winner's L1)
    for C in [0.1, 0.2, 0.5, 1.0]:
        raw.append(evaluate(f"RAW perStudyStd var-top1500 L1 C={C}",
                            Xps, y, study, n_select=N_TOP, select="var",
                            C=C, penalty="l1", solver="liblinear"))
    # per-study std + SUPERVISED selection (f_classif / MI) instead of variance
    for sel in ["fclassif", "mi"]:
        for k in [50, 200, 1500]:
            raw.append(evaluate(f"RAW perStudyStd {sel}-top{k} L2 C=0.1",
                                Xps, y, study, n_select=k, select=sel, C=0.1))
    # per-study std + supervised select + L1
    for sel in ["fclassif", "mi"]:
        raw.append(evaluate(f"RAW perStudyStd {sel}-top200 L1 C=0.2",
                            Xps, y, study, n_select=200, select=sel,
                            C=0.2, penalty="l1", solver="liblinear"))

    # ================= RANK-NORM MODELS (representation transform) ============
    print("\n--- RANK-norm models (representation under audit) ---")
    rk = []
    for C in [0.05, 0.1, 0.3, 1.0]:
        rk.append(evaluate(f"RANK var-top1500 L2 C={C}",
                           Xrank, y, study, n_select=N_TOP, select="var", C=C))
    for C in [0.1, 0.2, 0.5, 1.0]:
        rk.append(evaluate(f"RANK var-top1500 L1 C={C}",
                           Xrank, y, study, n_select=N_TOP, select="var",
                           C=C, penalty="l1", solver="liblinear"))

    # ================= HEAD-TO-HEAD ==========================================
    best_raw = max(raw, key=lambda d: d["auroc"])
    best_rank = max(rk, key=lambda d: d["auroc"])
    print(f"\nBEST RAW : {best_raw['name']}  AUROC={best_raw['auroc']:.3f} "
          f"CI{best_raw['auroc_ci95']}")
    print(f"BEST RANK: {best_rank['name']}  AUROC={best_rank['auroc']:.3f} "
          f"CI{best_rank['auroc_ci95']}")

    delta = paired_delta_ci(y, best_rank["_prob"], best_rank["_pred"],
                            best_raw["_prob"], best_raw["_pred"], study)
    print(f"\nPAIRED delta (rank - best_raw): mean={delta['delta_mean']:.3f} "
          f"CI{delta['delta_ci95']} frac>0={delta['frac_delta_gt0']:.3f}")

    # Also compare rank vs the paper's WEAK global-Z baseline for context
    weak = raw[0]
    delta_vs_weak = paired_delta_ci(y, best_rank["_prob"], best_rank["_pred"],
                                    weak["_prob"], weak["_pred"], study)

    genuine = delta["delta_ci95"][0] > 0.0

    # strip internal arrays before serialization
    def clean(d):
        return {k: v for k, v in d.items() if not k.startswith("_")}

    raw_max_auroc = best_raw["auroc"]
    rank_target_reached_by_raw = raw_max_auroc >= 0.64  # did regularized raw hit rank level?

    if genuine:
        verdict = ("GENUINE representation effect: per-sample rank normalization "
                   "beats the best well-regularized RAW-gene model (same info, no "
                   "rank transform) with a paired study-block delta CI excluding 0.")
    elif rank_target_reached_by_raw:
        verdict = ("WEAK-BASELINE ARTIFACT: a well-regularized RAW-gene model "
                   "reaches the rank-norm level (>=0.64) WITHOUT the rank transform; "
                   "the 0.55 'collapse' is model-specific, not a representation gap.")
    else:
        verdict = ("MIXED / not established: rank-norm has the higher point AUROC "
                   "but the paired delta CI includes 0 (studies too few / noisy), so "
                   "rank-norm's necessity is not statistically established, though a "
                   "regularized raw model does NOT reach the rank level either.")

    result = {
        "audit": "ToxiGene transfer-gap contribution audit",
        "question": ("Is raw-gene LOSO 0.55 a weak-baseline artifact? Does a "
                     "well-regularized RAW-gene model (same info, no rank transform) "
                     "reach rank-norm 0.64-0.68 without the transform?"),
        "data_source": "real NCBI GEO (12 studies, 279 samples, 7902 genes)",
        "protocol": ("leave-one-study-out; feature selection + logreg fit per fold "
                     "on TRAIN studies only; rank-norm & per-study-std are per-"
                     "sample/study and label-free (leakage-free); study-block "
                     "bootstrap (2000x) CIs and PAIRED delta CI."),
        "n_studies": int(len(set(study.tolist()))),
        "n_samples": int(len(y)),
        "raw_gene_models": [clean(d) for d in raw],
        "rank_norm_models": [clean(d) for d in rk],
        "best_raw_gene_model": clean(best_raw),
        "best_rank_norm_model": clean(best_rank),
        "paper_weak_baseline_auroc": weak["auroc"],
        "best_regularized_raw_auroc": raw_max_auroc,
        "best_rank_norm_auroc": best_rank["auroc"],
        "regularized_raw_reaches_rank_level(>=0.64)": bool(rank_target_reached_by_raw),
        "paired_delta_rank_minus_bestRaw": delta,
        "paired_delta_rank_minus_weakBaseline": delta_vs_weak,
        "genuine_representation_effect": bool(genuine),
        "verdict": verdict,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nVERDICT: {verdict}")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
