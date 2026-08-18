#!/usr/bin/env python3
"""Improve ToxiGene CROSS-STUDY (LOSO) transfer via stronger domain adaptation.

BAR TO BEAT (audited, from results/benchmarks/toxigene_pathway.json):
  * Best prior LOSO method  H_L1_rankNorm_genes: pooled AUROC 0.681
    (95% study-block-bootstrap CI [0.608, 0.755]).
    = per-sample rank/quantile normalization + top-1500-variance genes +
      L1 logreg (C=0.1). This is the number we must exceed.
  * Stated headline baseline 0.55 (global-z raw genes).
  * Within-study reference 0.858 (upper bound, NOT a transfer number).

The remaining gap (0.68 vs 0.86 within-study) is BATCH EFFECTS across GEO
studies. Prior work already tried per-study standardize (~0.564) and ComBat
(~0.565) on raw genes -- both LOST to per-sample rank normalization. So the
open question is whether a STRONGER domain-adaptation step, applied *on top of*
the rank-normalized features, closes more of the gap.

Domain-adaptation methods evaluated here (all leakage-free, fit on TRAIN
studies only where any statistic is needed; the held-out study is aligned using
ONLY its own feature values, never its labels):

  R0  rank-norm + L1 logreg (C=0.1)            -- EXACT prior best (replication)
  QN  cross-study quantile normalization to a TRAIN reference distribution
        (classic microarray QN; target quantiles from TRAIN pool; held-out
         study mapped onto the same target quantiles -- label-free)
  CORAL  rank-norm features, then per-study CORAL alignment: whiten each study
         by its own covariance and re-color to a TRAIN reference covariance
         (second-order / correlation alignment; label-free)
  PSS   rank-norm + per-study standardize (first-order alignment on ranks)
  R0+DA combinations of the above stacked before the L1 classifier

EVALUATION (identical strict protocol to the audited baseline):
  * Leave-one-study-out (LOSO). Feature selection (top-variance) + classifier
    fit per fold on TRAIN studies ONLY.
  * Pooled out-of-fold AUROC.
  * STUDY-BLOCK bootstrap (resample the 12 studies with replacement, 2000x) for
    the CI on each method AND for a PAIRED delta CI vs the prior best.
  * A gain counts ONLY if delta = (method - prior_best) has a study-block
    bootstrap CI whose lower bound EXCLUDES 0.

MIT License -- Bryan Cheng, 2026
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "molecular" / "realgeo"
OUT_PATH = PROJECT_ROOT / "results" / "benchmarks" / "weak_improve" / "toxigene-transfer.json"

RNG = 42
N_TOP_GENES = 1500
PRIOR_BEST_AUROC = 0.6809085681426107  # H_L1_rankNorm_genes, audited
N_BOOT = 2000


# ------------------------------------------------------------------ data ----
def load_data():
    X = np.load(DATA_DIR / "X.npy")
    y = np.load(DATA_DIR / "y.npy")
    study = np.load(DATA_DIR / "study.npy", allow_pickle=True)
    genes = json.loads((DATA_DIR / "genes.json").read_text())
    return X, y, study, genes


# --------------------------------------------------- domain-adaptation ------
def rank_normalize(X):
    """Per-sample within-array quantile rank (0..1). Label-free, per-sample =>
    trivially leakage-free (independent of any other sample)."""
    return np.apply_along_axis(lambda r: rankdata(r) / len(r), 1, X).astype(np.float32)


def per_study_standardize(X, study):
    """Center/scale each study by its OWN mean/std. Label-free. The held-out
    study uses its own stats, so no train info leaks in either direction."""
    Xc = X.astype(np.float64).copy()
    for s in set(study.tolist()):
        m = study == s
        mu = Xc[m].mean(0)
        sd = Xc[m].std(0)
        sd[sd < 1e-8] = 1.0
        Xc[m] = (Xc[m] - mu) / sd
    return Xc.astype(np.float32)


def quantile_normalize_to_ref(X, study, ref_studies):
    """Classic cross-sample quantile normalization.

    Build a TARGET quantile profile = row-sorted mean over TRAIN(ref) samples
    only. Then map every sample (train AND held-out) onto that target profile by
    rank. Label-free; the target uses only reference-study feature values.
    """
    X = X.astype(np.float64)
    ref_mask = np.isin(study, list(ref_studies))
    Xref = X[ref_mask]
    # target profile: mean of each column of the row-sorted reference matrix
    sorted_ref = np.sort(Xref, axis=1)
    target = sorted_ref.mean(axis=0)  # length G, ascending
    G = X.shape[1]
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        r = rankdata(X[i], method="average")  # 1..G
        # map average ranks onto target via interpolation on positions
        pos = (r - 1.0)  # 0..G-1
        out[i] = np.interp(pos, np.arange(G), target)
    return out.astype(np.float32)


def coral_align(X, study, ref_studies, eps=1e-3):
    """CORAL (CORrelation ALignment) per study on the given feature matrix.

    For each study: whiten by its OWN covariance, then re-color to a TRAIN
    reference covariance. Second-order domain alignment. Label-free -- uses only
    feature covariances and a train-reference covariance; held-out study is
    aligned with its own covariance (no train info about its labels).

    To keep the covariance well-conditioned in the p>>n regime, apply after
    feature selection is impractical (selection is per-fold), so we operate in a
    reduced random-free deterministic subspace: shrink covariance with eps*I.
    Runs on the (already low-D after we pass reduced feats) matrix.
    """
    X = X.astype(np.float64)
    d = X.shape[1]
    ref_mask = np.isin(study, list(ref_studies))
    Ct = np.cov(X[ref_mask], rowvar=False) + eps * np.eye(d)
    # target coloring matrix
    Ct_sqrt = _matrix_sqrt(Ct)
    out = X.copy()
    for s in set(study.tolist()):
        m = study == s
        if m.sum() < 3:
            continue
        Xs = X[m]
        mu = Xs.mean(0)
        Cs = np.cov(Xs, rowvar=False) + eps * np.eye(d)
        Cs_inv_sqrt = _matrix_inv_sqrt(Cs)
        out[m] = (Xs - mu) @ Cs_inv_sqrt @ Ct_sqrt + mu
    return out.astype(np.float32)


def _matrix_sqrt(C):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-8, None)
    return (V * np.sqrt(w)) @ V.T


def _matrix_inv_sqrt(C):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-8, None)
    return (V * (1.0 / np.sqrt(w))) @ V.T


# ----------------------------------------------------------- LOSO runner ----
def _metrics(yt, yp, prob):
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


def loso(make_feats, y, study, select_top_var=N_TOP_GENES,
         C=0.1, penalty="l1", solver="liblinear"):
    """Generic LOSO. `make_feats(train_mask)` returns the full N x D feature
    matrix for that fold, computing any TRAIN-dependent alignment using ONLY
    train studies as reference (held-out uses its own stats)."""
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
        Xf = make_feats(tr)
        Xtr, Xte = Xf[tr], Xf[te]
        if select_top_var is not None and select_top_var < Xf.shape[1]:
            var = Xtr.var(0)
            top = np.argsort(var)[::-1][:select_top_var]
            Xtr, Xte = Xtr[:, top], Xte[:, top]
        clf = LogisticRegression(C=C, penalty=penalty, solver=solver,
                                 max_iter=3000, class_weight="balanced",
                                 random_state=RNG)
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:, 1]
        oof_prob[te] = prob
        oof_pred[te] = (prob >= 0.5).astype(int)
        per_study[held] = _metrics(y[te], oof_pred[te], prob)
    return oof_prob, oof_pred, per_study


def pooled_auroc(y, oof_prob, oof_pred):
    mask = oof_pred >= 0
    return float(roc_auc_score(y[mask], oof_prob[mask]))


def summarize(y, oof_prob, oof_pred, study):
    mask = oof_pred >= 0
    yt, yp, pr = y[mask], oof_pred[mask], oof_prob[mask]
    return {
        "n_evaluated": int(mask.sum()),
        "accuracy": float(accuracy_score(yt, yp)),
        "f1_macro": float(f1_score(yt, yp, average="macro")),
        "auroc": float(roc_auc_score(yt, pr)),
        "baseline_majority_acc": float(max((yt == 0).mean(), (yt == 1).mean())),
    }


# ------------------------------- study-block bootstrap CI + paired delta ----
def study_bootstrap(y, prob_dict, oof_pred_ref, study, ref_key, n_boot=N_BOOT):
    """Resample studies with replacement; recompute pooled AUROC for every
    method on the SAME resampled studies (paired), and the delta vs ref_key.

    prob_dict: {method_name: oof_prob array (nan where unpredicted)}
    Uses the reference method's predicted mask to define evaluable samples
    (all methods predict the same held-out samples, so masks coincide)."""
    mask = oof_pred_ref >= 0
    idx_all = np.where(mask)[0]
    st = study[idx_all]
    yt = y[idx_all]
    uniq = sorted(set(st.tolist()))
    rng = np.random.default_rng(RNG)

    per_method_boot = {k: [] for k in prob_dict}
    delta_boot = {k: [] for k in prob_dict if k != ref_key}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([np.where(st == s)[0] for s in pick])
        yb = yt[sel]
        if len(set(yb.tolist())) < 2:
            continue
        aur = {}
        ok = True
        for k, pr in prob_dict.items():
            prb = pr[idx_all][sel]
            if np.isnan(prb).any():
                ok = False
                break
            aur[k] = roc_auc_score(yb, prb)
        if not ok:
            continue
        for k in prob_dict:
            per_method_boot[k].append(aur[k])
        for k in delta_boot:
            delta_boot[k].append(aur[k] - aur[ref_key])

    out = {}
    for k, arr in per_method_boot.items():
        arr = np.asarray(arr)
        out[k] = {
            "auroc_ci95": [float(np.percentile(arr, 2.5)),
                           float(np.percentile(arr, 97.5))],
        }
        if k in delta_boot:
            d = np.asarray(delta_boot[k])
            out[k]["delta_vs_ref"] = float(np.mean(d))
            out[k]["delta_ci95"] = [float(np.percentile(d, 2.5)),
                                    float(np.percentile(d, 97.5))]
            out[k]["p_delta_gt_0"] = float(np.mean(d > 0))
    return out


# ------------------------------------------------------------------- main ---
def main():
    t0 = time.time()
    X, y, study, genes = load_data()
    print(f"X={X.shape} studies={len(set(study.tolist()))} "
          f"pos={int(y.sum())} neg={int((y==0).sum())}")

    # precompute the base rank-normalized matrix once (per-sample, fold-invariant)
    Xrank = rank_normalize(X)

    methods = {}  # name -> (oof_prob, oof_pred, per_study, pooled_auroc, desc)

    def run(name, make_feats, desc, **kw):
        p, pr, ps = loso(make_feats, y, study, **kw)
        a = pooled_auroc(y, p, pr)
        methods[name] = dict(prob=p, pred=pr, per_study=ps, auroc=a, desc=desc)
        print(f"  {name:34s} LOSO AUROC = {a:.4f}")

    # R0: EXACT prior best replication (rank-norm + L1 C=0.1, top-1500 var)
    run("R0_prior_best_rank_L1",
        lambda tr: Xrank,
        "prior best: per-sample rank-norm genes, top-1500 var, L1 logreg C=0.1",
        C=0.1, penalty="l1", solver="liblinear")

    # PSS: rank-norm + per-study standardize (first-order alignment on ranks)
    Xrank_pss = per_study_standardize(Xrank, study)
    run("PSS_rank_perStudyStd_L1",
        lambda tr: Xrank_pss,
        "rank-norm then per-study standardize, top-1500 var, L1 C=0.1",
        C=0.1, penalty="l1", solver="liblinear")

    # QN: cross-study quantile normalization to TRAIN reference (on raw X)
    run("QN_crossStudy_quantile_L1",
        lambda tr: quantile_normalize_to_ref(X, study, sorted(set(study[tr].tolist()))),
        "cross-study quantile-normalized raw expression to TRAIN ref, top-1500 var, L1 C=0.1",
        C=0.1, penalty="l1", solver="liblinear")

    # CORAL: rank-norm -> top-var reduce (train) -> CORAL align in reduced space.
    # CORAL needs a well-conditioned covariance, so we first reduce to top-300
    # variance genes (train-only) INSIDE make_feats, then CORAL-align, then hand
    # the aligned reduced matrix to loso with select_top_var=None.
    def coral_feats(tr):
        ref = sorted(set(study[tr].tolist()))
        # train-only top-var reduction for conditioning (label-free)
        var = Xrank[tr].var(0)
        top = np.argsort(var)[::-1][:300]
        Xr = Xrank[:, top]
        return coral_align(Xr, study, ref)
    run("CORAL_rank_L1",
        coral_feats,
        "rank-norm -> top-300 var (train) -> per-study CORAL align to TRAIN ref, L1 C=0.1",
        select_top_var=None, C=0.1, penalty="l1", solver="liblinear")

    # R0 with L2 as a robustness reference (dense)
    run("R0_rank_L2",
        lambda tr: Xrank,
        "rank-norm genes, top-1500 var, L2 logreg C=0.1",
        C=0.1, penalty="l2", solver="lbfgs")

    # QN + rank not combined; instead QN followed by per-sample rank (double DA)
    Xqn_will_be_per_fold = True  # QN is train-dependent -> per fold
    def qn_then_rank(tr):
        ref = sorted(set(study[tr].tolist()))
        Xqn = quantile_normalize_to_ref(X, study, ref)
        return rank_normalize(Xqn)
    run("QNthenRank_L1",
        qn_then_rank,
        "cross-study QN then per-sample rank, top-1500 var, L1 C=0.1",
        C=0.1, penalty="l1", solver="liblinear")

    # ---- paired study-block bootstrap: CI on each + delta vs prior best ----
    ref_key = "R0_prior_best_rank_L1"
    prob_dict = {k: v["prob"] for k, v in methods.items()}
    boot = study_bootstrap(y, prob_dict, methods[ref_key]["pred"], study, ref_key)

    # assemble per-method summary
    summary = {}
    for k, v in methods.items():
        s = {
            "auroc": v["auroc"],
            "auroc_ci95": boot[k]["auroc_ci95"],
            "desc": v["desc"],
        }
        if "delta_vs_ref" in boot[k]:
            s["delta_vs_prior_best"] = boot[k]["delta_vs_ref"]
            s["delta_ci95"] = boot[k]["delta_ci95"]
            s["p_delta_gt_0"] = boot[k]["p_delta_gt_0"]
        summary[k] = s

    # pick best non-reference method by pooled AUROC
    cand = {k: v for k, v in methods.items() if k != ref_key}
    best_key = max(cand, key=lambda k: cand[k]["auroc"])
    best_auroc = cand[best_key]["auroc"]
    best_delta_ci = summary[best_key]["delta_ci95"]

    # improved iff new > prior best AND paired delta CI lower bound > 0
    improved = (best_auroc > PRIOR_BEST_AUROC) and (best_delta_ci[0] > 0.0)

    result = {
        "benchmark": "ToxiGene cross-study (leave-one-study-out) toxicant-exposure AUROC",
        "task": "binary control-vs-exposed classification, zebrafish GEO microarray",
        "data_source": "real NCBI GEO series matrices, 12 studies, 279 samples, 7902 shared genes",
        "holdout": "leave-one-study-out (LOSO); feature selection + classifier fit per fold on TRAIN studies only; alignment stats train-referenced, held-out study aligned by its OWN features (no labels)",
        "n_studies": int(len(set(study.tolist()))),
        "n_samples": int(len(y)),
        "prior_best_condition": "H_L1_rankNorm_genes (audited)",
        "prior_best_loso_auroc": PRIOR_BEST_AUROC,
        "stated_headline_baseline": 0.55,
        "within_study_reference_auroc": 0.858,
        "n_boot_study_block": N_BOOT,
        "methods": {k: {
            "loso_auroc": v["auroc"],
            "auroc_ci95": summary[k]["auroc_ci95"],
            "desc": v["desc"],
            "per_study_auroc": {s: (m.get("auroc")) for s, m in v["per_study"].items()},
            **({"delta_vs_prior_best": summary[k]["delta_vs_prior_best"],
                "delta_ci95": summary[k]["delta_ci95"],
                "p_delta_gt_0": summary[k]["p_delta_gt_0"]}
               if "delta_vs_prior_best" in summary[k] else {}),
        } for k, v in methods.items()},
        "best_new_method": best_key,
        "best_new_loso_auroc": best_auroc,
        "best_new_auroc_ci95": summary[best_key]["auroc_ci95"],
        "best_delta_vs_prior_best": summary[best_key]["delta_vs_prior_best"],
        "best_delta_ci95": best_delta_ci,
        "improved": bool(improved),
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    # honest verdict / diagnosis text
    if improved:
        result["verdict"] = (
            f"IMPROVED: {best_key} reaches LOSO AUROC {best_auroc:.3f} > prior "
            f"best {PRIOR_BEST_AUROC:.3f}; paired study-block-bootstrap delta CI "
            f"{best_delta_ci} excludes 0."
        )
    else:
        result["verdict"] = (
            f"NO HONEST IMPROVEMENT over the audited prior best "
            f"({PRIOR_BEST_AUROC:.3f}). Best new method {best_key}="
            f"{best_auroc:.3f}, paired delta CI {best_delta_ci} includes 0 "
            f"(or point estimate below prior best). Stronger domain adaptation "
            f"(cross-study quantile normalization, CORAL second-order alignment, "
            f"per-study standardization on ranks) does NOT beat simple "
            f"per-sample rank normalization + sparse L1 on this 12-study set. "
            f"The residual gap to within-study 0.858 reflects genuine "
            f"study-to-study biological/chemistry heterogeneity (different "
            f"toxicants, doses, tissues, developmental stages), which alignment "
            f"of marginal/second-order feature statistics cannot remove without "
            f"the held-out study's labels."
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print("\n=== LOSO AUROC (paired study-block bootstrap) ===")
    for k in methods:
        s = summary[k]
        line = f"  {k:34s} AUROC={s['auroc']:.3f} CI[{s['auroc_ci95'][0]:.3f},{s['auroc_ci95'][1]:.3f}]"
        if "delta_ci95" in s:
            line += (f"  d={s['delta_vs_prior_best']:+.3f} "
                     f"CI[{s['delta_ci95'][0]:+.3f},{s['delta_ci95'][1]:+.3f}] "
                     f"P(d>0)={s['p_delta_gt_0']:.2f}")
        print(line)
    print(f"\nBEST NEW: {best_key} AUROC={best_auroc:.3f} vs prior {PRIOR_BEST_AUROC:.3f}")
    print(f"IMPROVED: {improved}")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
