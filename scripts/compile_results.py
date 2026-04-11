#!/usr/bin/env python3
"""Results Compilation: Gather all experiment outputs into paper-ready summary.

Reads all available exp JSON results and produces:
  1. A comprehensive statistics table (JSON + LaTeX)
  2. A master results summary for paper writing
  3. A critique resolution checklist

MIT License — Bryan Cheng, 2026
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_BASE = PROJECT_ROOT / "results"
OUTPUT_DIR   = PROJECT_ROOT / "results" / "compiled"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: Could not load {path}: {e}")
        return None


def compile_all():
    print("=" * 65)
    print("SENTINEL RESULTS COMPILATION")
    print("=" * 65)

    compiled = {}

    # -----------------------------------------------------------------------
    # Exp9: Bootstrap CIs
    # -----------------------------------------------------------------------
    ci = load_json(RESULTS_BASE / "exp9_bootstrap" / "ci_results.json")
    if ci:
        compiled["exp9_bootstrap_ci"] = ci
        print("\n=== Exp9: Bootstrap 95% CIs ===")
        for model, res in ci.get("results", {}).items():
            pt  = res.get("point", res.get("auroc", res.get("r2", res.get("f1", "?"))))
            lo  = res.get("ci_lo", "?")
            hi  = res.get("ci_hi", "?")
            met = res.get("metric", "?")
            print(f"  {model}: {met}={pt:.4f} [{lo:.4f}, {hi:.4f}]")
    else:
        print("\n  [Exp9: Not yet complete]")

    # -----------------------------------------------------------------------
    # Exp10: MC Dropout
    # -----------------------------------------------------------------------
    mc = load_json(RESULTS_BASE / "exp10_mc_dropout" / "mc_results.json")
    if mc:
        compiled["exp10_mc_dropout"] = mc
        print("\n=== Exp10: MC Dropout Uncertainty ===")
        for model, res in mc.get("models", mc.items() if isinstance(mc, dict) else {}).items():
            if isinstance(res, dict):
                ece = res.get("ece", "?")
                unc = res.get("mean_uncertainty", "?")
                print(f"  {model}: ECE={ece}, uncertainty={unc}")
    else:
        print("\n  [Exp10: Not yet complete]")

    # -----------------------------------------------------------------------
    # Exp11: Label Noise / BioMotion Scrutiny
    # -----------------------------------------------------------------------
    ln = load_json(RESULTS_BASE / "exp11_label_noise" / "sensitivity_results.json")
    if ln:
        compiled["exp11_label_noise"] = ln
        print("\n=== Exp11: BioMotion Scrutiny ===")
        perm = ln.get("permutation_test", {})
        dist = ln.get("score_distribution", {})
        print(f"  True AUROC: {ln.get('true_auroc', '?'):.4f}")
        print(f"  Permutation p-value: {perm.get('p_value', '?'):.4f}")
        print(f"  Cohen's d: {dist.get('cohens_d', '?'):.4f}")
        for v in ln.get("verdicts", []):
            print(f"  ✓ {v}")
    else:
        print("\n  [Exp11: Not yet complete]")

    # -----------------------------------------------------------------------
    # Exp12: Multi-modal Integration
    # -----------------------------------------------------------------------
    inte = load_json(RESULTS_BASE / "exp12_integration" / "integration_results.json")
    if inte:
        compiled["exp12_integration"] = inte
        print("\n=== Exp12: Multi-Modal Integration ===")
        per = inte.get("per_combo", {})
        best = max((v for v in per.values() if v.get("method") == "fusion"),
                   key=lambda x: x["auroc"], default=None)
        if best:
            print(f"  Best fusion: {'+'.join(best['modalities'])} AUROC={best['auroc']:.4f}")
        gains = inte.get("marginal_modality_gains", {})
        for m, g in sorted(gains.items(), key=lambda x: -x[1]):
            print(f"  {m}: {g:+.4f} AUROC vs sensor-only")
    else:
        print("\n  [Exp12: Not yet complete]")

    # -----------------------------------------------------------------------
    # Exp13: PRPO Audit
    # -----------------------------------------------------------------------
    prpo = load_json(RESULTS_BASE / "exp13_prpo_audit" / "prpo_audit_results.json")
    if prpo:
        compiled["exp13_prpo_audit"] = prpo
        print("\n=== Exp13: PRPO Data Audit ===")
        pp = prpo.get("pre_post_2022_test", {})
        print(f"  Pre-2022 SpCond: {pp.get('median_pre_2022', '?'):.1f} µS/cm")
        print(f"  Post-2022 SpCond: {pp.get('median_post_2022', '?'):.1f} µS/cm")
        print(f"  Mann-Whitney p: {pp.get('p_value', '?'):.2e}")
        qf = prpo.get("qf_trend_correlation", {})
        print(f"  SpCond↔QF corr: r={qf.get('sc_qf_correlation', '?'):.3f}")
        for v in prpo.get("verdicts", []):
            print(f"  • {v}")
    else:
        print("\n  [Exp13: Not yet complete]")

    # -----------------------------------------------------------------------
    # Exp3: EPA Event Correlation
    # -----------------------------------------------------------------------
    epa = load_json(RESULTS_BASE / "exp3_epa_correlation" / "epa_results.json")
    if epa:
        compiled["exp3_epa"] = epa
        print("\n=== Exp3: EPA Event Detection ===")
        print(f"  Events scored: {epa.get('n_events_scored', '?')}/{epa.get('n_events_total', '?')}")
        print(f"  Median lead time: {epa.get('median_lead_time_h', '?'):.1f}h")
        print(f"  Mean lead time: {epa.get('mean_lead_time_h', '?'):.1f}h")
    else:
        print("\n  [Exp3: Not yet complete]")

    # -----------------------------------------------------------------------
    # Exp5: Explainability
    # -----------------------------------------------------------------------
    expl = load_json(RESULTS_BASE / "exp5_explainability" / "explainability_results.json")
    if expl:
        compiled["exp5_explainability"] = expl
        print("\n=== Exp5: Feature Importance ===")
        att = expl.get("attention_weights", {})
        for k, v in att.items():
            print(f"  Attention {k}: {v:.1%}")
        pert = expl.get("perturbation_sensitivity", {})
        for k, v in pert.items():
            if isinstance(v, (int, float)):
                print(f"  Perturbation Δ {k}: {v:.4f}")
    else:
        print("\n  [Exp5: Not yet complete]")

    # -----------------------------------------------------------------------
    # Exp7: Cross-modal CKA
    # -----------------------------------------------------------------------
    cka = load_json(RESULTS_BASE / "exp7_crossmodal" / "crossmodal_results.json")
    if cka:
        compiled["exp7_cka"] = cka
        print("\n=== Exp7: Cross-Modal Alignment ===")
        for pair, vals in cka.get("cka_pairs", {}).items():
            print(f"  CKA {pair}: {vals.get('linear_cka', '?'):.4f}")
    else:
        print("\n  [Exp7: Not yet complete]")

    # -----------------------------------------------------------------------
    # Exp8: NEON Trends
    # -----------------------------------------------------------------------
    trends = load_json(RESULTS_BASE / "exp8_neon_trends" / "trend_results.json")
    if trends:
        compiled["exp8_trends"] = trends
        print("\n=== Exp8: NEON Temporal Trends ===")
        sig = trends.get("significant_trends", [])
        print(f"  Significant trends: {len(sig)}")
        top = sorted(sig, key=lambda x: abs(x.get("slope", 0)), reverse=True)[:5]
        for t in top:
            print(f"  {t.get('site')}/{t.get('param')}: "
                  f"{t.get('slope', 0):+.1f}/yr (p={t.get('p_value', 1):.3f})")
    else:
        print("\n  [Exp8: Not yet complete]")

    # -----------------------------------------------------------------------
    # NEON Scan
    # -----------------------------------------------------------------------
    neon = load_json(RESULTS_BASE / "neon_anomaly_scan" / "neon_scan_results.json")
    if neon:
        compiled["neon_scan"] = neon
        print("\n=== NEON Anomaly Scan ===")
        print(f"  Sites processed: {neon.get('n_sites_success', '?')}/{neon.get('n_sites_processed', '?')}")
        print(f"  Total windows: {neon.get('total_windows', '?'):,}")
        top_sites = neon.get("top_sites_by_anomaly_score", [])
        for ts in top_sites[:5]:
            print(f"  {ts['site']}: max={ts['max_score']:.4f}, mean={ts['mean_score']:.4f}")
    else:
        print("\n  [NEON Scan: Not yet complete]")

    # -----------------------------------------------------------------------
    # Key metrics table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("KEY METRICS TABLE (for paper)")
    print("=" * 65)
    metrics_table = {
        "AquaSSM (USGS sensor)":  {"metric": "AUROC", "value": "TBD", "ci": "[TBD, TBD]"},
        "HydroViT (water temp)":  {"metric": "R²",    "value": "0.749", "ci": "[TBD, TBD]"},
        "MicroBiomeNet":          {"metric": "F1",    "value": "0.913", "ci": "[TBD, TBD]"},
        "ToxiGene":               {"metric": "F1",    "value": "0.894", "ci": "[TBD, TBD]"},
        "BioMotion (ECOTOX)":     {"metric": "AUROC", "value": "0.9999","ci": "[TBD, TBD]"},
        "Fusion (best combo)":    {"metric": "AUROC", "value": "0.638", "ci": "[TBD, TBD]"},
    }
    if ci:
        for model, res in ci.get("results", {}).items():
            clean = model.replace("_", " ").title()
            for k in metrics_table:
                if clean.lower() in k.lower() or model.lower() in k.lower():
                    pt = res.get("point", "?")
                    lo = res.get("ci_lo", "?")
                    hi = res.get("ci_hi", "?")
                    metrics_table[k]["value"] = f"{pt:.4f}"
                    metrics_table[k]["ci"] = f"[{lo:.4f}, {hi:.4f}]"

    print(f"{'Model':<30} {'Metric':<8} {'Value':>8} {'95% CI':>20}")
    print("-" * 70)
    for model, info in metrics_table.items():
        print(f"  {model:<28} {info['metric']:<8} {info['value']:>8} {info['ci']:>20}")

    # -----------------------------------------------------------------------
    # Critique resolution status
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("CRITIQUE RESOLUTION STATUS")
    print("=" * 65)
    critiques = [
        ("No CIs on metrics",          "✅" if ci else "🔄", "Exp9 bootstrap CI"),
        ("No uncertainty quantification", "✅" if mc else "🔄", "Exp10 MC Dropout"),
        ("BioMotion AUROC suspicious", "✅" if ln else "🔄", "Exp11 label noise"),
        ("Exp2 broken baseline",       "✅",                  "Exp12 proper fusion"),
        ("Cross-modal CKA near-zero",  "📝",                  "Exp7 documented"),
        ("AquaSSM mask shape bug",     "✅",                  "Fixed in neon_scan.py"),
        ("HydroViT weak multi-param",  "📝",                  "Documented"),
        ("NEON trend artifacts",       "✅" if prpo else "🔄","Exp13 PRPO audit"),
        ("Single training seed",       "🔶",                  "Exp10 MC proxy"),
    ]
    for crit, status, exp in critiques:
        print(f"  {status} {crit:<40} ({exp})")

    # Save compiled
    out = OUTPUT_DIR / "master_results.json"
    with open(out, "w") as f:
        json.dump(compiled, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    compile_all()
