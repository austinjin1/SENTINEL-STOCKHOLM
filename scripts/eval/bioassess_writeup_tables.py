#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Write-up tables for the M5 single test pass, computed only from the frozen run's saved artefacts
(comparisons_*.json, holm_A.json, pred_test*.parquet). Read-only; no registry writes.

    python -B scripts/eval/bioassess_writeup_tables.py --run-dir <final_run_dir> --out <json>
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ("log10_tp_ug_l", "log10_tn_mg_l", "log10_cond_us_cm", "log10_chloride_mg_l", "log10_turbidity_ntu", "ph_lab")
C = "Ensemble[BioSet[bioset_st_flat]+geography+RandomForest[community+geography]+TabPFNv2[community]]"
B = "Ensemble[BioSet[bioset_st_flat]+TabPFNv2[community]]"
A = "BioSet[bioset_st_flat]"


def r2(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def rmse(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((y[ok] - p[ok]) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.run_dir)
    rec = json.loads((d / "freeze_record.json").read_text())
    holm = json.loads((d / "holm_A.json").read_text())
    out = {"source_run": str(d), "targets": {}}
    comps = {b: json.loads((d / f"comparisons_{b}.json").read_text()) for b in ("A", "C", "B_all", "B_new", "B_resampled")}
    F = {b: pd.read_parquet(d / f"pred_test{b}.parquet") for b in ("A", "B", "C")}

    def comp(block, cand, kind, t):
        return next(x for x in comps[block] if x["candidate"] == cand and x["kind"] == kind and x["target"] == t)

    for t in TARGETS:
        ref = rec["sota_comparators"][t]
        o = {"sota_comparator": ref}
        for blk in ("A", "C"):
            FA = F[blk]
            y = FA[f"obs|{t}"].to_numpy(float)
            row = {}
            for cand, tag in ((C, "C"), (B, "B"), (A, "A")):
                x = comp(blk, cand, "sota", t)
                per_eco = {g: r2(y[m], FA[f"pred|{cand}|{t}"].to_numpy(float)[m]) - r2(y[m], FA[f"pred|{ref}|{t}"].to_numpy(float)[m])
                           for g in sorted(FA["ag_eco9"].unique()) for m in [FA["ag_eco9"].to_numpy() == g]}
                row[tag] = {"r2": r2(y, FA[f"pred|{cand}|{t}"].to_numpy(float)),
                            "delta_vs_sota": x["diff"], "ci": [x["ci_low"], x["ci_high"]], "p_two_sided": x["p_value"],
                            "per_seed_delta": x["per_seed_delta"], "sensitivity": x["sensitivity"],
                            "n_ecoregions_positive": int(sum(v > 0 for v in per_eco.values())), "n_ecoregions": len(per_eco),
                            "per_ecoregion_delta": per_eco,
                            "rmse": rmse(y, FA[f"pred|{cand}|{t}"].to_numpy(float))}
                if blk == "A":
                    k = f"sota|{cand}|{t}"
                    if k in holm:
                        row[tag]["holm"] = holm[k]
            row["comparator"] = {"r2": r2(y, FA[f"pred|{ref}|{t}"].to_numpy(float)), "rmse": rmse(y, FA[f"pred|{ref}|{t}"].to_numpy(float))}
            row["C"]["rmse_reduction_pct"] = 100 * (1 - row["C"]["rmse"] / row["comparator"]["rmse"])
            hm = comp(blk, A, "h_m5", t)
            row["h_m5"] = {"delta": hm["diff"], "ci": [hm["ci_low"], hm["ci_high"]], "p_two_sided": hm["p_value"],
                           "per_seed_delta": hm["per_seed_delta"],
                           "all_seeds_positive": all(v > 0 for v in hm["per_seed_delta"].values())}
            if blk == "A":
                row["h_m5"]["holm"] = holm.get(f"h_m5|{A}|{t}")
            for lbl in ("leave_bioset_out", "equal_size_baseline"):
                x = comp(blk, C, lbl, t)
                row[f"C_vs_{lbl}"] = {"delta": x["diff"], "ci": [x["ci_low"], x["ci_high"]],
                                      "all_seeds_positive": all(v > 0 for v in x["per_seed_delta"].values())}
            ranks = sorted(((r2(y, FA[c].to_numpy(float)), c.split("|")[1]) for c in FA.columns
                            if c.startswith("pred|") and c.endswith(f"|{t}")), reverse=True)
            row["top3"] = ranks[:3]
            o[f"test_{blk}"] = row
        for sub in ("B_all", "B_new", "B_resampled"):
            x = comp(sub, C, "sota", t)
            o[sub] = {"C_delta_vs_sota": x["diff"], "ci": [x["ci_low"], x["ci_high"]]}
        FB = F["B"]
        for sub, mask in (("B_all", np.ones(len(FB), bool)), ("B_new", ~FB["resampled"].to_numpy(bool)),
                          ("B_resampled", FB["resampled"].to_numpy(bool))):
            y = FB[f"obs|{t}"].to_numpy(float)[mask]
            sc = {c.split("|")[1]: r2(y, FB[c].to_numpy(float)[mask]) for c in FB.columns if c.startswith("pred|") and c.endswith(f"|{t}")}
            best = max(sc, key=sc.get)
            o[sub].update({"best_model": best, "best_r2": sc[best], "C_r2": sc[C], "rf_geography_r2": sc["RandomForest[geography]"],
                           "comparator_r2": sc[ref]})
        out["targets"][t] = o
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    for t, o in out["targets"].items():
        A_, C_ = o["test_A"], o["test_C"]
        print(f"{t:22s} A: C d={A_['C']['delta_vs_sota']:+.3f} [{A_['C']['ci'][0]:+.3f},{A_['C']['ci'][1]:+.3f}] holm={A_['C'].get('holm',{}).get('p_holm', float('nan')):.3g} "
              f"eco+ {A_['C']['n_ecoregions_positive']}/9 ecoCI {np.round(A_['C']['sensitivity']['ecoregion'],3)} rmse-{A_['C']['rmse_reduction_pct']:.1f}% | "
              f"Aalone d={A_['A']['delta_vs_sota']:+.3f} [{A_['A']['ci'][0]:+.3f},{A_['A']['ci'][1]:+.3f}] | HM5 d={A_['h_m5']['delta']:+.3f} seeds+={A_['h_m5']['all_seeds_positive']} "
              f"| TestC: C d={C_['C']['delta_vs_sota']:+.3f} [{C_['C']['ci'][0]:+.3f},{C_['C']['ci'][1]:+.3f}] HM5 d={C_['h_m5']['delta']:+.3f} [{C_['h_m5']['ci'][0]:+.3f},{C_['h_m5']['ci'][1]:+.3f}] seeds+={C_['h_m5']['all_seeds_positive']} "
              f"| B_all C d={o['B_all']['C_delta_vs_sota']:+.3f} best={o['B_all']['best_model'][:22]} {o['B_all']['best_r2']:.3f} geo={o['B_all']['rf_geography_r2']:.3f} C={o['B_all']['C_r2']:.3f}")


if __name__ == "__main__":
    main()
