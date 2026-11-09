#!/usr/bin/env python
"""Independent verification of the bloom-early-warning ONSET-ONLY claim.

Reproduces the onset-only new-bloom forecast AUROCs and adds the MISSING rigor:
a site-clustered PAIRED bootstrap on (fusion - best_single) restricted to the
onset-only subtask, to test whether multimodal optics+drivers GENUINELY add over
antecedent biomass (persistence-model), or whether the +0.02 point gain is noise.
Reuses the exact leakage-free feature build + OOF from bloom_early_warning.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bloom_early_warning as B  # noqa: E402


def main():
    df, sf = B.build()
    m = (df["chla"].notna() & df["next_chla"].notna()
         & (df["lead"] >= B.LEAD_LO) & (df["lead"] <= B.LEAD_HI))
    d = df[m].reset_index(drop=True)

    traj_sf = ["traj_" + c for c in sf]
    SAT = sf + traj_sf
    DRV = ["temp", "tp", "traj_temp"]
    PERS = ["chla", "prevd_chla", "traj_chla", "month_sin", "month_cos", "lead", "ante_gap"]
    Xsat = d[SAT].to_numpy("float64"); Xdrv = d[DRV].to_numpy("float64")
    Xpers = d[PERS].to_numpy("float64"); Xfus = np.hstack([Xsat, Xdrv, Xpers])
    sites = d["site"].astype(str).to_numpy()

    for name, thr in B.THRESHOLDS.items():
        y = (d["next_chla"].to_numpy() > thr).astype(int)
        state = (d["chla"].to_numpy() > thr).astype(int)
        om = state == 0                      # onset-only: t0 NOT already blooming
        yo, so = y[om], sites[om]
        if not (om.sum() >= 30 and 0 < yo.sum() < om.sum()):
            print(f"[{name}] too few onset pairs"); continue
        nso = min(5, len(np.unique(so)))

        def oof_sub(X):
            return B.within_oof(X[om], yo, so, nso)

        p = {"satellite": oof_sub(Xsat), "drivers": oof_sub(Xdrv),
             "persistence_model": oof_sub(Xpers), "fusion": oof_sub(Xfus)}
        auc = {k: round(float(roc_auc_score(yo, v)), 3) for k, v in p.items()}
        best = max(("satellite", "drivers", "persistence_model"), key=lambda k: auc[k])

        fus_ci = B.cluster_bootstrap_auc(yo, p["fusion"], so)
        vs_best = B.paired_delta(yo, p["fusion"], p[best], so)
        # also fusion vs optics+drivers-only (no antecedent chl) to bound persistence reliance
        Xnoper = np.hstack([Xsat, Xdrv, d[["month_sin", "month_cos", "lead", "ante_gap"]].to_numpy("float64")])
        p_noper = B.within_oof(Xnoper[om], yo, so, nso)
        auc_noper = round(float(roc_auc_score(yo, p_noper)), 3)

        print(f"\n[{name}] ONSET-ONLY (new-bloom) n={int(om.sum())} pos={int(yo.sum())} base={yo.mean():.2f}")
        print(f"  AUROC: sat={auc['satellite']} drv={auc['drivers']} "
              f"persistence_model={auc['persistence_model']} FUSION={auc['fusion']} "
              f"(fusion_no_antecedent_chl={auc_noper})")
        print(f"  fusion cluster-boot CI {fus_ci['ci95']} (excludes 0.5: {fus_ci['ci95'][0] > 0.5})")
        print(f"  fusion - best_single({best} {auc[best]}): delta={vs_best['delta_median']:+.3f} "
              f"CI{vs_best['ci95']} P(>0)={vs_best['P_delta_gt0']} "
              f"MULTIMODAL_ADDS_OVER_BEST_SINGLE={vs_best['excludes_0'] and vs_best['delta_median'] > 0}")


if __name__ == "__main__":
    main()
