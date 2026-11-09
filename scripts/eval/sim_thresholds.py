# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Conductivity × macroinvertebrate thresholds (protocol §8): national NRSA replication of the EPA
300 µS/cm benchmark (Central Appalachia positive control) + UK EA replication.

    python -B scripts/eval/sim_thresholds.py --workers 6 [--register] [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.core.registry import log_result
from sentinel.core.runlog import TrainingRun

OUT = paths.RUNS_ROOT / "sim" / "thresholds"
EPA_HC05 = 300.0


def _strata():
    from sentinel.analysis import thresholds as TH

    mg, ag = TH.nrsa_samples("genus", "wadeable")
    mf, af = TH.nrsa_samples("family", "wadeable")
    mb, abo = TH.nrsa_samples("genus", "boatable")
    S = {"appalachia_L3_69_70_genus": (mg[mg["us_l3code"].astype(str).isin(TH.APPALACHIA_L3)], ag),
         "national_wadeable_genus": (mg, ag), "national_wadeable_family": (mf, af),
         "national_boatable_genus": (mb, abo)}
    for e in sorted(mg["ag_eco9"].dropna().unique()):
        S[f"ecoregion_{e}_genus"] = (mg[mg["ag_eco9"] == e], ag)
    mu, au = TH.uk_samples("family")
    S["uk_england_family"] = (mu, au)
    out = {}
    for k, (m, a) in S.items():
        a = a.loc[m.index]
        a = a.loc[:, (a > 0).sum(axis=0) > 0]
        out[k] = (m, a)
    return out


def _job(args):
    name, meta, ab, n_boot_x, n_boot_t, n_perm, seed = args
    from sentinel.analysis import thresholds as TH

    res = {"stratum": name, "n_samples": int(len(meta)), "n_units": int(meta["unit"].nunique()),
           "bootstrap_unit": "site cluster (NRSA unique_id + coordinates <= 1 km)" if not name.startswith("uk")
           else "site (one sample per EA site)", "filters": {}}
    ref_mask = None
    if "reference_class" in meta:
        ref = meta["reference_class"].astype(str).eq("R").to_numpy()
        res["n_reference_samples"] = int(ref.sum())
        res["filters"]["reference_taxa"] = ("applied" if ref.sum() >= TH.MIN_REF_SAMPLES else
                                            f"NOT applied: {int(ref.sum())} reference samples < {TH.MIN_REF_SAMPLES}")
        if ref.sum() >= TH.MIN_REF_SAMPLES:
            ref_mask = (ab.to_numpy()[ref] > 0).any(axis=0)
            res["background_sc_p10_reference"] = float(np.quantile(meta["sc"].to_numpy()[ref], 0.10))
            res["background_sc_p25_reference"] = float(np.quantile(meta["sc"].to_numpy()[ref], 0.25))
    else:
        res["filters"]["reference_taxa"] = "NOT applicable (no reference classes)"
    res["sc_quantiles"] = {q: float(np.quantile(meta["sc"], q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
    # testability counts the taxa entering the PRIMARY estimate (reference-taxa HC05 when that filter applies)
    occ_ok = (ab.to_numpy() > 0).sum(axis=0) >= 25
    n_taxa25 = int((occ_ok & ref_mask).sum() if ref_mask is not None else occ_ok.sum())
    reasons = []
    if len(meta) < TH.MIN_SAMPLES:
        reasons.append(f"{len(meta)} samples < {TH.MIN_SAMPLES}")
    if res["n_units"] < TH.MIN_UNITS:
        reasons.append(f"{res['n_units']} site clusters < {TH.MIN_UNITS}")
    if n_taxa25 < TH.MIN_TAXA_HC05:
        reasons.append(f"{n_taxa25} taxa entering the primary HC05 (>= 25 occurrences) < {TH.MIN_TAXA_HC05}")
    res["n_taxa_ge25"] = n_taxa25
    res["testable"] = not reasons
    res["testability_reasons"] = reasons
    if len(meta) < 50:
        res["skipped"] = "fewer than 50 samples (no estimate)"
        return res
    ex = TH.extirpation(meta, ab, n_boot=n_boot_x, seed=seed)
    res["xc95_hc05_all_taxa"] = {k: v for k, v in ex.items() if k != "xc95"}
    res["xc95_by_taxon"] = ex["xc95"]
    if ref_mask is not None:
        exr = TH.extirpation(meta, ab, ref_mask=ref_mask, n_boot=n_boot_x, seed=seed)
        res["xc95_hc05_reference_taxa"] = {k: v for k, v in exr.items() if k != "xc95"}
    ti = TH.titan(meta, ab, n_perm=n_perm, n_boot=n_boot_t, seed=seed)
    if ti["n_decreasers"] > 0:
        res["titan_cp_minus_ci"] = TH.community_cp_ci(meta, ab, n_boot=min(200, n_boot_t), seed=seed)
    ti["taxa"].to_csv(OUT / f"titan_taxa_{name}.csv", index=False)
    res["titan"] = {k: v for k, v in ti.items() if k not in ("taxa", "cut_sc", "sum_z_minus", "sum_z_plus")}
    res["titan_curves"] = {"cut_sc": ti["cut_sc"].tolist(), "sum_z_minus": ti["sum_z_minus"].tolist(),
                           "sum_z_plus": ti["sum_z_plus"].tolist()}
    res["richness_baseline_sc"] = TH.richness_baseline(meta, ab)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-boot-xc", type=int, default=1000)
    ap.add_argument("--n-boot-titan", type=int, default=500)
    ap.add_argument("--n-perm", type=int, default=250)
    ap.add_argument("--quick", action="store_true", help="tiny bootstrap counts; never registered")
    ap.add_argument("--register", action="store_true")
    a = ap.parse_args(argv)
    if a.quick:
        a.n_boot_xc, a.n_boot_titan, a.n_perm, a.register = 20, 5, 20, False
    paths.ensure(OUT)
    strata = _strata()
    cfg = {k: v for k, v in vars(a).items() if k != "register"} | {"min_occ_xc95": 25, "titan_min_occ": 5,
                                                                  "min_split": 5, "n_cuts": 100, "min_ph": 6.0}
    with TrainingRun("sim", "conductivity_thresholds", config=cfg, data=["nars_streams", "uk_ea_ecology",
                                                                         "taxonomy_harmonised"],
                     plan_ref="sim_protocol §8", iteration="quick" if a.quick else "v1", artifacts_dir=str(OUT)) as run:
        jobs = [(k, m, ab, a.n_boot_xc, a.n_boot_titan, a.n_perm, 0) for k, (m, ab) in strata.items()]
        jobs.sort(key=lambda j: -len(j[1]))
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            results = list(ex.map(_job, jobs))
        res = {r["stratum"]: r for r in results}
        app = res.get("appalachia_L3_69_70_genus", {})
        prim = app.get("xc95_hc05_reference_taxa") or app.get("xc95_hc05_all_taxa") or {}
        if not app.get("testable", False):
            pc_status = "not testable (underpowered)"
        elif prim and prim["ci_low"] <= EPA_HC05 <= prim["ci_high"]:
            pc_status = "replicated"
        else:
            pc_status = "not replicated"
        pc = pc_status == "replicated"
        summary = {"positive_control_status": pc_status, "positive_control_replicated": pc,
                   "appalachia_testability_reasons": app.get("testability_reasons"),
                   "appalachia_filters": app.get("filters"), "appalachia_primary": prim,
                   "table": [{"stratum": k, "n": r["n_samples"], "n_units": r["n_units"],
                              "testable": r.get("testable"),
                              "hc05": (r.get("xc95_hc05_reference_taxa") or r.get("xc95_hc05_all_taxa") or {}).get("hc05"),
                              "hc05_ci": [(r.get("xc95_hc05_reference_taxa") or r.get("xc95_hc05_all_taxa") or {}).get(x)
                                          for x in ("ci_low", "ci_high")],
                              "hc05_all_taxa": r.get("xc95_hc05_all_taxa", {}).get("hc05"),
                              "titan_cp_minus": r.get("titan", {}).get("community_cp_minus"),
                              "n_decreasers": r.get("titan", {}).get("n_decreasers"),
                              "richness_baseline": r.get("richness_baseline_sc"),
                              "background_p10": r.get("background_sc_p10_reference")} for k, r in res.items()]}
        (OUT / ("thresholds_quick.json" if a.quick else "thresholds_results.json")).write_text(
            json.dumps({"summary": summary, "strata": res}, indent=1, default=str))
        print(pd.DataFrame(summary["table"]).to_string())
        print("positive control:", pc_status)
        run.metric("positive_control_replicated", float(pc))
        run.note(f"positive control status: {pc_status}")
        if a.register:
            art = [str(OUT / "thresholds_results.json")]
            for k, r in res.items():
                split = "uk_ea_2000_2026" if k.startswith("uk") else "nrsa_2008_2024"
                for key, label in (("xc95_hc05_reference_taxa", "XC95-HC05 (EPA 2011 method, reference-site taxa)"),
                                   ("xc95_hc05_all_taxa", "XC95-HC05 (EPA 2011 method, all taxa)")):
                    if key in r and np.isfinite(r[key]["hc05"]):
                        log_result(workstream="sim", experiment="conductivity_thresholds", model=label,
                                   metric="hc05_us_cm", value=r[key]["hc05"], ci_low=r[key]["ci_low"],
                                   ci_high=r[key]["ci_high"], ci_method=f"site-cluster bootstrap B={a.n_boot_xc}",
                                   split=split, n=r["n_samples"], target=k, config=cfg,
                                   data_manifests=["nars_streams", "uk_ea_ecology", "taxonomy_harmonised"],
                                   artifacts=art, notes="replication of EPA 2011 300 uS/cm benchmark (low novelty)"
                                   + ("" if r.get("testable") else "; NOT TESTABLE (underpowered): "
                                      + "; ".join(r.get("testability_reasons", []))),
                                   extra={"n_taxa": r[key]["n_taxa"], "n_units": r[key]["n_units"],
                                          "testable": r.get("testable"), "bootstrap_unit": "site cluster",
                                          "positive_control_status": pc_status if k.startswith("appalachia") else None})
                t = r.get("titan", {})
                if t and np.isfinite(t.get("community_cp_minus", np.nan)):
                    log_result(workstream="sim", experiment="conductivity_thresholds", model="TITAN-style sum(z-)",
                               metric="community_cp_minus_us_cm", value=t["community_cp_minus"], split=split,
                               n=r["n_samples"], target=k, config=cfg, artifacts=art,
                               data_manifests=["nars_streams", "uk_ea_ecology", "taxonomy_harmonised"],
                               extra={"n_decreasers": t["n_decreasers"], "n_increasers": t["n_increasers"]})
                if np.isfinite(r.get("richness_baseline_sc", np.nan)):
                    log_result(workstream="sim", experiment="conductivity_thresholds", model="richness LOWESS 5% drop",
                               metric="sc_5pct_richness_drop_us_cm", value=r["richness_baseline_sc"], split=split,
                               n=r["n_samples"], target=k, is_baseline=True, config=cfg, artifacts=art,
                               data_manifests=["nars_streams", "uk_ea_ecology", "taxonomy_harmonised"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
