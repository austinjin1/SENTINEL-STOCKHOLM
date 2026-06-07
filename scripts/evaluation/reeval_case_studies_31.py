#!/usr/bin/env python
"""Run REAL AquaSSM inference on all 31 documented case-study events.

Replaces the hardcoded literature lead times in results/case_studies_v3 with
actual model outputs where a continuous USGS NWIS station exists. Events with no
qualifying station are honestly recorded as untestable. The old hardcoded values
are preserved in case_studies_v3_OLD_hardcoded_paper.json for comparison.

Reuses the exact inference machinery from exp1_case_studies_real.py.
"""
import json
import importlib.util
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# import the real-inference machinery from the existing script
_spec = importlib.util.spec_from_file_location(
    "csreal", PROJECT_ROOT / "scripts" / "experiments" / "exp1_case_studies_real.py")
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)

# event_id -> (usgs_site, advisory_date, pre_days, post_days)
# NEON events use NEON site codes (handled separately / flagged).
SITES = {
    # Category A
    "lake_erie_hab_2023":            ("04199500", "2023-07-15", 60, 14),
    "toledo_water_crisis_2014":      ("04193500", "2014-08-02", 30, 7),
    "gulf_dead_zone_2023":           ("07374000", "2023-07-01", 90, 30),
    "chesapeake_bay_blooms_2023":    ("01578310", "2023-07-15", 90, 30),
    # Category C (research-validated) — best-known USGS continuous stations
    "grand_lake_st_marys_hab_2009":  ("04191500", "2009-07-15", 60, 14),
    "lake_erie_hab_2015":            ("04193500", "2015-08-04", 75, 14),
    "lake_okeechobee_hab_2016":      ("02276877", "2016-07-01", 60, 30),
    "lake_okeechobee_hab_2018":      ("02276877", "2018-07-01", 60, 30),
    "sf_bay_heterosigma_2022":       ("11447650", "2022-08-01", 60, 30),
    "klamath_river_hab_2021":        ("11530500", "2021-08-01", 60, 14),
    "utah_lake_hab_2016":            ("10167000", "2016-07-15", 60, 30),
    "utah_lake_hab_2018":            ("10167000", "2018-07-15", 60, 30),
    "mississippi_salinity_intrusion_2023": ("07374000", "2023-10-01", 60, 30),
    "delaware_river_salinity_2022":  ("01463500", "2022-11-01", 60, 30),
    "animas_river_amd_2015":         ("09361500", "2015-08-05", 30, 21),
    "neuse_river_hypoxia_2020_2022": ("02089500", "2022-08-15", 60, 30),
    "jordan_lake_hab_nc":            ("02097517", "2022-07-15", 45, 14),
    "iowa_nitrate_crisis":           ("05482000", "2015-05-01", 60, 14),
    "chesapeake_bay_hypoxia_2018":   ("01578310", "2018-07-20", 90, 30),
    "green_bay_hypoxia":             ("040851385", "2021-08-01", 60, 30),
    "saginaw_bay_hab":               ("04157005", "2021-08-01", 60, 30),
    "hudson_river_hab_2025":         ("01359165", "2025-08-01", 60, 30),
    "clear_lake_hab_2024":           ("11450000", "2024-08-01", 60, 30),
    "tar_creek_amd_oklahoma":        ("07185000", "2015-06-01", 60, 30),
    "lake_winnebago_hab":            ("04073500", "2021-08-01", 60, 30),
}
NEON_EVENTS = {
    "neon_pose_do_depletion_2025": "POSE",
    "neon_blde_storm_conductance_2024": "BLDE",
    "neon_mart_turbidity_2025": "MART",
    "neon_barc_eutrophication_2025": "BARC",
    "neon_leco_acid_runoff_2024": "LECO",
    "neon_sugg_conductance_2024": "SUGG",
}
THRESH = 0.10


def run_one(model, head, ev_id, site, advisory, pre, post):
    from datetime import timedelta
    adv = datetime.strptime(advisory, "%Y-%m-%d")
    start = (adv - timedelta(days=pre)).strftime("%Y-%m-%d")
    end = (adv + timedelta(days=post)).strftime("%Y-%m-%d")
    df = cs.fetch_usgs_iv(site, start, end)
    if df is None or len(df) < 200:
        return {"event_id": ev_id, "usgs_site": site, "tested": False,
                "reason": "no/insufficient USGS IV data", "n_records": 0 if df is None else len(df)}
    windows = cs.preprocess_for_aquassm(df)
    if not windows:
        return {"event_id": ev_id, "usgs_site": site, "tested": False,
                "reason": "no valid windows after QC", "n_records": len(df)}
    scores = cs.run_inference(model, head, windows)
    res = cs.analyse_event({"advisory_date": advisory}, scores, THRESH)
    return {"event_id": ev_id, "usgs_site": site, "tested": True,
            "n_records": len(df), "n_windows": len(windows),
            "max_prob": res.get("max_prob"),
            "detected": res.get("detected"),
            "lead_time_days": (res.get("lead_time_hours") or 0) / 24.0
                              if res.get("lead_time_hours") else None}


def main():
    model, head = cs.load_models()
    out = []
    for ev_id, (site, adv, pre, post) in SITES.items():
        print(f"\n>>> {ev_id}  (USGS {site}, advisory {adv})", flush=True)
        try:
            out.append(run_one(model, head, ev_id, site, adv, pre, post))
        except Exception as e:
            out.append({"event_id": ev_id, "usgs_site": site, "tested": False,
                        "reason": f"error: {type(e).__name__}: {str(e)[:80]}"})
        print("   ->", out[-1], flush=True)
    for ev_id, neon in NEON_EVENTS.items():
        out.append({"event_id": ev_id, "neon_site": neon, "tested": "neon_separate",
                    "reason": "real AquaSSM run on NEON sonde data (neon_anomaly_scan), not USGS"})
    tested = [r for r in out if r.get("tested") is True]
    real_leads = [r["lead_time_days"] for r in tested if r.get("lead_time_days")]
    summary = {
        "generated": datetime.utcnow().isoformat() + "Z",
        "n_events_total": 31,
        "n_usgs_attempted": len(SITES),
        "n_usgs_tested_ok": len(tested),
        "n_detected": sum(1 for r in tested if r.get("detected")),
        "mean_real_lead_days": (sum(real_leads) / len(real_leads)) if real_leads else None,
        "threshold": THRESH,
        "events": out,
    }
    outpath = PROJECT_ROOT / "results" / "case_studies_v3" / "case_studies_v3_REAL_inference.json"
    json.dump(summary, open(outpath, "w"), indent=2)
    print(f"\nWROTE {outpath}")
    print(f"USGS tested OK: {len(tested)}/{len(SITES)} | detected: {summary['n_detected']} | "
          f"mean real lead: {summary['mean_real_lead_days']}")


if __name__ == "__main__":
    main()
