# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Insight study E3 (protocol §9): Hare-2021 groundwater classes vs DO and thermal resilience.

    python -B scripts/eval/sim_e3.py [--register]      # needs $SENTINEL_RUNS/sim/insights/e1_sites.csv
"""

from __future__ import annotations

import argparse
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "6")   # review W7: <= 6 threads

import numpy as np
import pandas as pd

from sentinel.analysis.insights import e3
from sentinel.core import paths
from sentinel.core.registry import log_result
from sentinel.core.runlog import TrainingRun

OUT = paths.RUNS_ROOT / "sim" / "insights"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    a = ap.parse_args(argv)
    from sentinel.models.sim import data as D
    from sentinel.models.sim import runner

    d = runner.load_data("validation")
    cfg = {"thresholds": {"dam_lag_d": 40, "shallow_lag_d": 10, "deep_ar": 0.65, "max_ar": 1.1}, "n_boot": 2000}
    with TrainingRun("sim", "insight_E3", config=cfg, data=["sonde_daily", "forcings_daily", "streamcat"],
                     plan_ref="sim_protocol §9 E3", iteration="v1", artifacts_dir=str(OUT)) as run:
        cl = e3.classify_all(d)
        e1t = pd.read_csv(OUT / "e1_sites.csv", dtype={"site": str})
        tab = cl.merge(e1t[["site", "beta_bio_min", "beta_min", "beta_mean"]], on="site", how="left")
        pos = {s: i for i, s in enumerate(d.sites)}
        # declared proxy (protocol §9 E3, §12 2026-11-08): StreamCat NID dam storage in the watershed, top decile
        dam = np.array([d.static_raw[pos[s], D.STATIC_NAMES.index("log1p_damnidstorws")] for s in tab["site"]])
        tab["dam_storage_top_decile"] = dam >= np.nanquantile(dam, 0.9)
        tab.to_csv(OUT / "e3_sites.csv", index=False)
        res = {"class_counts": tab["hare_class"].value_counts().to_dict()}
        keep = tab[~tab["hare_class"].isin(["removed", "negative_lag", "dam"])]
        res["V1_thermal_sensitivity_shallow_minus_atmospheric"] = e3.class_difference(keep, "thermal_sensitivity", "shallow_gw", "atmospheric")
        res["V1_thermal_sensitivity_deep_minus_atmospheric"] = e3.class_difference(keep, "thermal_sensitivity", "deep_gw", "atmospheric")
        v1 = res["V1_thermal_sensitivity_shallow_minus_atmospheric"]
        res["V1_pass"] = bool(v1["ci_high"] < 0)
        for col in ("beta_bio_min", "beta_min"):
            res[f"test_{col}_shallow_minus_atmospheric"] = e3.class_difference(keep, col, "shallow_gw", "atmospheric")
            res[f"test_{col}_deep_minus_atmospheric"] = e3.class_difference(keep, col, "deep_gw", "atmospheric")
        nd = keep[~keep["dam_storage_top_decile"]]
        res["V2_no_dam_top_decile_beta_bio_min_shallow_minus_atm"] = e3.class_difference(nd, "beta_bio_min", "shallow_gw", "atmospheric")
        res["V3_hare_published_classes"] = ("not done: no site-level Hare 2021 class table is published (Zenodo "
                                            "doi:10.5281/zenodo.4313244 code archive checked 2026-11-08: example data only)")
        t = res["test_beta_bio_min_shallow_minus_atmospheric"]
        res["E3_two_sided_result"] = ("difference" if (t["ci_low"] > 0 or t["ci_high"] < 0) else "no detectable difference")
        (OUT / "e3_summary.json").write_text(json.dumps(res, indent=1, default=str))
        print(json.dumps(res, indent=1, default=str))
        run.metric("V1_pass", float(res["V1_pass"]))
        if a.register:
            base = dict(workstream="sim", experiment="insight_E3", model="Hare-2021 class contrast (median difference)",
                        split="train_val_sites_wy2008_2023", config=cfg, artifacts=[str(OUT / "e3_summary.json")],
                        data_manifests=["sonde_daily", "forcings_daily", "streamcat"],
                        ci_method="basin-group cluster bootstrap B=2000")
            for k, v in res.items():
                if isinstance(v, dict) and "diff" in v and np.isfinite(v["diff"]):
                    log_result(**base, metric=k, value=v["diff"], ci_low=v["ci_low"], ci_high=v["ci_high"],
                               n=v["n_a"] + v["n_b"], extra={"n_a": v["n_a"], "n_b": v["n_b"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
