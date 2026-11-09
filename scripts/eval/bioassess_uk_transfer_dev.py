#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""UK <-> US transfer DIAGNOSTIC on development data only (not the pre-registered transfer test).

* UK -> US: random forests trained on the UK EA pretraining pool (``role == pool_fit``; the reserved
  UK test blocks are never loaded) predict NRSA 2008-09 + 2013-14 development visits zero-shot.
* US -> UK: random forests trained on all NRSA development visits predict the UK pool *validation*
  blocks (``pool_val``), never the reserved UK test.

Representation and shared targets: ``sentinel.models.bioassess.transfer``. Metrics as in the CV
(UK -> US: pooled, anomaly and median per-ecoregion R², location-cluster CIs; US -> UK: pooled R²,
UK-site CIs). Logged to experiment 'bioassess_phase1_transfer_dev'; the pre-registered transfer
test (NRSA 2018-19, reserved UK blocks) is untouched.

    python -B scripts/eval/bioassess_uk_transfer_dev.py [--no-log] [--max-uk-rows 40000]
"""

from __future__ import annotations

import os

MAX_THREADS = 6   # CONTRIBUTING_V2 "Shared-host allocation" (2026-10-24 22:30)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.registry import log_result  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--max-uk-rows", type=int, default=40000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from sentinel.eval import bootstrap as EB
    from sentinel.eval import metrics as EM
    from sentinel.models.bioassess import evaluate as E
    from sentinel.models.bioassess import transfer as TR
    from sentinel.models.bioassess.data import assert_dev_only, load_nrsa
    from sentinel.models.bioassess.uk import load_uk

    us = load_nrsa()
    assert_dev_only(us.samples)
    uk = load_uk(include_reserved=False)
    if (uk.samples["role"] == "reserved_test").any():
        raise AssertionError("reserved UK test rows loaded")
    run_dir = paths.run_dir("bioassess", "transfer_dev")
    uk_fit = uk.samples.index[uk.samples["role"] == "pool_fit"]
    uk_val = uk.samples.index[uk.samples["role"] == "pool_val"]
    config = {"rf": TR.RF, "features": "harmonised family log10-abundance class", "min_family_n": 50,
              "max_uk_rows": a.max_uk_rows, "seed": a.seed}
    rows = []

    P = TR.uk_to_us(uk, us, uk_fit, us.samples.index, max_rows=a.max_uk_rows, seed=a.seed, n_jobs=MAX_THREADS)
    P.to_parquet(run_dir / "pred_uk_to_us_dev.parquet")
    region, cluster = us.samples["ag_eco9"].to_numpy(), us.samples["location_id"].to_numpy()
    for t in TR.SHARED:
        y, p = us.Y[t].to_numpy(float), P[t].to_numpy(float)
        s = E.summary(y, p, region)
        ci = E.cis(y, p, region, cluster, n_boot=a.n_boot)
        rows.append(("uk->us", "ukpool_fit->nrsa0814_dev", t, s, ci, "location cluster (NRSA)"))
        print(f"UK->US {t}: R2={s['r2']:.3f} [{ci['r2']['ci_low']:.3f},{ci['r2']['ci_high']:.3f}] "
              f"anom={s['r2_anomaly']:.3f} regmed={s['r2_region_median']:.3f} rho={s['spearman']:.3f}")

    Q = TR.us_to_uk(us, uk, us.samples.index, uk_val, seed=a.seed, n_jobs=MAX_THREADS)
    Q.to_parquet(run_dir / "pred_us_to_uk_poolval.parquet")
    for t_us, t_uk in TR.SHARED.items():
        yv = uk.Y.loc[uk_val, t_uk]
        ok = yv.notna().to_numpy()
        y, p = yv.to_numpy(float)[ok], Q[t_us].to_numpy(float)[ok]
        site = uk.samples.loc[uk_val[ok], "site_id"].to_numpy()
        b = EB.bootstrap_ci(EM.r2, y, p, groups=site, n_boot=a.n_boot)
        s = {"n": int(ok.sum()), "r2": EM.r2(y, p), "spearman": EM.spearman(y, p), "rmse": EM.rmse(y, p)}
        rows.append(("us->uk", "nrsa0814_dev->ukpool_val", t_us, s, {"r2": {"ci_low": b["ci_low"],
                                                                         "ci_high": b["ci_high"]}}, "UK site"))
        print(f"US->UK {t_us}: n={s['n']} R2={s['r2']:.3f} [{b['ci_low']:.3f},{b['ci_high']:.3f}] rho={s['spearman']:.3f}")

    pd.DataFrame([{"direction": d, "split": sp, "target": t, **{k: s.get(k) for k in ("n", "r2", "r2_anomaly",
                   "r2_region_median", "spearman", "rmse")}} for d, sp, t, s, _, _ in rows]).to_csv(
        run_dir / "transfer_dev_results.csv", index=False)
    if not a.no_log:
        for d, sp, t, s, ci, unit in rows:
            for m in [k for k in ("r2", "r2_anomaly", "r2_region_median") if k in ci]:
                log_result(workstream="bioassess", experiment="bioassess_phase1_transfer_dev",
                           model="RandomForest[family-log-class]", metric=m, value=float(s[m]), split=sp, n=s["n"],
                           ci_low=ci[m]["ci_low"], ci_high=ci[m]["ci_high"],
                           ci_method=f"percentile cluster bootstrap, unit = {unit}, B={a.n_boot}", target=t,
                           is_baseline=True, config=config,
                           data_manifests=["nars_streams", "uk_ea_ecology", "taxonomy_harmonised"],
                           artifacts=[str(run_dir / "transfer_dev_results.csv")],
                           notes="DEVELOPMENT transfer diagnostic only (NRSA 2008-14 / UK pool validation blocks); "
                                 "the pre-registered transfer test (NRSA 2018-19, reserved UK blocks) is untouched.",
                           extra={"direction": d, "spearman": s["spearman"], "rmse": s["rmse"]})
    print(f"artifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
