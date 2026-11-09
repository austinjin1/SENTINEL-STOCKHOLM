# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Lite baselines on the validation role (protocol §5): location kNN, location/season GBM,
RGBN band-ratio regressions, GBM on RGBN statistics (± context). Fits on train only; CPU (8 threads).

    python -B scripts/train/lite_baselines.py [--smoke]
"""

from __future__ import annotations

import argparse

from sentinel.models.lite import runner as R

R.cpu_threads(6)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.models.lite import baselines as B  # noqa: E402
from sentinel.models.lite import data as D  # noqa: E402
from sentinel.models.lite import evaluate as E  # noqa: E402
from sentinel.models.lite import features as F  # noqa: E402
from sentinel.models.lite import guard as G  # noqa: E402

EXPERIMENT = "lite_val"


def feature_frame(lab: pd.DataFrame, name: str, stats_path=None) -> pd.DataFrame:
    chips = np.asarray(D.cached_chips(lab, name))
    st = D.scene_stats(stats_path)
    p50 = D.scene_norm_params(lab, st)[:, :, 0]
    return F.chip_features(chips, lab["gsd_m"].to_numpy(), p50)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--track", default="original", choices=["original", "national"])
    a = ap.parse_args(argv)
    from sentinel.models.lite import national as N

    split, tr, va, stats_path, experiment = N.train_val_frames(a.track)
    if a.smoke:
        tr, va = tr.head(256), va.head(128)
    cfg = {"split_sha256": split["_sha256"], "gbm_params": B.GBM_PARAMS, "knn_k": 10, "smoke": a.smoke,
           "track": a.track}
    run_dir = R.run_directory("baselines" if a.track == "original" else "natbaselines", a.smoke)
    with R.training_run("baselines", cfg, run_dir, a.smoke) as run:
        Ftr, Fva = feature_frame(tr, "train", stats_path), feature_frame(va, "val", stats_path)
        Ftr.to_parquet(run_dir / "features_train.parquet")
        Fva.to_parquet(run_dir / "features_val.parquet")
        img_cols = list(Ftr.columns)
        Ctr = pd.concat([Ftr, tr[["lat", "lon", "doy"]]], axis=1)
        Cva = pd.concat([Fva, va[["lat", "lon", "doy"]]], axis=1)
        preds: dict[str, dict] = {}
        info: dict = {}
        for t in D.TARGETS:
            ytr = tr[f"y_{t}"].to_numpy(float)
            P = {}
            P["loc_knn"], info[f"loc_knn/{t}"] = B.location_knn(tr.lat, tr.lon, tr.site_id, ytr, va.lat, va.lon)
            P["loc_season_gbm"], info[f"loc_season_gbm/{t}"] = B.location_season_gbm(Ctr, ytr, Cva)
            # all pre-registered band-ratio forms; the best on val becomes 'band_ratio' (baseline-favouring, W3)
            yva_t = va[f"y_{t}"].to_numpy(float)
            fn_primary, _, hib = E.primary_metric(t)
            best = None
            for form in B.BAND_RATIO_FORMS[t]:
                pf, inf = B.band_ratio(Ftr, ytr, Fva, t, form)
                okv = np.isfinite(yva_t) & E.primary_mask(t, va["site_class"])
                v = fn_primary(yva_t[okv], pf[okv]) if okv.any() else np.nan
                inf["val_primary"] = v
                info[f"band_ratio_{form}/{t}"] = inf
                P[f"band_ratio_{form}"] = pf
                if best is None or (np.isfinite(v) and ((v > best[0]) if hib else (v < best[0]))):
                    best = (v, form)
            P["band_ratio"] = P[f"band_ratio_{best[1]}"]
            info[f"band_ratio/{t}"] = {"selected_form": best[1], "selected_on": "val primary metric"}
            P["gbm_rgbn"], info[f"gbm_rgbn/{t}"] = B.gbm(Ftr, ytr, Fva, cols=img_cols)
            P["gbm_rgbn_ctx"], info[f"gbm_rgbn_ctx/{t}"] = B.gbm(Ctr, ytr, Cva, cols=img_cols + ["lat", "lon", "doy"])
            P["train_median"] = np.full(len(va), np.nanmedian(ytr))
            preds[t] = P
        models = sorted({m for t in D.TARGETS for m in preds[t]})
        summary = {}
        for mname in models:
            scores = {t: E.score_primary(t, va[f"y_{t}"], preds[t][mname], va["water_body"], va["site_class"])
                      for t in D.TARGETS if mname in preds[t]}
            scores = {t: s for t, s in scores.items() if s["n"] > 0}
            summary[mname] = scores
            for t, s in scores.items():
                run.metric(f"{mname}/{t}/{s['primary']}", s["value"])
            R.save_predictions(run_dir / f"val_pred_{mname}.parquet", va,
                               {t: preds[t].get(mname, np.full(len(va), np.nan)) for t in D.TARGETS})
            if mname.startswith("band_ratio_"):
                continue    # individual forms are kept in summary.json; only the selected comparator is registered
            R.log_scores(scores, model=f"lite_{mname}", experiment=experiment, split="val", cfg=cfg,
                         artifacts=[run_dir / f"val_pred_{mname}.parquet"], smoke=a.smoke, is_baseline=True,
                         notes="M7 baseline, fitted on train role (NAIP <= 2021), scored on val water bodies")
        R.write_json(run_dir / "summary.json", {"summary": summary, "info": info, "split_sha256": split["_sha256"],
                                                  "smoke": a.smoke,
                                                  "n_train": len(tr), "n_val": len(va)})
        for mname, sc in summary.items():
            print(mname, {t: round(s["value"], 3) for t, s in sc.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
