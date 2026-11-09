#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Orbit baseline ladder on the development cells (orbit_protocol.md §4.2). Fits on fit rows
(train role x train years minus inner water bodies), tunes on inner rows, predicts inner + val cells.
Never loads protected rows (guard). Writes $SENTINEL_RUNS/orbit/<run>/baseline_preds.parquet (+ MDN mixture
quantiles) and info.json; metrics are computed and registered by scripts/train/orbit_compare.py.

    python -B scripts/train/orbit_baselines.py --run-name baselines_dev_v1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.core.runlog import TrainingRun  # noqa: E402
from sentinel.models.orbit import baselines as B  # noqa: E402
from sentinel.models.orbit import data as D  # noqa: E402
from sentinel.models.orbit import features as F  # noqa: E402
from sentinel.models.orbit import pipeline as PL  # noqa: E402
from sentinel.models.orbit.evaluate import mdsa_log10  # noqa: E402

E_mdsa = mdsa_log10
from sentinel.models.orbit.mdn import MDNRegressor  # noqa: E402

MDN_B7 = ("B01", "B02", "B03", "B04", "B05", "B06", "B07")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="baselines_dev_v3")
    ap.add_argument("--version", default=D.CACHE_VERSION)
    ap.add_argument("--fit-scheme", default="base", choices=["base", "recent", "track_u"],
                    help="fit cell: base = train role x <=2021; recent = train role x <=2023 (cycle 2)")
    ap.add_argument("--mdn-iter", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-mdn", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="train the MDNs on one GPU via sentinel.core.gpu (prefer 1)")
    a = ap.parse_args()
    device = "cpu"
    if a.gpu:
        PL.setup_gpu(f"orbit_{a.run_name}")
        device = "cuda"
    torch.set_num_threads(2 if a.gpu else 6)  # shared-host rule (CONTRIBUTING_V2)
    cfg = {"fit_scheme": a.fit_scheme, "mdn_b_grid": "bands {7,12} x n_iter {5k,10k,20k} x x_log {F,T}", "device": "cuda" if a.gpu else "cpu", "version": a.version, "mdn_iter": a.mdn_iter, "seed": a.seed, "gbm_grid_size": len(B.GBM_GRID),
           "knn_k_grid": [4, 8, 16, 32], "index_defs": {t: list(v) for t, v in B.INDEX_DEFS.items()}}
    run_dir = paths.run_dir("orbit", a.run_name)
    with TrainingRun(PL.WORKSTREAM, f"{'SMOKE/' if a.run_name.startswith('smoke/') else ''}orbit_baselines/{a.run_name}", config=cfg, data=[f"orbit_{a.version}", "gloria"],
                     plan_ref=PL.PLAN_REF, artifacts_dir=str(run_dir)) as run:
        df, pix, info, masks = PL.load_dev(a.version, a.fit_scheme)
        fit, inner = masks["fit"], masks["inner"]
        out_mask = ~fit
        print(f"fit {fit.sum()} inner {inner.sum()} predicted rows {out_mask.sum()}")
        y = F.log_targets(df)
        yf = {t: v[fit] for t, v in y.items()}
        yi = {t: v[inner] for t, v in y.items()}
        Ximg = F.image_features(df, pix)
        Xctx = B.location_season_matrix(df)
        Xboth = pd.concat([Ximg, Xctx], axis=1)
        preds, infos = {}, {}
        fitted = {}            # name -> predictor (persisted for the frozen test path)

        def add(name, fn, X, variants=False):
            fitted[name] = fn
            p = fn(df[out_mask], None if X is None else X[out_mask])
            for t in D.TARGETS:
                preds[(name, t)] = p[t]
            if variants:          # every grid member (val-level comparator selection, protocol §6; review W2)
                tag = "k" if name == "location_knn" else "v"
                for v in fn.variants:
                    kw = {"k": v} if name == "location_knn" else {"variant": v}
                    pv = fn(df[out_mask], None if X is None else X[out_mask], **kw)
                    for t in D.TARGETS:
                        preds[(f"{name}#{tag}{v}", t)] = pv[t]

        t0 = time.time()
        fn, infos["global_median"] = B.fit_global_median(df[fit], yf)
        add("global_median", fn, None)
        fn, infos["location_knn"] = B.fit_location_knn(df[fit], yf, df[inner], yi)
        add("location_knn", fn, None, variants=True)
        fn, infos["gbm_season_location"] = B.fit_gbm(Xctx[fit], yf, Xctx[inner], yi, seed=a.seed)
        add("gbm_season_location", fn, Xctx)
        fn, infos["index_opt"] = B.fit_index(Ximg[fit], yf, Ximg[inner], yi)
        add("index_opt", fn, Ximg)
        infos["index_opt"]["fallback_rate_predicted_rows"] = dict(fn.fallback_rate)
        fn, infos["gbm_imagery"] = B.fit_gbm(Ximg[fit], yf, Ximg[inner], yi, seed=a.seed)
        add("gbm_imagery", fn, Ximg, variants=True)
        fn, infos["gbm_imagery_context"] = B.fit_gbm(Xboth[fit], yf, Xboth[inner], yi, seed=a.seed)
        add("gbm_imagery_context", fn, Xboth, variants=True)
        print(f"tabular baselines done in {time.time() - t0:.0f}s")
        quant = {}
        if not a.skip_mdn:
            g = D.gloria_rows()
            # (name -> X_gloria or None, train mask, bands, divisor, n_iter, x_log). MDN_B grid selected on inner
            # rows (tuning parity with SatWQ-Net, protocol §4.2): bands {B01-B07, all 12} x iterations
            # {5k, 10k, 20k} x input {linear (official), log10}. mdn_b7 / mdn_b12 = published defaults.
            variants = {"mdn_gloria": (np.stack([g[b].to_numpy(float) / np.pi for b in MDN_B7], 1), None, MDN_B7, np.pi,
                                       a.mdn_iter, False)}
            for bn, bands in (("7", MDN_B7), ("12", D.BANDS)):
                for it in (5000, 10000, 20000):
                    for xl in (False, True):
                        nm = f"mdn_b{bn}" if (it == 10000 and not xl) else f"mdn_b{bn}_{it // 1000}k_{'log' if xl else 'lin'}"
                        variants[nm] = (None, fit, bands, 1.0, it if a.mdn_iter == 10000 else a.mdn_iter, xl)
            for name, (Xg, m, bands, div, n_iter, x_log) in variants.items():
                Xl = np.stack([df[f"rw3_{b}"].to_numpy(float) / div for b in bands], 1)
                infos[name] = {}
                for t in D.TARGETS:
                    if Xg is not None:
                        Xtr, ytr = Xg, np.log10(g[t].to_numpy(float))
                    else:
                        Xtr, ytr = Xl[m], y[t][m]
                    if np.isfinite(ytr).sum() < 50:
                        infos[name][t] = {"skipped": int(np.isfinite(ytr).sum())}
                        preds[(name, t)] = np.full(out_mask.sum(), np.nan)
                        continue
                    t1 = time.time()
                    mdn = MDNRegressor(n_iter=n_iter, seed=a.seed, device=device, x_log=x_log).fit(Xtr, ytr)
                    fitted[(name, t)] = mdn
                    preds[(name, t)] = mdn.predict(Xl[out_mask])
                    q = mdn.quantiles(Xl[out_mask], qs=(0.025, 0.16, 0.5, 0.84, 0.975))
                    for j, qq in enumerate(("q025", "q16", "q50", "q84", "q975")):
                        quant[(name, t, qq)] = q[:, j]
                    mi = inner & np.isfinite(y[t])
                    pin = mdn.predict(Xl[mi])
                    fin = np.isfinite(pin)          # rows with a missing band cannot be predicted (counted)
                    s_in = mdsa_log10(pin[fin] - y[t][mi][fin]) if fin.sum() else np.nan
                    infos[name][t] = {"n_train": mdn.n_fit, "inner_mdsa": float(s_in) if np.isfinite(s_in) else None,
                                      "inner_rows_unpredictable": int((~fin).sum()),
                                      "bands": list(bands), "n_iter": n_iter, "x_log": x_log,
                                      "seconds": round(time.time() - t1, 1)}
                    print(f"{name} {t}: n_train {mdn.n_fit} inner MdSA {s_in:.1f} ({time.time() - t1:.0f}s)")
            # MDN_B representative = best grid variant on inner, per target (selection on inner only)
            infos["mdn_b"] = {}
            grid_names = [v for v in variants if v.startswith("mdn_b")]
            ii = inner[out_mask]
            for t in D.TARGETS:
                # review: score every grid member on the inner rows ALL members can predict (identical rows)
                yy = y[t][out_mask][ii]
                common = np.isfinite(yy)
                for v in grid_names:
                    common &= np.isfinite(preds[(v, t)][ii])
                c = {v: (E_mdsa(preds[(v, t)][ii][common] - yy[common]) if common.sum() >= 10 else np.inf)
                     for v in grid_names}
                infos.setdefault("mdn_b_inner_common_rows", {})[t] = int(common.sum())
                best = min(c, key=c.get)
                preds[("mdn_b", t)] = preds[(best, t)]
                for qq in ("q025", "q16", "q50", "q84", "q975"):
                    if (best, t, qq) in quant:
                        quant[("mdn_b", t, qq)] = quant[(best, t, qq)]
                infos["mdn_b"][t] = {"variant": best, "inner": c}
        # official MDN-STREAM MSI weights (applied to L2A water-median rho/pi of B01-B07; chl-a and TSS only)
        from sentinel.models.orbit.mdn import OfficialMDN

        off = OfficialMDN(paths.RAW / "mdn_stream_weights" / "MSI" / "x2025")
        Xo = np.stack([df[f"rw3_{b}"].to_numpy(float) / np.pi for b in OfficialMDN.MSI_BANDS], 1)[out_mask]
        po = off.predict_log10(Xo)
        for t, pr_ in (("chla", "chl"), ("tss", "tss")):
            preds[("mdn_official", t)] = po[pr_]
        for t in ("turbidity", "secchi"):
            preds[("mdn_official", t)] = np.full(out_mask.sum(), np.nan)
        infos["mdn_official"] = {"weights": "MDN-STREAM Weights/MSI/1b79098d...zip (10 rounds)", "input": "rw3 B01-B07 / pi"}
        out = pd.DataFrame({f"{m}|{t}": v for (m, t), v in preds.items()})
        out.insert(0, "row_id", df.row_id.to_numpy()[out_mask])
        out.to_parquet(run_dir / "baseline_preds.parquet", index=False)
        if quant:
            qd = pd.DataFrame({f"{m}|{t}|{q}": v for (m, t, q), v in quant.items()})
            qd.insert(0, "row_id", df.row_id.to_numpy()[out_mask])
            qd.to_parquet(run_dir / "baseline_quantiles.parquet", index=False)
        try:      # persisted fitted baselines (cloudpickle: closures + MDN nets moved to CPU) for test scoring
            import cloudpickle

            for k, v in fitted.items():
                if isinstance(v, MDNRegressor):
                    v.net = v.net.cpu()
                    v.device = "cpu"
            with open(run_dir / "fitted_baselines.pkl", "wb") as fh:
                cloudpickle.dump({"fitted": fitted, "feature_columns": {"img": list(Ximg.columns),
                                                                        "ctx": list(Xctx.columns)}}, fh)
        except Exception as e:  # noqa: BLE001 — predictions are already written; record the failure loudly
            run.note(f"FAILED to persist fitted baselines: {type(e).__name__}: {e}")
            print(f"WARNING: could not persist fitted baselines: {e}")
        PL.write_json(run_dir / "info.json", {"config": cfg, "baselines": infos, "n_fit": int(fit.sum()),
                                              "n_inner": int(inner.sum())})
        for name in {m for m, _ in preds}:
            PL.log_config_tried(name, cfg, run_dir, "completed", {"kind": "baseline"})
        # quick inner diagnostics (not results)
        for (m, t), v in preds.items():
            if t == "chla":
                ii = inner[out_mask]
                yy = y[t][out_mask][ii]
                ok = np.isfinite(yy) & np.isfinite(v[ii])
                if "#" in m or not ok.any():
                    continue
                run.metric(f"inner_mdsa_chla_{m}", mdsa_log10(v[ii][ok] - yy[ok]))
        run.note("baseline predictions written; scoring/registry in orbit_compare.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
