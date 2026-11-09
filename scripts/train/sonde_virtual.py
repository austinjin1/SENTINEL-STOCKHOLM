#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Task N-b virtual sensing at unseen stations: GBM baseline vs VirtualLSTM (sentinel/models/sonde/virtual.py).

Fit: train-role sites × TRAIN block; early stopping: train-role sites × VAL block; score: val-role
sites × VAL block (unseen stations; the withheld sensor group never enters). Per-site KGE (original
units, sites with >= 60 days), median across sites; per-site KGE gain over GBM with a BASIN-GROUP
bootstrap CI (site bootstrap as sensitivity). Val results are registered (split 'val/...') unless smoke.

    python -B scripts/train/sonde_virtual.py --config configs/sonde/virtual.yaml
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.eval.bootstrap import bootstrap_ci
from sentinel.models.sonde import runner


def main() -> int:
    ap = runner.base_parser(__doc__)
    ap.add_argument("--gpu-prefer", type=int, default=0)
    ap.add_argument("--register", action="store_true")
    a = ap.parse_args()
    cfg = runner.apply_overrides(runner.load_config(a.config, smoke=a.smoke), a.set)
    seed = int(a.seed if a.seed is not None else cfg.get("seed", 0))
    gpu = runner.setup_gpu("sonde_virtual" + ("_smoke" if a.smoke else ""), prefer=a.gpu_prefer)
    from sentinel.models.sonde.features import FeatureBuilder
    from sentinel.models.sonde.virtual import VirtualGBM, VirtualLSTM, VLSTMConfig, site_kge, virtual_rows

    run_dir = runner.run_directory(f"virtual_s{seed}", cfg, a.run_name)
    log = runner.make_logger(run_dir / "log.txt")
    log(f"[virtual] GPU {gpu}")
    runner.log_config_tried("virtual_nb", cfg, run_dir)
    if a.register:
        pass  # checked after data preparation (needs split info)
    with runner.training_run("virtual_nb", cfg, run_dir) as run:
        data = runner.prepare(a, cfg, run_dir, training=True, log=log)
        if a.register:
            runner.assert_registrable(cfg, data.split_info)
        fb = FeatureBuilder(data.cube, data.clim_mean, data.clim_std, data.static)
        basin = {str(k): str(v) for k, v in data.split.basin_group.items()}
        res, summary = [], []
        for v in runner.model_variables(cfg, data):
            tr = virtual_rows(data, v, "train", ("train",))
            es = virtual_rows(data, v, "val", ("train",))
            ev = virtual_rows(data, v, "val", ("val",))
            if len(ev.y) < 200 or len(tr.y) < 5000:
                log(f"[virtual] {v}: too few rows (train {len(tr.y)}, eval {len(ev.y)}); skipped")
                continue
            g = VirtualGBM(num_threads=int(cfg.get("num_threads", 16)), seed=seed,
                           max_rounds=int(cfg.get("gbm_max_rounds", 2000)))
            gi = g.fit(fb, data, v, tr, es)
            p_g = g.predict(fb, data, v, ev)
            nn = VirtualLSTM(VLSTMConfig(**cfg.get("vlstm", {}), seed=seed))
            p_n, ni = nn.fit_predict(data, v, tr, es, ev, device="cuda", log=log)
            kg = site_kge(data, v, ev, p_g).rename(columns={"kge": "kge_gbm", "nse": "nse_gbm"})
            kn = site_kge(data, v, ev, p_n).rename(columns={"kge": "kge_lstm", "nse": "nse_lstm"})
            k = kg.merge(kn, on=["site_no", "n"])
            k["var"], k["gain"] = v, k["kge_lstm"] - k["kge_gbm"]
            res.append(k)
            if len(k) < 3:
                continue
            grp = np.array([basin.get(s, s) for s in k["site_no"]])
            rec = {"var": v, "n_sites": len(k), "n_basin_groups": int(np.unique(grp).size),
                   "median_kge_gbm": float(k["kge_gbm"].median()), "median_kge_lstm": float(k["kge_lstm"].median()),
                   "gbm_info": gi, "lstm_info": ni}
            for col, tag in (("gain", "gain"), ("kge_gbm", "kge_gbm"), ("kge_lstm", "kge_lstm")):
                x = k[col].to_numpy(float)
                ci = bootstrap_ci(np.median, x, groups=grp, n_boot=2000, seed=seed)
                cs = bootstrap_ci(np.median, x, n_boot=2000, seed=seed)
                rec.update({f"median_{tag}": ci["point"], f"{tag}_ci_low": ci["ci_low"], f"{tag}_ci_high": ci["ci_high"],
                            f"{tag}_site_ci_low": cs["ci_low"], f"{tag}_site_ci_high": cs["ci_high"]})
            rec["bar_met"] = bool(rec["gain_ci_low"] > 0)
            summary.append(rec)
            log(f"[virtual] {v}: sites {rec['n_sites']} groups {rec['n_basin_groups']} median KGE gbm "
                f"{rec['median_kge_gbm']:.3f} lstm {rec['median_kge_lstm']:.3f} gain {rec['median_gain']:+.3f} "
                f"[{rec['gain_ci_low']:+.3f},{rec['gain_ci_high']:+.3f}]")
        pd.concat(res).to_csv(run_dir / "site_kge_val.csv", index=False) if res else None
        sm = pd.DataFrame(summary)
        sm.to_csv(run_dir / "nb_summary_val.csv", index=False)
        if a.register and len(sm):
            from sentinel.core.registry import log_result

            for r in sm.to_dict("records"):
                for metric, val, lo, hi, model in (
                        ("median_site_kge_nb", r["median_kge_gbm"], r["kge_gbm_ci_low"], r["kge_gbm_ci_high"], "virtual_gbm"),
                        ("median_site_kge_nb", r["median_kge_lstm"], r["kge_lstm_ci_low"], r["kge_lstm_ci_high"], "virtual_lstm"),
                        ("median_site_kge_gain_vs_gbm_nb", r["median_gain"], r["gain_ci_low"], r["gain_ci_high"], "virtual_lstm")):
                    log_result(workstream=runner.WORKSTREAM, experiment="sonde_virtual_nb_val", model=model,
                               is_baseline=model == "virtual_gbm", metric=metric, value=val,
                               split=f"val/sonde_split_{data.split_info['sha256'][:12]}", n=r["n_sites"], ci_low=lo,
                               ci_high=hi, ci_method="basin-group cluster percentile bootstrap", target=r["var"],
                               config=runner.public_config(cfg), data_manifests=["sonde_daily", "forcings_daily"],
                               artifacts=[str(run_dir)], extra={"n_basin_groups": r["n_basin_groups"], "seed": seed})
        run.note(f"N-b variables scored: {list(sm['var']) if len(sm) else []}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
