#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Score and compare saved SENTINEL-Sonde forecasts on the COMMON rows of one block.

Each ``--forecast name=dir`` points at a run's forecasts/ directory (files ``<name>__<var>__<block>.npz``).
Forecasts are aligned onto the common rows (missing rows raise), scored with the naive baselines,
and compared with BASIN-GROUP cluster bootstraps (primary; site-level as sensitivity) of the median per-site
CRPS skill (primary metric) and the pooled CRPS skill 1 − ΣCRPS_model/ΣCRPS_ref. Protected blocks/sites need --allow-test AND
docs/preregistration/FREEZE_sonde.json; --register refuses smoke runs and non-final splits.

    python -B scripts/eval/sonde_score.py --config configs/sonde/common.yaml --forecast gbm=<run>/forecasts \
        --forecast lstm=<run>/forecasts [--eval-blocks val] [--register]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.eval.bootstrap import bootstrap_ci
from sentinel.models.sonde import common, runner
from sentinel.models.sonde.evaluate import score_forecast, site_metrics, skill_summary

REFS = ("damped_anomaly", "climatology", "persistence")


def pooled_skill_ci(samples: pd.DataFrame, model: str, ref: str, n_boot: int, seed: int, basin: dict) -> pd.DataFrame:
    """Pooled CRPS skill 1 − ΣCRPS_m/ΣCRPS_ref with a BASIN-GROUP cluster bootstrap CI (primary;
    unit = SondeSplit.basin_group) and a site-cluster CI (labelled sensitivity)."""
    rows = []
    for (v, h), g in samples[samples["model"].isin([model, ref])].groupby(["var", "h"]):
        a = g[g["model"] == model].set_index(["site_no", "issue_date"])["crps"]
        b = g[g["model"] == ref].set_index(["site_no", "issue_date"])["crps"]
        if a.empty:  # model does not forecast this variable at all (reported in the log)
            continue
        j = pd.concat([a.rename("m"), b.rename("r")], axis=1)
        if j.isna().any().any():
            raise ValueError(f"{model} vs {ref} {v}/h={h}: rows differ (common-row violation)")
        # per-site CRPS totals: resampling basin groups (or sites) of totals == resampling all their rows
        tot = j.groupby(level=0).sum()
        site = tot.index.to_numpy()
        grp = np.array([basin.get(x, x) for x in site])
        f = lambda m, r: 1 - m.sum() / r.sum()  # noqa: E731
        mv, rv = tot["m"].to_numpy(float), tot["r"].to_numpy(float)
        ci = bootstrap_ci(f, mv, rv, groups=grp, n_boot=n_boot, seed=seed)
        cs = bootstrap_ci(f, mv, rv, n_boot=n_boot, seed=seed)
        rows.append({"model": model, "ref": ref, "var": v, "h": int(h), "n_rows": len(j),
                     "n_sites": int(site.size), "n_basin_groups": int(np.unique(grp).size),
                     "pooled_crps_skill": ci["point"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                     "ci_unit": "basin_group", "site_ci_low": cs["ci_low"], "site_ci_high": cs["ci_high"]})
    return pd.DataFrame(rows)


def median_skill_ci(site: pd.DataFrame, model: str, ref: str, n_boot: int, seed: int, basin: dict) -> pd.DataFrame:
    """PRIMARY M1 metric (§15.2): median over sites of per-site CRPS skill 1 − CRPS_m/CRPS_ref.
    CI: BASIN-GROUP cluster bootstrap (resample whole basin groups of sites; primary) and a
    site bootstrap (labelled sensitivity). Val/test sites sit in few basin groups, so site-level
    CIs are too narrow."""
    from sentinel.models.sonde.evaluate import skill_vs

    rows = []
    sk = skill_vs(site, model, ref)
    for (v, h), g in sk.groupby(["var", "h"]):
        g = g[np.isfinite(g["skill"].to_numpy(float))]
        x = g["skill"].to_numpy(float)
        if x.size < 3:
            continue
        grp = np.array([basin.get(s_, s_) for s_ in g["site_no"]])
        ci = bootstrap_ci(np.median, x, groups=grp, n_boot=n_boot, seed=seed)
        cs = bootstrap_ci(np.median, x, n_boot=n_boot, seed=seed)
        rows.append({"model": model, "ref": ref, "var": v, "h": int(h), "n_sites": int(x.size),
                     "n_basin_groups": int(np.unique(grp).size),
                     "median_site_crps_skill": ci["point"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                     "ci_unit": "basin_group", "site_ci_low": cs["ci_low"], "site_ci_high": cs["ci_high"],
                     "frac_sites_positive": float((x > 0).mean())})
    return pd.DataFrame(rows)


def main() -> int:
    ap = runner.base_parser(__doc__)
    ap.add_argument("--forecast", action="append", required=True, help="name=forecast_dir (repeatable)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--register", action="store_true", help="log pooled skills + CIs to the results registry")
    ap.add_argument("--refs", default="", help="comma list of forecast labels that also serve as REFERENCES "
                    "(e.g. gbm,lstm_s0,chronos2_zs_uni): the M1 bar is skill vs the best of all references")
    ap.add_argument("--candidates", default="", help="comma list of candidate models (default: all forecasts)")
    ap.add_argument("--register-models", default="", help="comma list: only these models' rows are registered "
                    "(default: candidates) — avoids re-registering models already in the registry")
    a = ap.parse_args()
    cfg = runner.apply_overrides(runner.load_config(a.config, smoke=a.smoke), a.set)
    # --forecast label=dir (file prefix = label) or label=prefix@dir (e.g. lstm_s1=lstm@<run>/forecasts)
    fcs, prefix = {}, {}
    for x in a.forecast:
        lab, val = x.split("=", 1)
        pre, d = val.split("@", 1) if "@" in val else (lab, val)
        fcs[lab], prefix[lab] = d, pre
    run_dir = runner.run_directory("score", cfg, a.run_name)
    log = runner.make_logger(run_dir / "log.txt")
    data = runner.prepare(a, cfg, run_dir, need_blocks=("train",), log=log)
    if a.register:
        runner.assert_registrable(cfg, data.split_info)
    variables = runner.model_variables(cfg, data)
    for block in runner.eval_blocks(a):
        frames = []
        for v in variables:
            rows = data.rows(v, block)
            if not len(rows):
                continue
            for r in REFS:
                frames.append(score_forecast(v, common.baseline_forecast(data, v, rows, r), rows.score_mask, rows.obs,
                                             rows.site_no, rows.issue_date, rows.role, data.horizons, r))
            for name, d in fcs.items():
                f = Path(d) / f"{prefix[name]}__{v}__{block}.npz"
                if not f.is_file():
                    log(f"[score] {name}: no forecast for {v} ({f.name}); variable not scored for this model")
                    continue
                q = common.align_forecast(common.load_forecast(f), rows, data.horizons)
                frames.append(score_forecast(v, q, rows.score_mask, rows.obs, rows.site_no, rows.issue_date,
                                             rows.role, data.horizons, name))
        samples = pd.concat([f for f in frames if len(f)], ignore_index=True)
        samples.to_parquet(run_dir / f"samples_{block}.parquet", index=False, compression="zstd")
        site = site_metrics(samples)
        site["role"] = site["site_no"].map(dict(zip(data.sites, data.roles)))
        site.to_parquet(run_dir / f"site_metrics_{block}.parquet", index=False)
        extra_refs = [r for r in a.refs.split(",") if r]
        if set(extra_refs) - set(fcs):
            raise SystemExit(f"--refs must be forecast labels: {sorted(set(extra_refs) - set(fcs))}")
        cands = [c for c in a.candidates.split(",") if c] or list(fcs)
        refs_all = list(REFS) + extra_refs
        pairs = [(m, r) for m in fcs for r in (refs_all if m in cands else REFS) if r != m]
        summ = pd.concat([skill_summary(site, m, REFS) for m in fcs], ignore_index=True)
        summ.to_csv(run_dir / f"skill_summary_{block}.csv", index=False)
        basin = {str(k): str(v) for k, v in data.split.basin_group.items()}
        pooled = pd.concat([pooled_skill_ci(samples, m, r, a.n_boot, int(cfg.get("seed", 0)), basin)
                            for m, r in pairs if r in ("damped_anomaly",) + tuple(extra_refs)], ignore_index=True)
        pooled.to_csv(run_dir / f"pooled_skill_ci_{block}.csv", index=False)
        med = pd.concat([median_skill_ci(site[site["role"] == role] if role != "all" else site, m, r, a.n_boot,
                                         int(cfg.get("seed", 0)), basin).assign(site_role=role)
                         for role in ("all", "train", "val") for m, r in pairs], ignore_index=True)
        med.to_csv(run_dir / f"median_site_skill_ci_{block}.csv", index=False)
        # M1 bar (§15.2): for each candidate, the WORST case over all references (skill vs the best of them)
        bar = []
        for role in ("all", "val"):
            mm = med[(med["site_role"] == role) & med["model"].isin(cands) & med["ref"].isin(refs_all)]
            for (m, v, h), g in mm.groupby(["model", "var", "h"]):
                if set(g["ref"]) != set(refs_all) - {m}:
                    continue
                w = g.loc[g["ci_low"].idxmin()]
                bar.append({"site_role": role, "model": m, "var": v, "h": int(h), "binding_ref": w["ref"],
                            "skill_vs_binding": w["median_site_crps_skill"], "ci_low": w["ci_low"],
                            "ci_high": w["ci_high"], "min_skill_any_ref": float(g["median_site_crps_skill"].min()),
                            "n_basin_groups": int(g["n_basin_groups"].min()), "bar_met": bool(w["ci_low"] > 0)})
        bar = pd.DataFrame(bar)
        bar.to_csv(run_dir / f"m1_bar_{block}.csv", index=False)
        if len(bar):
            with pd.option_context("display.width", 220, "display.max_rows", 400, "display.float_format", "{:.3f}".format):
                log(f"[score] M1 bar on {block} (worst case over refs {refs_all}; basin-group CI):\n"
                    f"{bar[(bar['site_role'] == 'all') & (bar['h'] <= 7)].to_string(index=False)}")
        with pd.option_context("display.width", 200, "display.max_rows", 400, "display.float_format", "{:.3f}".format):
            log(f"[score] {'SMOKE ' if cfg['_smoke'] else ''}{block}: pooled CRPS skill vs damped anomaly "
                f"(site-bootstrap 95% CI; descriptive unless registered)\n"
                f"{pooled[pooled['ref'] == 'damped_anomaly'].to_string(index=False)}")
        if a.register:
            from sentinel.core.registry import log_result

            # registered: median per-site skill (PRIMARY) for all sites and for unseen (val-role) sites vs
            # every naive reference, and the pooled skill vs damped anomaly; everything else stays in CSVs
            reg_models = [c for c in a.register_models.split(",") if c] or cands
            for rec in med[med["site_role"].isin(["all", "val"]) & med["model"].isin(reg_models)].to_dict("records"):
                log_result(workstream=runner.WORKSTREAM, experiment=f"sonde_forecast_{block}", model=rec["model"],
                           metric=f"median_site_crps_skill_vs_{rec['ref']}_h{rec['h']}_{rec['site_role']}sites",
                           value=rec["median_site_crps_skill"], split=f"{block}/sonde_split_{data.split_info['sha256'][:12]}",
                           n=rec["n_sites"], ci_low=rec["ci_low"], ci_high=rec["ci_high"],
                           ci_method="basin-group cluster percentile bootstrap", target=rec["var"],
                           config=runner.public_config(cfg), data_manifests=["sonde_daily"], artifacts=[str(run_dir)],
                           notes=f"common rows; {fcs}",
                           extra={"n_basin_groups": rec["n_basin_groups"], "site_ci": [rec["site_ci_low"], rec["site_ci_high"]],
                                  "frac_sites_positive": rec["frac_sites_positive"], "site_role": rec["site_role"]})
            for rec in pooled[(pooled["ref"] == "damped_anomaly") & pooled["model"].isin(reg_models)].to_dict("records"):
                log_result(workstream=runner.WORKSTREAM, experiment=f"sonde_forecast_{block}", model=rec["model"],
                           metric=f"pooled_crps_skill_vs_{rec['ref']}_h{rec['h']}", value=rec["pooled_crps_skill"],
                           split=f"{block}/sonde_split_{data.split_info['sha256'][:12]}", n=rec["n_rows"], ci_low=rec["ci_low"],
                           ci_high=rec["ci_high"], ci_method="basin-group cluster percentile bootstrap", target=rec["var"],
                           extra={"n_basin_groups": rec["n_basin_groups"], "n_sites": rec["n_sites"],
                                  "site_ci": [rec["site_ci_low"], rec["site_ci_high"]]},
                           config=runner.public_config(cfg), data_manifests=["sonde_daily"],
                           artifacts=[str(run_dir)], notes=f"common rows; forecasts {fcs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
