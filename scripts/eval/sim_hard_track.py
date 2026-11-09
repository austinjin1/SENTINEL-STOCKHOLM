# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Sim hard track (protocol §4.6): a-priori difficulty strata + leave-HUC2-region-out hindcast.

    python -B scripts/eval/sim_hard_track.py --power                 # freeze definitions + power check (no model)
    python -B scripts/eval/sim_hard_track.py --runs base_s0 base_s1 base_s2 --tag base [--register]
    python -B scripts/eval/sim_hard_track.py --region-tag rh --region-seeds 0 1 2 [--register]

``--power`` writes docs/preregistration/sim_hard_track.json (definitions, region folds, per-cell power table,
sha256 of the per-site × day stratum flags of the evaluation cell). Scoring refuses to run if that file is
missing or its flag hash no longer matches. Validation cell only (val-role sites × WY2022–23).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "6")

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.core.registry import log_result
from sentinel.core.runlog import TrainingRun
from sentinel.data.sonde_splits import SondeSplit
from sentinel.models.sim import data as D
from sentinel.models.sim import guard, hindcast, runner
from sentinel.models.sim import strata as ST
from sentinel.models.sim.baselines import naive_baselines

FREEZE = paths.REPO_ROOT / "docs" / "preregistration" / "sim_hard_track.json"
PRIMARY = list(hindcast.PRIMARY_VARS)


def flag_hash(flags: dict, d, ev, block_days) -> str:
    h = hashlib.sha256()
    for s in ST.STRATA:
        h.update(s.encode())
        h.update(np.packbits(flags[s][ev][:, block_days]).tobytes())
    return h.hexdigest()


def power(d, split) -> dict:
    flags = ST.day_flags(d)
    ev = d.idx("val")
    vb = D.block_mask(d.dates, "val")
    nb = naive_baselines(d, hindcast.HORIZONS)
    recs = []
    for v in PRIMARY:
        for h in hindcast.PRIMARY_H:
            t_idx = np.flatnonzero(split.time_mask(d.dates, "val", horizon_days=h))
            ok = nb[v].scorable(t_idx, h)[ev]
            si, ti = np.nonzero(ok)
            rows = pd.DataFrame({"site": np.asarray(d.sites)[ev][si], "basin_group": d.basin_group[ev][si],
                                 "issue_date": d.dates[t_idx[ti]]})
            for s in ST.STRATA:
                m = ST.row_mask(rows, h, flags[s], d)
                rr = rows[m]
                n_s, n_g = int(rr["site"].nunique()), int(rr["basin_group"].nunique())
                recs.append({"stratum": s, "var": v, "h": h, "n_rows_upper_bound": int(m.sum()), "n_sites": n_s,
                             "n_groups": n_g, "powered": bool(n_s >= ST.MIN_SITES and n_g >= ST.MIN_GROUPS
                                                              and m.sum() >= ST.MIN_ROWS)})
    h2 = ST.site_huc2(d)
    folds = []
    for f, regs in ST.REGION_FOLDS.items():
        sel = np.isin(h2, regs)
        folds.append({"fold": f, "huc2": regs, "val_sites": int(((d.role == "val") & sel).sum()),
                      "val_groups": int(len(set(d.basin_group[(d.role == "val") & sel]))),
                      "train_cal_removed": int((np.isin(d.role, ["train", "cal"]) & sel).sum())})
    uncovered = sorted(set(h2[d.role == "val"]) - {r for v in ST.REGION_FOLDS.values() for r in v})
    doc = {"created": pd.Timestamp.now("UTC").isoformat(), "protocol": "sim_protocol.md §4.6",
           "split_sha256": guard.SPLIT_SHA256 if hasattr(guard, "SPLIT_SHA256") else None,
           "definitions": {"heatwave": f"target day in a run of >= {ST.HW_RUN} days with air-T anomaly >= site "
                                       f"train-block P{int(ST.HW_Q * 100)}",
                           "low_flow": f"observed Q(target) <= site train-block P{int(ST.LOW_Q * 100)}",
                           "high_flow": f"observed Q(target) >= site train-block P{int(ST.HIGH_Q * 100)}",
                           "power_rule": {"min_sites": ST.MIN_SITES, "min_groups": ST.MIN_GROUPS,
                                          "min_rows": ST.MIN_ROWS}},
           "region_folds": ST.REGION_FOLDS, "region_fold_table": folds, "val_huc2_not_in_any_fold": uncovered,
           "power_val_cell": recs, "n_val_sites": int(len(ev)),
           "stratum_flags_sha256_val_cell": flag_hash(flags, d, ev, vb)}
    doc["definition_sha256"] = ST.sha256_json({k: doc[k] for k in ("definitions", "region_folds",
                                                                   "stratum_flags_sha256_val_cell")})
    FREEZE.write_text(json.dumps(doc, indent=1))
    return doc


_UNDERPOWERED: set = set()


def _check(d) -> dict:
    if not FREEZE.exists():
        raise SystemExit("run --power first (freezes the hard-track definitions)")
    doc = json.loads(FREEZE.read_text())
    _UNDERPOWERED.update((r["stratum"], r["var"], int(r["h"])) for r in doc["power_val_cell"] if not r["powered"])
    flags = ST.day_flags(d)
    if flag_hash(flags, d, d.idx("val"), D.block_mask(d.dates, "val")) != doc["stratum_flags_sha256_val_cell"]:
        raise SystemExit("stratum flags changed since the freeze")
    return flags


def _verdict(sc: pd.DataFrame, stratum: str | None = None) -> dict:
    prim = sc[sc["model"].eq("sim_ec_site") & sc["var"].isin(PRIMARY) & sc["h"].isin(hindcast.PRIMARY_H)]
    out = {}
    for _, r in prim.iterrows():
        for ref in ("persistence", "climatology", "damped_anomaly", "physics_only"):
            out[f"{r['var']}_h{r['h']}_vs_{ref}"] = {"crpss": r[f"crpss_vs_{ref}"], "lo": r[f"crpss_vs_{ref}_lo"],
                                                    "hi": r[f"crpss_vs_{ref}_hi"], "pass": bool(r[f"crpss_vs_{ref}_lo"] > 0),
                                                    "n_rows": int(r["n_rows"]), "n_groups": int(r["n_groups"]),
                                                    "underpowered_by_frozen_rule": bool(stratum and (stratum, r["var"], int(r["h"]))
                                                                                         in _UNDERPOWERED)}
    out["track_pass_vs_persistence_and_climatology"] = bool(prim.shape[0] == 4 and all(
        v["pass"] for k, v in out.items() if isinstance(v, dict) and k.endswith(("persistence", "climatology"))))
    return out


def _register(sc, tag, track, cfg, art, n_boot):
    for _, r in sc.iterrows():
        if r["model"] not in hindcast.SIM_MODELS:
            continue
        for ref in hindcast.REFS:
            log_result(workstream="sim", experiment="sim_hindcast_hard_track",
                       model=f"SENTINEL-Sim v1 ({tag}):{r['model']}", metric=f"crpss_vs_{ref}_h{r['h']}",
                       value=r[f"crpss_vs_{ref}"], ci_low=r[f"crpss_vs_{ref}_lo"], ci_high=r[f"crpss_vs_{ref}_hi"],
                       ci_method=f"basin-group cluster bootstrap B={n_boot}, paired ratio", split=f"val/{track}",
                       n=int(r["n_rows"]), target=r["var"], config=cfg, data_manifests=D.DATA_MANIFESTS,
                       artifacts=art, notes=f"hard track ({track}); hindcast with observed forcings")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--region-tag", default=None, help="region runs are named <tag>_<fold>_s<seed>")
    ap.add_argument("--region-seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--register", action="store_true")
    a = ap.parse_args(argv)
    d = runner.load_data("validation")
    split = SondeSplit.load(guard.SPLIT_FILE)
    if a.power:
        doc = power(d, split)
        print(pd.DataFrame(doc["power_val_cell"]).to_string())
        print(json.dumps(doc["region_fold_table"], indent=1), "uncovered:", doc["val_huc2_not_in_any_fold"])
        print("definition_sha256", doc["definition_sha256"])
        return 0
    flags = _check(d)
    out = paths.ensure(paths.RUNS_ROOT / "sim" / "eval" / f"hard_{a.tag or a.region_tag}")
    cfg = {"runs": a.runs, "region_tag": a.region_tag, "region_seeds": a.region_seeds, "n_boot": a.n_boot,
           "definition_sha256": json.loads(FREEZE.read_text())["definition_sha256"]}
    res = {}
    with TrainingRun("sim", "sim_hindcast_hard_track", config=cfg, data=list(D.DATA_MANIFESTS),
                     plan_ref="sim_protocol §4.6", iteration=a.tag or a.region_tag, artifacts_dir=str(out)) as run:
        if a.runs:
            sim = runner.ensemble_sims([paths.RUNS_ROOT / "sim" / r / "sim_eval.npz" for r in a.runs], d)
            hc = hindcast.hindcast_rows(d, sim, d.idx("val"), "val", split, horizons=hindcast.PRIMARY_H,
                                        variables=PRIMARY)
            for s in ST.STRATA:
                sub = ST.subset(hc, {k: ST.row_mask(r["rows"], k[1], flags[s], d) for k, r in hc.items()})
                sc = hindcast.score(sub, n_boot=a.n_boot)
                sc.to_csv(out / f"scores_{s}.csv", index=False)
                res[s] = _verdict(sc, s)
                if a.register:
                    _register(sc, a.tag, s, cfg, [str(out / f"scores_{s}.csv")], a.n_boot)
        if a.region_tag:
            h2 = ST.site_huc2(d)
            parts = []
            for f, regs in ST.REGION_FOLDS.items():
                files = [paths.RUNS_ROOT / "sim" / f"{a.region_tag}_{f}_s{s}" / "sim_eval.npz" for s in a.region_seeds]
                missing = [str(x) for x in files if not x.exists()]
                if missing:
                    raise SystemExit(f"missing region-holdout runs: {missing}")
                for x in files:
                    rh = json.loads((x.parent / "region_holdout.json").read_text())
                    if sorted(rh["excluded_huc2"]) != sorted(regs):
                        raise SystemExit(f"{x.parent.name}: excluded {rh['excluded_huc2']} != fold {regs}")
                sim = runner.ensemble_sims(files, d)
                ev = np.flatnonzero((d.role == "val") & np.isin(h2, regs))
                part = hindcast.hindcast_rows(d, sim, ev, "val", split, horizons=hindcast.PRIMARY_H,
                                              variables=PRIMARY)
                parts.append(part)
                scf = hindcast.score(part, n_boot=a.n_boot)        # per fold (W-f: F1 is one basin group)
                scf.to_csv(out / f"scores_region_{f}.csv", index=False)
                res[f"region_fold_{f}"] = _verdict(scf)
            hc = ST.concat(parts)
            sc = hindcast.score(hc, n_boot=a.n_boot)
            sc.to_csv(out / "scores_region_holdout.csv", index=False)
            res["region_holdout"] = _verdict(sc)
            if a.register:
                _register(sc, f"{a.region_tag} {len(a.region_seeds)}-seed", "region_holdout", cfg,
                          [str(out / "scores_region_holdout.csv")], a.n_boot)
        (out / "hard_track_summary.json").write_text(json.dumps(res, indent=1, default=str))
        for k, v in res.items():
            run.metric(f"{k}_track_pass", float(v["track_pass_vs_persistence_and_climatology"]))
        print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
