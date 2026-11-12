#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Score the M6 SOTA levers (``docs/preregistration/hydromet_sota_protocol.md`` §6–§7).

    python -B scripts/eval/sota_hydromet_score.py inner [--register]          # decision table (inner groups only)
    python -B scripts/eval/sota_hydromet_score.py val --configs C2a,C1b [--register]   # descriptive val parity

* ``inner``: best-epoch inner MSE (7 M6 targets, the early-stopping criterion) and primary inner MSE (4 endpoints) of
  every lever run and of the reference ``ealstm_ws_v1_s{0,1,2}``; screening winner per lever; the "moves" rule
  (3-seed mean primary inner MSE ≤ reference − 0.006). No val number enters this table.
* ``val``: FINAL val-role sites × val block, primary endpoints. Lever EA-LSTM seed ensemble vs (a) the lever's own
  GBM bag (same data), (b) its physics members, (c) the reference EA-LSTM ``ealstm_ws`` and the frozen GBM family;
  paired basin-group bootstrap (B = 2,000), every-seed signs. Descriptive only (protocol §7).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY = ("temp_c_mean", "temp_c_7dadm", "do_mgl_mean", "do_mgl_min")
MOVE_THRESHOLD = 0.006
REF_RUNS = ("ealstm_ws_v1_s0/g0_s0", "ealstm_ws_v1_s1/g0_s1", "ealstm_ws_v1_s2/g0_s2")
LEVER_OF = {"C1a": 1, "C1b": 1, "C2a": 2, "C2b": 2, "C12": 12}
LEVER_CONFIGS = {1: ("C1a", "C1b"), 2: ("C2a", "C2b"), 12: ("C12",)}
SNAPSHOT_EXPERIMENT = "hydromet_sota/prereg_snapshot"
SNAPSHOT_DOCS = ("docs/preregistration/hydromet_sota_protocol.md", "docs/preregistration/hydromet_sota_lever_pins.json",
                 "docs/preregistration/hydromet_ea_protocol.md")


def check_snapshot() -> str:
    """Refuse to score unless the latest ``hydromet_sota/prereg_snapshot`` registry row equals the protocol, the pins and
    every lever configuration on disk (lead rule 2026-11-10: a new snapshot row follows any append, before the next
    score). Returns the id of that row. A guard only: it changes no number."""
    from sentinel.core import paths
    from sentinel.core.registry import file_sha256, read_registry

    snaps = [r for r in read_registry() if r.get("experiment") == SNAPSHOT_EXPERIMENT]
    if not snaps:
        raise SystemExit(f"no {SNAPSHOT_EXPERIMENT} row in the registry; refusing to score")
    last = sorted(snaps, key=lambda r: r["timestamp"])[-1]
    want = dict(last.get("extra", {}).get("sha256", {}))
    cfgs = sorted((paths.REPO_ROOT / "configs" / "hydromet_sota").glob("C*.yaml"))
    files = list(SNAPSHOT_DOCS) + [str(p.relative_to(paths.REPO_ROOT)) for p in cfgs]
    have = {x: file_sha256(paths.REPO_ROOT / x) for x in files}
    if have != want:
        diff = sorted(x for x in set(have) | set(want) if have.get(x) != want.get(x))
        raise SystemExit(f"snapshot row {last['id']} does not match the files on disk: {diff}; "
                         "log a new snapshot row before scoring")
    return str(last["id"])


def refuse_if_val_hold(what: str) -> None:
    """Val-cell scoring (§7, descriptive) waits while ``<runs>/hydromet/sota/HOLD_VAL_SCORING`` exists."""
    from sentinel.hydromet_sota import runner as SR

    f = SR.ROOT / "HOLD_VAL_SCORING"
    if f.exists():
        raise SystemExit(f"{what}: on hold ({f}): {f.read_text().strip()}")


def _best(hist):
    k = int(np.nanargmin([h["inner_val_loss"] for h in hist]))
    bt = hist[k]["inner_val_mse_by_target"]
    prim = float(np.mean([bt[v] for v in PRIMARY]))
    return k, float(hist[k]["inner_val_mse"]), prim, bt


def inner_table() -> pd.DataFrame:
    from sentinel.core import paths
    from sentinel.hydromet_sota import runner as SR

    rows = []
    H = paths.RUNS_ROOT / "hydromet"
    for r in REF_RUNS:
        t = json.loads((H / r / "train_info.json").read_text())
        k, allm, prim, bt = _best(t["history"])
        rows.append({"config": "reference_ealstm_ws", "lever": 0, "seed": int(r[-1]), "best_epoch": k,
                     "epochs": len(t["history"]), "inner_mse_7": allm, "inner_primary_mse": prim,
                     **{f"mse_{v}": bt.get(v) for v in bt}, "run": str(H / r)})
    for d in sorted(SR.ROOT.glob("sota_C*_s*")):
        m = re.fullmatch(r"sota_(C\w+?)_s(\d+)", d.name)
        f = d / "train_info.json"
        if not m or not f.exists():
            continue
        t = json.loads(f.read_text())
        k, allm, prim, bt = _best(t["history"])
        rows.append({"config": m.group(1), "lever": LEVER_OF.get(m.group(1), -1), "seed": int(m.group(2)),
                     "best_epoch": k, "epochs": len(t["history"]), "inner_mse_7": allm, "inner_primary_mse": prim,
                     **{f"mse_{v}": bt.get(v) for v in bt}, "run": str(d)})
    return pd.DataFrame(rows)


def decisions(tab: pd.DataFrame) -> dict:
    ref = tab[tab["config"] == "reference_ealstm_ws"]["inner_primary_mse"]
    out = {"reference_mean": float(ref.mean()), "reference_sd": float(ref.std(ddof=1)), "threshold": MOVE_THRESHOLD,
           "levers": {}}
    for lev, g in tab[tab["lever"] > 0].groupby("lever"):
        s0 = g[g["seed"] == 0].set_index("config")["inner_primary_mse"]
        complete = set(LEVER_CONFIGS.get(int(lev), ())) <= set(s0.index)
        win = s0.idxmin() if (len(s0) and complete) else None   # no winner before every screening run exists
        rec = {"screening_seed0": s0.round(5).to_dict(), "screening_complete": bool(complete), "screening_winner": win}
        if win is not None:
            w = g[g["config"] == win]
            rec["winner_seeds"] = sorted(w["seed"].tolist())
            if len(w) >= 3:
                mean3 = float(w["inner_primary_mse"].mean())
                rec.update({"winner_3seed_mean": mean3, "delta_vs_reference": mean3 - out["reference_mean"],
                            "moves": bool(mean3 - out["reference_mean"] <= -MOVE_THRESHOLD)})
        out["levers"][int(lev)] = rec
    return out


def cmd_inner(a) -> int:
    from sentinel.core.registry import log_result
    from sentinel.hydromet_sota import runner as SR

    snap = check_snapshot()
    print(f"prereg snapshot row {snap} matches the protocol, pins and lever configs on disk")
    tab = inner_table()
    dec = decisions(tab)
    dec["prereg_snapshot_row"] = snap
    od = SR.run_dir("score_inner_v1")
    tab.to_csv(od / "inner_table.csv", index=False)
    SR.write_json(od / "decisions.json", dec)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(tab[["config", "seed", "best_epoch", "epochs", "inner_mse_7", "inner_primary_mse"]].to_string(index=False))
    print(json.dumps(dec, indent=1))
    if a.register:
        from sentinel.core.registry import read_registry

        done = {(x["model"], x["metric"]) for x in read_registry() if x.get("experiment") == "M6SOTA/inner_v1"}
        for r in tab.itertuples(index=False):
            if r.config == "reference_ealstm_ws" or (f"hydromet_sota_{r.config}_s{r.seed}", "inner_primary_mse") in done:
                continue   # each run is registered once
            for met, val in (("inner_primary_mse", r.inner_primary_mse), ("inner_mse_7targets", r.inner_mse_7)):
                log_result(workstream=SR.WORKSTREAM, experiment="M6SOTA/inner_v1", model=f"hydromet_sota_{r.config}_s{r.seed}",
                           metric=met, value=float(val), split="train-role inner basin groups (256 sites, 20 groups) x train block",
                           n=256, target="primary4" if met == "inner_primary_mse" else "all7", is_baseline=False,
                           extra={"best_epoch": int(r.best_epoch), "epochs": int(r.epochs), "run": r.run,
                                  "reference_mean": dec["reference_mean"], "decision": dec["levers"].get(int(r.lever)),
                                  "prereg_snapshot_row": snap},
                           notes="SOTA lever screening (protocol hydromet_sota_protocol.md §6); inner groups only")
    return 0


def _merge_target_files(pred_dir: Path) -> list[Path]:
    """GBM parity writes one file per target; merge them (in memory) into one prediction dict per model."""
    return sorted(pred_dir.glob("*__val__*.npz"))


def load_preds(data, si, ti, dirs) -> dict:
    from sentinel.models.hydromet import evaluate as E

    preds = {}
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*__val.npz")) + _merge_target_files(d):
            pf = E.load_predictions(f)
            if pf["header"].get("split_sha256") != data.split.sha256() or pf["header"].get("smoke"):
                raise ValueError(f"{f}: wrong split or smoke")
            name = pf["header"]["model"]
            al = E.align_predictions(pf, data, si, ti)
            preds.setdefault(name, {}).update({v: x for v, x in al.items() if v in PRIMARY})
    return preds


def cmd_val(a) -> int:
    from sentinel.core import paths
    from sentinel.core.registry import log_result
    from sentinel.hydromet_sota import runner as SR
    from sentinel.hydromet_sota.data import load_val_cell
    from sentinel.models.hydromet import evaluate as E
    from sentinel.models.hydromet.pipeline import seed_ensembles

    refuse_if_val_hold("val scoring")
    snap = check_snapshot()
    print(f"prereg snapshot row {snap} matches the protocol, pins and lever configs on disk")
    H = paths.RUNS_ROOT / "hydromet"
    od = SR.run_dir(a.out)
    log = print
    data = load_val_cell(log=log)
    si = data.site_idx(["val"])
    ti = np.flatnonzero(data.block_mask("val"))
    cfgs = [c for c in a.configs.split(",") if c]
    dirs = [H / f"ealstm_ws_v1_s{k}" / "predictions" for k in range(3)] + \
        [H / f"gbmwsbag_v1_s{k}" / "predictions" for k in range(3)] + [H / f"gbmbag_v1_s{k}" / "predictions" for k in range(3)] + \
        [H / "basews_v1" / "predictions", H / "base_v2" / "predictions"]
    for c in cfgs:
        dirs += [SR.ROOT / f"sota_{c}_s{k}" / "predictions" for k in range(3)]
        dirs += [SR.ROOT / f"sota_gbm_{c}_s{k}" / "predictions" for k in range(3)]
        dirs += [SR.ROOT / f"sota_phys_{c}" / "predictions"]
    preds = load_preds(data, si, ti, dirs)
    ens, members = seed_ensembles(preds)
    preds.update(ens)
    keep = {n for n in preds if not re.fullmatch(r"(sota_gbm_.*|forcing_gbm_ws_bag|forcing_gbm_bag)_s\d+", n)}
    keep |= {n for n in preds if re.fullmatch(r"(sota_C\w+|ealstm_ws)_s\d+", n)}
    preds = {n: p for n, p in preds.items() if n in keep}
    site_df, rows = E.site_metrics(data, preds, np.arange(len(si)) if False else si, ti, "val", min_days=60,
                                   targets=PRIMARY)
    site_df.to_parquet(od / "site_metrics_val.parquet", index=False)
    SR.write_json(od / "rows_val.json", rows)
    summ = E.summarize(site_df, n_boot=a.n_boot, metrics=("kge", "rmse"))
    summ.to_csv(od / "summary_val.csv", index=False)
    phys_ref = {"temp_c_mean": "air2stream_regional_ws", "temp_c_7dadm": "air2stream_regional_ws",
                "do_mgl_mean": "dosat_ratio[air2stream_regional]_ws", "do_mgl_min": "dosat_ratio[air2stream_regional]_ws"}
    out = []
    for c in cfgs:
        lev = f"sota_{c}"
        if lev not in preds:
            log(f"{c}: no seed ensemble yet (members {members.get(lev)})")
            continue
        seeds = {lev: members.get(lev, [])}
        for v in PRIMARY:
            phys_members = [n for n in preds if n.startswith(f"sota_phys_{c}_") and v in preds[n]]
            comps = {
                "own_gbm_bag": f"sota_gbm_{c}_bag",
                "own_physics": (f"sota_phys_{c}_air2stream_regional_ws" if v.startswith("temp")
                                else f"sota_phys_{c}_dosat_ratio[air2stream_regional]_ws"),
                "reference_ealstm_ws": "ealstm_ws",
                "frozen_gbm_ws_bag": "forcing_gbm_ws_bag",
                "frozen_gbm_bag": "forcing_gbm_bag",
                "frozen_physics_ws": phys_ref[v],
            }
            if any("air2stream_q" in n for n in phys_members):
                comps["own_physics_q"] = (f"sota_phys_{c}_air2stream_q_regional_ws" if v.startswith("temp")
                                          else f"sota_phys_{c}_dosat_ratio[air2stream_q_regional]_ws")
            for label, ref in comps.items():
                if ref not in preds or v not in preds[ref]:
                    continue
                res, j = E._paired(site_df, lev, ref, v, "kge", a.n_boot, 0, groups=True)
                if res is None:
                    continue
                sd = []
                for sm in seeds[lev]:
                    r2, j2 = E._paired(site_df, sm, ref, v, "kge", 10, 0, groups=False)
                    sd.append(float(j2["a"].median() - j2["b"].median()) if len(j2) else np.nan)
                out.append({"config": c, "var": v, "vs": label, "comparator": ref, "n_sites": len(j),
                            "n_groups": int(j["basin_group"].nunique()), "kge_model": float(j["a"].median()),
                            "kge_comparator": float(j["b"].median()), "diff": res["diff"], "ci_low": res["ci_low"],
                            "ci_high": res["ci_high"], "p_value": res["p_value"], "seed_diffs": sd,
                            "every_seed_positive": bool(sd) and all(np.isfinite(sd)) and min(sd) > 0})
            # the GBM's own gain from the same lever data
            for lab, a_, b_ in (("gbm_gain_from_lever_data", f"sota_gbm_{c}_bag", "forcing_gbm_ws_bag"),):
                if a_ in preds and b_ in preds:
                    res, j = E._paired(site_df, a_, b_, v, "kge", a.n_boot, 0, groups=True)
                    if res is not None:
                        out.append({"config": c, "var": v, "vs": lab, "comparator": b_, "n_sites": len(j),
                                    "n_groups": int(j["basin_group"].nunique()), "kge_model": float(j["a"].median()),
                                    "kge_comparator": float(j["b"].median()), "diff": res["diff"],
                                    "ci_low": res["ci_low"], "ci_high": res["ci_high"], "p_value": res["p_value"],
                                    "seed_diffs": [], "every_seed_positive": None, "model": a_})
    res_df = pd.DataFrame(out)
    res_df.to_csv(od / "parity_val.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_rows", 200):
        if len(res_df):
            print(res_df[["config", "var", "vs", "n_sites", "n_groups", "kge_model", "kge_comparator", "diff", "ci_low",
                          "ci_high", "every_seed_positive"]].round(4).to_string(index=False))
    if a.register and len(res_df):
        for r in res_df.to_dict("records"):
            model = r.get("model") or f"hydromet_sota_{r['config']}"
            log_result(workstream=SR.WORKSTREAM, experiment="M6SOTA/val_v1", model=model,
                       metric=f"diff_median_site_kge_vs_{r['vs']}", value=float(r["diff"]), n=int(r["n_sites"]),
                       ci_low=float(r["ci_low"]), ci_high=float(r["ci_high"]), target=r["var"], is_baseline=False,
                       ci_method="paired percentile bootstrap resampling basin groups (B=%d); descriptive" % a.n_boot,
                       split="val-role sites x val block (FINAL split; descriptive, protocol hydromet_sota §7)",
                       extra={"comparator": r["comparator"], "kge_model": r["kge_model"], "kge_comparator": r["kge_comparator"],
                              "n_groups": r["n_groups"], "p_value": r["p_value"], "seed_diffs": r["seed_diffs"],
                              "every_seed_positive": r["every_seed_positive"]},
                       notes="M6 SOTA lever val parity (descriptive; val cell used in 6 earlier selection cycles)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inner")
    p.add_argument("--register", action="store_true")
    p = sub.add_parser("val")
    p.add_argument("--configs", required=True)
    p.add_argument("--out", default="score_val_v1")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--register", action="store_true")
    a = ap.parse_args()
    return cmd_inner(a) if a.cmd == "inner" else cmd_val(a)


if __name__ == "__main__":
    sys.exit(main())
