# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Paired water-body bootstrap comparisons on the val role (protocol §4/§6): student ensemble vs each
baseline (and vs another student, e.g. KD vs no-KD), Holm over the H1 family (3 targets × 2 baselines).

    python -B scripts/eval/lite_compare.py --student <run_dir> --baselines <baseline_run_dir> [--other <run_dir>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.eval.bootstrap import holm_bonferroni
from sentinel.models.lite import data as D
from sentinel.models.lite import evaluate as E
from sentinel.models.lite import runner as R

H1_BASELINES = ("band_ratio", "gbm_rgbn")
OTHER_BASELINES = ("gbm_rgbn_ctx", "loc_knn", "loc_season_gbm", "train_median")


SUBSET = ("val", "val_n")


def load(p: Path) -> pd.DataFrame:
    """Val predictions restricted to the roles in SUBSET (set by --subset). National-track runs also hold val_n."""
    df = pd.read_parquet(p)
    if set(df["role"]) - {"val", "val_n"}:
        raise ValueError(f"{p} holds non-val rows")
    return df[df["role"].isin(SUBSET)].set_index("naip_chip_id")


def run_split_sha(run_dir: Path) -> str:
    import json

    for f in ("summary.json",):
        j = json.loads((run_dir / f).read_text())
        sha = (j.get("cfg") or {}).get("split_sha256") or j.get("split_sha256")
        if sha:
            return sha
    raise ValueError(f"{run_dir} records no split_sha256")


def one_sided_p(r: dict, higher_is_better: bool) -> float:
    """One-sided p for 'student better' from the two-sided bootstrap p (review W2): p/2 if the point
    estimate favours the student, else 1 - p/2."""
    better = r["diff"] > 0 if higher_is_better else r["diff"] < 0
    return r["p_value"] / 2 if better else 1 - r["p_value"] / 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student", required=True)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--other", default=None, help="second student run (e.g. no-KD) for a paired student comparison")
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--subset", default="all", choices=["all", "val", "val_n"],
                    help="val roles to score: 'val' = the frozen 4-state val (original track), all = val + val_n")
    a = ap.parse_args(argv)
    from sentinel.models.lite import guard as G
    from sentinel.models.lite import national as N

    global SUBSET
    SUBSET = ("val", "val_n") if a.subset == "all" else (a.subset,)

    st = load(Path(a.student) / "val_pred_ensemble.parquet")
    name = Path(a.student).name
    frozen = {G.load_split()["_sha256"], N.load_split()["_sha256"]}       # original or national track (both frozen)
    for rd in [Path(a.student), Path(a.baselines)] + ([Path(a.other)] if a.other else []):
        if run_split_sha(rd) not in frozen:
            raise ValueError(f"{rd} was produced on split {run_split_sha(rd)[:12]}, not a frozen Lite split")
        if "smoke" in rd.parts:
            a.no_register = True          # smoke runs are never registered (W9)
    res, pvals = {}, {}
    comps = [(b, load(Path(a.baselines) / f"val_pred_{b}.parquet")) for b in H1_BASELINES + OTHER_BASELINES]
    if a.other:
        comps.append((f"student:{Path(a.other).name}", load(Path(a.other) / "val_pred_ensemble.parquet")))
    for bname, bdf in comps:
        if set(bdf.index) != set(st.index):
            raise ValueError(f"{bname}: chip sets differ ({len(set(bdf.index) ^ set(st.index))} chips)")
        bdf = bdf.reindex(st.index)
        for t in D.TARGETS:
            pm = E.primary_mask(t, st["site_class"])          # primary population (protocol §3)
            s2 = st[pm]
            r = E.compare(t, s2[f"y_{t}"], s2[f"p_{t}"], bdf.loc[pm, f"p_{t}"], s2["water_body"])
            okr = s2[f"y_{t}"].notna() & s2[f"p_{t}"].notna() & bdf.loc[pm, f"p_{t}"].notna()
            r["student"] = E.primary_metric(t)[0](s2.loc[okr, f"y_{t}"], s2.loc[okr, f"p_{t}"])
            r["population"] = E.PRIMARY_POPULATION[t]
            r["baseline"] = r["student"] - r["diff"]
            res[f"{bname}/{t}"] = r
            r["p_one_sided"] = one_sided_p(r, E.primary_metric(t)[2])
            if bname in H1_BASELINES and t in D.REG_TARGETS:
                pvals[f"{bname}/{t}"] = r["p_one_sided"]
    holm = holm_bonferroni(pvals)
    for k in res:
        # rejection counts only when the student is better AND Holm rejects on the one-sided p (W2)
        res[k]["holm_reject_H1"] = (bool(holm[k]) and res[k]["p_one_sided"] < 0.5) if k in holm else None
    national = any(Path(x).name.startswith("nat") for x in (a.student, a.baselines, a.other) if x)
    tag = f"_{Path(a.baselines).name}" + (f"_{Path(a.other).name}" if a.other else "") + f"_{a.subset}"
    out = Path(a.student) / (f"compare{tag}.json" if national
                             else f"compare{'_' + Path(a.other).name if a.other else ''}.json")
    R.write_json(out, res)
    for k, r in res.items():
        print(f"{k:42s} student {r['student']:.3f} base {r['baseline']:.3f} diff {r['diff']:+.3f} "
              f"[{r['ci_low']:+.3f},{r['ci_high']:+.3f}] p1={r['p_one_sided']:.3g} holm={r.get('holm_reject_H1')}")
    if not a.no_register:
        from sentinel.core.registry import log_result

        for k, r in res.items():
            if not np.isfinite(r["diff"]):
                continue
            bname, t = k.rsplit("/", 1)
            log_result(workstream=R.WORKSTREAM, experiment="lite_val_compare" + (f"_national_{a.subset}" if national else ""),
                       model=f"lite_student_{name}",
                       metric=f"{r['metric']}_diff_vs_{bname}", value=r["diff"], ci_low=r["ci_low"],
                       ci_high=r["ci_high"], ci_method="paired cluster bootstrap over water bodies (B=2000)",
                       split="val", n=r["n"], target=t, data_manifests=list(R.DATA_MANIFESTS), artifacts=[str(out)],
                       extra={"p_value": r["p_value"], "holm_reject_H1": r.get("holm_reject_H1"),
                              "p_one_sided": r["p_one_sided"], "student": r["student"], "baseline": r["baseline"],
                              "n_groups": r.get("n_groups"), "subset": a.subset,
                              "baselines_run": Path(a.baselines).name},
                       notes="negative diff = student better for MdSA; positive = better for balanced accuracy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
