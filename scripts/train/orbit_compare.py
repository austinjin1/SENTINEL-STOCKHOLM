#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Score SENTINEL-Orbit development predictions (orbit_protocol.md §5–§7) on the development cells with
**block** bootstrap CIs (the 50/200-km role block = primary resampling unit, RESEARCH_PLAN §15.1) and water-body
bootstrap as sensitivity; paired MdSA differences vs the H1 comparators; per-seed checks; interval coverage;
within-lake skill; bloom classes. Registers results (workstream M2-orbit) labelled "validation (development)".

H1 comparators per family (MDN_B grid, GBM imagery+context grid, location-kNN k grid) are the family members
with the lowest MdSA on the primary cell — selected on validation like SatWQ-Net configurations (tuning parity,
review W2; optimistic for the baselines, conservative for SENTINEL).

    python -B scripts/train/orbit_compare.py --baselines baselines_dev_v3 --satwq satwq_base_s0 satwq_base_s1 \
        satwq_base_s2 --label satwq_base --register
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.core.registry import log_result  # noqa: E402
from sentinel.models.orbit import data as D  # noqa: E402
from sentinel.models.orbit import evaluate as E  # noqa: E402
from sentinel.models.orbit import pipeline as PL  # noqa: E402
from sentinel.models.orbit.satwq import mixture_exceedance, mixture_quantiles  # noqa: E402

QS = (0.025, 0.16, 0.5, 0.84, 0.975)
FAMILIES = {"mdn_b": re.compile(r"^mdn_b(7|12)(_\d+k_(lin|log))?$"),
            "gbm_imagery_context": re.compile(r"^gbm_imagery_context(#v\d+)?$"),
            "location_knn": re.compile(r"^location_knn(#k\d+)?$")}
H1 = tuple(FAMILIES)
VARIANT = re.compile(r"(#[vk]\d+$)|(_\d+k_(lin|log)$)|(^mdn_b(7|12)$)")
PRIMARY_TARGET_CELL = {"chla": PL.PRIMARY_CELL, "secchi": PL.PRIMARY_CELL, "tss": "valxany_lake",
                       "turbidity": "valxany_lake"}


def load_satwq(run: str, row_ids: np.ndarray, kind: str = "preds"):
    z = np.load(paths.RUNS_ROOT / "orbit" / run / f"{kind}.npz")
    pos = pd.Series(np.arange(len(z["row_id"])), index=z["row_id"])
    ix = pos.loc[row_ids].to_numpy()
    return z["w"][ix], z["mu"][ix], z["sg"][ix]


def ensemble(mixes):
    w = np.concatenate([m[0] / len(mixes) for m in mixes], -1)
    return w, np.concatenate([m[1] for m in mixes], -1), np.concatenate([m[2] for m in mixes], -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--satwq", nargs="+", required=True, help="run names of the seeds")
    ap.add_argument("--label", required=True, help="name of the SatWQ-Net configuration")
    ap.add_argument("--version", default=D.CACHE_VERSION)
    ap.add_argument("--fit-scheme", default="base", choices=["base", "recent"],
                    help="fit cell: base = train role x <=2021; recent = train role x <=2023 (cycle 2)")
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--blend-with", default=None,
                    help="declared hybrid: SENTINEL model = 50/50 log10 mean of the SatWQ-Net median and this baseline's "
                         "INNER-selected representative (e.g. gbm_imagery_context); per-seed blends for the seed check")
    ap.add_argument("--n-boot", type=int, default=E.N_BOOT)
    a = ap.parse_args()
    df, pix, info, masks = PL.load_dev(a.version, a.fit_scheme)
    out_dir = paths.run_dir("orbit", f"compare_{a.label}" + ("" if a.fit_scheme == "base" else f"_fit-{a.fit_scheme}"))
    bp = pd.read_parquet(paths.RUNS_ROOT / "orbit" / a.baselines / "baseline_preds.parquet").set_index("row_id")
    bq_path = paths.RUNS_ROOT / "orbit" / a.baselines / "baseline_quantiles.parquet"
    bq = pd.read_parquet(bq_path).set_index("row_id") if bq_path.exists() else None
    base_models = sorted({c.split("|")[0] for c in bp.columns})
    rows, pairs, extra, chosen, row_store = [], [], [], {}, []
    # ---- pass 1: val-level comparator selection per family and target (primary cell of each target)
    for t in D.TARGETS:
        m = masks[PRIMARY_TARGET_CELL[t]]
        rid = df.row_id.to_numpy()[m]
        y = np.log10(df[t].to_numpy(float))[m]
        chosen[t] = {}
        for fam, rx in FAMILIES.items():
            cands = {}
            for bm in base_models:
                if rx.match(bm) and f"{bm}|{t}" in bp:
                    p = bp.loc[rid, f"{bm}|{t}"].to_numpy(float)
                    ok = np.isfinite(y) & np.isfinite(p)
                    if ok.sum() >= 20 and ok.sum() >= 0.95 * np.isfinite(y).sum():
                        cands[bm] = E.mdsa_log10(p[ok] - y[ok])
            if cands:
                best = min(cands, key=cands.get)
                chosen[t][fam] = {"model": best, "val_mdsa": cands[best], "n_candidates": len(cands)}
    PL.write_json(out_dir / "comparators_val_selected.json", chosen)
    # ---- pass 2: scoring
    for cell in [c for c in PL.EVAL_CELLS if c in masks]:
        m = masks[cell]
        sub = df[m]
        rid = sub.row_id.to_numpy()
        blocks, wbs = sub.block.to_numpy(), sub.waterbody.to_numpy()
        seeds = {r: load_satwq(r, rid) for r in a.satwq}
        variants = {"": ensemble(list(seeds.values()))}
        for kind in ("preds_ctx_off", "preds_rgbn"):
            try:
                variants["_" + kind.split("_", 1)[1]] = ensemble([load_satwq(r, rid, kind) for r in a.satwq])
            except (FileNotFoundError, KeyError):
                pass
        for j, t in enumerate(D.TARGETS):
            y = np.log10(sub[t].to_numpy(float))
            if np.isfinite(y).sum() < 20:
                continue
            preds = {bm: bp.loc[rid, f"{bm}|{t}"].to_numpy(float) for bm in base_models
                     if f"{bm}|{t}" in bp and not VARIANT.search(bm)}
            for fam, c in chosen.get(t, {}).items():
                preds[f"{fam}[val-sel]"] = bp.loc[rid, f"{c['model']}|{t}"].to_numpy(float)
            q_ens = {}
            for vname, (w, mu, sg) in variants.items():
                q = mixture_quantiles(w[:, j], mu[:, j], sg[:, j], QS)
                preds[f"{a.label}{vname}"] = q[:, 2]
                q_ens[vname] = q
            for r, (w, mu, sg) in seeds.items():
                preds[f"{a.label}@{r}"] = mixture_quantiles(w[:, j], mu[:, j], sg[:, j], (0.5,))[:, 0]
            if a.blend_with:
                if f"{a.blend_with}|{t}" not in bp:
                    raise KeyError(f"blend partner {a.blend_with} has no {t} predictions")
                partner = bp.loc[rid, f"{a.blend_with}|{t}"].to_numpy(float)
                preds[a.label] = 0.5 * preds[a.label] + 0.5 * partner
                for r in seeds:
                    preds[f"{a.label}@{r}"] = 0.5 * preds[f"{a.label}@{r}"] + 0.5 * partner
                for vname in list(q_ens):
                    if vname:
                        preds[f"{a.label}{vname}"] = 0.5 * preds[f"{a.label}{vname}"] + 0.5 * partner
            # models that cannot predict >= 95 % of rows (e.g. official MDN: chl-a/TSS only; band gaps) are scored
            # separately on their own rows; the common set is defined by the rest
            full = {k: v for k, v in preds.items() if np.isfinite(v[np.isfinite(y)]).mean() >= 0.95}
            part = {k: v for k, v in preds.items() if k not in full and np.isfinite(v).any()}
            tab, ok = E.score_models(y, full, blocks, n_boot=a.n_boot)
            for k, v in part.items():
                t2, _ = E.score_models(y, {k: v}, blocks, n_boot=a.n_boot)
                t2["note"] = "own rows (< 95 % coverage)"
                tab = pd.concat([tab, t2], ignore_index=True)
            tab.insert(0, "target", t)
            tab.insert(0, "cell", cell)
            cov = {}
            for vname, q in ({} if a.blend_with else q_ens).items():
                cov[f"{a.label}{vname}"] = (E.interval_coverage_mixture(y[ok], q[ok, 1], q[ok, 3]),
                                             E.interval_coverage_mixture(y[ok], q[ok, 0], q[ok, 4]))
            if bq is not None and "mdn_b" in base_models:
                vs = chosen.get(t, {}).get("mdn_b", {}).get("model")
                for mm, src in (("mdn_b", "mdn_b"), ("mdn_gloria", "mdn_gloria"), ("mdn_b[val-sel]", vs)):
                    if src and f"{src}|{t}|q16" in bq:
                        qq = {k: bq.loc[rid, f"{src}|{t}|{k}"].to_numpy(float) for k in ("q025", "q16", "q84", "q975")}
                        cov[mm] = (E.interval_coverage_mixture(y[ok], qq["q16"][ok], qq["q84"][ok]),
                                   E.interval_coverage_mixture(y[ok], qq["q025"][ok], qq["q975"][ok]))
            tab["cov68"] = tab.model.map(lambda k: cov.get(k, (np.nan, np.nan))[0])
            tab["cov95"] = tab.model.map(lambda k: cov.get(k, (np.nan, np.nan))[1])
            tab["within_lake_rho"] = [E.within_lake_spearman(y[ok], preds[k][ok], wbs[ok])["rho"]
                                      if k in full else np.nan for k in tab.model]
            tab["within_lake_n_lakes"] = E.within_lake_spearman(y[ok], preds[a.label][ok], wbs[ok])["n_lakes"]
            rows.append(tab)
            # per-row predictions on the scored rows (label, seeds and val-selected comparators) for the
            # power analysis (scripts/train/orbit_power.py); validation rows only
            keep = [k for k in full if k == a.label or k.startswith(f"{a.label}@") or k.endswith("[val-sel]")]
            row_store.append(pd.DataFrame({"cell": cell, "target": t, "row_id": rid[ok], "block": blocks[ok],
                                           "waterbody": wbs[ok], "y": y[ok], **{k: preds[k][ok] for k in keep}}))
            comps = [k for k in full if k != a.label and not k.startswith(f"{a.label}@") and not k.startswith(f"{a.label}_")]
            for comp in comps:
                rb = E.paired_mdsa(y[ok], preds[a.label][ok], preds[comp][ok], blocks[ok], n_boot=a.n_boot)
                rw = E.paired_mdsa(y[ok], preds[a.label][ok], preds[comp][ok], wbs[ok], n_boot=a.n_boot, seed=1)
                seed_diffs = [E.mdsa_log10(preds[f"{a.label}@{s}"][ok] - y[ok]) - E.mdsa_log10(preds[comp][ok] - y[ok])
                              for s in a.satwq]
                pairs.append({"cell": cell, "target": t, "model": a.label, "comparator": comp, **rb,
                              "wb_ci_low": rw["ci_low"], "wb_ci_high": rw["ci_high"], "wb_n_clusters": rw["n_clusters"],
                              "seed_diffs": seed_diffs, "all_seeds_better": bool(np.all(np.array(seed_diffs) < 0)),
                              "h1_comparator": comp.endswith("[val-sel]")})
            if t == "chla":
                w, mu, sg = variants[""]
                for thr in E.BLOOM_THRESHOLDS:
                    # blended model: rank score = blended point estimate (no mixture); else exceedance probability
                    pr = preds[a.label] if a.blend_with else mixture_exceedance(w[:, j], mu[:, j], sg[:, j], np.log10(thr))
                    extra.append({"cell": cell, "model": a.label, **E.bloom_scores(10 ** y, pr, thr)})
                    for comp in [k for k in full if k.endswith("[val-sel]")]:
                        extra.append({"cell": cell, "model": comp, **E.bloom_scores(10 ** y, preds[comp], thr)})
    PL.log_config_tried(a.label, {"satwq_runs": a.satwq, "blend_with": a.blend_with, "fit_scheme": a.fit_scheme,
                                  "baselines": a.baselines}, out_dir, "scored_on_val", {"kind": "val_evaluation"})
    res = pd.concat(rows, ignore_index=True)
    pr = pd.DataFrame(pairs)
    ex = pd.DataFrame(extra)
    res.to_csv(out_dir / "metrics.csv", index=False)
    pr.to_csv(out_dir / "paired.csv", index=False)
    ex.to_csv(out_dir / "bloom.csv", index=False)
    pd.concat(row_store, ignore_index=True).rename(columns=lambda c: c.replace(a.label, "SENTINEL")).to_parquet(
        out_dir / "val_row_preds.parquet", index=False)
    pd.set_option("display.width", 250)
    for cell in [c for c in PL.EVAL_CELLS if c in masks]:
        r = res[res.cell == cell]
        if r.empty:
            continue
        print(f"\n=== {cell} (validation, development; CI = block bootstrap) ===")
        print(r[~r.model.str.contains("@")][["target", "model", "n", "n_clusters", "mdsa", "mdsa_ci_low", "mdsa_ci_high",
                                             "sspb", "slope", "rmsle", "cov68", "cov95", "within_lake_rho"]]
              .round(3).to_string(index=False))
        p = pr[(pr.cell == cell) & pr.h1_comparator]
        if len(p):
            print(p[["target", "comparator", "diff", "ci_low", "ci_high", "wb_ci_low", "wb_ci_high", "p_value",
                     "all_seeds_better", "n_clusters"]].round(3).to_string(index=False))
    print(json.dumps(chosen, indent=1, default=str))
    if not ex.empty:
        print(ex[ex.cell == PL.PRIMARY_CELL].round(3).to_string(index=False))
    if a.register:
        exp = f"orbit_val_{a.version}" + ("" if a.fit_scheme == "base" else f"_fit-{a.fit_scheme}")
        from sentinel.core.registry import registry_frame

        reg = registry_frame()
        have = set()
        if len(reg):
            r_ = reg[reg.experiment == exp]
            have = set(zip(r_.model, r_.metric, r_.split, r_.target.fillna("")))
        import sentinel.core.registry as _R
        _log = _R.log_result

        def log_result(**kw):   # noqa: F811 — never duplicate an existing (experiment, model, metric, split, target)
            key = (kw["model"], kw["metric"], kw["split"], kw.get("target") or "")
            if key in have:
                return None
            have.add(key)
            return _log(**kw)
        common = {"workstream": PL.WORKSTREAM, "experiment": exp, "data_manifests": [f"orbit_{a.version}", "gloria"],
                  "artifacts": [str(out_dir)],
                  "notes": "validation (development) cell; NOT a claim; test never scored. Primary CI: role-block cluster "
                           "bootstrap (B=2000, percentile); water-body bootstrap in extra."}
        ci_m = "role-block cluster bootstrap (B=2000, percentile)"
        for r in res.itertuples():
            if "@" in r.model and r.cell != PL.PRIMARY_CELL:
                continue
            split = f"{r.cell}@{a.version}"
            is_base = not r.model.startswith(a.label)
            log_result(model=r.model, metric="mdsa", value=r.mdsa, split=split, n=r.n, ci_low=r.mdsa_ci_low,
                       ci_high=r.mdsa_ci_high, ci_method=ci_m, target=r.target, is_baseline=is_base,
                       extra={"n_clusters": r.n_clusters, "note": getattr(r, "note", "")}, **common)
            if r.cell in (PL.PRIMARY_CELL, PRIMARY_TARGET_CELL.get(r.target)) and "@" not in r.model:
                for met in ("sspb", "slope", "rmsle", "log_r2", "cov68", "cov95", "within_lake_rho"):
                    v = getattr(r, met)
                    if v is not None and np.isfinite(v):
                        log_result(model=r.model, metric=met, value=float(v), split=split, n=r.n, target=r.target,
                                   is_baseline=is_base, **common)
        for p in pr.itertuples():
            if not (p.h1_comparator or p.cell == PRIMARY_TARGET_CELL.get(p.target)):
                continue
            log_result(model=p.model, metric=f"mdsa_diff_vs_{p.comparator}", value=p.diff, split=f"{p.cell}@{a.version}",
                       n=p.n, ci_low=p.ci_low, ci_high=p.ci_high, target=p.target,
                       ci_method="paired role-block cluster bootstrap (B=2000); negative = SENTINEL better",
                       extra={"p_value": p.p_value, "seed_diffs": p.seed_diffs, "all_seeds_better": p.all_seeds_better,
                              "n_clusters": p.n_clusters, "waterbody_ci": [p.wb_ci_low, p.wb_ci_high]}, **common)
        for e in ex.itertuples():
            for met in ("auroc", "auprc", "recall_at_fpr"):
                v = getattr(e, met, np.nan)
                if v is not None and np.isfinite(v):
                    log_result(model=e.model, metric=f"bloom{int(e.threshold_ugL)}_{met}", value=float(v),
                               split=f"{e.cell}@{a.version}", n=e.n, target="chla", is_baseline=e.model != a.label,
                               **common)
        print(f"registered results under experiment {exp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
