# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Causal ladder step 1 — SIMULATION RECOVERY on declared synthetic data (protocol §7 step 1).

    python -B scripts/eval/sim_causal_step1.py --n-sites 20 --workers 6 --boot 50 [--register]
        [--pc-alpha 0.01] [--fdr-q Q] [--tag NAME] [--site-set dev|confirm] [--no-lpcmci]

Configuration selection (protocol §7 step 1, amended 2026-11-08): configurations are compared on the
20 "dev" synthetic sites (site-sampling seed 0, the sites of the failed v1 run); the selected configuration
must then pass the gate on 20 FRESH "confirm" sites (seed 1, disjoint from dev), which is the gate result.
Outputs go to $SENTINEL_RUNS/sim/causal_synthetic/<tag>/<site_set>/. Candidate tags and the selection rule:
sentinel/analysis/causal/ladder.py::STEP1_CANDIDATES / step1_selection.

Synthetic river variables are generated from known structural equations + the Sim physics module,
driven by real gridMET forcing of train-role sites (sentinel/analysis/causal/synthetic.py). Datasets are
written under $SENTINEL_RUNS/sim/causal_synthetic/ with a manifest flagged synthetic=True; registry rows
use split="synthetic". Never mixed with real-data results.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.core.manifest import write_manifest
from sentinel.core.registry import log_result
from sentinel.core.runlog import TrainingRun

OUT = paths.RUNS_ROOT / "sim" / "causal_synthetic"
TAU_MAX = 7


def _site_job(args):
    k, site, tair, sw, pr, dates, lat, boot, seed, methods, alpha, fdr_q, out = args
    from sentinel.analysis.causal import controls as C
    from sentinel.analysis.causal import discovery as DI
    from sentinel.analysis.causal import prep
    from sentinel.analysis.causal import synthetic as SY

    t0 = time.time()
    dates = pd.DatetimeIndex(dates)
    full = SY.generate(tair, sw, pr, dates, lat, seed=seed + k)
    obs = SY.add_gaps(full, seed=seed + k)
    obs.to_parquet(out / f"synthetic_site{k:02d}.parquet")
    eff = SY.true_links(tair, sw, pr, dates, lat, seed=seed + k)
    truth = SY.truth_set(eff)
    an = prep.anomalies(obs, np.ones(len(obs), bool))
    names = list(an.columns)
    x = an.to_numpy()
    scored = [p for p in eff if p not in (("dmax", "dmin"), ("dmin", "dmax"))]
    res = {"site_index": k, "source_site": site, "truth": sorted(map(list, truth)),
           "effects": {f"{a}->{b}": v for (a, b), v in eff.items()}}
    if "bagged" in methods:
        g, v = DI.bagged_pcmciplus(x, names, tau_max=TAU_MAX, pc_alpha=alpha, fdr_q=fdr_q, boot=boot, block=10,
                                   seed=seed + k)
        res["bagged_pcmciplus"] = C.recovery_scores(g[1:], v[1:], names, truth, scored)
        res["pcmciplus"] = C.recovery_scores(g[:1], v[:1], names, truth, scored)
        np.savez_compressed(out / f"graphs_bagged_site{k:02d}.npz", graphs=g, vals=v.astype("float16"), names=names)
    if "lpcmci" in methods:
        g, v = DI.run_lpcmci(x, names, tau_max=TAU_MAX, pc_alpha=alpha, constrained=False)
        res["lpcmci_unconstrained_uncorrected"] = True     # links at pc_alpha (no FDR)
        res["lpcmci_unconstrained"] = C.recovery_scores(g[None], v[None], names, truth, scored)
    if "lpcmci_fdr" in methods:
        g, v = DI.run_lpcmci(x, names, tau_max=TAU_MAX, pc_alpha=alpha, constrained=False,
                             fdr_q=alpha if fdr_q is None else fdr_q)
        res["lpcmci_fdr"] = C.recovery_scores(g[None], v[None], names, truth, scored)
        np.savez_compressed(out / f"graphs_lpcmci_fdr_site{k:02d}.npz", graphs=g[None], vals=v[None].astype("float16"),
                            names=names)
    g, v = DI.xcorr_baseline(x, names, tau_max=TAU_MAX, alpha=alpha if fdr_q is None else fdr_q)
    res["xcorr_baseline"] = C.recovery_scores(g[None], v[None], names, truth, scored)
    res["seconds"] = time.time() - t0
    return res


def _jpcmci(datasets, names, alpha):
    from sentinel.analysis.causal import discovery as DI

    return DI.run_jpcmciplus(datasets, names, tau_max=TAU_MAX, pc_alpha=alpha, fdr_q=alpha)


def pooled(results, key):
    tot = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for r in results:
        if key in r:
            for k in tot:
                tot[k] += r[key][k]
    return {**tot, "recall": tot["tp"] / max(tot["tp"] + tot["fn"], 1), "fpr": tot["fp"] / max(tot["fp"] + tot["tn"], 1),
            "precision": tot["tp"] / max(tot["tp"] + tot["fp"], 1), "n_sites": sum(key in r for r in results)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sites", type=int, default=20)
    ap.add_argument("--lpcmci-sites", type=int, default=10)
    ap.add_argument("--jpcmci-sites", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--boot", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pc-alpha", type=float, default=0.01)
    ap.add_argument("--fdr-q", type=float, default=None, help="BH level applied to the graph (default pc_alpha)")
    ap.add_argument("--tag", default="v2_a0.01")
    ap.add_argument("--site-set", default="dev", choices=["dev", "confirm", "confirm2"])
    ap.add_argument("--gate-method", default="bagged_pcmciplus", choices=["bagged_pcmciplus", "lpcmci_fdr"],
                    help="lpcmci_fdr: LPCMCI (unconstrained, FDR-pruned) on every site; protocol §12 v1.4")
    ap.add_argument("--no-bagged", action="store_true")
    ap.add_argument("--no-lpcmci", action="store_true")
    ap.add_argument("--no-jpcmci", action="store_true")
    ap.add_argument("--register", action="store_true")
    a = ap.parse_args(argv)
    from sentinel.models.sim import data as D
    from sentinel.models.sim import runner

    out = paths.ensure(OUT / a.tag / a.site_set)      # dev and confirm never overwrite each other
    d = runner.load_data("validation")
    tm = D.block_mask(d.dates, "train")
    dev = np.random.default_rng(0).choice(d.idx("train"), 20, replace=False)
    if a.site_set == "dev":
        sites = dev[: a.n_sites]
        gen_seed = a.seed
    else:   # fresh sites, disjoint from dev (and confirm2 also from confirm); different generator noise seed
        pool = np.setdiff1d(d.idx("train"), dev)
        conf = np.random.default_rng(1).choice(pool, 20, replace=False)
        if a.site_set == "confirm":
            sites = conf[: a.n_sites]
            gen_seed = a.seed + 1000
        else:
            sites = np.random.default_rng(2).choice(np.setdiff1d(pool, conf), a.n_sites, replace=False)
            gen_seed = a.seed + 2000
    jobs = []
    for k, i in enumerate(sites):
        f = d.forcing[i]
        methods = (() if a.no_bagged else ("bagged",)) + (("lpcmci",) if (k < a.lpcmci_sites and not a.no_lpcmci) else ())
        if a.gate_method == "lpcmci_fdr":
            methods = methods + ("lpcmci_fdr",)
        jobs.append((k, d.sites[i], 0.5 * (f[tm, 0] + f[tm, 1]), f[tm, 2], f[tm, 3], d.dates[tm].values, float(d.lat[i]),
                     a.boot, gen_seed, methods, a.pc_alpha, a.fdr_q, out))
    cfg = vars(a) | {"tau_max": TAU_MAX, "fdr": "fdr_bh (applied to graphs)", "block": 10, "effect_sd": 0.1,
                     "generator_seed": gen_seed}
    with TrainingRun("sim", "causal_step1_recovery", config=cfg, data=["forcings_daily (drivers only)", "SYNTHETIC"],
                     plan_ref="sim_protocol §7 step 1", iteration=f"{a.tag}/{a.site_set}", artifacts_dir=str(out)) as run:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            results = list(ex.map(_site_job, jobs))
        # J-PCMCI+ on the first few synthetic sites pooled (sites = datasets)
        from sentinel.analysis.causal import controls as C
        from sentinel.analysis.causal import prep
        from sentinel.analysis.causal import synthetic as SY

        jres = None
        if a.no_jpcmci:
            jres = {"skipped": True}
        else:
          try:
            dsets, truths, scored_all = [], set(), None
            for r in results[: a.jpcmci_sites]:
                obs = pd.read_parquet(out / f"synthetic_site{r['site_index']:02d}.parquet")
                dsets.append(prep.anomalies(obs, np.ones(len(obs), bool)).to_numpy())
                truths |= {tuple(t) for t in r["truth"]}
            names = SY.VARS
            g, v = _jpcmci(dsets, names, a.pc_alpha)
            scored = [(p, c) for p in names for c in names if p != c and c not in SY.EXOG
                      and (p, c) not in (("dmax", "dmin"), ("dmin", "dmax"))]
            jres = C.recovery_scores(g[None], v[None], names, truths, scored)
          except Exception as e:  # noqa: BLE001 — reported, not hidden
            jres = {"error": f"{type(e).__name__}: {e}"}
            run.note(f"J-PCMCI+ failed: {jres['error']}")
        summary = {m: pooled(results, m) for m in ("bagged_pcmciplus", "pcmciplus", "lpcmci_unconstrained",
                                                     "xcorr_baseline", "lpcmci_fdr")}
        summary["jpcmciplus_pooled"] = jres
        prim = summary[a.gate_method]
        summary["gate_pass"] = bool(prim["recall"] >= 0.8 and prim["fpr"] <= 0.05)
        summary["config"] = {"tag": a.tag, "site_set": a.site_set, "pc_alpha": a.pc_alpha, "gate_method": a.gate_method,
                             "fdr_q": a.pc_alpha if a.fdr_q is None else a.fdr_q, "boot": a.boot, "tau_max": TAU_MAX}
        (out / "step1_results.json").write_text(json.dumps({"summary": summary, "sites": results}, indent=1, default=str))
        files = sorted(out.glob("synthetic_site*.parquet"))
        write_manifest(f"sim_causal_synthetic_{a.tag}_{a.site_set}".replace(".", "p"), files=files, sources=["manifest:forcings_daily (exogenous drivers)"],
                       description="DECLARED SYNTHETIC: causal-ladder step-1 simulation-recovery datasets "
                                   "(river variables generated from known equations + Sim physics)",
                       license="MIT (generated)", synthetic=True,
                       stats={"n_sites": len(files), "generator": "sentinel/analysis/causal/synthetic.py", "seed": gen_seed,
                              "site_set": a.site_set})
        print(json.dumps(summary, indent=1, default=str))
        run.metric(f"recall_{a.gate_method}", prim["recall"])
        run.metric(f"fpr_{a.gate_method}", prim["fpr"])
        run.metric("gate_pass", float(summary["gate_pass"]))
        if a.register:
            for m, s in summary.items():
                if not isinstance(s, dict) or "recall" not in s:
                    continue
                for met in ("recall", "fpr", "precision"):
                    log_result(workstream="sim", experiment="causal_step1_recovery", model=m, metric=met,
                               value=s[met], split="synthetic", target=f"{a.tag}/{a.site_set}", n=s.get("tp", 0) + s.get("fn", 0) + s.get("fp", 0) + s.get("tn", 0),
                               config=cfg, data_manifests=[f"sim_causal_synthetic_{a.tag}_{a.site_set}".replace(".", "p")],
                               is_baseline=(m == "xcorr_baseline"), artifacts=[str(out / "step1_results.json")],
                               notes="SYNTHETIC DATA (declared, R9): simulation recovery; ordered pairs with river child",
                               extra={"synthetic": True, "gate_pass": summary["gate_pass"] if m == a.gate_method else None})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
