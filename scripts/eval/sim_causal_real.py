# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Causal ladder steps 2–3 on REAL data (protocol §7).

    python -B scripts/eval/sim_causal_real.py --phase controls  --step1-tag TAG --n-sites 40 --workers 6
    python -B scripts/eval/sim_causal_real.py --phase diel      --step1-tag TAG --n-sites 40 --workers 6
    python -B scripts/eval/sim_causal_real.py --phase discovery --step1-tag TAG --workers 6
    python -B scripts/eval/sim_causal_real.py --phase summarise --step1-tag TAG [--register]
    python -B scripts/eval/sim_causal_real.py --phase step4     --step1-tag TAG [--register]

``--step1-tag`` names the step-1 configuration whose CONFIRM-set gate passed
($SENTINEL_RUNS/sim/causal_synthetic/<tag>/step1_results.json, site_set "confirm", gate_pass true); its
pc_alpha / FDR level are used. Real-data phases refuse to run otherwise (review W2, protocol §7 step 1).

controls  : step 2 — train-role sites (random, seed 0) with tw, logq, dmax, dmin, logsc; per site four
            Bagged-PCMCI+ runs (primary constrained; unconstrained for N1; time-reversed unconstrained for the
            N2 asymmetry diagnostic; UNCONSTRAINED with the weather of a distant site added for N3) + the
            cross-correlation baseline.
diel      : step 2 diel controls (hourly light, T_w, D_h; one JJA summer per train-role site; τ_max 26 h).
step4     : step 4 — predictive gain + event agreement at val-role sites for the step-3 passers.
discovery : step 3 — primary constrained Bagged-PCMCI+ at every eligible train-role site (train block)
            and val-role site (WY2008–2023).
summarise : recall / FPR, replication table, evidence cards; registry rows (split "train_sites_train_block").
Per-site outputs are cached (resumable) under $SENTINEL_RUNS/sim/causal_real/.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

from sentinel.core import paths

OUT = paths.RUNS_ROOT / "sim" / "causal_real"
SYN = paths.RUNS_ROOT / "sim" / "causal_synthetic"
TAU, BOOT, BLOCK = 7, 50, 10
ALPHA, FDR_Q = 0.01, 0.01        # overwritten from the step-1 configuration (gate_config)
_D = None


def _code_sha() -> str:
    import hashlib

    h = hashlib.sha256()
    for rel in ("discovery.py", "controls.py", "prep.py", "diel.py"):
        h.update((paths.REPO_ROOT / "sentinel" / "analysis" / "causal" / rel).read_bytes())
    return h.hexdigest()[:12]


def gate_config(tag: str) -> dict:
    """Step-1 gate check (review W2): the named configuration must have passed on the confirm sites."""
    from sentinel.analysis.causal.ladder import step1_selection

    sel = step1_selection(SYN)
    if tag != sel["winner"]:
        raise SystemExit(f"'{tag}' is not the dev-selected configuration ({sel['winner']}); dev results {sel['dev']}")
    confirmed = [p.parent.name for p in SYN.glob("*/confirm/step1_results.json")]
    if len(confirmed) > 1:
        raise SystemExit(f"more than one configuration was run on the confirm sites: {confirmed}")
    f = SYN / tag / "confirm" / "step1_results.json"
    if not f.exists():
        raise SystemExit(f"step-1 results for configuration '{tag}' not found ({f})")
    s = json.loads(f.read_text())["summary"]
    c = s.get("config", {})
    if c.get("site_set") != "confirm" or not s.get("gate_pass"):
        raise SystemExit(f"step-1 gate not passed on the confirm sites for '{tag}' "
                         f"(site_set={c.get('site_set')}, gate_pass={s.get('gate_pass')}); no real-data causal run")
    if s["bagged_pcmciplus"]["n_sites"] < 20 or c.get("boot") != 50:
        raise SystemExit("confirm run incomplete (needs 20 sites, 50 bootstraps)")
    return {"step1_tag": tag, "pc_alpha": float(c["pc_alpha"]), "fdr_q": float(c["fdr_q"]),
            "recall": s["bagged_pcmciplus"]["recall"], "fpr": s["bagged_pcmciplus"]["fpr"]}


def _data():
    global _D
    if _D is None:
        from sentinel.models.sim import runner

        _D = runner.load_data("validation")
    return _D


def _matrix(i, block):
    from sentinel.analysis.causal import prep
    from sentinel.models.sim import data as D
    from sentinel.models.sim.dsst import water_year

    d = _data()
    if block == "train":
        days = D.block_mask(d.dates, "train")
    else:   # val-role sites: WY2008-2023 (train + val blocks)
        wy = water_year(d.dates)
        days = (wy >= 2008) & (wy <= 2023)
    return prep.site_matrix(d, i, days)


def _save(path: Path, g, v, names):
    np.savez_compressed(path, graphs=g, vals=v.astype("float16"), names=np.array(names))


def _controls_job(args):
    i, partner, ALPHA, FDR_Q = args
    from sentinel.analysis.causal import discovery as DI
    from sentinel.analysis.causal import prep

    d = _data()
    site = d.sites[i]
    f = OUT / "controls" / f"{site}.json"
    if f.exists():
        return json.loads(f.read_text())
    m = _matrix(i, "train")
    names = list(m.columns)
    x = m.to_numpy()
    res = {"site": site, "names": names, "basin_group": d.basin_group[i], "partner": d.sites[partner]}
    kw = dict(tau_max=TAU, pc_alpha=ALPHA, fdr_q=FDR_Q, boot=BOOT, block=BLOCK)
    g, v = DI.bagged_pcmciplus(x, names, seed=0, **kw)
    _save(OUT / "controls" / f"{site}_primary.npz", g, v, names)
    g, v = DI.bagged_pcmciplus(x, names, seed=1, constrained=False, **kw)
    _save(OUT / "controls" / f"{site}_unconstrained.npz", g, v, names)
    g, v = DI.bagged_pcmciplus(x[::-1].copy(), names, seed=2, constrained=False, **kw)
    _save(OUT / "controls" / f"{site}_reversed.npz", g, v, names)
    pw = prep.anomalies(prep.raw_site(d, partner)[["tair", "sw", "precip"]], prep.D.block_mask(d.dates, "train"))
    pw = pw.loc[m.index].add_prefix("far_")
    names_aug = names + list(pw.columns)
    xa = np.column_stack([x, pw.to_numpy()])
    # N3 runs UNCONSTRAINED (protocol §7; review A3)
    g, v = DI.bagged_pcmciplus(xa, names_aug, seed=3, constrained=False, **kw)
    _save(OUT / "controls" / f"{site}_augmented.npz", g, v, names_aug)
    sc = prep.cq_slope(d, i)
    res["cq_slope"] = sc
    res["dilution"] = bool(np.isfinite(sc[2]) and sc[2] < 0)
    f.write_text(json.dumps(res, default=str))
    return res


def _discovery_job(args):
    i, block, ALPHA, FDR_Q = args
    from sentinel.analysis.causal import discovery as DI

    d = _data()
    site = d.sites[i]
    f = OUT / "discovery" / f"{site}.npz"
    if f.exists():
        return site
    m = _matrix(i, block)
    if m is None:
        return None
    names = list(m.columns)
    g, v = DI.bagged_pcmciplus(m.to_numpy(), names, tau_max=TAU, pc_alpha=ALPHA, fdr_q=FDR_Q, boot=BOOT, block=BLOCK,
                               seed=0)
    _save(f, g, v, names)
    return site


def _diel_job(args):
    i, ALPHA, FDR_Q = args
    from sentinel.analysis.causal import diel as DL
    from sentinel.analysis.causal import discovery as DI

    d = _data()
    site = d.sites[i]
    f = OUT / "diel" / f"{site}.json"
    if f.exists():
        return json.loads(f.read_text())
    m = DL.site_matrix(d, i)
    if m is None:
        return None
    names = list(m.columns)
    x = m.to_numpy()
    kw = dict(tau_max=DL.TAU_MAX_H, pc_alpha=ALPHA, fdr_q=FDR_Q, boot=BOOT, block=24)
    g, v = DI.bagged_pcmciplus(x, names, seed=0, exogenous=("light",), **kw)
    pos = DL.score(g[1:], v[1:], names)
    gu, vu = DI.bagged_pcmciplus(x, names, seed=1, constrained=False, **kw)
    neg = DL.score_negative(gu[1:], vu[1:], names)
    res = {"site": site, "water_year": int(m.attrs["water_year"]), "n_hours_valid": int(np.isfinite(x).all(axis=1).sum()),
           "positive": pos, "negative": neg, "basin_group": d.basin_group[i]}
    f.write_text(json.dumps(res, default=str))
    return res


def eligible(d, roles, need):
    from sentinel.analysis.causal import prep
    from sentinel.models.sim import data as D

    out = []
    jj = {v: D.TARGETS.index(v) for v in ("temp_c_mean", "do_mgl_max", "do_mgl_min", "spcond_uscm_mean", "turb_fnu_mean")}
    tm = D.block_mask(d.dates, "train")
    for i in range(len(d.sites)):
        if d.role[i] not in roles:
            continue
        cnt = {k: int(np.isfinite(d.y[i, tm, j]).sum()) for k, j in jj.items()}
        if need == "controls":
            if min(cnt["temp_c_mean"], cnt["do_mgl_max"], cnt["do_mgl_min"], cnt["spcond_uscm_mean"]) >= prep.MIN_VALID + 365:
                out.append(i)
        else:
            if cnt["temp_c_mean"] >= prep.MIN_VALID and max(cnt["do_mgl_max"], cnt["spcond_uscm_mean"], cnt["turb_fnu_mean"]) >= prep.MIN_VALID:
                out.append(i)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["controls", "diel", "discovery", "summarise", "step4"])
    ap.add_argument("--step1-tag", required=True)
    ap.add_argument("--n-sites", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--register", action="store_true")
    a = ap.parse_args(argv)
    gc = gate_config(a.step1_tag)
    alpha, fdr_q = gc["pc_alpha"], gc["fdr_q"]
    for sub in ("controls", "discovery", "diel"):
        paths.ensure(OUT / sub)
    cfgf = OUT / "run_config.json"
    if cfgf.exists():      # cached per-site outputs must come from the same configuration
        old = json.loads(cfgf.read_text())
        if (old.get("step1_tag"), old.get("code_sha")) != (gc["step1_tag"], _code_sha()):
            raise SystemExit(f"cached outputs in {OUT} come from another configuration/code {old}; move them first")
    cfgf.write_text(json.dumps(gc | {"tau_max": TAU, "boot": BOOT, "block": BLOCK, "code_sha": _code_sha()}))
    d = _data()
    from sentinel.core.runlog import TrainingRun

    run_cfg = gc | {"phase": a.phase, "tau_max": TAU, "boot": BOOT, "block": BLOCK, "n_sites": a.n_sites}
    with TrainingRun("sim", f"causal_real_{a.phase}", config=run_cfg, data=["sonde_daily", "forcings_daily"]
                     + (["sonde_hourly"] if a.phase == "diel" else []), plan_ref="sim_protocol §7",
                     iteration=a.step1_tag, artifacts_dir=str(OUT)) as run:
        _phase(a, d, alpha, fdr_q, run, run_cfg)
    return 0


def _phase(a, d, alpha, fdr_q, run, run_cfg):
    reg = pd.read_parquet(paths.PROCESSED / "sites" / "sites.parquet")
    huc2 = reg.assign(s=reg["site_id"].astype(str)).set_index("s")["huc2"].astype(str)
    if a.phase == "controls":
        from sentinel.eval.splits import haversine_km

        el = eligible(d, ("train", "cal"), "controls")
        rs = np.random.default_rng(0)
        pick = sorted(rs.choice(el, min(a.n_sites, len(el)), replace=False).tolist())
        jobs = []
        for i in pick:
            far = [k for k in range(len(d.sites)) if huc2.get(d.sites[k]) != huc2.get(d.sites[i])
                   and haversine_km(d.lat[i], d.lon[i], d.lat[k], d.lon[k]) >= 1000]
            jobs.append((i, int(rs.choice(far)), alpha, fdr_q))
        (OUT / "controls_sites.json").write_text(json.dumps({"eligible": len(el), "picked": [d.sites[i] for i in pick]}))
        print(f"controls: {len(el)} eligible, running {len(jobs)}", flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for r in ex.map(_controls_job, jobs):
                print("done", r["site"], flush=True)
    elif a.phase == "discovery":
        tr = eligible(d, ("train", "cal"), "discovery")
        va = eligible(d, ("val",), "discovery")
        jobs = [(i, "train", alpha, fdr_q) for i in tr] + [(i, "val", alpha, fdr_q) for i in va]
        print(f"discovery: {len(tr)} train-role + {len(va)} val-role sites", flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for s in ex.map(_discovery_job, jobs, chunksize=2):
                print("done", s, flush=True)
    elif a.phase == "diel":
        from sentinel.analysis.causal import diel as DL

        el = [i for i in eligible(d, ("train", "cal"), "discovery") if DL.has_hourly(d.sites[i])]
        rs = np.random.default_rng(0)
        pick = sorted(rs.choice(el, min(a.n_sites, len(el)), replace=False).tolist())
        print(f"diel: {len(el)} eligible with hourly data, running {len(pick)}", flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            res = [r for r in ex.map(_diel_job, [(i, alpha, fdr_q) for i in pick]) if r]
        from sentinel.analysis.causal.ladder import diel_summary

        summ = diel_summary(res)
        (OUT / "diel_summary.json").write_text(json.dumps(summ, indent=1, default=str))
        print(json.dumps(summ, indent=1, default=str))
        if a.register:
            from sentinel.core.registry import log_result

            for r in summ["positive"] + summ["negative"]:
                log_result(workstream="sim", experiment="causal_step2_diel_controls", model="Bagged-PCMCI+ (ParCorr)",
                           metric="recall" if r["kind"] == "positive" else "fpr", value=r["rate"], n=r["n_sites"],
                           target=r["control"], split="train_sites_train_block", config=run_cfg,
                           artifacts=[str(OUT / "diel_summary.json")], data_manifests=["sonde_hourly", "forcings_daily"])
    elif a.phase == "summarise":
        from sentinel.analysis.causal.ladder import summarise

        summ = summarise(OUT, d, huc2, register=a.register, step1_pass=True, step1_tag=a.step1_tag)
        print(json.dumps(summ, indent=1, default=str)[:6000])
    else:   # step4
        from sentinel.analysis.causal.ladder import run_step4

        summ = run_step4(OUT, d, eligible(d, ("val",), "discovery"), register=a.register)
        print(json.dumps(summ, indent=1, default=str)[:6000])


if __name__ == "__main__":
    raise SystemExit(main())
