#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SINGLE-PASS M5-X test scorer (protocol ``bioassess_x_protocol.md`` §8–§9). DO NOT RUN before the lead's freeze.

Modes
-----
* (default) **test pass** — single-look enforcement in code (``sentinel/models/bioassess/xtestlook.py``, pattern of
  ``sentinel/models/hydromet/testlook.py``): accepts ONLY the lead-written ``docs/preregistration/bioassess_x_freeze.json``
  (never the candidate record; status ``frozen``; ``candidate_sha256`` and ``scoring_code_sha256`` must match),
  verifies file/package hashes and content-hash data pins, refuses if the registry holds any ``bioassess_x_test``
  row — all before the claim. Then setup (imports, data, GPU lock), then the claim: O_EXCL ``TEST_LOOK.json`` +
  claim-time registry row. Test rows load only under that open claim. Phase 1 predicts every unit (per-unit sha256
  cache), predictions are sealed (0444, ``SHA256SUMS``, O_EXCL ``PREDICTIONS_SAVED.json``); phase 2 scores the sealed
  predictions, flushes the registry once, seals the results and writes the O_EXCL ``SCORED.json`` marker.
* ``--resume RUN_DIR --authorised-by`` — the pre-registered single resume (crash before sealing; O_EXCL
  ``RESUMED.json``; only the claim row may exist; cached units reused, never refitted).
* ``--rescore-from-saved RUN_DIR --authorised-by`` — crash after sealing: score the sha-checked sealed predictions.
* ``--dry-run-dev [--max-folds N] [--cpu]`` — the identical code path on the development folds from the *candidate*
  record, with its own look directory and scratch registry inside the run directory (real markers untouched).

Statistics (all from the record): per block × target, C-X vs the per-target comparator on ``r2_anomaly`` (paired
location-cluster bootstrap, B = 10,000; one-sided rule p = two-sided p if Δ > 0 else 1); Holm over the 12 tests;
decision table §8 (per-seed Δ for seeds 0–2, region-unit paired CI, realised Holm-worst MDE vs 0.02). Secondary,
unadjusted: other candidates, pooled R², SENTINEL vs RF[geography], X-U by ``analysis_method``, 50-km-block CIs.

    python -B scripts/eval/bioassess_x_final.py --authorised-by "<lead>"
    python -B scripts/eval/bioassess_x_final.py --dry-run-dev [--max-folds 2]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "6"
os.environ.setdefault("SENTINEL_GPU_MIN_FREE_MB", "5000")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import importlib.metadata as md  # noqa: E402
import json  # noqa: E402
import stat  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.core.registry import file_sha256, log_result, read_registry  # noqa: E402
from sentinel.models.bioassess import xdev as XD  # noqa: E402
from sentinel.models.bioassess import xtestlook as XL  # noqa: E402
from sentinel.models.bioassess import xtrack as X  # noqa: E402

REPO = paths.REPO_ROOT
FREEZE = XL.FREEZE_FILE                  # the ONLY record a test pass accepts (lead-written)
CANDIDATE = XL.CANDIDATE_FILE            # development dry run only
EXPERIMENT = XL.EXPERIMENT
R_ENV = Path.home() / "miniconda3/envs/rbioinfer"
HOLM_Z = 3.71                      # (z_{1-0.05/24} + z_0.8), protocol §8
SESOI = 0.02


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def safe_log(**kw):
    """log_result, recording a non-finite value as 'not estimable' (metric suffix, value 1) instead of failing."""
    v = kw.get("value")
    if v is None or not np.isfinite(float(v)):
        kw = {**kw, "metric": kw["metric"] + "_not_estimable", "value": 1.0, "ci_low": None, "ci_high": None,
              "notes": kw.get("notes", "") + " [value not estimable on these rows]"}
    return log_result(**kw)


# ------------------------------------------------------------------------------------------------- record checks
def r_package_versions() -> dict:
    try:
        out = subprocess.run([str(R_ENV / "bin/Rscript"), "-e",
                              'cat(as.character(packageVersion("bio.infer")), as.character(packageVersion("gjam")), '
                              'R.version.string, sep="|")'], capture_output=True, text=True, timeout=300,
                             env={**os.environ, "PATH": f"{R_ENV}/bin:{os.environ.get('PATH', '')}"})
        b, g, r = out.stdout.strip().split("|")
        return {"bio.infer": b, "gjam": g, "R": r}
    except Exception as e:                      # noqa: BLE001
        return {"error": str(e)}


def package_versions(names) -> dict:
    out = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = None
    return out


def verify_record(rec: dict, check_data: bool = True) -> list[str]:
    bad = []
    for rel, h in rec["sha256"].items():
        p = REPO / rel if not rel.startswith("/") else Path(rel)
        if not p.exists() or file_sha256(p) != h:
            bad.append(f"hash changed: {rel}")
    if check_data:
        for f, h in rec["data_sha256"].items():
            if not Path(f).exists() or file_sha256(f) != h:
                bad.append(f"data changed: {f}")
    if package_versions(rec["package_versions"]) != rec["package_versions"]:
        bad.append("python package versions changed")
    if r_package_versions() != rec["r_package_versions"]:
        bad.append("R package versions changed")
    return bad


def registry_has(experiment: str, registry_path=None) -> bool:
    return any(r.get("experiment") == experiment for r in read_registry(registry_path))


def verify_freeze() -> dict:
    """Test pass: the lead-written FREEZE file only (``xtestlook.load_freeze``), registry refusal, file/package hashes
    and content-hash data pins — all before the claim, so a mismatch never burns the look."""
    try:
        fr = XL.load_freeze()
        if XL.registry_rows(EXPERIMENT):
            raise XL.SingleLookError("registry already holds bioassess_x_test rows — the M5-X look was used")
        if (XL.look_dir() / "TEST_LOOK.json").exists():
            raise XL.SingleLookError("the M5-X test look is already claimed")
        XL.verify_data_pins(fr)
    except XL.SingleLookError as e:
        raise SystemExit(str(e)) from e
    bad = verify_record(fr)
    if bad:
        raise SystemExit("record verification failed: " + "; ".join(bad[:10]))
    return fr


# ------------------------------------------------------------------------------------------------- unit cache
class Cache:
    def __init__(self, d: Path):
        self.d = d
        d.mkdir(parents=True, exist_ok=True)
        self.manifest_f = d / "manifest.json"
        self.m = json.loads(self.manifest_f.read_text()) if self.manifest_f.exists() else {}

    @staticmethod
    def key(*parts) -> str:
        return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:24]

    def get(self, k):
        f = self.d / f"{k}.parquet"
        if k in self.m:
            if not f.exists() or file_sha256(f) != self.m[k]:
                raise SystemExit(f"cached unit {k} was tampered with or lost")
            return pd.read_parquet(f)
        return None

    def put(self, k, df):
        f = self.d / f"{k}.parquet"
        df.to_parquet(f)
        self.m[k] = file_sha256(f)
        tmp = self.manifest_f.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.m))
        tmp.replace(self.manifest_f)


# ------------------------------------------------------------------------------------------------- predictors
def leaf_specs(spec) -> list[dict]:
    if spec["kind"] == "mean":
        return [x for m in spec["members"] for x in leaf_specs(m)]
    return [spec]


def spec_id(spec) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


class Predictor:
    def __init__(self, ctx, run_dir: Path, seeds, device, n_jobs: int = 5):
        self.ctx, self.run_dir, self.seeds, self.device, self.n_jobs = ctx, run_dir, tuple(seeds), device, n_jobs
        self.cache = Cache(run_dir / "cache")
        self.TM = XD.TaxonMatrix(ctx["D"])
        self.prep = None

    def leaf(self, spec, block, k, fo, seed) -> pd.DataFrame:
        stochastic = spec["kind"] == "bioset" or (spec["kind"] == "baseline" and spec["name"] in XD.STOCHASTIC)
        s_eff = seed if stochastic else 0
        key = self.cache.key(block, spec_id(spec), fo.g, s_eff)
        got = self.cache.get(key)
        if got is not None:
            return got
        S, frozen, D = self.ctx["S"], self.ctx["frozen"], self.ctx["D"]
        X.assert_fit_ids(XD.keys_of(fo.train), S, frozen)              # fit guard (protocol §7)
        T = list(XD.TARGETS[block])
        if spec["kind"] == "baseline":
            P = pd.DataFrame(index=fo.test, columns=T, dtype=float)
            for t in T:
                v = spec["variant"].get(t, "same") if isinstance(spec.get("variant"), dict) else "same"
                P[t] = XD.fit_predict(spec["name"], block, t, D, self.TM, fo.train, fo.test, variant=v, seed=s_eff,
                                      device=self.device, n_jobs=self.n_jobs)
        elif spec["kind"] == "bioset":
            from sentinel.models.bioassess import xbioset as XB

            if self.prep is None:
                self.prep = XB.Prepared(D)
            preds, _ = XB.fit_fold(self.prep, spec["config"], block, fo, S, frozen, seeds=(s_eff,), device=self.device,
                                   pretrain_root=self.run_dir / "pretrain", fold_index=k, log=log)
            P = preds[s_eff]
        elif spec["kind"] == "r":
            P = self.r_method(spec["method"], block, k, fo)
        else:
            raise ValueError(spec)
        self.cache.put(key, P)
        log("unit", block, spec.get("name") or spec.get("config") or spec.get("method"), fo.g, f"seed{s_eff}")
        return P

    def r_method(self, method, block, k, fo) -> pd.DataFrame:
        import importlib.util

        spec_ = importlib.util.spec_from_file_location("xdev_script", REPO / "scripts/eval/bioassess_x_dev.py")
        mod = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(mod)
        d = self.run_dir / "r_in" / block / f"fold{k:02d}"
        mod.write_r_fold(d, block, fo, self.ctx["D"], self.TM)
        out = self.run_dir / "r_out" / method / block / f"fold{k:02d}.csv"
        env = {**os.environ, "PATH": f"{R_ENV}/bin:{os.environ.get('PATH', '')}", "OMP_NUM_THREADS": "1"}
        if method == "bioinfer":
            tax = self.run_dir / "r_out" / "bioinfer" / "taxonomy.rds"
            if not tax.exists():
                tax.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"name": sorted(self.ctx["D"].tokens["name"].dropna().astype(str).unique())}).to_csv(
                    tax.parent / "names.csv", index=False)
                subprocess.run([str(R_ENV / "bin/Rscript"), str(REPO / "scripts/eval/bioassess_x_bioinfer.R"),
                                "taxonomy", str(tax.parent / "names.csv"), str(tax)], check=True, env=env)
            cmd = [str(R_ENV / "bin/Rscript"), str(REPO / "scripts/eval/bioassess_x_bioinfer.R"), "fold", str(d),
                   str(tax), str(out)]
        else:
            cmd = [str(R_ENV / "bin/Rscript"), str(REPO / "scripts/eval/bioassess_x_gjam.R"), "fold", str(d), str(out)]
        subprocess.run(cmd, check=True, env=env)
        P = pd.read_csv(out).set_index("visit_id").reindex(fo.test)
        return P.reindex(columns=list(XD.TARGETS[block]))

    def model(self, spec, block, folds, seed) -> pd.DataFrame:
        if spec["kind"] == "mean":
            ms = [self.model(m, block, folds, seed) for m in spec["members"]]
            return sum(ms) / len(ms)
        return pd.concat([self.leaf(spec, block, k, fo, seed) for k, fo in enumerate(folds)])


# ------------------------------------------------------------------------------------------------- statistics
def block50(lat, lon) -> np.ndarray:
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    r = np.floor((lat + 90) * 111.195 / 50).astype(int)
    c = np.floor((lon + 180) * 111.195 * np.cos(np.radians(lat)) / 50).astype(int)
    return np.array([f"{a}_{b}" for a, b in zip(r, c)])


def holm(p: dict) -> dict:
    keys = sorted(p, key=p.get)
    m, out, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


def label(row) -> tuple[str, str]:
    """Decision table §8 (first matching row); a Δ that cannot be estimated is labelled as such."""
    if not np.isfinite(row.get("delta", 0.0)) or not np.isfinite(row["ci_low"]) or not np.isfinite(row["ci_high"]):
        return "Not estimable", "no finite paired estimate on these rows"
    if row["holm_rejected"]:
        fails = [n for n, ok in (("per-seed Δ > 0 for seeds 0-2", row["all_seeds_positive"]),
                                 ("region-unit paired CI > 0", row["region_ci_low"] > 0)) if not ok]
        return ("SOTA (M5-X, {}, within-region)".format(row["block"]), "") if not fails else \
            ("Above, not robust", "; ".join(fails))
    if row["ci_high"] < 0:
        return "Below", ""
    if row["ci_low"] <= 0 <= row["ci_high"] and row["realised_mde"] > SESOI:
        return "Inconclusive (underpowered)", ""
    return "Competitive", ""


def score_blocks(pred: dict, truth: dict, meta: dict, rec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """pred[block][model][seed] -> frame; returns (confirmatory table, marginal table)."""
    from sentinel.models.bioassess import evaluate as EV

    nb, nbc = int(rec["n_boot"]), int(rec["n_boot_confirmatory"])
    conf, marg = [], []
    for b, models in pred.items():
        Y, mt = truth[b], meta[b]
        region, cluster = mt["region"].to_numpy(), mt["cluster"].to_numpy()
        blk50 = block50(mt["lat"], mt["lon"])
        for name, seeds in models.items():
            mean = sum(seeds.values()) / len(seeds)
            for t in XD.TARGETS[b]:
                y, p = Y[t].to_numpy(float), mean[t].to_numpy(float)
                ok = np.isfinite(y) & np.isfinite(p)
                if ok.sum() < 3:
                    continue
                sm = EV.summary(y[ok], p[ok], region[ok])
                ci = EV.cis(y[ok], p[ok], region[ok], cluster[ok], n_boot=nb, metrics=("r2_anomaly", "r2"))
                marg.append({"block": b, "model": name, "target": t, "n": int(ok.sum()), "r2_anomaly": sm["r2_anomaly"],
                             "r2": sm["r2"], "rmse": sm["rmse"], "spearman": sm["spearman"],
                             "ci_low": ci["r2_anomaly"]["ci_low"], "ci_high": ci["r2_anomaly"]["ci_high"],
                             "r2_ci": [ci["r2"]["ci_low"], ci["r2"]["ci_high"]],
                             "per_seed_r2_anomaly": {s: EV.anomaly_r2(y[ok], f[t].to_numpy(float)[ok], region[ok])
                                                     for s, f in seeds.items()}})
        cand = rec["candidates"][b]["confirmatory"]["name"]
        for t in XD.TARGETS[b]:
            comp = rec["comparators"][b][t]["name"]
            y = Y[t].to_numpy(float)
            A, Bm = models[cand], models[comp]
            pa, pb = (sum(A.values()) / len(A))[t].to_numpy(float), (sum(Bm.values()) / len(Bm))[t].to_numpy(float)
            ok = np.isfinite(y) & np.isfinite(pa) & np.isfinite(pb)
            pr = EV.paired_vs(y[ok], pa[ok], pb[ok], region[ok], cluster[ok], metric="r2_anomaly", n_boot=nbc)
            prr = EV.paired_vs(y[ok], pa[ok], pb[ok], region[ok], region[ok], metric="r2_anomaly", n_boot=nbc)
            pr50 = EV.paired_vs(y[ok], pa[ok], pb[ok], region[ok], blk50[ok], metric="r2_anomaly", n_boot=nb)
            per_seed = {s: EV.anomaly_r2(y[ok], A[s][t].to_numpy(float)[ok], region[ok])
                        - EV.anomaly_r2(y[ok], Bm[s if s in Bm else 0][t].to_numpy(float)[ok], region[ok]) for s in A}
            signs = []
            for g in np.unique(region[ok]):
                m = region[ok] == g
                signs.append(EV.anomaly_r2(y[ok][m], pa[ok][m], region[ok][m])
                             - EV.anomaly_r2(y[ok][m], pb[ok][m], region[ok][m]))
            d = pr["diff"]
            conf.append({"block": b, "target": t, "candidate": cand, "comparator": comp, "n": int(ok.sum()), "delta": d,
                         "ci_low": pr["ci_low"], "ci_high": pr["ci_high"], "p_two_sided": pr["p_value"],
                         "p_one_sided_rule": pr["p_value"] if d > 0 else 1.0,
                         "region_ci_low": prr["ci_low"], "region_ci_high": prr["ci_high"],
                         "block50_ci": [pr50["ci_low"], pr50["ci_high"]],
                         "per_seed_delta": per_seed, "all_seeds_positive": all(v > 0 for v in per_seed.values()),
                         "region_signs_positive": int(np.sum(np.array(signs) > 0)), "n_regions": len(signs),
                         "realised_mde": HOLM_Z * (pr["ci_high"] - pr["ci_low"]) / 3.92})
    C = pd.DataFrame(conf)
    if len(C):
        adj = holm({(r.block, r.target): r.p_one_sided_rule for r in C.itertuples()})
        C["p_holm"] = [adj[(r.block, r.target)] for r in C.itertuples()]
        C["holm_rejected"] = C["p_holm"] < float(rec["alpha"])
        lab = [label(r) for _, r in C.iterrows()]
        C["label"], C["label_reason"] = [x[0] for x in lab], [x[1] for x in lab]
    return C, pd.DataFrame(marg)


def secondary(pred, truth, meta, rec) -> list[dict]:
    """Unadjusted secondary contrasts: other candidates vs comparator, C-X vs RF[geography], X-U by method."""
    from sentinel.models.bioassess import evaluate as EV

    out = []
    for b, models in pred.items():
        Y, mt = truth[b], meta[b]
        region, cluster = mt["region"].to_numpy(), mt["cluster"].to_numpy()
        cand = rec["candidates"][b]["confirmatory"]["name"]
        pairs = [(c["name"], rec["comparators"][b][t]["name"], t, "secondary candidate vs comparator")
                 for c in rec["candidates"][b]["secondary"] for t in XD.TARGETS[b]]
        pairs += [(cand, "rf_geography", t, "C-X vs RF[geography]") for t in XD.TARGETS[b]]
        for a_, b_, t, what in pairs:
            if a_ not in models or b_ not in models:
                continue
            y = Y[t].to_numpy(float)
            pa = (sum(models[a_].values()) / len(models[a_]))[t].to_numpy(float)
            pb = (sum(models[b_].values()) / len(models[b_]))[t].to_numpy(float)
            strata = {"all": np.ones(len(y), bool)}
            if b == "X-U":
                strata["ANLA"] = (mt["analysis_method"] == "ANLA").to_numpy()
                strata["non-ANLA"] = ~strata["ANLA"]
            for sname, m in strata.items():
                ok = m & np.isfinite(y) & np.isfinite(pa) & np.isfinite(pb)
                if ok.sum() < 20:
                    continue
                pr = EV.paired_vs(y[ok], pa[ok], pb[ok], region[ok], cluster[ok], metric="r2_anomaly",
                                  n_boot=int(rec["n_boot"]))
                out.append({"block": b, "what": what, "a": a_, "b": b_, "target": t, "stratum": sname,
                            "n": int(ok.sum()), "delta": pr["diff"], "ci_low": pr["ci_low"], "ci_high": pr["ci_high"],
                            "p_two_sided": pr["p_value"], "note": "secondary, unadjusted"})
    return out


# ------------------------------------------------------------------------------------------------- the pass
def seal(run_dir: Path) -> None:
    files = sorted(p for p in run_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    lines = [f"{file_sha256(p)}  {p.relative_to(run_dir)}" for p in files]
    (run_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    for p in files + [run_dir / "SHA256SUMS"]:
        p.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def setup(dry: bool, cpu: bool):
    import torch

    torch.set_num_threads(2)
    device = "cpu"
    if not (dry and cpu):
        from sentinel.core.gpu import acquire_gpu

        acquire_gpu("m5x_final" + ("_dryrun" if dry else ""), min_free_mb=5000)
        device = "cuda"
    from sentinel.models.chemtox.tabpfn_model import load_tabpfn

    load_tabpfn()
    S, frozen, D = XD.load_context()
    return device, S, frozen, D


def design_and_folds(dry, S, frozen, D, max_folds):
    if dry:
        folds, DD = XD.read_folds(XD.DEV_DIR / "folds_dev_v1.json", S, frozen), D
    else:
        Dts = {b: XD.load_test_rows(S, b) for b in XD.BLOCKS}          # requires the open claim (xtestlook)
        DD = D
        for b in XD.BLOCKS:
            DD = XD.combine(DD, Dts[b])
        folds = {b: XD.test_folds(S, frozen, D, Dts[b], b) for b in XD.BLOCKS}
    if max_folds:
        folds = {b: fl[:max_folds] for b, fl in folds.items()}
    for b in XD.BLOCKS:
        for fo in folds[b]:
            X.assert_fit_ids(XD.keys_of(fo.train), S, frozen)
    return DD, folds


def model_specs(rec, b) -> dict:
    specs = {c["name"]: c["spec"] for c in [rec["candidates"][b]["confirmatory"], *rec["candidates"][b]["secondary"]]}
    specs.update({c["name"]: c["spec"] for c in rec["comparators"][b].values()})
    specs["rf_geography"] = {"kind": "baseline", "name": "rf_geography", "variant": {}}
    for name, spec in (rec.get("event_track", {}).get("threshold_baselines", {}).get(b, {})).items():
        specs.setdefault(name, spec)                   # event track (v8): threshold-on-baseline models
    return specs


def predict_events(rec, ctx, folds, P, pred_dir: Path) -> None:
    """Event track (v8, secondary): per-fold prevalence/climatology and the direct challenger (powered events only),
    written with the other predictions so they are sealed together."""
    from sentinel.events import bioassess_x as XE

    D = ctx["D"]
    for (b, code) in XE.TRACK:
        if b not in folds:
            continue
        rows = []
        for fo in folds[b]:
            pv, cm = XE.fold_prevalence_climatology(D, b, code, fo)
            rows.append(pd.DataFrame({"prevalence": pv, "climatology": cm}, index=fo.test))
        out = pred_dir / b / f"event_clim_{code}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(rows).to_parquet(out)
        if (b, code) not in XE.POWERED:
            continue
        for s in rec["seeds"]:
            parts = []
            for fo in folds[b]:
                key = P.cache.key(b, "event_direct", code, fo.g, s)
                got = P.cache.get(key)
                if got is None:
                    X.assert_fit_ids(XD.keys_of(fo.train), ctx["S"], ctx["frozen"])
                    got = pd.DataFrame({"p": XE.fold_direct(D, P.TM, b, code, fo, s, P.n_jobs)}, index=fo.test)
                    P.cache.put(key, got)
                parts.append(got)
            pd.concat(parts).to_parquet(pred_dir / b / f"event_direct_{code}__s{s}.parquet")


def score_events(rec, ctx, folds, pred_dir: Path, out_dir: Path, *, split: str, reg, record_sha: str) -> dict:
    """Event track scoring from the sealed predictions + the frozen development calibration (secondary)."""
    from sentinel.events import bioassess_x as XE

    et = rec["event_track"]
    cf = Path(et["calibration_file"])
    if file_sha256(cf) != et["calibration_sha256"]:
        raise SystemExit("event calibration file changed since the freeze")
    calib = pd.read_parquet(cf)
    DD, out = ctx["D"], {}
    for (b, code), (t, _, _) in XE.TRACK.items():
        if b not in folds:
            continue
        ids = pd.Index(np.concatenate([fo.test for fo in folds[b]]))
        cl = DD.samples.loc[ids, "location_id"].astype(str).to_numpy()
        yt = DD.Y.loc[ids, t].to_numpy(float)
        ye = np.full(len(yt), np.nan)
        ye[np.isfinite(yt)] = XE.labels(yt[np.isfinite(yt)], b, code)
        cx = rec["candidates"][b]["confirmatory"]["name"]
        probs = {}
        for name, mname in [("C-X:" + cx, cx)] + [(m, m) for m in XE.COMPARATORS]:
            mu = sum(pd.read_parquet(pred_dir / b / f"{mname}__s{s}.parquet").reindex(ids)[t].to_numpy(float)
                     for s in rec["seeds"]) / len(rec["seeds"])
            c = calib[(calib["block"] == b) & (calib["event"] == code) & (calib["model"] == name)]
            raw = XE.derived_probability(mu, float(c["sd_all"].iloc[0]), b, code)
            probs[name] = XE.frozen_calibrated(c["p_raw"].to_numpy(float), c["y"].to_numpy(float), raw)
        clim = pd.read_parquet(pred_dir / b / f"event_clim_{code}.parquet").reindex(ids)
        probs["prevalence"], probs["climatology"] = clim["prevalence"].to_numpy(), clim["climatology"].to_numpy()
        if (b, code) in XE.POWERED:
            raw = sum(pd.read_parquet(pred_dir / b / f"event_direct_{code}__s{s}.parquet").reindex(ids)["p"].to_numpy(float)
                      for s in rec["seeds"]) / len(rec["seeds"])
            c = calib[(calib["block"] == b) & (calib["event"] == code) & (calib["model"] == "direct")]
            probs["direct"] = XE.frozen_calibrated(c["p_raw"].to_numpy(float), c["y"].to_numpy(float), raw)
        ok = np.isfinite(ye)
        pw = XE.power_flag(ye[ok], cl[ok])
        status = "descriptive only" if ((b, code) in XE.DESCRIPTIVE_ONLY or pw["flag"] != "F") else "secondary"
        cxn = "C-X:" + cx
        mets = {n: XE.metrics(ye[ok], p[ok], cl[ok], p_ref=probs["climatology"][ok]) for n, p in probs.items()}
        comps = {} if status == "descriptive only" else \
            {f"{cxn} - {n}": XE.compare(ye[ok], probs[cxn][ok], probs[n][ok], cl[ok]) for n in probs if n != cxn}
        out[f"{b}|{code}"] = {"power_test": pw, "status": status, "metrics": mets, "contrasts": comps}
        for n, v in mets.items():
            for k in ("auprc", "brier", "bss"):
                if k in v:
                    val, lo, hi = XE.metric_value(v, k)
                    safe_log(workstream="bioassess", experiment=EXPERIMENT, model=n, metric=f"event_{k}",
                             value=val, split=f"{b}_{split}", target=code, n=int(ok.sum()), registry_path=reg,
                             ci_low=lo, ci_high=hi,
                             ci_method="location-cluster bootstrap B=2000 (events.state_metrics)",
                             is_baseline=not n.startswith("C-X"),
                             notes=f"M5-X event track, SECONDARY ({status}); amendment v8",
                             extra={"power_test": pw, "record_sha256": record_sha})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "events.json").write_text(json.dumps(out, indent=1, default=lambda o: o.tolist()
                                                    if hasattr(o, "tolist") else str(o)))
    return out


def predict_all(rec, ctx, folds, run_dir, device, pred_dir: Path) -> None:
    """Phase 1: every unit (cached per unit, sha-verified), then one prediction file per (block, model, seed)."""
    P = Predictor(ctx, run_dir, rec["seeds"], device)
    for b in XD.BLOCKS:
        for name, spec in model_specs(rec, b).items():
            for s in rec["seeds"]:
                out = pred_dir / b / f"{name}__s{s}.parquet"
                out.parent.mkdir(parents=True, exist_ok=True)
                P.model(spec, b, folds[b], s).to_parquet(out)
    if rec.get("event_track", {}).get("status") == "active":
        predict_events(rec, ctx, folds, P, pred_dir)


def score_saved(rec, ctx, folds, pred_dir: Path, out_dir: Path, *, split: str, reg, record_sha: str) -> pd.DataFrame:
    """Phase 2: score the sealed predictions; one registry flush; write results."""
    DD = ctx["D"]
    models, truth, meta = {}, {}, {}
    for b in XD.BLOCKS:
        ids = pd.Index(np.concatenate([fo.test for fo in folds[b]]))
        truth[b] = pd.DataFrame({t: DD.Y.loc[ids, t].to_numpy(float) for t in XD.TARGETS[b]}, index=ids)
        s = DD.samples.loc[ids]
        meta[b] = pd.DataFrame({"region": s["ag_eco9"].astype(str), "cluster": s["location_id"].astype(str),
                                "lat": s["lat"], "lon": s["lon"],
                                "analysis_method": s.get("analysis_method", pd.Series(None, index=ids))}, index=ids)
        models[b] = {}
        for name in model_specs(rec, b):
            models[b][name] = {sd: pd.read_parquet(pred_dir / b / f"{name}__s{sd}.parquet").reindex(ids)
                               for sd in rec["seeds"]}
    C, M = score_blocks(models, truth, meta, rec)
    sec = secondary(models, truth, meta, rec)
    out_dir.mkdir(parents=True, exist_ok=True)
    C.to_json(out_dir / "confirmatory.json", orient="records", indent=1, default_handler=str)
    M.to_json(out_dir / "marginal.json", orient="records", indent=1, default_handler=str)
    (out_dir / "secondary.json").write_text(json.dumps(sec, indent=1, default=float))
    common = {"n_configurations": rec["n_configurations_tried"], "selection_note": rec["selection_note"],
              "record_sha256": record_sha}
    for r in M.itertuples():
        safe_log(workstream="bioassess", experiment=EXPERIMENT, model=r.model, metric="r2_anomaly",
                 value=r.r2_anomaly, split=f"{r.block}_{split}", target=r.target, n=r.n, ci_low=r.ci_low,
                 ci_high=r.ci_high, ci_method=f"percentile location-cluster bootstrap B={rec['n_boot']}",
                 is_baseline=not r.model.startswith(("bsx_", "ens_")), registry_path=reg,
                 notes="M5-X single pass (marginal)", extra={"r2": r.r2, "rmse": r.rmse, **common})
    for r in C.itertuples():
        safe_log(workstream="bioassess", experiment=EXPERIMENT, model=f"{r.candidate} - {r.comparator}",
                 metric="delta_r2_anomaly_confirmatory", value=r.delta, split=f"{r.block}_{split}", target=r.target,
                 n=r.n, ci_low=r.ci_low, ci_high=r.ci_high,
                 ci_method=f"paired location-cluster bootstrap B={rec['n_boot_confirmatory']}", registry_path=reg,
                 notes=f"M5-X confirmatory: {r.label}" + (f" ({r.label_reason})" if r.label_reason else ""),
                 extra={"p_holm": r.p_holm, "p_one_sided_rule": r.p_one_sided_rule, "per_seed_delta": r.per_seed_delta,
                        "region_ci": [r.region_ci_low, r.region_ci_high], "block50_ci": r.block50_ci,
                        "realised_mde": r.realised_mde, "region_signs_positive": r.region_signs_positive,
                        "n_regions": r.n_regions, **common})
    if rec.get("event_track", {}).get("status") == "active":
        score_events(rec, ctx, folds, pred_dir, out_dir, split=split, reg=reg, record_sha=record_sha)
    return C


def run_pass(rec: dict, *, dry: bool, run_dir: Path, authorised_by: str, max_folds: int | None, resume: bool = False,
             rescore: bool = False, cpu: bool = False) -> int:
    """Test pass (dry=False) or development dry run (dry=True: candidate record, dev folds, its own look directory
    and scratch registry inside ``run_dir``; the real look directory and registry are never touched)."""
    root = run_dir / "look" if dry else None
    reg = run_dir / "registry_dryrun.jsonl" if dry else None
    record_sha = file_sha256(CANDIDATE if dry else FREEZE)
    run_dir.mkdir(parents=True, exist_ok=True)
    device, S, frozen, D = setup(dry, cpu)                               # setup before any claim
    if rescore:
        pred_dir = XL.check_sealed(root=root)
    else:
        if resume:
            XL.claim_resume(authorised_by, run_dir, root=root, registry_path=reg)
        else:
            XL.claim({**rec, "_path": str(CANDIDATE if dry else FREEZE), "_sha256": record_sha}, authorised_by, run_dir,
                     root=root, registry_path=reg)
        pred_dir = run_dir / "predictions"
    DD, folds = design_and_folds(dry, S, frozen, D, max_folds)
    ctx = {"S": S, "frozen": frozen, "D": DD}
    if not rescore:
        predict_all(rec, ctx, folds, run_dir, device, pred_dir)
        XL.seal_predictions(pred_dir, root=root)
        pred_dir = XL.check_sealed(root=root)
    C = score_saved(rec, ctx, folds, pred_dir, run_dir / "results", split="dry_run_dev" if dry else "test", reg=reg,
                    record_sha=record_sha)
    info = {"record_sha256": record_sha, "dry_run": dry, "folds": {b: len(f) for b, f in folds.items()},
            "test_rows_loaded": DD.info if not dry else None, "finished": dt.datetime.now().isoformat(),
            "pretrain_sha256": {str(p.relative_to(run_dir)): file_sha256(p) for p in run_dir.rglob("encoder.pt")}}
    (run_dir / "results" / "pass_info.json").write_text(json.dumps(info, indent=1, default=str))
    seal(run_dir / "results")
    XL.mark_scored({"results": str(run_dir / "results"), "n_confirmatory": int(len(C))}, root=root)
    log("scored and sealed", run_dir)
    if len(C):
        log(C[["block", "target", "delta", "ci_low", "ci_high", "p_holm", "label"]].to_string())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--authorised-by")
    ap.add_argument("--dry-run-dev", action="store_true")
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--cpu", action="store_true", help="dry run only: no GPU lock (smoke runs without GPU models)")
    ap.add_argument("--resume", help="RUN_DIR of the claimed pass (pre-registered single resume)")
    ap.add_argument("--rescore-from-saved", help="RUN_DIR whose sealed predictions are scored (crash after sealing)")
    a = ap.parse_args()
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    if a.dry_run_dev:
        if a.resume or a.rescore_from_saved:
            raise SystemExit("resume / rescore are test-pass options")
        rec = json.loads(CANDIDATE.read_text())
        bad = verify_record(rec)
        if bad:
            raise SystemExit("candidate record verification failed: " + "; ".join(bad[:10]))
        XL.verify_data_pins(rec)
        run_dir = paths.RUNS_ROOT / "bioassess" / f"x_final_dryrun_dev_{stamp}"
        return run_pass(rec, dry=True, run_dir=run_dir, authorised_by="dry-run (development data)",
                        max_folds=a.max_folds, cpu=a.cpu)
    if a.max_folds or a.cpu:
        raise SystemExit("--max-folds / --cpu are for the development dry run only")
    if not a.authorised_by:
        raise SystemExit("--authorised-by is required for the test pass")
    try:
        fr = XL.load_freeze()                                   # FREEZE only, never the candidate
        XL.verify_data_pins(fr)
    except XL.SingleLookError as e:
        raise SystemExit(str(e)) from e
    bad = verify_record(fr)
    if bad:
        raise SystemExit("record verification failed: " + "; ".join(bad[:10]))
    if a.resume:
        return run_pass(fr, dry=False, run_dir=Path(a.resume), authorised_by=a.authorised_by, max_folds=None,
                        resume=True)
    if a.rescore_from_saved:
        return run_pass(fr, dry=False, run_dir=Path(a.rescore_from_saved), authorised_by=a.authorised_by,
                        max_folds=None, rescore=True)
    fr = verify_freeze()
    run_dir = paths.RUNS_ROOT / "bioassess" / f"x_final_{stamp}"
    return run_pass(fr, dry=False, run_dir=run_dir, authorised_by=a.authorised_by, max_folds=None)


if __name__ == "__main__":
    raise SystemExit(main())
