# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Single test scoring of SENTINEL-Lite (protocol §8): run ONCE, after FREEZE_lite.json exists and the lead has
authorised it. Original 4-state track: Test-A (`test`, headline) and Test-B (`test_time`).

    python -B scripts/eval/lite_score_test.py --allow-test            # the one real run (needs the freeze)
    python -B scripts/eval/lite_score_test.py --dry-run-val --freeze docs/preregistration/FREEZE_lite_candidate.json
        # code-path check on the VAL role with the candidate file: nothing registered, no test row loaded

The freeze file (keys beyond guard.FREEZE_KEYS):
- ``runs``: {primary, nokd, robust_variant, baselines} run directories;
- ``weights_sha256``: {primary_seed{0,1,2}, nokd_seed{0,1,2}, robust_variant_seed0}. Every checkpoint is verified
  before use;
- ``deployable``: "primary_seed0";
- ``band_ratio_forms``: the val-selected form per target;
- ``teacher_manifest``: the Lite-safe teacher, used for the red-edge gap.

What is computed, per eval role:
1. Primary-population MdSA (balanced accuracy for chl-a classes) with water-body cluster-bootstrap CIs, for the
   primary ensemble, each primary seed, the deployable seed-0 model, the no-KD ensemble, the robust variant, and
   every baseline (refitted on train exactly as in `lite_baselines.py`).
2. **H1** (Test-A): paired water-body bootstrap, primary ensemble vs band-ratio and vs GBM-RGBN for Secchi,
   turbidity and TSS. One-sided p, Holm over the 6 comparisons; a rejection counts only when the student is
   better.
3. **H2** (Test-A): primary ensemble vs no-KD ensemble, Holm over 3.
4. §15.1 seed rule: each primary seed vs each H1 baseline (sign of the gain per seed), and the 3-seed spread.
5. Secondary slices: unseen water body × 2022–2023, chips > 10 km from every train site, and by site class (via
   ``score_primary``).
6. Robustness grid (§7) for the deployable model, the primary ensemble and the robust variant (secondary).
7. Red-edge gap (descriptive): the Lite-safe teacher on S2 (all bands, context off) vs the primary ensemble on
   NAIP, on the same chips with an S2 chip within ±1 d.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentinel.models.lite import runner as R

R.cpu_threads(6)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from sentinel.eval.bootstrap import holm_bonferroni  # noqa: E402
from sentinel.eval.splits import nearest_distance_km  # noqa: E402
from sentinel.models.lite import augment as A  # noqa: E402
from sentinel.models.lite import baselines as B  # noqa: E402
from sentinel.models.lite import data as D  # noqa: E402
from sentinel.models.lite import evaluate as E  # noqa: E402
from sentinel.models.lite import features as F  # noqa: E402
from sentinel.models.lite import guard as G  # noqa: E402
from sentinel.models.lite.train import Inputs, build_model, predict  # noqa: E402

H1_BASELINES = ("band_ratio", "gbm_rgbn")


def feature_frame(lab: pd.DataFrame, name: str) -> pd.DataFrame:
    chips = np.asarray(D.cached_chips(lab, name))
    return F.chip_features(chips, lab["gsd_m"].to_numpy(), D.scene_norm_params(lab, D.scene_stats())[:, :, 0])


def baseline_preds(tr: pd.DataFrame, ev: pd.DataFrame, ev_name: str, forms: dict) -> dict:
    Ftr, Fev = feature_frame(tr, "train"), feature_frame(ev, ev_name)
    img = list(Ftr.columns)
    Ctr = pd.concat([Ftr, tr[["lat", "lon", "doy"]]], axis=1)
    Cev = pd.concat([Fev, ev[["lat", "lon", "doy"]]], axis=1)
    out = {}
    for t in D.TARGETS:
        ytr = tr[f"y_{t}"].to_numpy(float)
        P = {"band_ratio": B.band_ratio(Ftr, ytr, Fev, t, forms[t])[0],
             "gbm_rgbn": B.gbm(Ftr, ytr, Fev, cols=img)[0],
             "gbm_rgbn_ctx": B.gbm(Ctr, ytr, Cev, cols=img + ["lat", "lon", "doy"])[0],
             "loc_knn": B.location_knn(tr.lat, tr.lon, tr.site_id, ytr, ev.lat, ev.lon)[0],
             "train_median": np.full(len(ev), np.nanmedian(ytr))}
        out[t] = P
    return out


def load_models(run: Path, seeds, freeze: dict, key: str, device: str):
    ms = []
    for s in seeds:
        ck = run / f"model_seed{s}.pt"
        G.verify_weights(freeze, f"{key}_seed{s}", ck)
        c = torch.load(ck, map_location="cpu", weights_only=False)
        cfg = {**c["cfg"], "pretrained": False, "gloria_init": None, "amp": False}
        pj = run / "pool.json"
        if pj.exists():
            cfg["pool"] = json.loads(pj.read_text())["pool"]
        m = build_model(cfg)
        m.load_state_dict(c["state_dict"])
        ms.append((s, m.to(device).eval(), cfg))
    return ms


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--dry-run-val", action="store_true", help="score the val role with the candidate file")
    ap.add_argument("--freeze", default=None)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--n-boot", type=int, default=E.N_BOOT)
    ap.add_argument("--smoke", action="store_true", help="with --dry-run-val only: 200 val chips, 20 bootstraps")
    a = ap.parse_args(argv)
    if a.smoke and not a.dry_run_val:
        raise SystemExit("--smoke only with --dry-run-val")
    if a.smoke:
        a.n_boot = 20
    split = G.load_split()
    if a.dry_run_val:
        roles = ("val",)
        freeze = json.loads(Path(a.freeze).read_text())
        register = False
    else:
        roles = ("test", "test_time")
        acc = G.check_access(roles, allow_test=a.allow_test, split=split, freeze_file=a.freeze)
        freeze = acc["freeze"]
        register = True
    if a.device == "cuda":
        R.setup_gpu("lite_score_test")
    runs = {k: Path(v) for k, v in freeze["runs"].items()}
    ev_all = D.role_frame(split, roles, allow_test=not a.dry_run_val, training=False, freeze_file=a.freeze)
    tr = D.role_frame(split, ("train",), training=True)
    out_dir = R.run_directory("test_scoring" if register else "dryrun_test_scoring", smoke=not register)
    prim = load_models(runs["primary"], (0, 1, 2), freeze, "primary", a.device)
    nokd = load_models(runs["nokd"], (0, 1, 2), freeze, "nokd", a.device)
    robust = load_models(runs["robust_variant"], (0,), freeze, "robust_variant", a.device)
    results = {"freeze_sha256": freeze.get("_sha256"), "split_sha256": split["_sha256"], "roles": list(roles),
               "dry_run_val": a.dry_run_val, "by_role": {}}
    rows = []
    for role in roles:
        ev = ev_all[ev_all["role"] == role].reset_index(drop=True)
        cname = f"eval_{role}"
        if a.smoke:
            ev = ev.head(200).reset_index(drop=True)
            cname = f"smoke_{role}"
        chips = np.asarray(D.cached_chips(ev, cname))
        inp = Inputs(chips, D.scene_norm_params(ev, D.scene_stats()), F.pooling_weights(chips, ev["gsd_m"].to_numpy(), 4),
                     None, a.device)
        y = ev[[f"y_{t}" for t in D.TARGETS]].to_numpy(float)

        def ens(models, pert=None):
            return np.mean([predict(m, inp, c, perturb=pert)[0] for _, m, c in models], 0)

        P = {f"primary_seed{s}": predict(m, inp, c)[0] for s, m, c in prim}
        P["primary_ensemble"] = np.mean([P[f"primary_seed{s}"] for s, _, _ in prim], 0)
        P["nokd_ensemble"] = ens(nokd)
        P["robust_variant_seed0"] = ens(robust)
        bp = baseline_preds(tr, ev, cname, freeze["band_ratio_forms"])
        res = {"n_chips": int(len(ev)), "scores": {}, "h1": {}, "h2": {}, "seed_rule": {}, "slices": {}}
        names = list(P) + list(bp[D.TARGETS[0]])
        for t_i, t in enumerate(D.TARGETS):
            for nm in names:
                p = P[nm][:, t_i] if nm in P else bp[t][nm]
                s = E.score_primary(t, y[:, t_i], p, ev["water_body"], ev["site_class"], n_boot=a.n_boot)
                res["scores"][f"{nm}/{t}"] = s
                rows.append(dict(role=role, model=nm, target=t, **{k: s[k] for k in ("value", "ci_low", "ci_high", "n")}))
        # H1 / H2 / seed rule on the primary populations
        pv1, pv2 = {}, {}
        for t_i, t in enumerate(D.REG_TARGETS):
            j = D.TARGETS.index(t)
            pm = E.primary_mask(t, ev["site_class"])
            hib = E.primary_metric(t)[2]
            for b in H1_BASELINES:
                r = E.compare(t, y[pm, j], P["primary_ensemble"][pm, j], bp[t][b][pm], ev.loc[pm, "water_body"],
                              n_boot=a.n_boot)
                better = r["diff"] > 0 if hib else r["diff"] < 0
                r["p_one_sided"] = r["p_value"] / 2 if better else 1 - r["p_value"] / 2
                res["h1"][f"{b}/{t}"] = r
                pv1[f"{b}/{t}"] = r["p_one_sided"]
                res["seed_rule"][f"{b}/{t}"] = {
                    f"seed{s}": E.compare(t, y[pm, j], P[f"primary_seed{s}"][pm, j], bp[t][b][pm],
                                          ev.loc[pm, "water_body"], n_boot=a.n_boot)["diff"] for s, _, _ in prim}
            r = E.compare(t, y[pm, j], P["primary_ensemble"][pm, j], P["nokd_ensemble"][pm, j],
                          ev.loc[pm, "water_body"], n_boot=a.n_boot)
            better = r["diff"] > 0 if hib else r["diff"] < 0
            r["p_one_sided"] = r["p_value"] / 2 if better else 1 - r["p_value"] / 2
            res["h2"][t] = r
            pv2[t] = r["p_one_sided"]
        for fam, pv in (("h1", pv1), ("h2", pv2)):
            holm = holm_bonferroni(pv)
            for k in pv:
                res[fam][k]["holm_reject"] = bool(holm[k]) and res[fam][k]["p_one_sided"] < 0.5
        for k, v in res["seed_rule"].items():
            v["all_seeds_better"] = all(d < 0 for d in v.values() if isinstance(d, float))
        # secondary slices
        trs = tr.drop_duplicates("site_id")
        far = nearest_distance_km(ev["lat"], ev["lon"], trs["lat"], trs["lon"]) > 10.0
        slices = {"gt10km_from_train": far}
        if role == "test":
            slices["future_years"] = ev["naip_year"].astype(int).to_numpy() >= 2022
        for sn, msk in slices.items():
            msk = np.asarray(msk, bool)
            if msk.sum() < 10:
                continue
            res["slices"][sn] = {f"{nm}/{t}": E.score_primary(t, y[msk, j], (P[nm][:, j] if nm in P else bp[t][nm])[msk],
                                                              ev.loc[msk, "water_body"], ev.loc[msk, "site_class"],
                                                              n_boot=a.n_boot)["value"]
                                 for j, t in enumerate(D.TARGETS)
                                 for nm in ("primary_ensemble", "primary_seed0", "band_ratio", "gbm_rgbn")}
        # robustness (§7), primary populations
        yr = y.copy()
        for j, t in enumerate(D.TARGETS):
            yr[~E.primary_mask(t, ev["site_class"]), j] = np.nan
        rob = {}
        for label, models in (("deployable_seed0", prim[:1]), ("primary_ensemble", prim), ("robust_variant_seed0", robust)):
            rr = {}
            for kind, level in [("clean", 0)] + A.ROBUSTNESS_GRID:
                draws = range(A.GAIN_DRAWS) if kind == "gain" else range(1)
                mus = [ens(models, lambda x, k=kind, l=level, s=s: A.perturb(x, k, l, seed=s)) for s in draws]
                rr[f"{kind}:{level}"] = {t: float(np.mean([E.primary_metric(t)[0](yr[np.isfinite(yr[:, j]), j],
                                                                                   mu[np.isfinite(yr[:, j]), j])
                                                           for mu in mus])) for j, t in enumerate(D.TARGETS)}
            rob[label] = rr
        res["robustness"] = rob
        # red-edge gap (descriptive)
        try:
            from sentinel.models.orbit.teacher import OrbitTeacher

            man = json.loads(Path(freeze["teacher_manifest"]).read_text())
            for c, sha in zip(man["checkpoints"], man["sha256"]):
                if G.artifact_sha256(c) != sha:
                    raise RuntimeError("teacher sha mismatch")
            has = ev["s2_chip_id"].notna().to_numpy()
            if has.any():
                tch = OrbitTeacher(man["checkpoints"], device=a.device)
                tp = tch.predict_store(D.teacher_store().root, chip_ids=ev.loc[has, "s2_chip_id"].unique().tolist(),
                                       workers=2).set_index("chip_id")
                gap = {}
                for j, t in enumerate(D.REG_TARGETS):
                    col = f"imgonly_{t}_log10_q50"
                    pt = ev["s2_chip_id"].map(tp[col]) if col in tp else pd.Series(np.nan, index=ev.index)
                    m = has & np.isfinite(pt.to_numpy(float)) & np.isfinite(y[:, D.TARGETS.index(t)]) \
                        & E.primary_mask(t, ev["site_class"])
                    if m.sum() >= 10:
                        fn = E.primary_metric(t)[0]
                        gap[t] = {"n": int(m.sum()), "teacher_s2": fn(y[m, D.TARGETS.index(t)], pt.to_numpy(float)[m]),
                                  "student_naip": fn(y[m, D.TARGETS.index(t)], P["primary_ensemble"][m, D.TARGETS.index(t)])}
                res["red_edge_gap"] = gap
        except Exception as e:  # descriptive only; never blocks the primary results
            res["red_edge_gap"] = {"error": repr(e)}
        results["by_role"][role] = res
        R.save_predictions(out_dir / f"pred_{role}.parquet", ev, {f"{nm}:{t}": P[nm][:, j] for nm in P
                                                                  for j, t in enumerate(D.TARGETS)})
    R.write_json(out_dir / "test_results.json", results)
    pd.DataFrame(rows).to_csv(out_dir / "scores.csv", index=False)
    if register:
        from sentinel.core.registry import log_result

        for r in rows:
            if np.isfinite(r["value"]):
                log_result(workstream=R.WORKSTREAM, experiment="lite_test", model=f"lite_{r['model']}",
                           metric=E.primary_metric(r["target"])[1], value=r["value"], ci_low=r["ci_low"],
                           ci_high=r["ci_high"], ci_method="cluster bootstrap over water bodies (B=2000)",
                           split=r["role"], n=r["n"], target=r["target"], is_baseline=not r["model"].startswith(
                               ("primary", "nokd", "robust")), data_manifests=list(R.DATA_MANIFESTS),
                           artifacts=[str(out_dir)], extra={"freeze_sha256": freeze.get("_sha256")})
        for role, res in results["by_role"].items():
            for fam in ("h1", "h2"):
                for k, r in res[fam].items():
                    log_result(workstream=R.WORKSTREAM, experiment=f"lite_test_{fam}", model="lite_primary_ensemble",
                               metric=f"{r['metric']}_diff_vs_{k.split('/')[0] if fam == 'h1' else 'nokd'}",
                               value=r["diff"], ci_low=r["ci_low"], ci_high=r["ci_high"],
                               ci_method="paired cluster bootstrap over water bodies (B=2000)", split=role, n=r["n"],
                               target=k.split("/")[-1], data_manifests=list(R.DATA_MANIFESTS), artifacts=[str(out_dir)],
                               extra={"p_one_sided": r["p_one_sided"], "holm_reject": r["holm_reject"]})
    print(json.dumps({role: {k: round(v["value"], 2) for k, v in res["scores"].items()
                             if k.split("/")[0] in ("primary_ensemble", "primary_seed0", "band_ratio", "gbm_rgbn")}
                      for role, res in results["by_role"].items()}, indent=1))
    print("written:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
