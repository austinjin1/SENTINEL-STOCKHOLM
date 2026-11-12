# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Single test look of SENTINEL-Lite, NATIONAL track (lite_protocol.md §12; lite_national_protocol.md N1–N6).
Generated from the frozen 4-state scorer ``lite_score_test.py`` (unchanged). The differences are:
- the national-2 split and loaders;
- Test-N (``test_n``, confirmatory) and ``test_time_n`` (descriptive);
- merged DOQQ scene stats;
- registry experiments ``lite_national_test*``;
- ``national_testlook`` (its own look, freeze and code hash);
- the national extras: N1 strata, N2 per-target n flags, and N3 events (E3a_turb with one pre-declared endpoint;
  E4a and E3a_tss descriptive).

Run ONCE after the lead writes ``docs/preregistration/FREEZE_lite_national.json`` and authorises it:
    /home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/eval/national_lite_score_test.py --allow-test
Dry run (val + val_n, nothing claimed or registered):
    ... national_lite_score_test.py --dry-run-val --candidate docs/preregistration/FREEZE_lite_national_candidate.json
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

from sentinel.eval.splits import nearest_distance_km  # noqa: E402
from sentinel.models.lite import augment as A  # noqa: E402
from sentinel.models.lite import baselines as B  # noqa: E402
from sentinel.models.lite import data as D  # noqa: E402
from sentinel.models.lite import evaluate as E  # noqa: E402
from sentinel.models.lite import export as X  # noqa: E402
from sentinel.models.lite import features as F  # noqa: E402
from sentinel.models.lite import guard as G  # noqa: E402
from sentinel.models.litev2 import tests_mask as TM  # noqa: E402
from sentinel.models.lite import national_testlook as T  # noqa: E402
from sentinel.models.lite import national2 as N2  # noqa: E402
from sentinel.models.lite.train import Inputs, build_model, predict  # noqa: E402

H1_BASELINES = ("band_ratio", "gbm_rgbn")
ALL_BASELINES = ("band_ratio", "gbm_rgbn", "gbm_rgbn_ctx", "loc_knn", "loc_season_gbm", "train_median")
ENSEMBLES = {"primary": "primary_ensemble", "nokd": "nokd_ensemble", "rgb_ablation": "rgb_ablation_ensemble",
             "gloria_ablation": "gloria_ablation_ensemble", "gloria_base": "gloria_base_ensemble"}
CARRIER = "deployable_int8_seed0"
ROB_MODELS = ("primary_seed0", "primary_ensemble", "robust_variant_seed0")


def col(model: str, target: str) -> str:
    return f"pred__{model}__{target}"


# ===================================================================================== phase 1: predictions
def feature_frame(lab: pd.DataFrame, name: str) -> pd.DataFrame:
    chips = np.asarray(D.cached_chips(lab, name))
    return F.chip_features(chips, lab["gsd_m"].to_numpy(), D.scene_norm_params(lab, D.scene_stats(N2.scene_stats_path()))[:, :, 0])


def baseline_preds(tr: pd.DataFrame, ev: pd.DataFrame, ev_name: str, forms: dict) -> dict:
    Ftr, Fev = feature_frame(tr, "train"), feature_frame(ev, ev_name)
    img = list(Ftr.columns)
    Ctr = pd.concat([Ftr, tr[["lat", "lon", "doy"]]], axis=1)
    Cev = pd.concat([Fev, ev[["lat", "lon", "doy"]]], axis=1)
    out = {}
    for t in D.TARGETS:
        ytr = tr[f"y_{t}"].to_numpy(float)
        out[("band_ratio", t)] = B.band_ratio(Ftr, ytr, Fev, t, forms[t])[0]
        out[("gbm_rgbn", t)] = B.gbm(Ftr, ytr, Fev, cols=img)[0]
        out[("gbm_rgbn_ctx", t)] = B.gbm(Ctr, ytr, Cev, cols=img + ["lat", "lon", "doy"])[0]
        out[("loc_knn", t)] = B.location_knn(tr.lat, tr.lon, tr.site_id, ytr, ev.lat, ev.lon)[0]
        out[("loc_season_gbm", t)] = B.location_season_gbm(Ctr, ytr, Cev)[0]
        out[("train_median", t)] = np.full(len(ev), np.nanmedian(ytr))
    return out


def load_torch(freeze: dict, name: str, device: str):
    spec = freeze["models"][name]
    run = Path(spec["run"])
    pj = run / "pool.json"
    pool = None
    if pj.exists():                                   # review W4: pool.json changes predictions -> must be hashed
        G.verify_weights(freeze, f"{name}_pool_json", pj)
        pool = json.loads(pj.read_text())["pool"]
    ms = []
    for s in spec["seeds"]:
        ck = run / f"model_seed{s}.pt"
        G.verify_weights(freeze, f"{name}_seed{s}", ck)
        c = torch.load(ck, map_location="cpu", weights_only=False)
        cfg = {**c["cfg"], "pretrained": False, "gloria_init": None, "amp": False}
        if pool:
            cfg["pool"] = pool
        m = build_model(cfg)
        m.load_state_dict(c["state_dict"])
        ms.append((s, m.to(device).eval(), cfg))
    return ms


def centre_crop_np(chips: np.ndarray, pw: np.ndarray, crop: int):
    if not crop or crop >= chips.shape[-1]:
        return chips, pw
    c0 = (chips.shape[-1] - crop) // 2
    oc = np.ascontiguousarray(chips[..., c0:c0 + crop, c0:c0 + crop])
    ow = pw[:, c0 // 4:(c0 + crop) // 4, c0 // 4:(c0 + crop) // 4]
    return oc, ow / np.clip(ow.sum((1, 2), keepdims=True), 1e-8, None)


def predict_role(freeze: dict, ev: pd.DataFrame, tr: pd.DataFrame, cname: str, device: str, log=print) -> tuple:
    chips = np.asarray(D.cached_chips(ev, cname))
    pw = F.pooling_weights(chips, ev["gsd_m"].to_numpy(), 4)
    inp = Inputs(chips, D.scene_norm_params(ev, D.scene_stats(N2.scene_stats_path())), pw, None, device)
    keep = ["naip_chip_id", "water_body", "site_id", "site_class", "naip_year", "lat", "lon", "s2_chip_id", "role",
            "hylak_id"]
    out = ev[keep + [f"y_{t}" for t in D.TARGETS]].copy()
    trs = tr.drop_duplicates("site_id")
    out["gt10km_from_train"] = nearest_distance_km(ev["lat"], ev["lon"], trs["lat"], trs["lon"]) > 10.0
    # review W8: water bodies with no Lite-safe teacher row within 5 km (coordinates only)
    if T.sha_file(Path(freeze["teacher_manifest"])) != freeze["teacher_manifest_sha256"]:
        raise G.ProtectedDataError("teacher manifest sha256 differs from the freeze")
    man = json.loads(Path(freeze["teacher_manifest"]).read_text())
    if G.artifact_sha256(man["seen_points"]) != man["seen_points_sha256"]:
        raise G.ProtectedDataError("teacher seen_points sha mismatch")
    seen = pd.read_parquet(man["seen_points"]).drop_duplicates(["lat", "lon"])
    dk = pd.Series(nearest_distance_km(ev["lat"], ev["lon"], seen["lat"], seen["lon"]), index=ev.index)
    out["teacher_gt5km_wb"] = (dk.groupby(ev["water_body"]).transform("min") > 5.0).to_numpy()
    preds, not_run = {}, {}
    loaded = {}
    for name, spec in freeze["models"].items():
        if spec.get("kind") != "torch":
            continue
        if not spec.get("run") or not Path(spec["run"]).is_dir():
            not_run[name] = spec.get("not_run_reason", "run missing")
            continue
        ms = load_torch(freeze, name, device)
        loaded[name] = ms
        mus = []
        for s, m, cfg in ms:
            mu, sd = predict(m, inp, cfg)
            mus.append(mu)
            for j, t in enumerate(D.TARGETS):
                preds[col(f"{name}_seed{s}", t)] = mu[:, j]
                preds[f"sig__{name}_seed{s}__{t}"] = sd[:, j]
        if name in ENSEMBLES:
            for j, t in enumerate(D.TARGETS):
                preds[col(ENSEMBLES[name], t)] = np.mean(mus, 0)[:, j]
        log(f"[predict] {name}: {len(ms)} seeds")
    # ONNX deployable (review C5): hash-checked, scored on the training crop
    for name, spec in freeze["models"].items():
        if spec.get("kind") != "onnx":
            continue
        G.verify_weights(freeze, name, spec["path"])
        oc, ow = centre_crop_np(chips, pw, int(spec.get("crop", 0)))
        mu = X.onnx_predict(spec["path"], oc, ow, len(D.TARGETS), threads=6)
        for j, t in enumerate(D.TARGETS):
            preds[col(name, t)] = mu[:, j]
        log(f"[predict] {name} (onnx)")
    for (b, t), p in baseline_preds(tr, ev, cname, freeze["band_ratio_forms"]).items():
        preds[col(b, t)] = p
    # robustness grid (§7)
    rob_sets = {"primary_seed0": loaded["primary"][:1], "primary_ensemble": loaded["primary"],
                "robust_variant_seed0": loaded.get("robust_variant", [])}
    for label, ms in rob_sets.items():
        if not ms:
            not_run[f"robustness/{label}"] = "model missing"
            continue
        for kind, level in A.ROBUSTNESS_GRID:
            for d in range(A.GAIN_DRAWS if kind == "gain" else 1):
                mu = np.mean([predict(m, inp, c, perturb=lambda x, k=kind, l=level, s=d: A.perturb(x, k, l, seed=s))[0]
                              for _, m, c in ms], 0)
                for j, t in enumerate(D.TARGETS):
                    preds[f"rob__{label}__{kind}:{level}__{d}__{t}"] = mu[:, j]
        log(f"[predict] robustness {label}")
    # red-edge gap: Lite-safe teacher, all bands, context off (review W4: manifest hash checked against the freeze)
    try:
        from sentinel.models.orbit.teacher import OrbitTeacher

        mp = Path(freeze["teacher_manifest"])
        if T.sha_file(mp) != freeze["teacher_manifest_sha256"]:
            raise G.ProtectedDataError("teacher manifest sha256 differs from the freeze")
        man = json.loads(mp.read_text())
        for c, sha in zip(man["checkpoints"], man["sha256"]):
            if G.artifact_sha256(c) != sha:
                raise G.ProtectedDataError("teacher checkpoint sha mismatch")
        # lead ruling 2026-11-10 (conflict C2): no Sentinel-2-input model is scored on a row of an M2 test-role water
        # body dated <= 2023-12-31 until M2S/test is sealed. Such rows are dropped by keys; a hard failure otherwise.
        blocked = TM.teacher_score_blocked(ev)
        out["teacher_blocked_m2_test_wb"] = blocked
        ids = TM.teacher_chip_ids(ev, blocked)
        TM.assert_teacher_scorable(ev, ids, blocked)
        log(f"[predict] teacher: {int(blocked.sum())} rows on M2 test-role water bodies not scored; {len(ids)} chips")
        if ids:
            tch = OrbitTeacher(man["checkpoints"], device=device)
            tp = tch.predict_store(D.teacher_store().root, chip_ids=ids, workers=2).set_index("chip_id")
            s2_ok = ev["s2_chip_id"].where(~blocked)
            for t in D.REG_TARGETS:
                c_ = f"imgonly_{t}_log10_q50"
                preds[col("teacher_s2_imgonly", t)] = s2_ok.map(tp[c_]).to_numpy(float) if c_ in tp \
                    else np.full(len(ev), np.nan)
    except TM.MaskError:     # never swallowed: a blocked row reached the teacher step
        raise
    except Exception as e:   # descriptive only
        not_run["red_edge_gap"] = repr(e)
    if freeze.get("events_national"):          # N3: climatology raw P for E3a_turb from train sites only
        try:
            from sentinel.events import lite as EL

            G.verify_weights(freeze, "events_national_calibrators", freeze["events_national"]["path"])
            preds["evraw__climatology__E3a_turb"] = EL.climatology_p(tr, EL.labels(tr["y_turbidity"], "E3a_turb"),
                                                                    ev["lat"], ev["lon"])
        except Exception as e:
            not_run["events_national_climatology"] = f"not run: {e!r}"
    if freeze.get("events"):
        try:
            preds.update(event_predictions(freeze, ev, tr, inp, loaded["primary"], preds))
        except Exception as e:   # D2: a secondary-track failure never burns the look; H1/H2/H-G6 still complete
            not_run["events"] = f"not run: {e!r}"
    out = pd.concat([out.reset_index(drop=True), pd.DataFrame(preds)], axis=1)
    return out, not_run


# ===================================================================================== event track (protocol §13)
def event_predictions(freeze: dict, ev: pd.DataFrame, tr: pd.DataFrame, inp, primary, preds: dict) -> dict:
    """Raw and frozen-calibrated P for every E4a route plus descriptive tiers / E3 (route i). All fitted
    artifacts (heads, comparator residual SDs, calibrators) come from the hashed event files in the freeze."""
    from sentinel.events import lite as EL

    evf = freeze["events"]
    if T.sha_file(Path(EL.EV().__file__)) != evf["events_module_sha256"]:
        raise G.ProtectedDataError("sentinel/eval/events.py differs from the released version in the freeze")
    art = {}
    for k in ("heads", "comparators", "calibrators"):
        G.verify_weights(freeze, f"events_{k}", evf[k])
        art[k] = json.loads(Path(evf[k]).read_text())
    seeds = [s for s, _, _ in primary]
    mu = lambda t: [preds[col(f"primary_seed{s}", t)] for s in seeds]  # noqa: E731
    sg = lambda t: [preds[f"sig__primary_seed{s}__{t}"] for s in seeds]  # noqa: E731
    raw = {"derived": EL.derived_p_ensemble(mu("chla"), sg("chla"), "E4a"),
           "direct": np.mean([EL.head_p(EL.pooled_embedding(m, inp, c), art["heads"][str(s)]) for s, m, c in primary], 0),
           "band_ratio": EL.derived_p(preds[col("band_ratio", "chla")],
                                      np.full(len(ev), art["comparators"]["band_ratio"]["resid_sd"]), "E4a"),
           "gbm_rgbn": EL.derived_p(preds[col("gbm_rgbn", "chla")],
                                    np.full(len(ev), art["comparators"]["gbm_rgbn"]["resid_sd"]), "E4a"),
           "climatology": EL.climatology_p(tr, EL.labels(tr["y_chla"], "E4a"), ev["lat"], ev["lon"])}
    out = {}
    for r, p in raw.items():
        out[f"evraw__{r}__E4a"] = p
        out[f"evcal__{r}__E4a"] = EL.calibrator_from_dict(art["calibrators"][r]).transform(p)
    for code in EL.E4A["tiers"]:
        out[f"evraw__derived__{code}"] = EL.derived_p_ensemble(mu("chla"), sg("chla"), code)
    for e in EL.E3:
        out[f"evraw__derived__{e['code']}"] = EL.derived_p_ensemble(mu(e["target"]), sg(e["target"]), e["code"])
    return out


def score_events(df: pd.DataFrame, confirmatory: bool, n_boot: int) -> dict:
    """E4a (lakes), DESCRIPTIVE ONLY (protocol §13 v1.7): state metrics per route (frozen-calibrated P) with CIs and
    the four paired comparisons derived − {climatology, band-ratio, direct, GBM} (ΔBrier / ΔAUPRC with CIs). No
    decision boolean and no claim depends on E4a. Tiers and E3: descriptive (raw route-(i) P). ``confirmatory`` is
    only recorded (Test-A vs Test-B)."""
    from sentinel.events import lite as EL

    ev = EL.EV()
    if "evcal__derived__E4a" not in df:
        return {"not_run": "event predictions missing"}
    y = EL.labels(df["y_chla"], "E4a")
    pop = E.population_mask(df["site_class"], "lake") & np.isfinite(y)
    g = df.loc[pop, "water_body"].to_numpy()
    res = {"E4a": {"n": int(pop.sum()), "n_pos": int(np.nansum(y[pop])),
                   "n_wb_pos": int(df.loc[pop & (y == 1), "water_body"].nunique()), "confirmatory": confirmatory}}
    for r in ("derived", "direct", "band_ratio", "gbm_rgbn", "climatology"):
        res["E4a"][r] = ev.state_metrics(y[pop], df.loc[pop, f"evcal__{r}__E4a"].to_numpy(), groups=g, n_boot=n_boot)
    for c in ("direct", "band_ratio", "gbm_rgbn", "climatology"):
        res["E4a"][f"derived_vs_{c}"] = ev.compare_routes(y[pop], df.loc[pop, "evcal__derived__E4a"].to_numpy(),
                                                         df.loc[pop, f"evcal__{c}__E4a"].to_numpy(), groups=g,
                                                         n_boot=n_boot)
    res["E4a"]["power_flag"] = ev.power_flag(res["E4a"]["n_pos"], res["E4a"]["n_wb_pos"])
    desc = {}
    for code in EL.E4A["tiers"]:
        yt = EL.labels(df["y_chla"], code)
        m = E.population_mask(df["site_class"], "lake") & np.isfinite(yt)
        if 0 < np.nansum(yt[m]) < m.sum():
            desc[code] = ev.state_metrics(yt[m], df.loc[m, f"evraw__derived__{code}"].to_numpy(),
                                          groups=df.loc[m, "water_body"].to_numpy(), n_boot=n_boot)
    for e in EL.E3:
        ye = EL.labels(df[f"y_{e['target']}"], e["code"])
        m = np.isfinite(ye)
        desc[e["code"]] = {"n_pos": int(np.nansum(ye[m])), "n_wb_pos": int(df.loc[m & (ye == 1), "water_body"].nunique())}
        if 0 < np.nansum(ye[m]) < m.sum():
            desc[e["code"]]["metrics"] = ev.state_metrics(ye[m], df.loc[m, f"evraw__derived__{e['code']}"].to_numpy(),
                                                          groups=df.loc[m, "water_body"].to_numpy(), n_boot=n_boot)
    res["descriptive"] = desc
    return res


# ===================================================================================== phase 2: scoring
def pmask(df: pd.DataFrame, t: str) -> np.ndarray:
    return E.primary_mask(t, df["site_class"]) & df[f"y_{t}"].notna().to_numpy()


def cmp(df, t, a, b, n_boot):
    m = pmask(df, t) & df[col(a, t)].notna().to_numpy() & df[col(b, t)].notna().to_numpy()
    return E.compare_one_sided(t, df.loc[m, f"y_{t}"], df.loc[m, col(a, t)], df.loc[m, col(b, t)],
                               df.loc[m, "water_body"], n_boot=n_boot)


def composite(rej: dict, key) -> bool:
    """Secchi AND (turbidity OR TSS); ``rej`` maps target -> bool."""
    return bool(rej.get("secchi")) and (bool(rej.get("turbidity")) or bool(rej.get("tss")))


def score_role(df: pd.DataFrame, freeze: dict, confirmatory: bool, n_boot: int) -> dict:
    models = sorted({c.split("__")[1] for c in df.columns if c.startswith("pred__")})
    res = {"n_chips": int(len(df)), "scores": {}, "lake_only": {}, "secondary_comparisons": {}, "slices": {}}
    for m in models:
        for t in D.TARGETS:
            c = col(m, t)
            if c not in df or df[c].notna().sum() == 0:
                continue
            res["scores"][f"{m}/{t}"] = E.score_primary(t, df[f"y_{t}"], df[c], df["water_body"], df["site_class"],
                                                        n_boot=n_boot)
            if t in ("turbidity", "tss"):
                lk = (df["site_class"] == "lake_reservoir").to_numpy() & df[f"y_{t}"].notna().to_numpy() & df[c].notna().to_numpy()
                if lk.sum() >= 5:
                    res["lake_only"][f"{m}/{t}"] = E.score(t, df.loc[lk, f"y_{t}"], df.loc[lk, c], df.loc[lk, "water_body"],
                                                           n_boot=n_boot)
    have = lambda m: all(col(m, t) in df for t in D.REG_TARGETS)  # noqa: E731
    fam = {}
    # H1 (ensemble) and H1-deployable (INT8 carrier) families; H2
    for fname, student in (("h1", "primary_ensemble"), ("h1_deployable", CARRIER)):
        if not have(student):
            fam[fname] = {"not_run": f"{student} missing"}
            continue
        rr = {f"{b}/{t}": cmp(df, t, student, b, n_boot) for t in D.REG_TARGETS for b in H1_BASELINES}
        if confirmatory:
            E.holm_with_ci(rr)
            rej = {t: all(rr[f"{b}/{t}"]["reject"] for b in H1_BASELINES) for t in D.REG_TARGETS}
            seeds = {k: {f"seed{s}": cmp(df, k.split("/")[1], f"primary_seed{s}", k.split("/")[0], 0)["diff"]
                         for s in freeze["models"]["primary"]["seeds"]} for k in rr}
            for k, v in seeds.items():
                v["all_seeds_better"] = all(x < 0 for x in v.values())
            seed_ok = {t: all(seeds[f"{b}/{t}"]["all_seeds_better"] for b in H1_BASELINES) for t in D.REG_TARGETS}
            fam[fname] = {"comparisons": rr, "rejected_by_target": rej, "seed_rule": seeds,
                          "verdict": composite(rej, None),
                          "verdict_with_seed_rule": composite({t: rej[t] and seed_ok[t] for t in rej}, None)}
        else:
            fam[fname] = {"comparisons": rr, "descriptive": True}
    if have("primary_ensemble") and have("nokd_ensemble"):
        rr = {t: cmp(df, t, "primary_ensemble", "nokd_ensemble", n_boot) for t in D.REG_TARGETS}
        if confirmatory:
            E.holm_with_ci(rr)
            per_seed = {t: {f"seed{s}": cmp(df, t, f"primary_seed{s}", f"nokd_seed{s}", 0)["diff"]
                            for s in freeze["models"]["primary"]["seeds"]} for t in D.REG_TARGETS}
            every = {t: all(v < 0 for v in per_seed[t].values()) for t in D.REG_TARGETS}
            rej = {t: rr[t]["reject"] for t in D.REG_TARGETS}
            fam["h2"] = {"comparisons": rr, "per_seed": per_seed, "every_seed": every, "verdict": composite(rej, None),
                         "verdict_with_every_seed": composite({t: rej[t] and every[t] for t in rej}, None)}
        else:
            fam["h2"] = {"comparisons": rr, "descriptive": True}
    else:
        fam["h2"] = {"not_run": "nokd ensemble missing"}
    res["families"] = fam
    # secondary paired comparisons (no Holm): ensemble and carrier vs every baseline; ablations; robust variant
    sc = res["secondary_comparisons"]
    for student in ("primary_ensemble", CARRIER):
        if have(student):
            for b in ALL_BASELINES:
                for t in D.REG_TARGETS:
                    sc[f"{student}_vs_{b}/{t}"] = cmp(df, t, student, b, n_boot)
    for a_, b_, tag in (("rgb_ablation_ensemble", "primary_ensemble", "rgb_only_vs_rgbn"),
                        ("gloria_ablation_ensemble", "gloria_base_ensemble", "gloria_vs_none"),
                        ("robust_variant_seed0", "primary_seed0", "robust_vs_primary_clean"),
                        (CARRIER, "deployable_fp32_seed0", "int8_vs_fp32")):
        if have(a_) and have(b_):
            for t in D.REG_TARGETS:
                sc[f"{tag}/{t}"] = cmp(df, t, a_, b_, n_boot)
        else:
            sc[tag] = {"not_run": f"{a_} or {b_} missing"}
    g25 = {m: {t: f"rob__{m}__glint:0.25__0__{t}" for t in D.REG_TARGETS} for m in ("robust_variant_seed0", "primary_seed0")}
    if all(c in df for m in g25 for c in g25[m].values()):
        for t in D.REG_TARGETS:
            m = pmask(df, t)
            sc[f"robust_vs_primary_glint25/{t}"] = E.compare_one_sided(
                t, df.loc[m, f"y_{t}"], df.loc[m, g25["robust_variant_seed0"][t]], df.loc[m, g25["primary_seed0"][t]],
                df.loc[m, "water_body"], n_boot=n_boot)
    # W8 sensitivity: H2 restricted to water bodies with no teacher row within 5 km (descriptive)
    if have("primary_ensemble") and have("nokd_ensemble"):
        sub = df[df["teacher_gt5km_wb"].to_numpy(bool)]
        for t in D.REG_TARGETS:
            if pmask(sub, t).sum() >= 10:
                sc[f"h2_teacher_gt5km/{t}"] = cmp(sub, t, "primary_ensemble", "nokd_ensemble", n_boot)
    # label (§15.1) for the carrier
    if confirmatory and have(CARRIER):
        beats = {t: all(sc[f"{CARRIER}_vs_{b}/{t}"]["ci_excludes_0_better"] for b in ALL_BASELINES) for t in D.REG_TARGETS}
        seed_ok = fam.get("h1", {}).get("seed_rule", {})
        seeds_t = {t: all(seed_ok.get(f"{b}/{t}", {}).get("all_seeds_better", False) for b in H1_BASELINES)
                   for t in D.REG_TARGETS}
        from sentinel.core import paths as _paths

        novelty = (_paths.REPO_ROOT / "docs" / "novelty_checks" / "lite.md").is_file()
        fic = novelty and composite({t: beats[t] and seeds_t[t] for t in beats}, None)
        best_worse = any(sc[f"{CARRIER}_vs_{b}/secchi"]["ci_low"] > 0 for b in ALL_BASELINES)
        res["label"] = {"label": "first-in-class" if fic else ("negative" if best_worse else "competitive"),
                        "beats_every_baseline": beats, "seed_rule": seeds_t, "novelty_check_present": novelty,
                        "sota_applicable": False}
    # slices
    for sn in ("gt10km_from_train", "future_years", "teacher_gt5km_wb"):
        m = df[sn].to_numpy(bool) if sn != "future_years" else (df["naip_year"].astype(int) >= 2022).to_numpy()
        if sn == "future_years" and not confirmatory:
            continue                     # (Test-N spans all years; test_time_n is itself the future block)
        if m.sum() < 10:
            continue
        sub = df[m]
        res["slices"][sn] = {f"{mm}/{t}": E.score_primary(t, sub[f"y_{t}"], sub[col(mm, t)], sub["water_body"],
                                                          sub["site_class"], n_boot=n_boot)["value"]
                             for t in D.TARGETS for mm in ("primary_ensemble", CARRIER, "band_ratio", "gbm_rgbn")
                             if col(mm, t) in sub}
    # robustness table (§7)
    rob = {}
    for m in ROB_MODELS:
        tab = {"clean:0": {t: res["scores"].get(f"{m}/{t}", {}).get("value") for t in D.TARGETS}}
        for kind, level in A.ROBUSTNESS_GRID:
            vals = {}
            for t in D.TARGETS:
                cs = [c for c in df.columns if c.startswith(f"rob__{m}__{kind}:{level}__") and c.endswith(f"__{t}")]
                if cs:
                    mk = pmask(df, t)
                    vals[t] = float(np.mean([E.primary_metric(t)[0](df.loc[mk, f"y_{t}"], df.loc[mk, c]) for c in cs]))
            if vals:
                tab[f"{kind}:{level}"] = vals
        rob[m] = tab
    res["robustness"] = rob
    # red-edge gap (descriptive): same chips, primary populations
    gap = {}
    for t in D.REG_TARGETS:
        c = col("teacher_s2_imgonly", t)
        if c in df:
            m = pmask(df, t) & df[c].notna().to_numpy()
            if m.sum() >= 10:
                fn = E.primary_metric(t)[0]
                gap[t] = {"n": int(m.sum()), "teacher_s2": fn(df.loc[m, f"y_{t}"], df.loc[m, c]),
                          "student_ensemble_naip": fn(df.loc[m, f"y_{t}"], df.loc[m, col("primary_ensemble", t)]),
                          "carrier_int8_naip": fn(df.loc[m, f"y_{t}"], df.loc[m, col(CARRIER, t)]) if col(CARRIER, t) in df else None}
    res["red_edge_gap"] = gap
    return res


# ===================================================================================== national extras (N1–N3)
STRATA = {"X1_small_wb": None, "X2_turbidity_gt_p90": ("turbidity", 1.5103, "gt"), "X3_tss_gt_p90": ("tss", 1.6990, "gt"),
          "X4_secchi_lt_p10": ("secchi", -0.2607, "lt")}
EXTRA_MODELS = ("primary_ensemble", CARRIER, "band_ratio", "gbm_rgbn", "gbm_rgbn_ctx")


def n_flags(sub: pd.DataFrame, t: str) -> dict:
    """N2: >= 10 chips per target for interpretation; >= 5 water bodies for a paired comparison."""
    m = sub[f"y_{t}"].notna() & E.primary_mask(t, sub["site_class"])
    n, nwb = int(m.sum()), int(sub.loc[m, "water_body"].nunique())
    return {"n": n, "n_wb": nwb, "interpretable": n >= 10, "paired_ok": n >= 10 and nwb >= 5}


def national_extras(df: pd.DataFrame, freeze: dict, confirmatory: bool, n_boot: int) -> dict:
    out = {"strata": {}, "slice_n": {}, "events": {}}
    # N2 per-target n for the standard slices
    for sn in ("gt10km_from_train", "teacher_gt5km_wb"):
        if sn in df:
            sub = df[df[sn].to_numpy(bool)]
            out["slice_n"][sn] = {t: n_flags(sub, t) for t in D.REG_TARGETS}
    # N1 strata (thresholds fixed from national-1 train)
    for name, rule in STRATA.items():
        if rule is None:
            m = (df["site_class"] == "lake_reservoir").to_numpy() & df["hylak_id"].isna().to_numpy() \
                if "hylak_id" in df else np.zeros(len(df), bool)
            targets = D.REG_TARGETS
        else:
            t0, thr, op = rule
            y = df[f"y_{t0}"].to_numpy(float)
            m = np.isfinite(y) & ((y > thr) if op == "gt" else (y < thr))
            targets = (t0,)
        sub = df[m]
        res = {}
        for t in targets:
            fl = n_flags(sub, t)
            r = {"n_flags": fl}
            if fl["interpretable"]:
                for mm in EXTRA_MODELS:
                    if col(mm, t) in sub:
                        mk = E.primary_mask(t, sub["site_class"]) & sub[f"y_{t}"].notna().to_numpy()
                        r[mm] = float(E.primary_metric(t)[0](sub.loc[mk, f"y_{t}"], sub.loc[mk, col(mm, t)]))
            if fl["paired_ok"]:
                for st in ("primary_ensemble", CARRIER):
                    for b in ("band_ratio", "gbm_rgbn", "gbm_rgbn_ctx"):
                        if col(st, t) in sub and col(b, t) in sub:
                            r[f"{st}_vs_{b}"] = cmp(sub, t, st, b, n_boot)
            res[t] = r
        out["strata"][name] = res
    # N7 numeric chl-a (secondary) and 80% prediction intervals for every target
    try:
        from sentinel.eval.bootstrap import bootstrap_ci

        seeds = freeze["models"]["primary"]["seeds"]
        out["numeric_chla"], out["intervals"] = {}, {}
        for mm in ("primary_ensemble", CARRIER, "primary_seed0", "band_ratio", "gbm_rgbn", "gbm_rgbn_ctx", "train_median"):
            c = col(mm, "chla")
            if c not in df:
                continue
            mk = E.population_mask(df["site_class"], "lake") & df["y_chla"].notna().to_numpy() & df[c].notna().to_numpy()
            if mk.sum() < 10:
                out["numeric_chla"][mm] = {"n": int(mk.sum()), "interpretable": False}
                continue
            y_, p_ = df.loc[mk, "y_chla"].to_numpy(float), df.loc[mk, c].to_numpy(float)
            r = E.reg_report(y_, p_)
            ci = bootstrap_ci(E.mdsa_log, y_, p_, groups=df.loc[mk, "water_body"].to_numpy(), n_boot=n_boot)
            r.update(mdsa_ci_low=ci["ci_low"], mdsa_ci_high=ci["ci_high"], n_wb=int(df.loc[mk, "water_body"].nunique()))
            out["numeric_chla"][mm] = r
        for st in ("primary_ensemble", CARRIER):
            for b in ("band_ratio", "gbm_rgbn", "gbm_rgbn_ctx"):
                if col(st, "chla") in df and col(b, "chla") in df:
                    mk = E.population_mask(df["site_class"], "lake") & df["y_chla"].notna().to_numpy()
                    sub = df[mk]
                    r = E.compare_one_sided("secchi", sub["y_chla"], sub[col(st, "chla")], sub[col(b, "chla")],
                                            sub["water_body"], n_boot=n_boot)    # "secchi" -> MdSA metric
                    r["metric"] = "mdsa"
                    out["numeric_chla"][f"{st}_vs_{b}"] = r
        # intervals: per-seed sigma; ensemble sigma^2 = mean sigma^2 + var(mu) (as the student summary)
        for t in D.TARGETS:
            pop = E.population_mask(df["site_class"], "lake" if t in ("secchi", "chla") else "all")
            mus = np.stack([df[col(f"primary_seed{s}", t)].to_numpy(float) for s in seeds])
            sgs = np.stack([df[f"sig__primary_seed{s}__{t}"].to_numpy(float) for s in seeds])
            ens_mu, ens_sd = mus.mean(0), np.sqrt((sgs ** 2).mean(0) + mus.var(0))
            mk = pop & df[f"y_{t}"].notna().to_numpy()
            if mk.sum() < 10:
                out["intervals"][t] = {"n": int(mk.sum()), "interpretable": False}
                continue
            yv = df.loc[mk, f"y_{t}"].to_numpy(float)
            z = 1.2815515655446004
            out["intervals"][t] = {
                "n": int(mk.sum()),
                "ensemble_cov80": float(np.mean(np.abs(yv - ens_mu[mk]) <= z * ens_sd[mk])),
                "ensemble_median_width_log10": float(np.median(2 * z * ens_sd[mk])),
                "seed0_cov80": float(np.mean(np.abs(yv - mus[0][mk]) <= z * sgs[0][mk])),
                "seed0_median_width_log10": float(np.median(2 * z * sgs[0][mk])),
                "nominal": 0.80}
    except Exception as e:
        out["numeric_chla"] = {"not_run": repr(e)}
    # N3 events
    try:
        from sentinel.events import lite as EL

        ev = EL.EV()
        evn = freeze.get("events_national") or {}
        seeds = freeze["models"]["primary"]["seeds"]
        mu = lambda t: [df[col(f"primary_seed{s}", t)].to_numpy(float) for s in seeds]  # noqa: E731
        sg = lambda t: [df[f"sig__primary_seed{s}__{t}"].to_numpy(float) for s in seeds]  # noqa: E731
        g_all = df["water_body"].to_numpy()
        # E3a_turb: one pre-declared endpoint (N3)
        y = EL.labels(df["y_turbidity"], "E3a_turb")
        okm = np.isfinite(y)
        rec = {"n": int(okm.sum()), "n_pos": int(np.nansum(y[okm])),
               "n_wb_pos": int(df.loc[okm & (y == 1), "water_body"].nunique())}
        rec["power_ok"] = rec["n_pos"] >= 30 and rec["n_wb_pos"] >= 10
        cals = json.loads(Path(evn["path"]).read_text()) if evn else {}
        if "E3a_turb" in cals and 0 < rec["n_pos"] < rec["n"]:
            cal = {k: EL.calibrator_from_dict(v) for k, v in cals["E3a_turb"].items()}
            p_der = cal["derived"].transform(EL.derived_p_ensemble(mu("turbidity"), sg("turbidity"), "E3a_turb"))
            p_clim = cal["climatology"].transform(df["evraw__climatology__E3a_turb"].to_numpy(float)) \
                if "evraw__climatology__E3a_turb" in df else np.full(len(df), np.nan)
            rec["derived"] = ev.state_metrics(y[okm], p_der[okm], groups=g_all[okm], n_boot=n_boot)
            rec["climatology"] = ev.state_metrics(y[okm], p_clim[okm], groups=g_all[okm], n_boot=n_boot)
            cr = ev.compare_routes(y[okm], p_der[okm], p_clim[okm], groups=g_all[okm], n_boot=n_boot)
            d = cr["delta_brier"]
            rec["endpoint"] = {"diff": d.get("diff"), "ci_low": d.get("ci_low"), "ci_high": d.get("ci_high"),
                               "derived_better": bool(confirmatory and rec["power_ok"] and np.isfinite(d.get("ci_high", np.nan))
                                                      and d["ci_high"] < 0),
                               "rule": "N3: derived better iff the paired water-body CI of ΔBrier(derived − climatology) "
                                       "lies below 0, on Test-N, with >= 30 positives and >= 10 water bodies; otherwise descriptive",
                               "descriptive_only": not (confirmatory and rec["power_ok"])}
            rec["delta_auprc"] = cr["delta_auprc"]
        out["events"]["E3a_turb"] = rec
        # E4a and E3a_tss: descriptive (raw derived P)
        for code, t, popm in (("E4a", "chla", E.population_mask(df["site_class"], "lake")), ("E3a_tss", "tss", None)):
            yy = EL.labels(df[f"y_{t}"], code)
            mk = np.isfinite(yy) & (popm if popm is not None else True)
            r = {"n": int(mk.sum()), "n_pos": int(np.nansum(yy[mk])), "descriptive_only": True}
            if 0 < r["n_pos"] < r["n"]:
                p = EL.derived_p_ensemble(mu(t), sg(t), code)
                r["derived"] = ev.state_metrics(yy[mk], p[mk], groups=g_all[mk], n_boot=n_boot)
            out["events"][code] = r
    except Exception as e:      # events never stop H1/H2/H-G6 (v1.7 rule)
        out["events"]["not_run"] = repr(e)
    return out


def h_g6(res_conf: dict, freeze: dict) -> dict:
    G.verify_weights(freeze, "export_report_seed0", freeze["export_report"])   # budgets come from a hashed report
    rep = json.loads(Path(freeze["export_report"]).read_text())
    budget = rep["budget"]
    fam = res_conf["families"]
    h1d = fam.get("h1_deployable", {}).get("verdict_with_seed_rule", False)
    h2 = fam.get("h2", {}).get("verdict_with_every_seed", False)
    gap = bool(res_conf.get("red_edge_gap")) and any(np.isfinite(v["teacher_s2"]) for v in res_conf["red_edge_gap"].values())
    return {"budgets_met": all(budget.values()), "budgets": budget, "h1_deployable": bool(h1d), "h2": bool(h2),
            "red_edge_gap_reported": gap, "H_G6": bool(all(budget.values()) and h1d and h2 and gap)}


def score_all(pdir: Path, freeze: dict, roles, confirm_role: str, n_boot: int) -> dict:
    out = {"freeze_sha256": freeze.get("_sha256"), "roles": list(roles), "by_role": {}}
    for role in roles:
        df = pd.read_parquet(pdir / f"pred_{role}.parquet")
        out["by_role"][role] = score_role(df, freeze, confirmatory=(role == confirm_role), n_boot=n_boot)
        out["by_role"][role]["national_extras"] = national_extras(df, freeze, confirmatory=(role == confirm_role),
                                                                 n_boot=n_boot)
    out["H_G6"] = h_g6(out["by_role"][confirm_role], freeze)
    nr = pdir / "not_run.json"
    out["not_run"] = {**(freeze.get("not_run") or {}), **(json.loads(nr.read_text()) if nr.exists() else {})}
    return out


def registry_rows(results: dict) -> list[dict]:
    rows = []
    for role, res in results["by_role"].items():
        for k, s in res["scores"].items():
            m, t = k.split("/")
            if np.isfinite(s.get("value", np.nan)):
                rows.append(dict(experiment="lite_national_test", model=f"lite_{m}", metric=s["primary"], value=s["value"],
                                 ci_low=s["ci_low"], ci_high=s["ci_high"], split=role, n=s["n"], target=t,
                                 is_baseline=m in ALL_BASELINES))
        for fname, f in res["families"].items():
            for k, r in (f.get("comparisons") or {}).items():
                t = k.split("/")[-1]
                rows.append(dict(experiment=f"lite_national_test_{fname}", model="lite_" + ("primary_ensemble" if fname != "h1_deployable" else CARRIER),
                                 metric=f"{r['metric']}_diff_vs_{k.split('/')[0] if '/' in k else 'nokd_ensemble'}",
                                 value=r["diff"], ci_low=r["ci_low"], ci_high=r["ci_high"], split=role, n=r["n"], target=t,
                                 is_baseline=False, extra={"p_one_sided": r["p_one_sided"], "holm": r.get("holm"),
                                                           "reject": r.get("reject"), "confirmatory": "holm" in r}))
        e4 = (res.get("events") or {}).get("E4a")
        if e4:
            for r in ("derived", "direct", "band_ratio", "gbm_rgbn", "climatology"):
                for k in ("auprc", "brier"):
                    m = e4[r][k]
                    if np.isfinite(m["point"]):
                        rows.append(dict(experiment="lite_national_test_events", model=f"lite_E4a_{r}", metric=k,
                                         value=float(m["point"]), ci_low=m["ci_low"], ci_high=m["ci_high"], split=role,
                                         n=int(e4[r]["n"]), target="E4a", is_baseline=r not in ("derived", "direct"),
                                         extra={"ece": e4[r].get("ece"), "n_pos": e4[r]["n_pos"]}))
            for c in ("direct", "band_ratio", "gbm_rgbn", "climatology"):
                for k in ("delta_brier", "delta_auprc"):
                    d = e4[f"derived_vs_{c}"][k]
                    if np.isfinite(d.get("diff", np.nan)):
                        rows.append(dict(experiment="lite_national_test_events", model="lite_E4a_derived", metric=f"{k}_vs_{c}",
                                         value=float(d["diff"]), ci_low=d["ci_low"], ci_high=d["ci_high"], split=role,
                                         n=int(e4[f"derived_vs_{c}"]["n"]), target="E4a", is_baseline=False,
                                         extra={"p_value": d.get("p_value"), "descriptive": True}))
    conf = results["H_G6"]
    rows.append(dict(experiment="lite_national_test_verdicts", model="lite_" + CARRIER, metric="H_G6",
                     value=float(conf["H_G6"]), split="test_n", n=None, target=None, is_baseline=False, extra=conf))
    for role, res in results["by_role"].items():
        e3 = ((res.get("national_extras") or {}).get("events") or {}).get("E3a_turb") or {}
        if "endpoint" in e3 and np.isfinite(e3["endpoint"].get("diff", np.nan)):
            ep = e3["endpoint"]
            rows.append(dict(experiment="lite_national_test_events", model="lite_E3a_turb_derived",
                             metric="delta_brier_vs_climatology", value=float(ep["diff"]), ci_low=ep["ci_low"],
                             ci_high=ep["ci_high"], split=role, n=e3.get("n"), target="E3a_turb", is_baseline=False,
                             extra={"derived_better": ep.get("derived_better"), "confirmatory": role == "test_n",
                                    "power_ok": e3.get("power_ok")}))
    return rows


def flush_registry(rows: list[dict], out_dir: Path, freeze: dict) -> None:
    from sentinel.core.registry import log_result

    for r in rows:
        extra = {**(r.pop("extra", None) or {}), "freeze_sha256": freeze.get("_sha256")}
        log_result(workstream=R.WORKSTREAM, ci_method="cluster bootstrap over water bodies (B=2000)",
                   data_manifests=list(R.DATA_MANIFESTS), artifacts=[str(out_dir)], extra=extra, **r)


# ===================================================================================== main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--rescore-from-saved", action="store_true")
    ap.add_argument("--dry-run-val", action="store_true")
    ap.add_argument("--candidate", default=None, help="dry run only: the candidate file")
    ap.add_argument("--smoke", action="store_true", help="dry run only: 150 chips, 20 bootstraps")
    a = ap.parse_args(argv)
    split = N2.load_split()
    if a.dry_run_val:
        freeze = json.loads(Path(a.candidate).read_text())
        freeze["_sha256"] = T.sha_file(a.candidate)
        T.check_packages(freeze)                       # the dry run exercises the same pre-claim checks
        T.verify_data_pins(freeze)
        T.verify_artifacts(freeze)
        roles, confirm_role = ("val_n", "val"), "val_n"
        n_boot = 20 if a.smoke else int(freeze["n_boot"])
        out_dir = R.run_directory("dryrun_test_scoring", smoke=True)
        pdir = out_dir / "predictions"
        pdir.mkdir(parents=True, exist_ok=True)
    else:
        if a.smoke or a.candidate:
            raise SystemExit("--smoke/--candidate are dry-run options; test mode reads only FREEZE_lite.json")
        if not a.allow_test:
            raise SystemExit("test mode needs --allow-test (after the lead's authorisation)")
        freeze = T.load_frozen()                       # C2 + W-b: FREEZE_FILE only, FROZEN, authorised, packages
        T.verify_data_pins(freeze)                     # C3 + W-a: before the claim, frozen == live loader paths
        T.verify_artifacts(freeze)                     # D2: every hashed artifact and run dir, before the claim
        roles, confirm_role = ("test_n", "test_time_n"), "test_n"
        n_boot = int(freeze["n_boot"])                 # W3: from the freeze
        out_dir = T.look_dir()
        if a.rescore_from_saved:
            pdir = T.check_sealed(freeze)              # D1: registry anchor, this freeze, no prior scoring rows
            results = score_all(pdir, freeze, roles, confirm_role, n_boot)
            R.write_json(out_dir / "test_results.json", results)
            flush_registry(registry_rows(results), out_dir, freeze)
            T.mark_scored({"H_G6": results["H_G6"], "rescored_from_saved": True})
            print(json.dumps(results["H_G6"], indent=1))
            return 0
        if freeze.get("device", "cpu") != "cpu":      # E2: refused before the claim
            raise SystemExit("the frozen device must be cpu (GPU budget 0/1 shared; review W3)")
        T.claim(freeze)                                # C1
        pdir = T.pred_dir()
        pdir.mkdir(parents=True, exist_ok=True)
    device = freeze.get("device", "cpu")
    if device != "cpu":
        raise SystemExit("the frozen device must be cpu (GPU budget 0/1 shared; review W3)")
    not_run = {}
    import signal

    def _term(signum, frame):                          # E2: SIGTERM -> exception, so the abort marker is written
        raise SystemExit(f"terminated by signal {signum}")

    signal.signal(signal.SIGTERM, _term)
    try:   # phase 1 (claimed look): any crash before sealing is recorded (ABORTED.json + registry row), then raised
        if a.dry_run_val:
            ev_all = N2.role_frame(split, roles)
        else:
            ev_all = T.load_protected(freeze, roles)
        tr = N2.role_frame(split, ("train",))
        for role in roles:
            ev = ev_all[ev_all["role"] == role].reset_index(drop=True)
            cname = f"eval_{role}"
            if a.smoke:
                ev, cname = ev.head(150).reset_index(drop=True), f"smoke_{role}"
            dfp, nr = predict_role(freeze, ev, tr, cname, device)
            not_run.update({f"{role}/{k}": v for k, v in nr.items()})
            dfp.to_parquet(pdir / f"pred_{role}.parquet", index=False)     # C5: every model and baseline, per row
        R.write_json(pdir / "not_run.json", not_run)
        if not a.dry_run_val:
            T.seal_predictions(freeze)                 # D1: + lite_test_sealed registry anchor; closes test access
    except BaseException as e:                         # E2: also KeyboardInterrupt / SystemExit (SIGTERM)
        if not a.dry_run_val:
            T.mark_aborted(freeze, repr(e))
        raise
    results = score_all(pdir, freeze, roles, confirm_role, n_boot)
    R.write_json(out_dir / ("test_results.json" if not a.dry_run_val else "dryrun_results.json"), results)
    if not a.dry_run_val:
        flush_registry(registry_rows(results), out_dir, freeze)        # only after everything was scored
        T.mark_scored({"H_G6": results["H_G6"]})
    print(json.dumps({"H_G6": results["H_G6"], "not_run": not_run}, indent=1, default=str))
    print("written:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
