# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Lite event track, validation stage (lite_protocol.md §13; event_layer_plan.md §2): train and val roles
only, CPU.

For the selected student (``FREEZE_lite_candidate.json`` models.primary), this script:
1. computes the seed μ/σ and the water-pooled decoder embeddings on train and val;
2. fits the route-(ii) head per seed on train chips with chl-a labels;
3. fits the comparators on train (band-ratio and GBM-RGBN chl-a regressions with cross-validated residual SD;
   location climatology);
4. builds raw P for every route on val;
5. computes water-body cross-fitted recalibrated val probabilities, then ``events.state_metrics`` and
   ``events.compare_routes``;
6. fits the final Platt calibrators on all of val (frozen for the test look).

Outputs go to ``$SENTINEL_RUNS/lite/events_e4a/``: ``heads.json``, ``comparators.json``, ``calibrators.json``,
``val_event_preds.parquet``, ``val_event_metrics.json``. Registry experiment: ``events_lite_val``.

    python -B scripts/eval/events_lite.py [--no-register]
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

from sentinel.core import paths  # noqa: E402
from sentinel.events import lite as EL  # noqa: E402
from sentinel.models.lite import baselines as B  # noqa: E402
from sentinel.models.lite import data as D  # noqa: E402
from sentinel.models.lite import evaluate as E  # noqa: E402
from sentinel.models.lite import features as F  # noqa: E402
from sentinel.models.lite import guard as G  # noqa: E402
from sentinel.models.lite.train import Inputs, build_model, predict  # noqa: E402

OUT = paths.RUNS_ROOT / "lite" / "events_e4a"
PRIMARY = paths.RUNS_ROOT / "lite" / "student_mnv4s_raw_crop128_kd03_20261108-152950"
BASELINES = paths.RUNS_ROOT / "lite" / "baselines_20261024-221014"
ROUTES = ("derived", "direct", "band_ratio", "gbm_rgbn", "climatology")


def inputs(lab: pd.DataFrame, name: str) -> Inputs:
    chips = np.asarray(D.cached_chips(lab, name))
    return Inputs(chips, D.scene_norm_params(lab, D.scene_stats()), F.pooling_weights(chips, lab["gsd_m"].to_numpy(), 4),
                  None, "cpu")


def feats(lab: pd.DataFrame, name: str) -> pd.DataFrame:
    chips = np.asarray(D.cached_chips(lab, name))
    return F.chip_features(chips, lab["gsd_m"].to_numpy(), D.scene_norm_params(lab, D.scene_stats())[:, :, 0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    split = G.load_split()
    lab = D.role_frame(split, ("train", "val"), training=True)
    G.assert_no_protected(lab["role"])
    tr, va = lab[lab.role == "train"].reset_index(drop=True), lab[lab.role == "val"].reset_index(drop=True)
    itr, iva = inputs(tr, "train"), inputs(va, "val")
    j_chl = D.TARGETS.index("chla")
    mus, sds, heads, p_head = {}, {}, {}, []
    y_tr = EL.labels(tr["y_chla"], "E4a")
    for s in (0, 1, 2):
        ck = torch.load(PRIMARY / f"model_seed{s}.pt", map_location="cpu", weights_only=False)
        cfg = {**ck["cfg"], "pretrained": False, "gloria_init": None, "amp": False}
        m = build_model(cfg)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        mu, sd = predict(m, iva, cfg)
        mus[s], sds[s] = mu, sd
        emb_tr, emb_va = EL.pooled_embedding(m, itr, cfg), EL.pooled_embedding(m, iva, cfg)
        heads[s] = EL.fit_head(emb_tr, y_tr)
        p_head.append(EL.head_p(emb_va, heads[s]))
        print(f"seed {s}: head n_train {heads[s]['n_train']} pos {heads[s]['n_pos']}", flush=True)
    # comparators (train only)
    forms = json.loads((BASELINES / "summary.json").read_text())["info"]
    Ftr, Fva = feats(tr, "train"), feats(va, "val")
    img = list(Ftr.columns)
    ytr_chl = tr["y_chla"].to_numpy(float)
    br_form = forms["band_ratio/chla"]["selected_form"]
    br = lambda Xa, ya, Xb: B.band_ratio(Xa, ya, Xb, "chla", br_form)[0]  # noqa: E731
    gb = lambda Xa, ya, Xb: B.gbm(Xa, ya, Xb, cols=img)[0]  # noqa: E731
    comp = {"band_ratio": {"form": br_form, "resid_sd": EL.cv_residual_sd(br, Ftr, ytr_chl, tr["water_body"])},
            "gbm_rgbn": {"resid_sd": EL.cv_residual_sd(gb, Ftr, ytr_chl, tr["water_body"])},
            "climatology": {"k": EL.CLIM_K}}
    raw = {"derived": EL.derived_p_ensemble([mus[s][:, j_chl] for s in mus], [sds[s][:, j_chl] for s in sds], "E4a"),
           "direct": np.mean(p_head, 0),
           "band_ratio": EL.derived_p(br(Ftr, ytr_chl, Fva), np.full(len(va), comp["band_ratio"]["resid_sd"]), "E4a"),
           "gbm_rgbn": EL.derived_p(gb(Ftr, ytr_chl, Fva), np.full(len(va), comp["gbm_rgbn"]["resid_sd"]), "E4a"),
           "climatology": EL.climatology_p(tr, y_tr, va["lat"], va["lon"])}
    y_va = EL.labels(va["y_chla"], "E4a")
    pop = E.population_mask(va["site_class"], EL.E4A["population"]) & np.isfinite(y_va)
    ev = EL.EV()
    cal_va, calibrators, metrics = {}, {}, {"E4a": {}}
    for r in ROUTES:
        cal_va[r] = np.full(len(va), np.nan)
        cal_va[r][pop] = EL.crossfit_calibrated(raw[r][pop], y_va[pop], va.loc[pop, "water_body"])
        calibrators[r] = EL.calibrator_to_dict(EL.fit_calibrator(raw[r][pop], y_va[pop]))
        metrics["E4a"][r] = ev.state_metrics(y_va[pop], cal_va[r][pop], groups=va.loc[pop, "water_body"].to_numpy())
    metrics["E4a"]["derived_vs_direct"] = ev.compare_routes(y_va[pop], cal_va["derived"][pop], cal_va["direct"][pop],
                                                           groups=va.loc[pop, "water_body"].to_numpy())
    for c in ("band_ratio", "gbm_rgbn", "climatology"):
        metrics["E4a"][f"derived_vs_{c}"] = ev.compare_routes(y_va[pop], cal_va["derived"][pop], cal_va[c][pop],
                                                             groups=va.loc[pop, "water_body"].to_numpy())
    metrics["E4a"]["power_flag"] = ev.power_flag(int(np.nansum(y_va[pop])), int(va.loc[pop & (y_va == 1), "water_body"].nunique()))
    # sensitivity tiers and E3 (derived route only, descriptive)
    desc = {}
    for code in EL.E4A["tiers"]:
        yt = EL.labels(va["y_chla"], code)
        p = EL.derived_p_ensemble([mus[s][:, j_chl] for s in mus], [sds[s][:, j_chl] for s in sds], code)
        okt = E.population_mask(va["site_class"], "lake") & np.isfinite(yt)
        desc[code] = ev.state_metrics(yt[okt], p[okt], groups=va.loc[okt, "water_body"].to_numpy())
    for e in EL.E3:
        j = D.TARGETS.index(e["target"])
        ye = EL.labels(va[f"y_{e['target']}"], e["code"])
        oke = np.isfinite(ye)
        p = EL.derived_p_ensemble([mus[s][:, j] for s in mus], [sds[s][:, j] for s in sds], e["code"])
        desc[e["code"]] = {"n_pos": int(np.nansum(ye[oke])), "n_wb_pos": int(va.loc[oke & (ye == 1), "water_body"].nunique()),
                           "metrics": ev.state_metrics(ye[oke], p[oke], groups=va.loc[oke, "water_body"].to_numpy())
                           if 0 < np.nansum(ye[oke]) < oke.sum() else "no positives"}
    metrics["descriptive"] = desc
    R.write_json(OUT / "heads.json", {str(s): h for s, h in heads.items()})
    R.write_json(OUT / "comparators.json", comp)
    R.write_json(OUT / "calibrators.json", calibrators)
    R.write_json(OUT / "val_event_metrics.json", metrics)
    pv = va[["naip_chip_id", "water_body", "site_class", "y_chla"]].copy()
    for r in ROUTES:
        pv[f"raw_{r}"], pv[f"cal_{r}"] = raw[r], cal_va[r]
    pv.to_parquet(OUT / "val_event_preds.parquet", index=False)
    R.write_json(OUT / "provenance.json", {"primary_run": str(PRIMARY), "baselines_run": str(BASELINES),
                                            "split_sha256": split["_sha256"], "events_module_sha256": G.artifact_sha256(
                                                Path(EL.EV().__file__)), "added": "2026-11-09, after val model selection "
                                            "(selection score used Secchi/turbidity/TSS only; chl-a never entered S)"})
    if not a.no_register:
        from sentinel.core.registry import log_result

        for r in ROUTES:
            m = metrics["E4a"][r]
            for k in ("auprc", "brier"):
                if np.isfinite(m[k]["point"]):
                    log_result(workstream=R.WORKSTREAM, experiment="events_lite_val", model=f"lite_E4a_{r}", metric=k,
                               value=float(m[k]["point"]), ci_low=m[k]["ci_low"], ci_high=m[k]["ci_high"], split="val",
                               n=int(m["n"]), target="E4a", is_baseline=r not in ("derived", "direct"),
                               artifacts=[str(OUT)], data_manifests=list(R.DATA_MANIFESTS),
                               ci_method="water-body cluster bootstrap (events.state_metrics v1.0); cross-fitted Platt",
                               extra={"ece": m.get("ece"), "n_pos": m["n_pos"], "power_flag": m["power_flag"]})
        for c in ("direct", "band_ratio", "gbm_rgbn", "climatology"):
            cr = metrics["E4a"][f"derived_vs_{c}"]
            for k in ("delta_brier", "delta_auprc"):
                d = cr[k]
                if np.isfinite(d.get("diff", np.nan)):
                    log_result(workstream=R.WORKSTREAM, experiment="events_lite_val", model="lite_E4a_derived",
                               metric=f"{k}_vs_{c}", value=float(d["diff"]), ci_low=d["ci_low"], ci_high=d["ci_high"],
                               split="val", n=int(cr["n"]), target="E4a", artifacts=[str(OUT)],
                               data_manifests=list(R.DATA_MANIFESTS),
                               ci_method="paired water-body bootstrap (events.compare_routes v1.0)",
                               extra={"p_value": d.get("p_value")})
    print(json.dumps({r: {k: metrics["E4a"][r][k]["point"] for k in ("auprc", "brier")} for r in ROUTES}, indent=1))
    print(json.dumps({c: {k: (round(metrics["E4a"][f"derived_vs_{c}"][k]["diff"], 4),
                              round(metrics["E4a"][f"derived_vs_{c}"][k]["ci_low"], 4),
                              round(metrics["E4a"][f"derived_vs_{c}"][k]["ci_high"], 4)) for k in ("delta_brier", "delta_auprc")}
                      for c in ("direct", "band_ratio", "gbm_rgbn", "climatology")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
