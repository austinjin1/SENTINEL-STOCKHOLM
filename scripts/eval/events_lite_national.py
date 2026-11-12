# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Validation stage of the national event track (lite_national_protocol.md N3): train / val / val_n of national-2
only, CPU.

- **E3a_turb** (turbidity > 50 NTU, snapshot, all water bodies).
  - Derived route: P from the selected national student's seed μ/σ, seed-averaged. Climatology: positive rate of
    the 10 nearest train sites.
  - One Platt calibrator per route is fitted on val ∪ val_n and frozen. Val metrics use water-body cross-fitted
    calibration. The pre-declared endpoint is ΔBrier(derived − climatology); on val it is descriptive.
- **E4a** (lakes) and **E3a_tss**: derived raw P, descriptive.

Output: ``$SENTINEL_RUNS/lite/events_national/{calibrators.json, val_event_metrics.json}``; registry experiment
``events_lite_national_val``.

    python -B scripts/eval/events_lite_national.py --run $SENTINEL_RUNS/lite/<natstudent run> [--no-register]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentinel.models.lite import runner as R

R.cpu_threads(6)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.events import lite as EL  # noqa: E402
from sentinel.models.lite import data as D  # noqa: E402
from sentinel.models.lite import evaluate as E  # noqa: E402
from sentinel.models.lite import features as F  # noqa: E402
from sentinel.models.lite import national2 as N2  # noqa: E402
from sentinel.models.lite.train import Inputs, build_model, predict  # noqa: E402

OUT = paths.RUNS_ROOT / "lite" / "events_national"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    run = Path(a.run)
    split = N2.load_split()
    tr = N2.role_frame(split, ("train",))
    va = N2.role_frame(split, ("val", "val_n"))
    chips = np.asarray(D.cached_chips(va, "val"))
    inp = Inputs(chips, D.scene_norm_params(va, D.scene_stats(N2.scene_stats_path())),
                 F.pooling_weights(chips, va["gsd_m"].to_numpy(), 4), None, "cpu")
    mus, sds = [], []
    for ck in sorted(run.glob("model_seed*.pt")):
        c = torch.load(ck, map_location="cpu", weights_only=False)
        cfg = {**c["cfg"], "pretrained": False, "gloria_init": None, "amp": False}
        cfg.pop("ssl_init", None)                   # full weights are loaded below
        m = build_model(cfg)
        m.load_state_dict(c["state_dict"])
        mu, sd = predict(m.eval(), inp, cfg)
        mus.append(mu)
        sds.append(sd)
    jt, js, jc = D.TARGETS.index("turbidity"), D.TARGETS.index("tss"), D.TARGETS.index("chla")
    ev = EL.EV()
    g = va["water_body"].to_numpy()
    y = EL.labels(va["y_turbidity"], "E3a_turb")
    ok = np.isfinite(y)
    raw = {"derived": EL.derived_p_ensemble([m_[:, jt] for m_ in mus], [s_[:, jt] for s_ in sds], "E3a_turb"),
           "climatology": EL.climatology_p(tr, EL.labels(tr["y_turbidity"], "E3a_turb"), va["lat"], va["lon"])}
    cal_va, cals, met = {}, {}, {"E3a_turb": {"n": int(ok.sum()), "n_pos": int(np.nansum(y[ok])),
                                              "n_wb_pos": int(va.loc[ok & (y == 1), "water_body"].nunique())}}
    for r, p in raw.items():
        cal_va[r] = np.full(len(va), np.nan)
        cal_va[r][ok] = EL.crossfit_calibrated(p[ok], y[ok], g[ok])
        cals[r] = EL.calibrator_to_dict(EL.fit_calibrator(p[ok], y[ok]))
        met["E3a_turb"][r] = ev.state_metrics(y[ok], cal_va[r][ok], groups=g[ok])
    met["E3a_turb"]["derived_vs_climatology"] = ev.compare_routes(y[ok], cal_va["derived"][ok], cal_va["climatology"][ok],
                                                                 groups=g[ok])
    for code, j, pop in (("E4a", jc, E.population_mask(va["site_class"], "lake")), ("E3a_tss", js, None)):
        yy = EL.labels(va[f"y_{D.TARGETS[j]}"], code)
        mk = np.isfinite(yy) & (pop if pop is not None else True)
        p = EL.derived_p_ensemble([m_[:, j] for m_ in mus], [s_[:, j] for s_ in sds], code)
        met[code] = {"n": int(mk.sum()), "n_pos": int(np.nansum(yy[mk])), "descriptive_only": True,
                     "derived_raw": ev.state_metrics(yy[mk], p[mk], groups=g[mk]) if 0 < np.nansum(yy[mk]) < mk.sum() else None}
    (OUT / "calibrators.json").write_text(json.dumps({"E3a_turb": cals, "run": str(run)}, indent=1))
    R.write_json(OUT / "val_event_metrics.json", met)
    if not a.no_register:
        from sentinel.core.registry import log_result

        for r in raw:
            for k in ("auprc", "brier"):
                x = met["E3a_turb"][r][k]
                if np.isfinite(x["point"]):
                    log_result(workstream=R.WORKSTREAM, experiment="events_lite_national_val", model=f"lite_E3a_turb_{r}",
                               metric=k, value=float(x["point"]), ci_low=x["ci_low"], ci_high=x["ci_high"], split="val",
                               n=met["E3a_turb"]["n"], target="E3a_turb", is_baseline=r != "derived", artifacts=[str(OUT)],
                               data_manifests=["lite_labels_national2"],
                               ci_method="water-body cluster bootstrap (events v1.0); cross-fitted Platt")
    d = met["E3a_turb"]["derived_vs_climatology"]["delta_brier"]
    print(json.dumps({"E3a_turb": {r: {k: round(met["E3a_turb"][r][k]["point"], 3) for k in ("auprc", "brier")} for r in raw},
                      "delta_brier": [round(d["diff"], 4), round(d["ci_low"], 4), round(d["ci_high"], 4)],
                      "counts": {k: met["E3a_turb"][k] for k in ("n", "n_pos", "n_wb_pos")}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
