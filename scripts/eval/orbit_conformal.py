#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Split-conformal interval recalibration, fitted on validation only (orbit_protocol.md §5b; secondary).

Fits one water-body-weighted CQR offset per (model, target, level) for the SatWQ-Net seed-ensemble mixture and the
val-selected MDN_B, on the declared calibration cells (chl-a, Secchi: valxval_lake; TSS, turbidity: valxany_lake,
as the §2.2 selection cells). Registers raw and cross-fitted (5 water-body folds) conformal coverage and width, and
writes the frozen artifact ``conformal_orbit_<version>.json`` (sha-pinned in the freeze). Never loads protected rows.

    python -B scripts/eval/orbit_conformal.py --version dev_v5 --satwq ... --baselines baselines_dev_v5 \
        --comparators-json .../comparators_val_selected.json --label final_v5 --register
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.models.orbit import conformal as CF  # noqa: E402
from sentinel.models.orbit import data as D  # noqa: E402
from sentinel.models.orbit import evaluate as E  # noqa: E402
from sentinel.models.orbit import pipeline as PL  # noqa: E402

QS = (0.025, 0.16, 0.5, 0.84, 0.975)
CAL_CELL = {"chla": "valxval_lake", "secchi": "valxval_lake", "tss": "valxany_lake", "turbidity": "valxany_lake"}
MDN_B7 = ("B01", "B02", "B03", "B04", "B05", "B06", "B07")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--satwq", nargs="+", required=True)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--comparators-json", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--fit-scheme", default="base", choices=["base", "track_u"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--register", action="store_true")
    a = ap.parse_args()
    import cloudpickle
    import torch

    from sentinel.core.runlog import TrainingRun
    from sentinel.events import orbit as EO
    from sentinel.models.orbit.satwq import mixture_quantiles

    torch.set_num_threads(2)
    R = paths.RUNS_ROOT / "orbit"
    out_dir = paths.ensure(R / (f"conformal_{a.label}" + ("" if a.fit_scheme == "base" else f"_{a.fit_scheme}")))
    cfg = {"method": "CQR, water-body-weighted quantile (orbit_protocol.md §5b)", "cal_cells": CAL_CELL,
           "levels": list(CF.LEVELS), "n_folds": CF.N_FOLDS, "version": a.version, "satwq": a.satwq,
           "baselines": a.baselines, "fit_scheme": a.fit_scheme}
    with TrainingRun(PL.WORKSTREAM, f"conformal_{a.label}", config=cfg, data=[f"orbit_{a.version}"],
                     plan_ref="orbit_protocol.md §5b", iteration="conformal-v1", artifacts_dir=str(out_dir)) as run:
        df, pix, _, masks = PL.load_dev(a.version, a.fit_scheme)
        use = masks["valxany_lake"]
        sub = df[use].reset_index(drop=True)
        psub = {k: v[use] for k, v in pix.items()}
        cells = {c: masks[c][use] for c in ("valxval_lake", "valxany_lake")}
        (w, mu, sg), _ = EO.satwq_mixture_embedding([R / r / "model.pt" for r in a.satwq], sub, psub, "cpu")
        fitted = cloudpickle.loads((R / a.baselines / "fitted_baselines.pkl").read_bytes())["fitted"]
        comps = json.loads(Path(a.comparators_json).read_text())
        art = {"version": a.version, "fit_scheme": a.fit_scheme, "satwq_runs": a.satwq, "baselines": a.baselines,
               "method": cfg["method"], "cal_cells": CAL_CELL, "mdn_models": {}, "offsets": {"SENTINEL": {}, "mdn_b": {}}}
        rows = []
        for j, t in enumerate(D.TARGETS):
            y = np.log10(sub[t].to_numpy(float))
            qsrc = {"SENTINEL": dict(zip(QS, mixture_quantiles(w[:, j], mu[:, j], sg[:, j], QS).T))}
            mdn = comps.get(t, {}).get("mdn_b", {}).get("model")
            if mdn and (mdn, t) in fitted:
                bands = D.BANDS if mdn.startswith("mdn_b12") else MDN_B7
                Xl = np.stack([sub[f"rw3_{b}"].to_numpy(float) for b in bands], 1)
                qsrc["mdn_b"] = dict(zip(QS, fitted[(mdn, t)].quantiles(Xl, QS).T))
                art["mdn_models"][t] = mdn
            m = cells[CAL_CELL[t]] & np.isfinite(y)
            for model, Q in qsrc.items():
                ok = m & np.all([np.isfinite(v) for v in Q.values()], 0)
                if ok.sum() < 10:
                    art["offsets"][model][t] = None
                    run.note(f"{model}/{t}: {int(ok.sum())} calibration rows < 10 -> no conformal offset (raw only)")
                    continue
                yy, g, bl = y[ok], sub.waterbody.to_numpy()[ok], sub.block.to_numpy()[ok]
                Qo = {q: v[ok] for q, v in Q.items()}
                art["offsets"][model][t] = CF.fit(yy, Qo, g)
                cf = CF.crossfit_intervals(yy, Qo, g)
                for lvl, (lo, hi) in CF.LEVELS.items():
                    for kind, (L, H) in (("raw", (Qo[lo], Qo[hi])), ("conformal_crossfit", cf[lvl])):
                        cov = CF.coverage(yy, L, H)
                        ci = E.cluster_ci(lambda a_, b_, c_: CF.coverage(a_, b_, c_), (yy, L, H), bl, a.n_boot)
                        rows.append({"model": model, "target": t, "cell": CAL_CELL[t], "level": lvl, "kind": kind,
                                     "n": int(ok.sum()), "n_waterbodies": int(len(np.unique(g))), "coverage": cov,
                                     "cov_ci_low": ci[0], "cov_ci_high": ci[1],
                                     "width_log10": float(np.nanmean(H - L)),
                                     "offset_log10": art["offsets"][model][t][lvl]})
        import pandas as pd

        res = pd.DataFrame(rows)
        res.to_csv(out_dir / "conformal_val.csv", index=False)
        print(res.round(3).to_string(index=False))
        ap_ = out_dir / f"conformal_orbit_{a.version}.json"
        ap_.write_text(json.dumps(art, indent=1, default=float))
        run.note(f"artifact {ap_} sha256 {hashlib.sha256(ap_.read_bytes()).hexdigest()[:12]}")
        PL.log_config_tried("conformal_intervals", cfg, out_dir, "completed", {"note": "secondary interval method (§5b)"})
        if a.register:
            from sentinel.core.registry import log_result

            for r in res.itertuples():
                log_result(workstream=PL.WORKSTREAM, experiment=f"orbit_conformal_val_{a.version}",
                           model=f"{a.label}:{r.model}", metric=f"cov{r.level}_{r.kind}", value=float(r.coverage),
                           split=f"{r.cell}@{a.version}", n=int(r.n), target=r.target, ci_low=float(r.cov_ci_low),
                           ci_high=float(r.cov_ci_high), ci_method="role-block cluster bootstrap (B=2000)",
                           is_baseline=r.model != "SENTINEL", data_manifests=[f"orbit_{a.version}"],
                           artifacts=[str(ap_)], extra={"width_log10": r.width_log10, "offset_log10": r.offset_log10},
                           notes="validation (development), conformal intervals §5b (secondary)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
