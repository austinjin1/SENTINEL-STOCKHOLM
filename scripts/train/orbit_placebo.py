#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Shuffled-imagery placebo (orbit_protocol.md §7 B): on a development cell, permute the imagery (pixels +
geometry) among rows of the same water body and calendar month (context unchanged), re-predict with the
SatWQ-Net seed ensemble and compare chl-a MdSA, bloom AUROC and within-lake rho with the unshuffled
predictions. If the scores do not degrade, imagery is not being used. Development cells only (guard).

    python -B scripts/train/orbit_placebo.py --satwq satwq_base_s0 satwq_base_s1 satwq_base_s2 --label satwq_base
"""

from __future__ import annotations

import argparse
import os
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--satwq", nargs="+", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--cell", default=PL.PRIMARY_CELL)
    ap.add_argument("--n-perm", type=int, default=5)
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="CPU inference, no GPU lock")
    a = ap.parse_args()
    if not a.cpu:
        PL.setup_gpu(f"orbit_placebo_{a.label}")
    import torch

    torch.set_num_threads(2)

    from sentinel.models.orbit.satwq import mixture_exceedance, mixture_quantiles
    from sentinel.models.orbit.teacher import OrbitTeacher

    df, pix, info, masks = PL.load_dev()
    m = masks[a.cell]
    sub = df[m].reset_index(drop=True)
    sp = {k: v[m] for k, v in pix.items()}
    t = OrbitTeacher([paths.RUNS_ROOT / "orbit" / r / "model.pt" for r in a.satwq], device="cpu" if a.cpu else "cuda")
    y = np.log10(sub.chla.to_numpy(float))
    groups = sub.waterbody.to_numpy()
    key = sub.waterbody + "|" + pd.to_datetime(sub.date).dt.strftime("%Y-%m")
    img_cols = ["sun_zenith", "sun_azimuth", "view_zenith", "view_azimuth", "aot_c3", "jrc_dist_land_m", "scl_c3_n_water",
                "platform"]

    def score(dfx, px):
        def ds(nm):
            return PL.make_set(dfx, px, nm)
        w, mu, sg = t._mixture(ds, ctx_off=False)
        j = D.TARGETS.index("chla")
        med = mixture_quantiles(w[:, j], mu[:, j], sg[:, j], (0.5,))[:, 0]
        ok = np.isfinite(y)
        out = {"mdsa": E.mdsa_log10(med[ok] - y[ok]),
               "within_lake_rho": E.within_lake_spearman(y[ok], med[ok], groups[ok])["rho"]}
        for thr in E.BLOOM_THRESHOLDS:
            pr = mixture_exceedance(w[:, j], mu[:, j], sg[:, j], np.log10(thr))
            out[f"bloom{int(thr)}_auroc"] = E.bloom_scores(10 ** y, pr, thr).get("auroc", np.nan)
        return out

    real = score(sub, sp)
    keys = {"waterbody_month": key, "waterbody": sub.waterbody}
    res = {"real": real, "n_rows": int(len(sub)), "levels": {}}
    out = paths.run_dir("orbit", f"placebo_{a.label}")
    for level, kk in keys.items():
        rs = np.random.default_rng(0)
        perms, moved = [], []
        for p in range(a.n_perm):
            order = np.arange(len(sub))
            for _, ix in sub.groupby(kk.to_numpy()).indices.items():
                if len(ix) > 1:
                    order[ix] = rs.permutation(ix)
            moved.append(int((order != np.arange(len(sub))).sum()))
            dfx = sub.copy()
            dfx[img_cols] = sub[img_cols].to_numpy()[order]
            px = {k: v[order] for k, v in sp.items()}
            perms.append(score(dfx, px))
        pl = pd.DataFrame(perms)
        res["levels"][level] = {"placebo_mean": pl.mean().to_dict(), "placebo_sd": pl.std().to_dict(),
                                "rows_moved_mean": float(np.mean(moved))}
        if a.register:
            for k in real:
                log_result(workstream=PL.WORKSTREAM, experiment="orbit_placebo_dev_v1", model=a.label,
                           metric=f"{k}_placebo_{level}", value=float(pl[k].mean()), split=f"{a.cell}@dev_v1",
                           n=len(sub), target="chla", artifacts=[str(out)],
                           notes=f"validation (development); mean over {a.n_perm} imagery permutations within {level}; "
                                 f"mean {np.mean(moved):.0f} rows moved")
    if a.register:
        for k in real:
            log_result(workstream=PL.WORKSTREAM, experiment="orbit_placebo_dev_v1", model=a.label, metric=f"{k}_real",
                       value=real[k], split=f"{a.cell}@dev_v1", n=len(sub), target="chla", artifacts=[str(out)],
                       notes="validation (development); unshuffled reference for the placebo")
    PL.write_json(out / f"placebo_{a.cell}.json", res)
    print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
