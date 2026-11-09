#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Track R sanity check (orbit_protocol.md §7 R): GLORIA in-situ radiometry (S2A-simulated bands), 5-fold
GroupKFold by GLORIA Dataset_ID over the train-role, <= 2021 pretraining rows only. Compares the MDN
re-implementation (B01-B07 Rrs) with the SatWQ-Net spectral encoder (single pixel, B01-B8A, no context).
Not a SOTA arena (in-situ, noise-limited); comparability check with the MDN literature (~46-55% MdSA
leave-one-dataset-out, Pahlevan et al. 2022).

    python -B scripts/train/orbit_gloria_trackr.py --register
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
from sentinel.core.runlog import TrainingRun  # noqa: E402
from sentinel.models.orbit import data as D  # noqa: E402
from sentinel.models.orbit import evaluate as E  # noqa: E402
from sentinel.models.orbit import pipeline as PL  # noqa: E402

MDN_B7 = ("B01", "B02", "B03", "B04", "B05", "B06", "B07")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--mdn-iter", type=int, default=10000)
    ap.add_argument("--register", action="store_true")
    a = ap.parse_args()
    PL.setup_gpu("orbit_gloria_trackr")
    import torch
    from sklearn.model_selection import GroupKFold

    from sentinel.models.orbit import satwq as S
    from sentinel.models.orbit.mdn import MDNRegressor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)  # shared-host rule: GPU jobs use 2 CPU threads
    cfg = vars(a)
    run_dir = paths.run_dir("orbit", "gloria_trackr")
    with TrainingRun(PL.WORKSTREAM, "orbit_gloria_trackr", config=cfg, data=["gloria"], plan_ref=PL.PLAN_REF,
                     artifacts_dir=str(run_dir)) as run:
        g = D.gloria_rows()
        ylog = np.stack([np.log10(g[t].to_numpy(float)) for t in D.TARGETS], 1)
        pred = {m: np.full(ylog.shape, np.nan) for m in ("mdn_reimpl", "satwq_spectral")}
        gkf = GroupKFold(n_splits=a.folds)
        for k, (tr, te) in enumerate(gkf.split(g, groups=g.dataset_id)):
            Xm = np.stack([g[b].to_numpy(float) / np.pi for b in MDN_B7], 1)
            for j, t in enumerate(D.TARGETS):
                ok = np.isfinite(ylog[tr, j])
                if ok.sum() < 50:
                    continue
                mdn = MDNRegressor(n_iter=a.mdn_iter, seed=k, device=device).fit(Xm[tr][ok], ylog[tr, j][ok])
                pred["mdn_reimpl"][te, j] = mdn.predict(Xm[te])
            norm = PL.Norm(np.zeros((len(tr), len(PL.GEO_COLS))), ylog[tr])
            model = S.SatWQNet(n_geo=len(PL.GEO_COLS), n_ctx=len(PL.CTX_COLS)).to(device)
            aug = {"mult_noise": 0.02, "add_noise": 0.0, "band_add_noise": 0.0, "ctx_off": True}
            S.run_epochs(model, PL.gloria_set(g.iloc[tr], norm), inner_set=None, epochs=a.epochs, lr=1e-3, batch=256,
                         device=device, seed=k, aug=aug, y_mean=norm.y_mean, y_std=norm.y_std)
            q = S.predict_quantiles(model, PL.gloria_set(g.iloc[te], norm), device, norm.y_mean, norm.y_std,
                                    qs=(0.5,), ctx_off=True)
            pred["satwq_spectral"][te] = q[:, :, 0]
            print(f"fold {k} done")
        rows = []
        for j, t in enumerate(D.TARGETS):
            y = ylog[:, j]
            preds = {m: p[:, j] for m, p in pred.items()}
            tab, ok = E.score_models(y, preds, g.dataset_id.to_numpy())
            tab.insert(0, "target", t)
            rows.append(tab)
            if a.register:
                for r in tab.itertuples():
                    log_result(workstream=PL.WORKSTREAM, experiment="orbit_trackR_gloria", model=r.model, metric="mdsa",
                               value=r.mdsa, split=f"GLORIA GroupKFold({a.folds}) by Dataset_ID (train role, <=2021)",
                               n=r.n, ci_low=r.mdsa_ci_low, ci_high=r.mdsa_ci_high,
                               ci_method="GLORIA-dataset cluster bootstrap (B=2000)", target=t,
                               is_baseline=r.model == "mdn_reimpl", artifacts=[str(run_dir)],
                               notes="Track R sanity check (in-situ radiometry, simulated S2A bands); not a SOTA arena")
        res = pd.concat(rows)
        res.to_csv(run_dir / "trackr.csv", index=False)
        print(res.round(3).to_string(index=False))
        for r in res.itertuples():
            run.metric(f"mdsa_{r.target}_{r.model}", r.mdsa)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
