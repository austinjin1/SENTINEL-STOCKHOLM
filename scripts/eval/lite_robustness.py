# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Robustness suite on the val role (protocol §7): glint, haze, resolution, ±5% per-band gain (5 sign draws),
NIR misregistration. Scores the seed ensemble of a student run; MdSA degradation relative to clean.

    python -m sentinel.core.gpu --name lite_robust --prefer 2 -- python -B scripts/eval/lite_robustness.py --run <run_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentinel.models.lite import runner as R

R.cpu_threads(2)   # GPU job (shared-host rule)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sentinel.models.lite import augment as A  # noqa: E402
from sentinel.models.lite import data as D  # noqa: E402
from sentinel.models.lite import evaluate as E  # noqa: E402
from sentinel.models.lite import features as F  # noqa: E402
from sentinel.models.lite import guard as G  # noqa: E402
from sentinel.models.lite.train import Inputs, build_model, predict  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="cpu: no GPU lock, 6 threads")
    ap.add_argument("--pool", default=None, choices=[None, "mean", "median"],
                    help="override chip pooling (default: the run's pool.json, else mean)")
    a = ap.parse_args(argv)
    if a.device == "cuda":
        R.setup_gpu("lite_robustness")
    else:
        torch.set_num_threads(6)
    run = Path(a.run)
    if "smoke" in run.parts:
        a.no_register = True          # review W9
    cks = sorted(run.glob("model_seed*.pt"))
    split = G.load_split()
    va = D.role_frame(split, ("val",), training=True)
    G.assert_no_protected(va["role"])
    chips = np.asarray(D.cached_chips(va, "val"))
    inp = Inputs(chips, D.scene_norm_params(va, D.scene_stats()), F.pooling_weights(chips, va["gsd_m"].to_numpy(), 4),
                 None, a.device)
    y = va[[f"y_{t}" for t in D.TARGETS]].to_numpy(float)
    for j, t in enumerate(D.TARGETS):            # primary populations only (protocol §3)
        y[~E.primary_mask(t, va["site_class"]), j] = np.nan
    models = []
    for ck in cks:
        c = torch.load(ck, map_location="cpu", weights_only=False)
        cfg = {**c["cfg"], "pretrained": False, "gloria_init": None}
        pj = run / "pool.json"
        pool = a.pool or (json.loads(pj.read_text())["pool"] if pj.exists() else "mean")
        cfg["pool"] = pool
        if cfg.get("smoke"):
            a.no_register = True
        m = build_model(cfg)
        m.load_state_dict(c["state_dict"])
        models.append((m.to(a.device).eval(), cfg))

    def ens(pert):
        return np.mean([predict(m, inp, cfg, perturb=pert)[0] for m, cfg in models], 0)

    grid = [("clean", 0)] + A.ROBUSTNESS_GRID
    res = {}
    for kind, level in grid:
        draws = range(A.GAIN_DRAWS) if kind == "gain" else range(1)
        mus = [ens(lambda x, k=kind, l=level, s=s: A.perturb(x, k, l, seed=s)) for s in draws]
        res[f"{kind}:{level}"] = {t: float(np.mean([E.primary_metric(t)[0](y[np.isfinite(y[:, j]), j],
                                                                          mu[np.isfinite(y[:, j]), j]) for mu in mus]))
                                  for j, t in enumerate(D.TARGETS)}
    clean = res["clean:0"]
    for k, v in res.items():
        v["rel_degradation"] = {t: (v[t] - clean[t]) / clean[t] if t in D.REG_TARGETS else (clean[t] - v[t]) / clean[t]
                                for t in D.TARGETS}
        print(k, {t: round(v[t], 3) for t in D.TARGETS}, {t: round(x, 3) for t, x in v["rel_degradation"].items()})
    R.write_json(run / "robustness_val.json", res)
    if not a.no_register:
        from sentinel.core.registry import log_result

        for k, v in res.items():
            for j, t in enumerate(D.TARGETS):
                log_result(workstream=R.WORKSTREAM, experiment="lite_robustness_val", model=f"lite_student_{run.name}",
                           metric=f"{E.primary_metric(t)[1]}@{k}", value=v[t], split="val", target=t,
                           n=int(np.isfinite(y[:, j]).sum()), data_manifests=list(R.DATA_MANIFESTS),
                           artifacts=[str(run / "robustness_val.json")],
                           extra={"rel_degradation": v["rel_degradation"][t]},
                           notes="seed ensemble; perturbation applied to input DN at inference (augment.perturb)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
