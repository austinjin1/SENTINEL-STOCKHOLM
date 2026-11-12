#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Pipeline smoke test for SONDE-HF / R-NH (train-role sites × TRAIN block only; perfect-prog observed weather
without an error model; 2 short epochs). Not a result: no score is computed, no forecast file and no registry row is
written. Checks shapes, memory, GPU throughput and the checkpoint/resume path before the step-1 and CV runs."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sentinel.core import paths  # noqa: E402


def main() -> int:
    from sentinel.models.sonde import gpu_budget, runner
    from sentinel.models.sonde_v2 import nwp as NWP
    from sentinel.models.sonde_v2 import train_v2 as TV
    from sentinel.models.sonde_v2.data_v2 import V2Tensors
    from sentinel.models.sonde_v2.prep import NOT_MODELLED, observed_weather_fields, prepare_v2

    kind = sys.argv[1] if len(sys.argv) > 1 else "sonde_hf"
    gpu_budget.widen(10000)
    gpu = runner.setup_gpu(f"sonde_v2_smoke_{kind}", prefer=0)
    t0 = time.time()
    data = prepare_v2(blocks=("train",), roles=("train",))
    sreg = pd.read_parquet(paths.PROCESSED / "sites" / "sites.parquet").set_index("site_key")["utc_offset_h"]
    utc = {s: float(sreg.get(f"USGS-{s}", -6.0)) for s in data.sites}
    fit = data.site_idx(["train"])
    ten = V2Tensors(data, variables=[v for v in data.cube.variables if v not in NOT_MODELLED], L=14,
                    use_weather=True, hourly=(kind == "sonde_hf"), utc_offset=utc, stat_sites=fit)
    ten.set_products("train", NWP.observed_future(observed_weather_fields(data), 7), stat_sites=fit)
    print(f"[smoke] GPU {gpu}; tensors built in {(time.time() - t0) / 60:.1f} min; hourly sites "
          f"{0 if ten.hourly is None else ten.hourly.shape[0]}", flush=True)
    si, ti = ten.train_rows(fit, "train", ["temp_c_mean"] if kind == "rnh" else None)
    rs = np.random.default_rng(0)
    pick = rs.choice(si.size, 20000, replace=False)
    b = ten.batch(si[pick[:256]], ti[pick[:256]], "train")
    for key, val in b.items():
        arrs = val if isinstance(val, list) else [val]
        for a_ in arrs:
            if np.issubdtype(np.asarray(a_).dtype, np.floating) and key != "y":
                print(f"[smoke] input {key}: shape {np.shape(a_)} non-finite {int((~np.isfinite(a_)).sum())}",
                      flush=True)
    cfg = TV.TrainConfig(kind=kind, steps_per_epoch=50, max_epochs=2, es_max_samples=2000,
                         rnh_target="temp_c_mean" if kind == "rnh" else None,
                         head="cmal" if kind == "rnh" else "quantile")
    ck = paths.RUNS_ROOT / "sonde" / "v2_smoke" / kind
    ck.mkdir(parents=True, exist_ok=True)
    for f in ck.glob("*.pt"):
        f.unlink()
    t1 = time.time()
    res = TV.train(cfg, ten, si[pick[:15000]], ti[pick[:15000]], si[pick[15000:]], ti[pick[15000:]], ck,
                   max_minutes=30)
    import torch
    print(f"[smoke] {kind}: {res['epoch']} epochs, {(time.time() - t1) / 100:.2f} s/step incl. ES; peak VRAM "
          f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB; history {res['history']}", flush=True)
    model = TV.build_model(cfg, ten).cuda()
    model.load_state_dict(torch.load(ck / "best.pt"))
    q = TV.predict(model, cfg, ten, si[pick[:512]], ti[pick[:512]], "train", "cuda")
    print(f"[smoke] predict shape {q.shape}, monotone {bool((np.diff(q, axis=-1) >= -1e-4).all())}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
