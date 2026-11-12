# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Memory test that fixes the row cap of S10 (TabPFN) — lead order of 2026-11-10, item 6: the cap is 10,000 rows;
"only if 10,000 rows x 98 features does not fit in one 11 GB GPU, use the largest of 10,000 / 8,000 / 6,000 that
fits, decided by a memory test on training rows only with no validation row loaded, and registered before any
validation number".

    python -m sentinel.core.gpu --name litev2_tabpfn_memtest --min-free-mb 10000 -- \
        python -B scripts/train/litev2_tabpfn_memtest.py --register

What is run, per candidate cap, in a child process (a CUDA crash then costs one candidate, not the test): the S10
fit of the target with the most uncensored training rows (package defaults, seed 0, rows chosen by the water-body
rule of ``tabular.tabpfn_keep``) and the prediction of one chunk of ``tabular.TABPFN_CHUNK`` *training* rows — the
chunk size S10 uses. The predictions are discarded: only their count and a finite flag are kept, no metric is
computed. The first candidate that completes is the cap. Only the v2 training rows are loaded (the row loader is
asked for training roles only). The test is a fit of a v2 comparator, so it needs the lead's stamp. On a GPU the
whole test is one job of block ``tabpfn`` in the GPU-hour ledger (``launch.gpu_job``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

CAPS = (10_000, 8_000, 6_000)
EXPERIMENT = "lite/tabpfn_memtest"


def worker(cap: int, device: str) -> dict:
    import numpy as np

    from sentinel.models.lite import runner as R

    R.cpu_threads(6)
    from sentinel.models.litev2 import harness as H
    from sentinel.models.litev2 import labels2b as L
    from sentinel.models.litev2 import rows as RW
    from sentinel.models.litev2 import stamp
    from sentinel.models.litev2 import tabular as T

    stamp.require_stamp("the TabPFN memory test (a fit of comparator S10 on training rows)")
    tr = RW.train_rows()
    if set(tr["role"]) - {"train", "train_p2"}:
        raise RuntimeError("a non-training row reached the memory test")
    X = H.feature_store(tr, "train", None)["X"][T.parity_columns()].to_numpy(np.float32)
    n_unc = {t: int(tr[f"y_{t}"].notna().sum()) for t in L.TARGETS}
    target = max(n_unc, key=n_unc.get)
    out = {"cap": int(cap), "target": target, "n_uncensored": n_unc, "n_features": int(X.shape[1]),
           "chunk": int(T.TABPFN_CHUNK), "device": device}
    import torch

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        out["gpu_name"] = torch.cuda.get_device_name(0)
        out["gpu_total_mb"] = int(torch.cuda.get_device_properties(0).total_memory // 2**20)
    t0 = time.time()
    p, info = T.fit_predict_tabpfn(X, tr[f"y_{target}"].to_numpy(float), tr["water_body"].to_numpy(),
                                   X[:T.TABPFN_CHUNK], cap=cap, device=device, seeds=(0,))
    out.update({"n_fit": info["n_fit"], "n_water_bodies_fit": info["n_water_bodies_fit"],
                "tabpfn_version": info["tabpfn_version"], "n_predicted": int(len(p)),
                "predictions_finite": bool(np.isfinite(p).all()), "seconds": round(time.time() - t0, 1)})
    if device == "cuda":
        out["peak_allocated_mb"] = int(torch.cuda.max_memory_allocated() // 2**20)
        out["peak_reserved_mb"] = int(torch.cuda.max_memory_reserved() // 2**20)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--register", action="store_true", help="write the registry row (once)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--gpu-budget-h", type=float, default=0.5, help="GPU-hour budget of the test (ledger)")
    a = ap.parse_args(argv)
    if a.worker is not None:
        print("MEMTEST " + json.dumps(worker(a.worker, a.device)), flush=True)
        return 0

    from sentinel.core.registry import read_registry
    from sentinel.models.lite import runner as R
    from sentinel.models.litev2 import stamp

    stamp.require_stamp("the TabPFN memory test (a fit of comparator S10 on training rows)")
    if a.register and any(r.get("workstream") == R.WORKSTREAM and r.get("experiment") == EXPERIMENT
                          for r in read_registry()):
        raise SystemExit("the TabPFN row cap is already registered; it is decided once")
    if a.device == "cuda":
        from sentinel.models.litev2 import launch as LA

        with LA.gpu_job("litev2_tabpfn_memtest", "tabpfn", a.gpu_budget_h):
            tests, cap = run_candidates(a.device)
    else:
        tests, cap = run_candidates(a.device)
    return finish(a, tests, cap)


def run_candidates(device: str) -> tuple[list, int | None]:
    tests, cap = [], None
    for c in CAPS:
        r = subprocess.run([sys.executable, "-B", __file__, "--worker", str(c), "--device", device],
                           capture_output=True, text=True, timeout=3600)
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("MEMTEST ")]
        rec = json.loads(line[-1][8:]) if line else {"cap": c}
        rec["returncode"] = r.returncode
        rec["fits"] = bool(r.returncode == 0 and line and rec.get("predictions_finite")
                           and rec.get("n_predicted", 0) > 0)
        if not rec["fits"]:
            rec["stderr_tail"] = r.stderr[-600:]
        tests.append(rec)
        print(json.dumps(rec), flush=True)
        if rec["fits"]:
            cap = c
            break
    return tests, cap


def finish(a, tests: list, cap: int | None) -> int:
    from sentinel.core import paths
    from sentinel.core.registry import log_result
    from sentinel.models.lite import runner as R

    if cap is None:
        raise SystemExit("none of 10,000 / 8,000 / 6,000 rows fits: report to the lead, S10 is not run")
    out = {"cap": cap, "candidates": list(CAPS), "tests": tests,
           "rule": "largest of 10,000 / 8,000 / 6,000 whose fit and one prediction chunk complete on one 11 GB GPU; "
                   "training rows only"}
    dst = paths.run_dir("lite", "counts_arch_v2") / "tabpfn_memtest.json"
    R.write_json(dst, out)
    print("cap", cap, "written to", dst)
    if a.register:
        row = log_result(workstream=R.WORKSTREAM, experiment=EXPERIMENT, model="litev2_S10_tabpfn",
                         metric="tabpfn_row_cap", value=cap, split="train", n=tests[-1].get("n_fit"),
                         artifacts=[str(dst)],
                         notes="row cap of S10 fixed by a memory test on training rows only (no validation row loaded, "
                               "no metric computed), before any validation number (lead order 2026-11-10, item 6)",
                         extra={"written_by": "M7", **out})
        print("registry row", row["id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
