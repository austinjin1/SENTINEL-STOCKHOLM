# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Pre-registered model selection for SENTINEL-Lite (protocol §5, review W4) — validation only.

S = mean over {Secchi, turbidity, TSS} of (student val MdSA / GBM-RGBN val MdSA) on the primary populations
(Secchi lakes/reservoirs; turbidity, TSS all water bodies); lower is better; ties (|ΔS| < 0.01) go to the
smaller model (parameter count from the checkpoint). Reads every ``student_*/summary.json`` (non-smoke) and
the baseline summary; writes ``$SENTINEL_RUNS/lite/selection.json`` and prints the ranking.

    python -B scripts/eval/lite_select.py --baselines $SENTINEL_RUNS/lite/baselines_20261024-221014
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentinel.core import paths

TARGETS = ("secchi", "turbidity", "tss")
TIE = 0.01


def n_params(run: Path) -> int:
    import torch

    ck = next(iter(sorted(run.glob("model_seed*.pt"))), None)
    if ck is None:
        return 10 ** 12
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    return int(sum(v.numel() for k, v in sd.items() if hasattr(v, "numel") and "running" not in k
                   and "num_batches" not in k))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--runs", nargs="*", default=None, help="student run dirs (default: all student_* runs)")
    ap.add_argument("--out", default="selection.json", help="file name under $SENTINEL_RUNS/lite")
    a = ap.parse_args(argv)
    root = paths.RUNS_ROOT / "lite"
    base = json.loads((Path(a.baselines) / "summary.json").read_text())
    gbm = base["summary"]["gbm_rgbn"]
    runs = [Path(r) for r in a.runs] if a.runs else sorted(p for p in root.glob("student_*") if
                                                           (p / "summary.json").exists())
    rows = []
    for r in runs:
        s = json.loads((r / "summary.json").read_text())["scores"]
        ratios = {t: s[t]["value"] / gbm[t]["value"] for t in TARGETS}
        rows.append({"run": r.name, "S": sum(ratios.values()) / len(TARGETS), "ratios": ratios,
                     "mdsa": {t: s[t]["value"] for t in TARGETS}, "n_params": n_params(r)})
    rows.sort(key=lambda x: x["S"])
    best = rows[0]
    ties = [x for x in rows if x["S"] - best["S"] < TIE]
    chosen = min(ties, key=lambda x: (x["n_params"], x["S"]))
    out = {"rule": "protocol §5: S = mean(student/GBM-RGBN val MdSA) over secchi, turbidity, tss; tie <0.01 -> "
                   "smaller model", "gbm_rgbn": {t: gbm[t]["value"] for t in TARGETS}, "ranking": rows,
           "selected": chosen["run"], "n_configurations": len(rows)}
    (root / a.out).write_text(json.dumps(out, indent=1))
    for x in rows:
        print(f"{x['S']:.3f}  {x['run']:<55} " + " ".join(f"{t}={x['mdsa'][t]:.1f}" for t in TARGETS)
              + f"  params={x['n_params']}")
    print("selected:", chosen["run"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
