#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D7 TroutRefuge driver (docs/preregistration/downstream_troutrefuge.md). Validation only.

    python -B scripts/train/downstream_troutrefuge.py --freeze-split
    python -B scripts/train/downstream_troutrefuge.py --build-only     # counts + power (labels of train/val cycles only)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.models.downstream import common as C  # noqa: E402
from sentinel.models.downstream import troutrefuge as TT  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freeze-split", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--phase1", action="store_true", help="fit train cycles, score validation (protocol §3-§4)")
    ap.add_argument("--phase1b", action="store_true", help="MSST comparator + CQR intervals (protocol §8, 2026-11-10)")
    a = ap.parse_args(argv)
    if a.freeze_split:
        print(json.dumps({"split_sha256": TT.freeze()}))
        return 0
    split, sha = C.load_split(TT.APP, expect_sha=TT.SPLIT_SHA)
    if a.phase1:
        from sentinel.models.downstream import troutrefuge_models as TM

        C.limit_threads()
        with C.cpu_slot("troutrefuge_phase1"):
            res = TM.phase1(split, sha)
        print(json.dumps({"primary": res["primary_challenger"], "selection": {k: v["best_inner_auroc"] for k, v in res["selection"].items()},
                          "sets": {s: {k: round(v["auroc"], 4) for k, v in r["scores"].items()} for s, r in res["sets"].items()}}, indent=1))
        return 0
    if a.phase1b:
        from sentinel.models.downstream import troutrefuge_models as TM

        C.limit_threads()
        with C.cpu_slot("troutrefuge_phase1b"):
            res = TM.phase1b(split, sha)
        print(json.dumps({"gbm_repro_max_abs_diff": res["gbm_repro_max_abs_diff"], "msst_binary_rule": res["msst_binary_rule"],
                          "sets": {s: {"msst_auroc": round(r["msst_rule"]["auroc"], 4),
                                       "paired": {k: [round(v["diff"], 4), round(v["ci_low"], 4), round(v["ci_high"], 4), v["p_value"],
                                                      [round(x, 4) for x in v["per_seed_diff"]]] for k, v in r["paired"].items()},
                                       "holm_six": {k: v["reject"] for k, v in r["holm_six"].items()}} for s, r in res["sets"].items()},
                          "numeric": {k: (v if not isinstance(v, dict) else
                                          {"qgbm_rmse": round(v["models"]["qgbm_med"]["rmse"], 3), "msst_rmse": round(v["models"]["msst"]["rmse"], 3),
                                           "msst_r2": round(v["models"]["msst"]["r2"], 3), "msst_n": v["models"]["msst"]["n"],
                                           "rmse_diff": [round(v["rmse_diff_qgbm_minus_msst"][x], 3) for x in ("diff", "ci_low", "ci_high")],
                                           "pi90": {t: [round(p["coverage"], 3), round(p["mean_width_c"], 2)] for t, p in v["pi90"].items()}})
                                      for k, v in res["numeric"].items()}}, indent=1))
        return 0
    u = C.drop_protected(TT.load_units())
    C.assert_no_test(u)
    out = {"split_sha256": sha, "y_cold": TT.power_check(u, "y_cold"), "y_salmonid": TT.power_check(u, "y_salmonid")}
    C.write_json(C.run_dir(TT.APP, "build") / "build_summary.json", out)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
