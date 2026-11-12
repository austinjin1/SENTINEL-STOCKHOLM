#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D4 TapRisk driver (docs/preregistration/downstream_taprisk.md). Validation only.

    python -B scripts/train/downstream_taprisk.py --freeze-split
    python -B scripts/train/downstream_taprisk.py --build-only        # labels, counts, power check (no fit, no StreamCat)
    python -B scripts/train/downstream_taprisk.py --reproduce         # Pennino-design reproduction (protocol §5)
    python -B scripts/train/downstream_taprisk.py --phase1 --task d4a_gw_nitrate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.models.downstream import common as C  # noqa: E402

C.limit_threads()

from sentinel.models.downstream import taprisk as TR  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freeze-split", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--reproduce", action="store_true")
    ap.add_argument("--phase1", action="store_true")
    ap.add_argument("--it2", action="store_true", help="protocol §10/§11 it2 (corrected: C5, C6, W3)")
    ap.add_argument("--numeric", action="store_true", help="protocol §11 numeric track (UCMR4 HAA5/TOC/bromide), validation")
    ap.add_argument("--task", default="d4a_gw_nitrate", choices=list(TR.TASKS))
    a = ap.parse_args(argv)
    if a.freeze_split:
        sha = TR.freeze()
        split, _ = C.load_split(TR.APP)
        from collections import Counter
        print(json.dumps({"split_sha256": sha, "state_roles": Counter(split["assignments"]["state_role"].values())}))
        return 0
    split, sha = C.load_split(TR.APP, expect_sha=TR.SPLIT_SHA)
    if a.build_only:
        out = {"split_sha256": sha}
        for task in TR.TASKS:
            df = TR.build(task, split, with_streamcat=False)
            out[task] = TR.power_check(df)
        C.write_json(C.run_dir(TR.APP, "build") / "build_summary.json", out)
        print(json.dumps(out, indent=1))
        return 0
    if a.numeric:
        from sentinel.models.downstream import taprisk_numeric as TN

        with C.cpu_slot("taprisk_numeric"):
            res = TN.validate(split, sha)
        print(json.dumps({t: {"r2": {m: round(v["r2"], 3) for m, v in r["models"].items()}, "pi90": r["pi90"]}
                          for t, r in res["targets"].items()}, indent=1))
        return 0
    if a.it2:
        from sentinel.models.downstream import taprisk_it2 as T2

        with C.cpu_slot(f"taprisk_it2_{a.task}"):
            res = T2.run(a.task, split, sha)
        print(json.dumps({"selection": res["selection"],
                          "auroc": {es: {k: round(v, 4) for k, v in r["auroc"].items()} for es, r in res["sets"].items()}},
                         indent=1))
        return 0
    if a.reproduce or a.phase1:
        from sentinel.models.downstream import taprisk_models as TM

        with C.cpu_slot(f"taprisk_{'reproduce' if a.reproduce else a.task}"):
            res = TM.reproduce(split, sha) if a.reproduce else TM.phase1(a.task, split, sha)
        print(json.dumps(res.get("summary", res), indent=1, default=str)[:4000])
        return 0
    ap.error("choose a stage")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
