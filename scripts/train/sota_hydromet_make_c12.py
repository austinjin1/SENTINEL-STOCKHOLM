#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Write ``configs/hydromet_sota/C12.yaml`` only if the §6 gate passes (``hydromet_sota_protocol.md`` §4: C12 = best
lever-1 data + best lever-2 option, run only if BOTH levers move the held-out-group loss).

    python -B scripts/train/sota_hydromet_make_c12.py     # reads $SENTINEL_RUNS/hydromet/sota/score_inner_v1/decisions.json

Composition (mechanical): data flags ``lever1a``/``weak`` and ``weak_share`` from the lever-1 winner; ``aux_q``/
``q_only_sites`` and ``aux`` from the lever-2 winner; everything else is the common base recipe.
"""

from __future__ import annotations

import json
import sys

import yaml


def main() -> int:
    from sentinel.core import paths
    from sentinel.hydromet_sota import runner as SR

    dec = json.loads((SR.ROOT / "score_inner_v1" / "decisions.json").read_text())
    l1, l2 = dec["levers"].get("1", {}), dec["levers"].get("2", {})
    if not (l1.get("moves") and l2.get("moves")):
        print(f"gate not passed (lever 1 moves={l1.get('moves')}, lever 2 moves={l2.get('moves')}): C12 is not run")
        return 1
    d = paths.REPO_ROOT / "configs" / "hydromet_sota"
    a = yaml.safe_load((d / f"{l1['screening_winner']}.yaml").read_text())
    b = yaml.safe_load((d / f"{l2['screening_winner']}.yaml").read_text())
    c = dict(a, model="sota_C12", lever_id="C12", lever=12,
             data={"lever1a": a["data"]["lever1a"], "weak": a["data"]["weak"], "q_only_sites": b["data"]["q_only_sites"],
                   "aux_q": b["data"]["aux_q"]},
             weak_share=a["weak_share"], aux=b["aux"])
    out = d / "C12.yaml"
    if out.exists():
        raise SystemExit(f"{out} exists")
    hdr = ("# MIT License — Bryan Cheng, 2026\n# Part of SENTINEL v2\n# M6 SOTA lever C12 (protocol §4): written by "
           f"sota_hydromet_make_c12.py from {l1['screening_winner']} (lever 1) + {l2['screening_winner']} (lever 2).\n")
    out.write_text(hdr + yaml.safe_dump(c, sort_keys=False))
    print(out.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
