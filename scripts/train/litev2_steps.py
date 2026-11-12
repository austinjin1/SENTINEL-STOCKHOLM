#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""One-shot steps of M7 architecture round v2 (lead order of 2026-11-10 20:20 EST). CPU only.

    python -B scripts/train/litev2_steps.py row-checks      # Q5 (i), (ii) on the keys of the real rows: counts only
    python -B scripts/train/litev2_steps.py ledger          # GPU-hours logged so far against the 45 GPU-h ceiling
    python -B scripts/train/litev2_steps.py register-init   # after the re-runs of config 3 and of its no-KD twin,
                                                            # before the first slot job: sha256 of the six checkpoints
    python -B scripts/train/litev2_steps.py e3a-count       # once: E3a_turb positives and water bodies on val_n

``register-init`` and ``e3a-count`` need the lead's stamp and write one registry row each; a second ``e3a-count`` is
refused, a second ``register-init`` with other hashes is refused. ``row-checks`` reads keys only (chip id, water
body, site, position, HydroLAKES id, role) and prints counts; ``ledger`` reads the ledger file.
"""

from __future__ import annotations

import argparse
import json

STEPS = ("row-checks", "ledger", "register-init", "e3a-count")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=STEPS)
    a = ap.parse_args(argv)
    from sentinel.models.lite import runner as R

    R.cpu_threads(6)
    from sentinel.core import paths
    from sentinel.models.litev2 import checks as C
    from sentinel.models.litev2 import launch as LA
    from sentinel.models.litev2 import stamp

    if a.step == "ledger":
        print(json.dumps(LA.ledger_summary(), indent=1))
        return 0
    if a.step == "row-checks":
        tr = C.train_keys()
        out = {"refit_table": C.assert_refit_clean(tr), "one_grouping": C.assert_one_grouping(tr)}
        print(json.dumps(out, indent=1))
        return 0
    stamp.require_stamp(f"step {a.step} of the v2 round")
    if a.step == "register-init":
        shas = LA.register_init_checkpoints()
        print(json.dumps(shas, indent=1))
        return 0
    out = C.e3a_turb_count()
    dst = paths.run_dir("lite", "counts_arch_v2") / "e3a_turb_count_national2b.json"
    R.write_json(dst, out)
    print(json.dumps(out, indent=1))
    print("written to", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
