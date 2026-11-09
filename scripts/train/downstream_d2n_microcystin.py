#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Driver for the national WQP track `d2n_microcystin` (docs/preregistration/downstream_d2n_microcystin.md).

    python -B scripts/train/downstream_d2n_microcystin.py --freeze-split   # once, before any fit
    python -B scripts/train/downstream_d2n_microcystin.py --build-only     # counts + power check (no fit)
    python -B scripts/train/downstream_d2n_microcystin.py --phase1         # fit train, score val only
Test rows are dropped before any feature is built (common.drop_protected); no test scoring path exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.models.downstream import common as C  # noqa: E402

C.limit_threads()

from sentinel.models.downstream import d2n_microcystin as M  # noqa: E402
from sentinel.models.downstream import wqp_track as T  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(T.run_cli(M))
