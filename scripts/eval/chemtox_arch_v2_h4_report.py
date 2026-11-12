#!/usr/bin/env python
"""M4 arch-v2 H4 (BCF/BCFD/BAF) join-and-QC report, through the guarded reader.

**This script opens no parquet and takes no output path.** Every read is owned by
``arch_v2_guard.load_h4_rows()``, which is one of the five sanctioned readers of a protected table
for T1/T6 work; the join and QC rules live in the pure module
``sentinel.models.chemtox.arch_v2_h4``. The reader writes its own exclusion report, with exclusive
creation and a counter, under a fixed directory of the runs root, 0444 (ruling G-8) — so there is
no ``--out`` to pass and nothing here can be overwritten.

It cannot run until the lead has written the pins file: ``load_h4_rows()`` calls
``verify_freeze()`` first, which requires a lead-written registry row vouching for the pins (G-1).

    python -B scripts/eval/chemtox_arch_v2_h4_report.py --tracks T1 T6
"""

from __future__ import annotations

import argparse
import json

from sentinel.models.chemtox import arch_v2_guard as G


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracks", nargs="+", default=["T1", "T6"], choices=["T1", "T6"],
                    help="which track's AUX-LEAK filters to apply")
    a = ap.parse_args()

    for track in a.tracks:
        rows, rep = G.load_h4_rows(track)
        steps = rep["admissible_for_ablation_A5"][track]
        print(f"=== H4 / {track} ===")
        print(f"  report        : {rep['report_path']}")
        print(f"  pins sha256   : {rep.get('pins_sha256', '(in the report)')}")
        print(f"  cell rule     : {rep['cell_rule']}")
        print(f"  unfiltered    : {rep['unfiltered_usable']['rows']} rows (pre-stamp figure)")
        for key in G.SEVEN_EXCLUSIONS:
            print(f"  removed {key:26s}: {steps.get('removed_' + key)}")
        print(f"  admissible    : {steps['admissible_rows']} rows / "
              f"{steps['admissible_chemicals']} chemicals / {steps['admissible_species']} species")
        print(f"  re-check      : {json.dumps(rep['recheck'], sort_keys=True)}")
        print(f"  returned rows : {len(rows)}")
        print(f"  NOTE          : {rep['superseded_counts']}")


if __name__ == "__main__":
    main()
