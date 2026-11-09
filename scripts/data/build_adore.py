#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build processed('adore'): per-challenge statistics and the official split assignments.

For every ADORE challenge file: rows, chemicals, species, taxon groups, target summary, and per
split column the train/test row counts, chemical counts and train<->test chemical overlap. Writes
  challenge_summary.json      nested per-challenge statistics
  split_overlap.csv           one row per (challenge, split)
  split_assignments.parquet   official assignments in long form (challenge, result_id, split,
                              assignment, role, fold) — the only split source M4 may use
and the manifest ``adore``. Also checks that ``result_conc1_mean_mol_log`` == log10(
``result_conc1_mean_mol``) (the research computation used the latter).

    python -B scripts/data/build_adore.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data.sources import adore as A  # noqa: E402

log = logging.getLogger("build_adore")


def main() -> int:
    out = paths.processed("adore")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out / "build.log"), logging.StreamHandler()],
    )
    summaries, rows, assigns = {}, [], []
    for ch in A.CHALLENGES:
        df = A.load_challenge(ch)
        s = A.summarize_challenge(df)
        y = df[A.TARGET].to_numpy(dtype=float)
        ymol = np.log10(df["result_conc1_mean_mol"].to_numpy(dtype=float))
        s["target_vs_log10_mol_max_abs_diff"] = float(np.nanmax(np.abs(y - ymol)))
        s["n_rows_missing_target"] = int(np.isnan(y).sum())
        s["duplicate_result_ids"] = int(df[A.ROW_KEY].duplicated().sum())
        summaries[ch] = s
        for sp, st in s["splits"].items():
            rows.append(
                {
                    "challenge": ch,
                    "split": sp,
                    **{k: v for k, v in st.items() if k != "assignment_counts"},
                    "assignment_counts": json.dumps(st["assignment_counts"]),
                }
            )
        sa = A.split_assignments(df)
        sa.insert(0, "challenge", ch)
        assigns.append(sa)
        occ = s["splits"].get(A.PRIMARY_SPLIT, {})
        log.info(
            "%-12s rows=%6d chem=%5d species=%4d | occurrence train/test %s/%s overlap %s | splits: %s",
            ch,
            s["rows"],
            s["chemicals"],
            s["species"],
            occ.get("n_train"),
            occ.get("n_test"),
            occ.get("chem_overlap"),
            ", ".join(f"{k.replace('split_', '')}:{v['chem_overlap']}" for k, v in s["splits"].items()),
        )
    (out / "challenge_summary.json").write_text(json.dumps(summaries, indent=2))
    ov = pd.DataFrame(rows)
    ov.to_csv(out / "split_overlap.csv", index=False)
    sa = pd.concat(assigns, ignore_index=True)
    sa.to_parquet(out / "split_assignments.parquet", index=False, compression="zstd")
    files = [out / "challenge_summary.json", out / "split_overlap.csv", out / "split_assignments.parquet"]
    bench = {
        ch: {
            "rows": summaries[ch]["rows"],
            "chemicals": summaries[ch]["chemicals"],
            "species": summaries[ch]["species"],
            "splits": {
                sp: {k: v[k] for k in ("n_train", "n_test", "n_chem_train", "n_chem_test", "chem_overlap")}
                for sp, v in summaries[ch]["splits"].items()
            },
        }
        for ch in A.CHALLENGES
    }
    write_manifest(
        "adore",
        files=files,
        sources=[A.CKAN_PACKAGE_URL],
        description="ADORE challenge statistics and official split assignments (long form). "
        "Derived from manifest adore_raw; never re-split.",
        license=A.LICENSE,
        stats={
            "raw_manifest": "adore_raw",
            "target": A.TARGET,
            "primary_split": A.PRIMARY_SPLIT,
            "secondary_split": A.SECONDARY_SPLIT,
            "challenges": bench,
        },
    )
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
