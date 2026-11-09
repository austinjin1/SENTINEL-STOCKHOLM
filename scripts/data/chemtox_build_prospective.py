#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build the ChemTox **prospective** test set (ECOTOX records ADORE does not contain).

The ADORE test rows are spent (one pre-registered pass, ``docs/preregistration/chemtox_test_plan.md``).
This script builds a new, previously unseen test set with ADORE's own curation rules — see
:mod:`sentinel.models.chemtox.prospective` for the rules, the tracks and the exclusion logic — and
writes it with a manifest and a sha256, so the pre-registration
(``docs/preregistration/chemtox_prospective_plan.md``) can freeze the file before any model is
scored on it. No model is fitted here and no target value is printed.

    python -B scripts/data/chemtox_build_prospective.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.models.chemtox import prospective as P  # noqa: E402

log = logging.getLogger("chemtox_prospective")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output directory (default processed/chemtox_prospective)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_dir = Path(args.out) if args.out else paths.processed("chemtox_prospective")
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out_dir / "build.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)

    df, report = P.build()
    out_path = out_dir / "prospective.parquet"
    df.to_parquet(out_path, index=False)
    sha = P.file_hash(out_path)
    report["file"] = {"path": str(out_path), "sha256": sha, "n_rows": int(len(df)),
                      "columns": list(df.columns)}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))

    write_manifest(
        "chemtox_prospective",
        files=[out_path],
        sources=["ECOTOX release 2026-09-15 (processed/ecotox/ecotox_aquatic.parquet)",
                 "ADORE release DOI 10.25678/0008C9 (raw/adore/release) — exclusion reference only",
                 "processed/chemicals/chem_ids.parquet"],
        description=("Prospective ChemTox test set: ECOTOX acute LC50/EC50 records curated by "
                     "ADORE's rules, restricted to chemicals absent from ADORE (CAS and InChIKey) "
                     "or to publications after ADORE's last reference year; tracks fish_novel "
                     "(primary), crust_novel, algae_novel, noncarbon, temporal."),
        license="ECOTOX: US EPA public domain; ADORE CC BY 4.0 (used only to define exclusions)",
        synthetic=False,
        stats={k: report[k] for k in ("adore", "curation", "pool", "tracks", "novel_union")},
    )

    for name, rep in report["tracks"].items():
        log.info("track %-12s rows=%-6s chemicals=%-5s species=%-4s refs=%s",
                 name, rep.get("n_rows"), rep.get("n_chemicals"), rep.get("n_species"),
                 rep.get("n_references"))
    log.info("wrote %s (sha256 %s)", out_path, sha)
    print(json.dumps({"path": str(out_path), "sha256": sha,
                      "tracks": {k: v.get("n_rows") for k, v in report["tracks"].items()},
                      "chemicals": {k: v.get("n_chemicals") for k, v in report["tracks"].items()}},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
