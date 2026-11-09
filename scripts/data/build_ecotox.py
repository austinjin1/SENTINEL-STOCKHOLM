#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build processed('ecotox'): tidy aquatic acute/chronic endpoint table from the ECOTOX ASCII release.

Input: the converted release under $SENTINEL_DATA/raw/ecotox/release_<date>/ (download_ecotox.py).
Output ($SENTINEL_DATA/processed/ecotox/):
  ecotox_aquatic.parquet   one row per ECOTOX result: aquatic tests (organism_habitat == Water),
                           fish / crustaceans / algae / insects / molluscs / amphibians,
                           L(E)C50/IC50, ECx/LCx (x <= 25), NOEC/LOEC/NOEL/LOEL/MATC; concentration in
                           mg/L and mol/L where convertible, duration in hours, taxonomy, CAS/DTXSID
  references_aquatic.parquet  bibliographic records cited by those rows
  unit_inventory.csv       every concentration unit seen, with row counts and how it was handled
  summary.json
Molar <-> mass conversion needs the ``chemicals`` dataset (MW); run build_chemicals.py and then this
script again (the first pass converts only within mass or molar units).

    python -B scripts/data/build_ecotox.py [--release 2026-09-15]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import read_manifest, write_manifest  # noqa: E402
from sentinel.data import chemtox_ecotox as CE  # noqa: E402
from sentinel.data.sources import ecotox as E  # noqa: E402

log = logging.getLogger("build_ecotox")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default=None)
    a = ap.parse_args()
    out = paths.processed("ecotox")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out / "build.log"), logging.StreamHandler()],
    )
    tdir = E.table_dir(a.release)
    release = tdir.name.replace("release_", "")
    log.info("ECOTOX release %s (%s)", release, tdir)
    tests = E.read_table("tests", release=release)
    results = E.read_table("results", release=release)
    species = E.read_table("species", release=release)
    chemicals = E.read_table("chemicals", release=release)
    references = E.read_table("references", release=release)
    media = E.read_table(
        "media_characteristics",
        release=release,
        columns=[
            "result_id",
            "media_ph_mean",
            "media_temperature_mean",
            "media_temperature_unit",
            "media_hardness_mean",
            "media_hardness_unit",
            "dissolved_oxygen_mean",
            "dissolved_oxygen_unit",
            "media_salinity_mean",
            "media_salinity_unit",
        ],
    )
    chem_path = paths.PROCESSED / "chemicals" / "chem_ids.parquet"
    chem = pd.read_parquet(chem_path) if chem_path.exists() else None
    if chem is None:
        log.warning(
            "chemicals dataset not built: no mg/L <-> mol/L cross conversion (re-run after build_chemicals)"
        )
    n_aq_tests = int((tests["organism_habitat"] == "Water").sum())
    df = CE.build_aquatic(
        tests, results, species, chemicals, references=references, media=media, chem_ids=chem
    )
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("string")
    df.to_parquet(out / "ecotox_aquatic.parquet", index=False, compression="zstd")
    refs = references[references["reference_number"].isin(df["reference_number"].dropna().unique())]
    refs.to_parquet(out / "references_aquatic.parquet", index=False, compression="zstd")
    inv = (
        df.groupby(["conc1_unit", "conc_basis", "conc_unit_core", "conc_unit_kind"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    inv.to_csv(out / "unit_inventory.csv", index=False)
    has_val = df["conc_value"].notna()
    summary = {
        "release": release,
        "n_aquatic_tests_in_release": n_aq_tests,
        "rows": int(len(df)),
        "rows_with_value": int(has_val.sum()),
        "chemicals": int(df["cas"].nunique()),
        "species": int(df["species_number"].nunique()),
        "references": int(df["reference_number"].nunique()),
        "rows_by_taxon_group": {k: int(v) for k, v in df["taxon_group"].value_counts().items()},
        "species_by_taxon_group": {
            k: int(v) for k, v in df.groupby("taxon_group")["species_number"].nunique().items()
        },
        "rows_by_endpoint": {k: int(v) for k, v in df["endpoint_norm"].value_counts().items()},
        "rows_by_exposure_class": {k: int(v) for k, v in df["exposure_class"].value_counts().items()},
        "rows_by_water_type": {
            str(k): int(v) for k, v in df["water_type"].value_counts(dropna=False).items()
        },
        "conc_conversion": {k: int(v) for k, v in df.loc[has_val, "conc_conversion"].value_counts().items()},
        "frac_rows_with_mg_L": float(df.loc[has_val, "conc_mg_L"].notna().mean()),
        "frac_rows_with_mol_L": float(df.loc[has_val, "conc_mol_L"].notna().mean()),
        "censored_rows": int(df["censored"].sum()),
        "censor_kind": {str(k): int(v) for k, v in df["censor_kind"].value_counts().items()},
        "value_flags": {str(k): int(v) for k, v in df["conc_value_flag"].value_counts().items()},
        "mw_source": {str(k): int(v) for k, v in df["mw_source"].value_counts(dropna=False).items()},
        "rows_with_duration_h": int(df["duration_h"].notna().sum()),
        "chemicals_with_structure": (
            int(df.loc[df["chem_smiles"].notna(), "cas"].nunique()) if "chem_smiles" in df.columns else None
        ),
        "n_distinct_units": int(df["conc1_unit"].nunique()),
        "chem_ids_used": chem is not None,
    }
    pulls = sorted((paths.RAW / "ecotox" / "data_pull").glob("aquatic_all_*_std_subset.parquet"))
    if pulls:
        pull = pd.read_parquet(pulls[-1])
        summary["epa_crosscheck"] = {"pull_file": pulls[-1].name, **CE.crosscheck_epa_standardised(df, pull)}
        log.info("EPA cross-check: %s", json.dumps(summary["epa_crosscheck"], default=str)[:2000])
    else:
        log.info("no EPA data-pull subset found (scripts/data/download_ecotox_pull.py); cross-check skipped")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("summary: %s", json.dumps(summary))
    raw_stats = read_manifest("ecotox_raw")["stats"]
    write_manifest(
        "ecotox",
        files=[
            out / f
            for f in (
                "ecotox_aquatic.parquet",
                "references_aquatic.parquet",
                "unit_inventory.csv",
                "summary.json",
            )
        ],
        sources=[E.INDEX_URL + raw_stats.get("zip", "")],
        description="Tidy ECOTOX aquatic endpoint table (LC50/EC50/IC50, ECx, NOEC/LOEC...) for fish, "
        "crustaceans, algae, insects, molluscs and amphibians; mg/L + mol/L; hours.",
        license=E.LICENSE,
        stats=summary
        | {"raw_manifest": "ecotox_raw", "chem_manifest": "chemicals" if chem is not None else None},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
