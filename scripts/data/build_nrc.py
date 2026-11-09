#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build processed('nrc_incidents') from the NRC yearly workbooks in raw('nrc').

Outputs ($SENTINEL_DATA/processed/nrc_incidents/):
  reports.parquet         one row per NRC report (SEQNOS) — time, type, cause, location, lat/lon,
                          medium, water body, impact flags/counts + report-level material flags
  materials.parquet       one row per (report, material) — CHRIS code, CAS (validated), name,
                          amounts released / in water with units (+ litres / kg where defined)
  nrc_incidents.parquet   flat join of the two, plus the CAS -> chemical-identity join
                          (DTXSID, SMILES, InChIKey, MW, structure class) when the ``chemicals``
                          dataset exists (run scripts/data/build_chemicals.py, then re-run this)
  coverage_by_year.csv    per file-year share of water-reaching reports with coordinates / CAS
  summary.json
All ten sheets are also archived as strings in $SENTINEL_DATA/interim/nrc/sheets/<SHEET>.parquet.

    python -B scripts/data/build_nrc.py [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data import chemtox_nrc as C  # noqa: E402
from sentinel.data.sources import nrc as N  # noqa: E402

log = logging.getLogger("build_nrc")


def parse_one(path: str) -> dict:
    name = Path(path).name
    book = N.read_workbook(path)
    rep = N.tidy_reports(book, source_file=name)
    mat = N.tidy_materials(book, source_file=name)
    sheets = {k: N.sheet_as_strings(v, source_file=name) for k, v in book.items() if k in N.SHEETS}
    counts = {k: int(len(v)) for k, v in book.items()}
    return {
        "name": name,
        "reports": rep,
        "materials": mat,
        "sheets": sheets,
        "sheet_rows": counts,
        "within_sheet_duplicates": rep.attrs.get("within_sheet_duplicates", {}),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    raw = paths.raw("nrc")
    out = paths.processed("nrc_incidents")
    sheets_dir = paths.ensure(paths.interim("nrc") / "sheets")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out / "build.log"), logging.StreamHandler()],
    )
    files = sorted(raw.glob("CY*.xlsx"), key=lambda p: N.year_from_filename(p.name))
    log.info("%d workbooks", len(files))
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(parse_one, [str(f) for f in files]):
            log.info(
                "%s: %d reports, %d material rows", res["name"], len(res["reports"]), len(res["materials"])
            )
            results.append(res)
    reports = pd.concat([r["reports"] for r in results], ignore_index=True)
    materials = pd.concat([r["materials"] for r in results], ignore_index=True)
    # archive sheets
    for sheet in N.SHEETS:
        parts = [r["sheets"][sheet] for r in results if sheet in r["sheets"]]
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(
                sheets_dir / f"{sheet}.parquet", index=False, compression="zstd"
            )
    # duplicates across / within files
    dup_mask = reports["report_id"].duplicated(keep=False)
    dup = reports[dup_mask]
    n_dup_ids = int(dup["report_id"].nunique())
    log.info("report ids appearing more than once: %d (rows %d)", n_dup_ids, len(dup))
    reports = reports[reports["report_id"].notna()]
    # keep the row from the latest file for duplicated ids (later files carry later edits)
    reports = reports.sort_values(["file_year"]).drop_duplicates("report_id", keep="last")
    mat_keep = materials.merge(
        reports[["report_id", "source_file"]], on=["report_id", "source_file"], how="inner"
    )
    n_mat_dropped = len(materials) - len(mat_keep)
    materials = mat_keep
    orphan = ~materials["report_id"].isin(reports["report_id"])
    chem_path = paths.PROCESSED / "chemicals" / "chem_ids.parquet"
    chem = pd.read_parquet(chem_path) if chem_path.exists() else None
    if chem is None:
        log.warning(
            "chemicals dataset not built yet: CAS -> identity join skipped (re-run after build_chemicals)"
        )
    flags = C.report_level_flags(reports, materials, chem)
    reports = reports.merge(flags, on="report_id", how="left")
    flat = C.flat_table(reports, materials, chem)
    reports["_order"] = pd.to_numeric(reports["report_id"], errors="coerce")
    reports = reports.sort_values(["file_year", "_order"]).drop(columns="_order").reset_index(drop=True)
    reports.to_parquet(out / "reports.parquet", index=False, compression="zstd")
    materials.to_parquet(out / "materials.parquet", index=False, compression="zstd")
    flat.to_parquet(out / "nrc_incidents.parquet", index=False, compression="zstd")
    cov = C.coverage_by_file_year(reports)
    cov.to_csv(out / "coverage_by_year.csv", index=False)
    overall = C.coverage_stats(reports)
    cy24 = C.coverage_stats(reports[reports["file_year"] == 2024])
    summary = {
        "n_files": len(files),
        "n_reports": int(len(reports)),
        "n_material_rows": int(len(materials)),
        "n_flat_rows": int(len(flat)),
        "duplicate_report_ids": n_dup_ids,
        "material_rows_dropped_with_duplicate_reports": int(n_mat_dropped),
        "orphan_material_rows": int(orphan.sum()),
        "file_years": [int(reports["file_year"].min()), int(reports["file_year"].max())],
        "incident_datetime_parsed_frac": float(reports["incident_datetime_local"].notna().mean()),
        "coverage_overall": overall,
        "coverage_2024": cy24,
        "chem_join": None
        if chem is None
        else {
            "materials_with_valid_cas": int(materials["cas"].notna().sum()),
            "unique_valid_cas": int(materials["cas"].nunique()),
            "material_rows_with_cas_in_chem_table": int(
                materials["cas"].dropna().isin(set(chem["cas"])).sum()
            ),
            "material_rows_with_structure": int(flat["chem_smiles"].notna().sum()),
            "unique_cas_with_structure": int(flat.loc[flat["chem_smiles"].notna(), "cas"].nunique()),
            "reached_water_rows_with_valid_cas": int(
                (flat["if_reached_water"].fillna(False).astype(bool) & flat["cas"].notna()).sum()
            ),
            "reached_water_rows_with_structure": int(
                (flat["if_reached_water"].fillna(False).astype(bool) & flat["chem_smiles"].notna()).sum()
            ),
            "reached_water_structure_class_counts": {
                str(k): int(v)
                for k, v in flat.loc[
                    flat["if_reached_water"].fillna(False).astype(bool), "chem_structure_class"
                ]
                .value_counts()
                .items()
            },
            "structure_class_counts_material_rows": {
                str(k): int(v) for k, v in flat["chem_structure_class"].value_counts().items()
            },
        },
        "invalid_cas_strings": int(
            (
                materials["cas_raw"].notna()
                & materials["cas"].isna()
                & ~materials["cas_raw"].str.fullmatch(r"0+-0+-0").fillna(False)
            ).sum()
        ),
        "placeholder_cas_000000-00-0": int(
            materials["cas_raw"].str.fullmatch(r"0+-0+-0").fillna(False).sum()
        ),
        "sheet_rows": {r["name"]: r["sheet_rows"] for r in results},
        "within_sheet_duplicate_seqnos": {
            r["name"]: r["within_sheet_duplicates"]
            for r in results
            if any(r["within_sheet_duplicates"].values())
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("coverage 2024: %s", json.dumps(cy24))
    log.info("coverage overall: %s", json.dumps(overall))
    outs = [
        out / f
        for f in (
            "reports.parquet",
            "materials.parquet",
            "nrc_incidents.parquet",
            "coverage_by_year.csv",
            "summary.json",
        )
    ]
    outs += sorted(sheets_dir.glob("*.parquet"))
    write_manifest(
        "nrc_incidents",
        files=outs,
        sources=[N.BASE_URL, N.DICTIONARY_URL],
        description="USCG NRC incident reports 1990-present (initial, unvalidated): tidy "
        "reports, materials and flat join with CAS -> chemical identity.",
        license=N.LICENSE,
        stats={k: v for k, v in summary.items() if k != "sheet_rows"}
        | {"raw_manifest": "nrc_raw", "chem_manifest": "chemicals" if chem is not None else None},
    )
    log.info("done: %d reports, %d material rows", len(reports), len(materials))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
