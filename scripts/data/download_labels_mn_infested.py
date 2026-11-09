#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download the Minnesota DNR Infested Waters List and build MusselGuard listing tables.

The workbook is small (~120 KB) and is kept in raw('mn_infested') under its list date
(``infested-waters_<YYYY-MM-DD>.xlsx``) so every vintage stays rebuildable. The DNR web page
(definitions and headline counts) is snapshotted next to it.

    python -B scripts/data/download_labels_mn_infested.py
    python -B scripts/data/download_labels_mn_infested.py --xlsx <saved vintage.xlsx>   # rebuild offline
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data.sources import downstream_dreissena as D  # noqa: E402
from sentinel.data.sources import events_common as C  # noqa: E402

log = logging.getLogger("mn_infested")

RESEARCH_COUNTS = {"rows": 1526, "zm_confirmed_lakes_wetlands": 398, "zm_connected": 302}  # docs/research/07


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx", type=Path, default=None,
                    help="parse this saved workbook instead of downloading")
    a = ap.parse_args()

    raw = paths.raw("mn_infested")
    out = paths.processed("labels_mn_infested")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(raw / "download.log"), logging.StreamHandler()])
    retrieved = C.retrieval_date()
    headers = {}
    if a.xlsx is None:
        headers = C.http_head(D.MN_XLSX_URL)
        tmp = C.download_file(D.MN_XLSX_URL, raw / "infested-waters.download.xlsx", overwrite=True)
        version = D.mn_list_version(C.read_xlsx(tmp, 0, header=None).iloc[0, 0]) or retrieved
        xlsx = raw / f"infested-waters_{version}.xlsx"
        tmp.replace(xlsx)
        log.info("downloaded %s (%s bytes, Last-Modified %s)", xlsx.name, headers.get("content_length"),
                 headers.get("last_modified"))
        page = raw / f"infested_page_{retrieved}.html"
        page.write_text(C.get_text(D.MN_PAGE_URL))
        terms = raw / f"disclaimers_and_policies_{retrieved}.html"
        terms.write_text(C.get_text(D.MN_TERMS_URL))
    else:
        xlsx = a.xlsx
        pages = sorted(raw.glob("infested_page_*.html"))
        page = pages[-1] if pages else None

    df, meta = D.parse_mn_infested(xlsx)
    descriptions = D.mn_column_descriptions(xlsx)
    page_counts = D.mn_page_counts(page.read_text()) if page else {}
    desc_path = C.save_json({"list_version": meta["list_version"], "column_descriptions": descriptions},
                            raw / f"column_descriptions_{meta['list_version']}.json")

    zm = df[df["species_common"] == "zebra mussel"].reset_index(drop=True)
    f_all = C.write_parquet(df, out / "mn_infested_waters.parquet")
    f_zm = C.write_parquet(zm, out / "mn_infested_zebra_mussel.parquet")

    def vc(s):
        return {str(k): int(v) for k, v in s.fillna("(missing)").value_counts().items()}

    zm_conf = zm[zm["listing_basis"] == "confirmed"]
    stats = {
        "retrieval_date": retrieved, "list_version": meta["list_version"], "title": meta["title"],
        "http_headers": headers, "source_file": str(xlsx), "n_rows": len(df),
        "n_waterbody_species_rows_by_species": vc(df["species_common"]),
        "listing_basis": vc(df["listing_basis"]), "confirmed_flag": vc(df["confirmed_flag"]),
        "dow_class": vc(df["dow_class"]),
        "year_listed_min": int(df["year_listed"].min()), "year_listed_max": int(df["year_listed"].max()),
        "rows_by_year_listed": {int(k): int(v) for k, v in
                                df["year_listed"].value_counts().sort_index().items()},
        "zebra_mussel": {
            "n_rows": len(zm), "listing_basis": vc(zm["listing_basis"]),
            "basis_by_dow_class": {f"{b}|{c}": int(n) for (b, c), n in
                                   zm.groupby(["listing_basis", "dow_class"]).size().items()},
            "n_unique_dow8_confirmed": int(zm_conf["dow8"].nunique()),
            "year_confirmed_min": int(zm_conf["year_confirmed"].min()),
            "year_confirmed_max": int(zm_conf["year_confirmed"].max()),
            "listing_lag_years_quantiles": {str(q): float(zm_conf["listing_lag_years"].dropna().astype(float)
                                                          .quantile(q)) for q in (0.1, 0.5, 0.9)},
            "n_negative_listing_lag": int((zm_conf["listing_lag_years"] < 0).sum()),
        },
        "web_page_counts": page_counts, "research_counts_2026_10_22": RESEARCH_COUNTS,
        "extra_cells": df.loc[df["extra_cell"].notna(), ["row_number", "extra_cell"]].to_dict("records"),
        "parse_meta": meta,
    }
    C.save_json(stats, out / "mn_infested_stats.json")
    terms_files = sorted(raw.glob("disclaimers_and_policies_*.html"))[-1:]
    files = [f_all, f_zm, out / "mn_infested_stats.json", xlsx, desc_path, *terms_files]
    files += [page] if page else []
    write_manifest(
        "labels_mn_infested", files=files, sources=[D.MN_XLSX_URL, D.MN_PAGE_URL, D.MN_TERMS_URL],
        description="Minnesota DNR Infested Waters List (official regulatory list): one row per water "
                    "body x aquatic invasive species with year listed, year first confirmed or "
                    "'connected' basis, DOW number (CC-LLLL[-BB] and 8-digit); zebra-mussel-only view "
                    "for MusselGuard rolling tests.",
        license="Minnesota DNR web content: DNR copyright notice (mndnr.gov/copyright, "
                "aboutdnr/disclaimers_and_policies.html) allows viewing/printing for personal and fair use; "
                "republication requires DNR permission -> research use only, do not redistribute the raw "
                "workbook; provided 'as is' without warranty",
        synthetic=False, stats=stats)
    log.info("done: %d rows (%d zebra mussel), list version %s", len(df), len(zm), meta["list_version"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
