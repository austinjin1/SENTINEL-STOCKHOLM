#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download EPA UCMR 4 occurrence data and build drinking-water precursor/toxin tables (TapRisk, D4).

Outputs in processed('labels_ucmr'):
  ucmr4_source_water.parquet   raw/source-water (SR) total organic carbon + bromide (DBP precursors)
  ucmr4_haa.parquet            HAA5 / HAA6Br / HAA9 group results (distribution system, DS)
  ucmr4_cyanotoxins.parquet    total microcystins, anatoxin-a, cylindrospermopsin (+ congeners) at EP
  ucmr4_haa_ade.parquet        disinfectant type / residual / treatment info per HAA sample event
  ucmr4_cyanotoxin_ade.parquet bloom / toxin questionnaire + treatment info per cyanotoxin sample event
  ucmr4_zipcodes.parquet       ZIP codes served per PWS
The 10.6 MB zip is kept in raw('ucmr') (< 100 MB rule). UCMR 5 (2023-2025) is PFAS + lithium
only and has no TOC/bromide; ``--check-ucmr5`` re-verifies that and records it in the manifest.

    python -B scripts/data/download_labels_ucmr.py --check-ucmr5
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.core.registry import file_sha256  # noqa: E402
from sentinel.data.sources import downstream_drinking_water as D  # noqa: E402
from sentinel.data.sources import events_common as C  # noqa: E402

log = logging.getLogger("labels_ucmr")


def summary(df: pd.DataFrame) -> dict:
    g = df.groupby("analyte")
    return {
        "n_rows": int(len(df)), "n_pws": int(df["pwsid"].nunique()),
        "by_analyte": {k: {"n": int(len(v)), "n_pws": int(v["pwsid"].nunique()),
                           "frac_censored": round(float(v["censored"].mean()), 4),
                           "n_detected": int((~v["censored"]).sum())} for k, v in g},
        "facility_water_type": df["facility_water_type"].value_counts(dropna=False).rename(index=str).to_dict(),
        "sample_point_type": df["sample_point_type"].value_counts(dropna=False).rename(index=str).to_dict(),
        "date_min": str(df["collection_date"].min().date()), "date_max": str(df["collection_date"].max().date()),
        "collection_date_flags": df["collection_date_flag"].value_counts().to_dict(),
        "units": df["unit"].value_counts(dropna=False).rename(index=str).to_dict(),
        "n_states": int(df["state"].nunique()),
    }


def check_ucmr5(tmp: Path) -> dict:
    """Download the UCMR 5 zip to a temp dir, list its contaminants, delete it."""
    head = C.http_head(D.UCMR5_ZIP_URL)
    p = C.download_file(D.UCMR5_ZIP_URL, tmp / "ucmr5-occurrence-data.zip", overwrite=True)
    contaminants: set[str] = set()
    spt: set[str] = set()
    with zipfile.ZipFile(p) as z, z.open("UCMR5_All.txt") as fh:
        for chunk in pd.read_csv(fh, sep="\t", dtype=str, keep_default_na=False, encoding="latin-1",
                                 usecols=["Contaminant", "SamplePointType"], chunksize=500_000):
            contaminants |= set(chunk["Contaminant"])
            spt |= set(chunk["SamplePointType"])
        members = [i.filename for i in z.infolist()]
    p.unlink()
    has_precursors = any(c.lower() in ("total organic carbon", "bromide") for c in contaminants)
    return {"url": D.UCMR5_ZIP_URL, "last_modified": head["last_modified"], "bytes": head["content_length"],
            "members": members, "contaminants": sorted(contaminants), "sample_point_types": sorted(spt),
            "has_toc_or_bromide": has_precursors,
            "decision": "included" if has_precursors else "not included (no TOC/bromide; PFAS + lithium only)"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-ucmr5", action="store_true")
    a = ap.parse_args()
    raw, out = paths.raw("ucmr"), paths.processed("labels_ucmr")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(raw / "download.log"), logging.StreamHandler()])
    retrieved = C.retrieval_date()
    head = C.http_head(D.UCMR4_ZIP_URL)
    zpath = C.download_file(D.UCMR4_ZIP_URL, raw / "ucmr4-occurrence-data.zip")
    C.save_json(dict(head, retrieved=retrieved, sha256=file_sha256(zpath), members=D.zip_members(zpath)),
                raw / "ucmr4_zip_info.json")
    log.info("UCMR4 zip %s (%d bytes, Last-Modified %s)", zpath, zpath.stat().st_size, head["last_modified"])

    sw = D.ucmr4_source_water(zpath)
    haa, cy = D.ucmr4_split_all(zpath)
    haa_ade = D.tidy_ucmr_ade(D.read_ucmr_txt(zpath, "UCMR4_HAA_AddtlDataElem.txt"))
    cy_ade = D.tidy_ucmr_ade(D.read_ucmr_txt(zpath, "UCMR4_Cyanotoxin_AddtlDataElem.txt"))
    zips = D.read_ucmr_txt(zpath, "UCMR4_ZIPCodes.txt")
    zips = pd.DataFrame({"pwsid": C.str_id(zips["PWSID"]), "zipcode": C.str_id(zips["ZIPCODE"], width=5)})
    for name, df in (("ucmr4_source_water", sw), ("ucmr4_haa", haa), ("ucmr4_cyanotoxins", cy),
                     ("ucmr4_haa_ade", haa_ade), ("ucmr4_cyanotoxin_ade", cy_ade), ("ucmr4_zipcodes", zips)):
        C.write_parquet(df, out / f"{name}.parquet")
        log.info("%s: %d rows", name, len(df))

    stats = {"retrieved": retrieved, "url": D.UCMR4_ZIP_URL, "zip_last_modified": head["last_modified"],
             "zip_sha256": file_sha256(zpath), "zip_bytes": zpath.stat().st_size,
             "source_water": summary(sw), "haa": summary(haa), "cyanotoxins": summary(cy),
             "source_water_pws_by_facility_water_type":
                 sw.groupby("facility_water_type")["pwsid"].nunique().to_dict(),
             "n_haa_ade": int(len(haa_ade)), "n_cyanotoxin_ade": int(len(cy_ade)), "n_zipcodes": int(len(zips)),
             "ade_undecoded_responses": int(pd.concat([haa_ade, cy_ade])["response_desc"].isna().sum()),
             "expected_source_water_pws": 3444,
             "verified_source_water_pws": int(sw["pwsid"].nunique()) == 3444}
    if a.check_ucmr5:
        stats["ucmr5_check"] = check_ucmr5(paths.interim("ucmr"))
        log.info("UCMR5: %s", stats["ucmr5_check"]["decision"])
    C.save_json(stats, out / "build_stats.json")
    write_manifest("labels_ucmr", files=sorted(out.glob("*.parquet")) + [out / "build_stats.json", zpath,
                                                                          raw / "ucmr4_zip_info.json"],
                   sources=[D.UCMR4_ZIP_URL, D.UCMR_PAGE_URL] + ([D.UCMR5_ZIP_URL] if a.check_ucmr5 else []),
                   description="EPA UCMR 4 (2018-2020 monitoring) source-water TOC and bromide (DBP precursors), "
                               "HAA group results, cyanotoxin results (censoring kept: censored + detection_limit "
                               "= MRL), additional data elements (treatment, disinfectant, bloom questionnaire) and "
                               "ZIP codes served.",
                   license="US Government work (EPA), public domain", stats=stats)
    log.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
