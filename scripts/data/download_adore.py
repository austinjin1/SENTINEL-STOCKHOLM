#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download the official ADORE release (Eawag ERIC, DOI 10.25678/0008C9) and convert it to Parquet.

Steps: CKAN ``package_show`` (saved as ckan_package.json) -> download README, LICENSE and the
396 MB zip, verifying each file's sha256 against the CKAN record -> convert every CSV/TSV member
to Parquet under $SENTINEL_DATA/raw/adore/release/<member path>.parquet -> manifest ``adore_raw``.
The zip is deleted after a successful conversion (raw text is not kept, CONTRIBUTING_V2.md) unless
``--keep-zip``; its URL, size and sha256 stay in the manifest so it can be re-fetched and verified.

    python -B scripts/data/download_adore.py [--keep-zip] [--zip PATH_TO_EXISTING_ZIP]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data import chemtox_http as H  # noqa: E402
from sentinel.data.sources import adore as A  # noqa: E402

log = logging.getLogger("download_adore")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-zip", action="store_true")
    ap.add_argument("--zip", default=None, help="use an already-downloaded zip (sha256 is still verified)")
    a = ap.parse_args()
    out = paths.raw("adore")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out / "download.log"), logging.StreamHandler()],
    )
    s = H.make_session()
    pkg = H.get(A.CKAN_PACKAGE_URL, session=s).json()
    (out / "ckan_package.json").write_text(json.dumps(pkg, indent=2))
    meta = pkg["result"]
    log.info(
        "CKAN: %s | metadata_modified %s | %d resources",
        meta["title"],
        meta["metadata_modified"],
        len(meta["resources"]),
    )
    downloads = []
    for res in meta["resources"]:
        name = Path(res["url"]).name
        dest = out / name
        if name.endswith(".zip") and a.zip:
            dest = Path(a.zip)
        rec = H.download(
            res["url"],
            dest,
            expected_sha256=res.get("hash") or None,
            expected_size=int(res["size"]) if res.get("size") else None,
            session=s,
        )
        rec.update(
            {
                "ckan_resource_id": res["id"],
                "ckan_name": res["name"],
                "ckan_last_modified": res.get("last_modified"),
                "ckan_sha256": res.get("hash"),
            }
        )
        log.info(
            "%s: %d bytes sha256 %s (CKAN %s) %s",
            name,
            rec["bytes"],
            rec["sha256"][:12],
            str(res.get("hash"))[:12],
            "reused" if rec.get("reused_existing") else "downloaded",
        )
        downloads.append(rec)
    zrec = next(r for r in downloads if r["path"].endswith(".zip"))
    rel = A.release_dir()
    members = A.convert_release_zip(zrec["path"], rel, log=log)
    (out / "release_members.json").write_text(json.dumps(members, indent=2))
    checks = A.verify_release(zrec["path"], members, log=log)
    (out / "conversion_check.json").write_text(json.dumps(checks, indent=2))
    n_bad = sum(c["mismatches"] != 0 for c in checks)
    if n_bad:
        log.error("%d converted tables differ from their CSV text; zip kept", n_bad)
        a.keep_zip = True
    files = [Path(m["path"]) for m in members] + [
        out / "ckan_package.json",
        out / "release_members.json",
        out / "conversion_check.json",
    ]
    files += [Path(r["path"]) for r in downloads if not r["path"].endswith(".zip")]
    stats = {
        "ckan_metadata_modified": meta["metadata_modified"],
        "doi": meta.get("doi"),
        "downloads": downloads,
        "n_members": len(members),
        "n_tables": sum(1 for m in members if m["rows"] is not None),
        "conversion_cells_checked": sum(c["cells"] for c in checks),
        "conversion_mismatches": sum(max(c["mismatches"], 0) for c in checks),
        "conversion_tables_with_mismatch": n_bad,
        "zip_deleted_after_conversion": not a.keep_zip,
    }
    write_manifest(
        "adore_raw",
        files=files,
        sources=[A.CKAN_PACKAGE_URL] + [r["url"] for r in downloads],
        description="ADORE release (Schür et al. 2023): every CSV/TSV member converted "
        "losslessly to Parquet (zstd); README/LICENSE/tree files copied.",
        license=A.LICENSE,
        stats=stats,
    )
    if not a.keep_zip and not a.zip and not n_bad:
        Path(zrec["path"]).unlink()
        log.info("deleted %s after conversion", zrec["path"])
    log.info("done: %d members", len(members))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
