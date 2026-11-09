#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download EPA NARS tabular files (NRSA 2008-09/2013-14/2018-19/2023-24, NLA 2007/2012/2017/2022).

1. fetch + parse the official data hub page (catalog of every file, saved with the raw data);
2. HEAD each data/metadata file (Content-Length, Last-Modified, ETag);
3. reuse a local copy from ``--reuse-dir`` only when its file name AND byte size equal the
   official file's Content-Length; otherwise download (sequential, polite);
4. write ``manifests/nars_raw.json`` (sha256 of every file, request parameters, retrieval date).

Raw output: $SENTINEL_DATA/raw/nars/<survey_code>/<original file name>

    python -B scripts/data/download_nars.py [--surveys nrsa1819 nla22] [--reuse-dir DIR]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data import bio_common as BC  # noqa: E402
from sentinel.data.sources import nars as N  # noqa: E402

log = logging.getLogger("download_nars")


def find_reusable(reuse_dir: Path | None, name: str, size: int | None) -> Path | None:
    if reuse_dir is None or size is None or not reuse_dir.exists():
        return None
    for p in reuse_dir.rglob(name):
        if p.is_file() and p.stat().st_size == size:
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surveys", nargs="*", default=list(N.SURVEYS.values()))
    ap.add_argument("--reuse-dir", type=Path, default=None,
                    help="directory with previously downloaded copies (matched by name + size)")
    ap.add_argument("--sleep", type=float, default=0.5)
    a = ap.parse_args()

    out = paths.raw("nars")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / "download.log"), logging.StreamHandler()])
    today = dt.date.today().isoformat()
    idx = paths.ensure(out / "_index")
    page = BC.http_request("GET", N.DATA_PAGE_URL).text
    page_path = idx / f"nars_data_page_{today}.html"
    page_path.write_text(page, encoding="utf-8")
    cat = N.parse_data_page(page)
    cat.to_csv(idx / "nars_catalog_all.csv", index=False)
    sel = N.select_files(cat, a.surveys)
    log.info("catalog: %d rows total, %d selected for %s", len(cat), len(sel), a.surveys)

    # actions of earlier runs (a file copied from --reuse-dir in run 1 is 'present' in run 2)
    prev_log = idx / "nars_download_log.csv"
    prev = {}
    if prev_log.exists():
        pl = pd.read_csv(prev_log, dtype=str, keep_default_na=False)
        prev = dict(zip(pl["path"], pl["action"]))
    records = []
    for _, row in sel.iterrows():
        for kind in ("data", "meta"):
            url = row[f"{kind}_url"]
            name = row[f"{kind}_name"]
            if not isinstance(url, str) or not url:
                continue
            dest = paths.ensure(out / row["survey_code"]) / name
            rec = {"survey_code": row["survey_code"], "indicator": row["indicator"], "kind": kind,
                   "label": row["label"], "url": url, "path": str(dest)}
            try:
                info = BC.head_info(url)
            except RuntimeError as e:
                log.error("HEAD failed %s: %s", url, e)
                info = {"content_length": None, "last_modified": None, "etag": None}
            rec.update({k: info.get(k) for k in ("content_length", "last_modified", "etag")})
            size = info.get("content_length")
            if dest.exists() and (size is None or dest.stat().st_size == size):
                earlier = prev.get(str(dest), "")
                rec["action"] = earlier if earlier.startswith(("reused", "downloaded")) else "present"
            else:
                src = find_reusable(a.reuse_dir, name, size)
                if src is not None:
                    shutil.copy2(src, dest)
                    rec["action"] = f"reused:{src}"
                else:
                    try:
                        BC.download_file(url, dest, expected_bytes=size)
                        rec["action"] = "downloaded"
                    except RuntimeError as e:
                        log.error("download failed %s: %s", url, e)
                        rec["action"] = f"failed: {e}"
                    time.sleep(a.sleep)
            if dest.exists():
                rec["bytes"] = dest.stat().st_size
            records.append(rec)
            log.info("%s %s %s (%s bytes)", row["survey_code"], rec["action"][:40], name, rec.get("bytes"))
    # tables published only inside a bundle (nars.INCLUDE_ARCHIVES) are extracted next to it
    import zipfile
    extracted = []
    for rec in list(records):
        name = Path(rec["path"]).name
        if rec["kind"] == "data" and name in N.INCLUDE_ARCHIVES and Path(rec["path"]).exists():
            with zipfile.ZipFile(rec["path"]) as zf:
                if zf.testzip():
                    raise RuntimeError(f"CRC failure in {rec['path']}")
                for member in N.INCLUDE_ARCHIVES[name]:
                    dest = Path(rec["path"]).parent / member
                    if not dest.exists():
                        dest.write_bytes(zf.read(member))
                    extracted.append(str(dest))
                    records.append({**rec, "kind": "extracted", "url": f"{rec['url']}#{member}",
                                    "path": str(dest), "bytes": dest.stat().st_size,
                                    "action": f"extracted_from:{name}"})
    recs = pd.DataFrame(records)
    recs.to_csv(idx / "nars_download_log.csv", index=False)
    ok = recs[recs["path"].map(lambda p: Path(p).exists())]
    failed = recs[~recs.index.isin(ok.index)]
    files = sorted(set(ok["path"])) + [str(page_path), str(idx / "nars_catalog_all.csv"),
                                        str(idx / "nars_download_log.csv")]
    write_manifest(
        "nars_raw", files=files, sources=[N.DATA_PAGE_URL] + sorted(set(ok["url"])),
        description="EPA NARS raw tabular files (data + metadata) for NRSA 2008-09, 2013-14, 2018-19, "
                    "2023-24 and NLA 2007, 2012, 2017, 2022, as published on the NARS data hub. "
                    "Shapefile/bundle zips excluded.",
        license="US Government work (EPA); no license statement on the hub page",
        stats={"retrieved": today, "surveys": a.surveys, "n_files": len(ok),
               "n_failed": len(failed), "failed_urls": failed["url"].tolist(),
               "n_reused": int(ok["action"].astype(str).str.startswith("reused").sum()),
               "reuse_rule": "same file name and byte size as official Content-Length",
               "extracted_from_archives": extracted,
               "catalog_rows_total": len(cat), "catalog_rows_selected": len(sel),
               "last_modified": json.loads(ok.drop_duplicates("path").set_index("path")["last_modified"].to_json())},
    )
    log.info("done: %d files, %d failed", len(ok), len(failed))
    return 0 if failed.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
