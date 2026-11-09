#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download openly licensed hazard / water-mask label sets for the M7 drone student (docs/research/08 §b.1).

Licences were checked on the hosting records on 2026-10-22 (docs/licenses.md):
  * MARIDA (Zenodo 5151941, CC BY 4.0) — Sentinel-2 marine debris / sediment / foam patches.
  * MADOS (Zenodo 10664073, CC BY 4.0) — Sentinel-2 marine pollution incl. oil spills.
  * Oil Spill Drone, De Kerf et al. 2024 (Zenodo 10555314, CC BY 4.0) — drone RGB oil segmentation.
  * Chesapeake Land Cover (LILA, CDLA-Permissive-1.0) — 1 m land-cover labels incl. water aligned
    with NAIP; all label tiles (~0.6 GB) + NAIP 'naip-new' for the val/test tiles only (~15 GB).
  * FloodNet (CDLA-Permissive-1.0) — hosted on a Dropbox shared folder; attempted, see log.
  * LADOS — distributed only through Roboflow Universe (account required): NOT downloaded.

    python -B scripts/data/download_naip_hazard_labels.py --sets marida mados dekerf chesapeake floodnet
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data.sources.aquacorr import download_file  # noqa: E402

log = logging.getLogger("hazard_labels")
UA = {"User-Agent": "SENTINEL-v2 research (public-data download)"}
ZENODO = {"marida": ("5151941", "CC-BY-4.0", "MARIDA: Marine Debris Archive (Kikaki et al. 2022)"),
          "mados": ("10664073", "CC-BY-4.0", "MADOS: Marine Debris and Oil Spill (Kikaki et al. 2024)"),
          "dekerf": ("10555314", "CC-BY-4.0", "Oil Spill Drone: annotated RGB images of oil spills in a port "
                                               "environment (De Kerf et al. 2024)")}
CHES_BUCKET = "public-datasets-lila"
CHES_PREFIX = "lcmcvpr2019/cvpr_chesapeake_landcover/"
FLOODNET_URL = ("https://www.dropbox.com/scl/fo/k33qdif15ns2qv2jdxvhx/ANGaa8iPRhvlrvcKXjnmNRc"
                "?rlkey=ao2493wzl1cltonowjdbrnp7f&dl=1")
MIN_FREE_GB = 160


def free_gb(p: Path) -> float:
    return shutil.disk_usage(p).free / 1e9


def zenodo_set(key: str, root: Path) -> dict:
    rec_id, lic, title = ZENODO[key]
    rec = requests.get(f"https://zenodo.org/api/records/{rec_id}", headers=UA, timeout=120).json()
    got_lic = (rec.get("metadata", {}).get("license") or {}).get("id", "")
    if got_lic.lower() != lic.lower():
        raise RuntimeError(f"{key}: licence on record is {got_lic!r}, expected {lic}")
    out = paths.ensure(root / key)
    (out / "zenodo_record.json").write_text(json.dumps(rec, indent=1))
    files = []
    for f in rec["files"]:
        if free_gb(out) < MIN_FREE_GB + f["size"] / 1e9:
            raise RuntimeError(f"not enough free disk for {f['key']}")
        md5 = (f.get("checksum") or "").split(":", 1)[-1] or None
        log.info("%s: downloading %s (%.0f MB)", key, f["key"], f["size"] / 1e6)
        files.append(download_file(f["links"]["self"], out / f["key"], md5, log=log.info))
    write_manifest(f"hazard_labels_{key}", files=files + [out / "zenodo_record.json"],
                   sources=[f"https://doi.org/10.5281/zenodo.{rec_id}"], description=title, license=lic,
                   stats={"zenodo_record": rec_id, "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    return {"set": key, "files": [p.name for p in files], "license": lic}


def gcs_list(bucket: str, prefix: str) -> list[dict]:
    items, token = [], None
    while True:
        params = {"prefix": prefix, "maxResults": 1000, "fields": "items(name,size,md5Hash),nextPageToken"}
        if token:
            params["pageToken"] = token
        d = requests.get(f"https://storage.googleapis.com/storage/v1/b/{bucket}/o", params=params,
                         headers=UA, timeout=120).json()
        items += d.get("items", [])
        token = d.get("nextPageToken")
        if not token:
            return items


def chesapeake(root: Path, naip_splits: tuple[str, ...] = ("val", "test")) -> dict:
    out = paths.ensure(root / "chesapeake_landcover")
    items = gcs_list(CHES_BUCKET, CHES_PREFIX)
    (out / "listing.json").write_text(json.dumps(items))
    want = []
    for it in items:
        n = it["name"]
        if n.endswith("_lc.tif") or n.endswith("_nlcd.tif"):
            want.append(it)
        elif n.endswith("_naip-new.tif") and any(f"-{s}_tiles/" in n for s in naip_splits):
            want.append(it)
    log.info("chesapeake: %d files (%.1f GB)", len(want), sum(int(i["size"]) for i in want) / 1e9)
    files = []
    for k, it in enumerate(want):
        rel = it["name"][len(CHES_PREFIX):]
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if free_gb(out) < MIN_FREE_GB:
            raise RuntimeError("free disk below threshold; stopping chesapeake download")
        if not (dest.exists() and dest.stat().st_size == int(it["size"])):
            download_file(f"https://storage.googleapis.com/{CHES_BUCKET}/{it['name']}", dest, None, log=log.info)
        files.append(dest)
        if k % 100 == 0:
            log.info("chesapeake %d/%d", k, len(want))
    write_manifest("hazard_labels_chesapeake_landcover", files=files, hash_files=False,
                   sources=["https://lila.science/datasets/chesapeakelandcover",
                            f"https://storage.googleapis.com/{CHES_BUCKET}/{CHES_PREFIX}"],
                   description="Chesapeake Land Cover (Robinson et al. 2019): all 1 m Chesapeake Conservancy "
                               "land-cover label tiles (class 1 = water) and NLCD tiles for 732 tiles; NAIP "
                               "'naip-new' imagery only for the val/test tiles. Remaining NAIP is on Planetary "
                               "Computer.",
                   license="CDLA-Permissive-1.0 (labels); NAIP/Landsat public domain",
                   stats={"n_files": len(files), "naip_splits": list(naip_splits),
                          "gcs_md5": {Path(i['name']).name: i.get('md5Hash') for i in want}})
    return {"set": "chesapeake", "n_files": len(files)}


def floodnet(root: Path) -> dict:
    """Dropbox serves the shared folder as one zip (FloodNet.zip, ~13 GB) via a redirect; it does
    not honour HTTP Range, so an interrupted download restarts from zero."""
    import zipfile
    out = paths.ensure(root / "floodnet")
    if free_gb(out) < MIN_FREE_GB + 14:
        raise RuntimeError("not enough free disk for FloodNet (13 GB)")
    dest = download_file(FLOODNET_URL, out / "FloodNet.zip", None, log=log.info)
    with zipfile.ZipFile(dest) as z:
        names = z.namelist()
    info = {"n_members": len(names), "bytes": dest.stat().st_size,
            "top_level": sorted({n.split("/")[0] for n in names})[:20]}
    (out / "contents.json").write_text(json.dumps(info, indent=1))
    write_manifest("hazard_labels_floodnet", files=[dest], hash_files=False,
                   sources=["https://github.com/BinaLab/FloodNet-Supervised_v1.0", FLOODNET_URL],
                   description="FloodNet supervised v1.0 (Rahnemoonfar et al. 2021): UAV (DJI Mavic Pro) RGB "
                               "after Hurricane Harvey, 2,343 images, 10 classes incl. water and flooded road/building.",
                   license="CDLA-Permissive-1.0", stats=info)
    return {"set": "floodnet", "downloaded": True, **info}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=["marida", "dekerf", "mados", "chesapeake", "floodnet"])
    a = ap.parse_args()
    root = paths.raw("hazard_labels")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(root / "download.log"), logging.StreamHandler()])
    results = []
    for s in a.sets:
        try:
            if s in ZENODO:
                results.append(zenodo_set(s, root))
            elif s == "chesapeake":
                results.append(chesapeake(root))
            elif s == "floodnet":
                results.append(floodnet(root))
        except Exception as e:  # noqa: BLE001 — record and continue with the other sets
            log.error("%s failed: %r", s, e)
            results.append({"set": s, "error": repr(e)})
    (root / "summary.json").write_text(json.dumps(results, indent=1, default=str))
    log.info("done: %s", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
