#!/usr/bin/env python3
"""Download the GRQA (Global River Quality Archive) dataset.

Source: Zenodo DOI 10.5281/zenodo.7056647
Pre-harmonized: 17M+ records, 43 water quality parameters, global rivers.
This is the easiest large-scale real water quality dataset to obtain.

MIT License — Bryan Cheng, 2026
"""

import os
import time
import zipfile
from pathlib import Path

import requests

DATA_DIR = Path("data/raw/grqa")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# GRQA Zenodo record — get the latest version's files
ZENODO_RECORD = "7056647"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_download_links():
    """Get file download links from Zenodo API."""
    log(f"Querying Zenodo record {ZENODO_RECORD}...")
    resp = requests.get(ZENODO_API, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    files = []
    for f in data.get("files", []):
        files.append({
            "filename": f["key"],
            "size": f["size"],
            "url": f["links"]["self"],
            "checksum": f.get("checksum", ""),
        })
    return files


def download_file(url, dest, expected_size=None):
    """Download a file with progress reporting."""
    if dest.exists():
        actual = dest.stat().st_size
        if expected_size and actual == expected_size:
            log(f"  Already exists: {dest.name} ({actual / 1e6:.1f} MB)")
            return True
        else:
            log(f"  Incomplete: {dest.name} ({actual} vs {expected_size}), re-downloading")

    log(f"  Downloading: {dest.name} ({expected_size / 1e6:.1f} MB)..." if expected_size else f"  Downloading: {dest.name}...")

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    last_report = 0

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192 * 16):
            f.write(chunk)
            downloaded += len(chunk)
            if total and downloaded - last_report > 50e6:  # Report every 50MB
                pct = downloaded / total * 100
                log(f"    {downloaded / 1e6:.0f}/{total / 1e6:.0f} MB ({pct:.0f}%)")
                last_report = downloaded

    log(f"  Done: {dest.name} ({downloaded / 1e6:.1f} MB)")
    return True


def extract_zip(zip_path, dest_dir):
    """Extract a zip file."""
    log(f"  Extracting: {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    log(f"  Extracted to {dest_dir}")


def main():
    log("=" * 60)
    log("GRQA Dataset Download")
    log("=" * 60)

    files = get_download_links()
    log(f"Found {len(files)} files in Zenodo record:")
    total_size = 0
    for f in files:
        size_mb = f["size"] / 1e6
        total_size += f["size"]
        log(f"  {f['filename']:40s} {size_mb:8.1f} MB")
    log(f"  {'TOTAL':40s} {total_size / 1e6:8.1f} MB")

    # Download all files
    for f in files:
        dest = DATA_DIR / f["filename"]
        download_file(f["url"], dest, f["size"])

    # Extract any zip files
    for zf in DATA_DIR.glob("*.zip"):
        extract_zip(zf, DATA_DIR)

    # Report what we got
    log("\n" + "=" * 60)
    log("GRQA Download Complete!")
    log("=" * 60)
    csv_files = list(DATA_DIR.rglob("*.csv"))
    parquet_files = list(DATA_DIR.rglob("*.parquet"))
    log(f"CSV files: {len(csv_files)}")
    log(f"Parquet files: {len(parquet_files)}")

    # Quick data inspection
    import pandas as pd
    for f in csv_files[:3]:
        try:
            df = pd.read_csv(f, nrows=5)
            log(f"  {f.name}: {df.shape[1]} columns, preview: {list(df.columns[:8])}")
        except Exception as e:
            log(f"  {f.name}: error reading ({e})")

    for f in parquet_files[:3]:
        try:
            df = pd.read_parquet(f, engine="pyarrow")
            log(f"  {f.name}: {len(df):,} rows, {df.shape[1]} columns")
            log(f"    Columns: {list(df.columns[:10])}")
        except Exception as e:
            log(f"  {f.name}: error reading ({e})")


if __name__ == "__main__":
    main()
