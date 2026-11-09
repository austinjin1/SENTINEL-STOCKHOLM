#!/usr/bin/env python3
"""Master data acquisition script for SENTINEL.

Orchestrates downloading from all data sources in the correct order.
"""

import argparse
import logging
from pathlib import Path

from sentinel.utils.config import load_config
from sentinel.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Download all SENTINEL data sources")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--data-dir", default="data/", help="Root data directory")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["satellite", "sensor", "microbial", "molecular", "ecotox", "case_studies"],
        help="Which data sources to download",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip already downloaded data")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config(args.config)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    steps = {
        "satellite": _download_satellite,
        "sensor": _download_sensor,
        "microbial": _download_microbial,
        "molecular": _download_molecular,
        "ecotox": _download_ecotox,
        "case_studies": _download_case_studies,
    }

    for source in args.sources:
        if source not in steps:
            logger.warning(f"Unknown source: {source}, skipping")
            continue
        logger.info(f"{'='*60}")
        logger.info(f"Downloading: {source}")
        logger.info(f"{'='*60}")
        try:
            steps[source](config, data_dir, skip_existing=args.skip_existing)
            logger.info(f"Completed: {source}")
        except Exception as e:
            logger.error(f"Failed to download {source}: {e}")
            raise


def _download_satellite(config: dict, data_dir: Path, skip_existing: bool = False):
    from sentinel.data.satellite.download import SatelliteDownloader

    downloader = SatelliteDownloader(config=config["data"]["satellite"], output_dir=data_dir / "raw" / "satellite")
    downloader.download_sentinel2(skip_existing=skip_existing)
    downloader.download_landsat_thermal(skip_existing=skip_existing)


def _download_sensor(config: dict, data_dir: Path, skip_existing: bool = False):
    from sentinel.data.sensor.download import SensorDownloader

    downloader = SensorDownloader(config=config["data"]["sensor"], output_dir=data_dir / "raw" / "sensor")
    downloader.discover_stations()
    downloader.download_all(skip_existing=skip_existing)


def _download_microbial(config: dict, data_dir: Path, skip_existing: bool = False):
    from sentinel.data.microbial.download import MicrobialDownloader

    downloader = MicrobialDownloader(output_dir=data_dir / "raw" / "microbial")
    downloader.download_epa_nars()
    downloader.download_emp()
    downloader.search_ncbi_sra()


def _download_molecular(config: dict, data_dir: Path, skip_existing: bool = False):
    from sentinel.data.molecular.download import MolecularDownloader

    downloader = MolecularDownloader(config=config["data"]["molecular"], output_dir=data_dir / "raw" / "molecular")
    downloader.download_geo_datasets()
    downloader.download_ctd()


def _download_ecotox(config: dict, data_dir: Path, skip_existing: bool = False):
    from sentinel.data.ecotox.download import EcotoxDownloader

    downloader = EcotoxDownloader(output_dir=data_dir / "raw" / "ecotox")
    downloader.download_bulk()


def _download_case_studies(config: dict, data_dir: Path, skip_existing: bool = False):
    from sentinel.data.case_studies.collector import CaseStudyCollector

    collector = CaseStudyCollector(
        config=config["evaluation"]["case_studies"],
        output_dir=data_dir / "raw" / "case_studies",
    )
    collector.collect_all()


if __name__ == "__main__":
    main()
