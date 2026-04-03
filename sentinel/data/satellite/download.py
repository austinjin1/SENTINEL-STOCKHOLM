"""
Satellite imagery download for SENTINEL.

Supports two backends:
  1. Google Earth Engine (GEE) — Sentinel-2 L2A and Landsat 8/9 TIRS
  2. Microsoft Planetary Computer STAC API — Sentinel-2 L2A

Imagery is exported as 512x512 tiles at 10 m resolution (5.12 km x 5.12 km).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from sentinel.utils.config import load_config
from sentinel.utils.logging import get_logger, make_progress

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

SENTINEL2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
LANDSAT_THERMAL_BANDS = ["ST_B10"]

SENTINEL2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
LANDSAT8_COLLECTION = "LANDSAT/LC08/C02/T1_L2"
LANDSAT9_COLLECTION = "LANDSAT/LC09/C02/T1_L2"


@dataclass
class AOI:
    """Axis-aligned bounding box in WGS-84."""

    west: float
    south: float
    east: float
    north: float

    def to_list(self) -> list[float]:
        return [self.west, self.south, self.east, self.north]

    def to_geojson(self) -> dict[str, Any]:
        return {
            "type": "Polygon",
            "coordinates": [
                [
                    [self.west, self.south],
                    [self.east, self.south],
                    [self.east, self.north],
                    [self.west, self.north],
                    [self.west, self.south],
                ]
            ],
        }


@dataclass
class DownloadRequest:
    """Parameters for a satellite download job."""

    aoi: AOI
    start_date: date
    end_date: date
    cloud_max_pct: float = 20.0
    tile_size_px: int = 512
    resolution_m: float = 10.0
    bands: list[str] = field(default_factory=lambda: list(SENTINEL2_BANDS))
    output_dir: Path = Path("data/satellite/raw")
    include_thermal: bool = True


# ---------------------------------------------------------------------------
# Google Earth Engine backend
# ---------------------------------------------------------------------------


class GEEDownloader:
    """Download Sentinel-2 and Landsat thermal tiles via Google Earth Engine."""

    def __init__(self, project: str | None = None) -> None:
        import ee

        try:
            ee.Initialize(project=project)
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=project)
        self._ee = ee
        logger.info("Google Earth Engine initialized")

    def search_sentinel2(self, req: DownloadRequest) -> list[dict[str, Any]]:
        """Search Sentinel-2 L2A collection and return image metadata."""
        ee = self._ee
        aoi = ee.Geometry.Rectangle(req.aoi.to_list())
        collection = (
            ee.ImageCollection(SENTINEL2_COLLECTION)
            .filterBounds(aoi)
            .filterDate(req.start_date.isoformat(), req.end_date.isoformat())
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", req.cloud_max_pct))
            .select(req.bands)
            .sort("system:time_start")
        )
        count = collection.size().getInfo()
        logger.info(f"Found {count} Sentinel-2 scenes matching criteria")

        metadata = []
        image_list = collection.toList(count)
        for i in range(count):
            img = ee.Image(image_list.get(i))
            props = img.getInfo()["properties"]
            metadata.append(
                {
                    "id": props.get("system:index", f"scene_{i}"),
                    "datetime": datetime.utcfromtimestamp(
                        props["system:time_start"] / 1000
                    ).isoformat(),
                    "cloud_pct": props.get("CLOUDY_PIXEL_PERCENTAGE", -1),
                    "source": "GEE",
                    "collection": SENTINEL2_COLLECTION,
                }
            )
        return metadata

    def search_landsat_thermal(self, req: DownloadRequest) -> list[dict[str, Any]]:
        """Search Landsat 8/9 TIRS collections."""
        ee = self._ee
        aoi = ee.Geometry.Rectangle(req.aoi.to_list())
        results: list[dict[str, Any]] = []
        for collection_id in (LANDSAT8_COLLECTION, LANDSAT9_COLLECTION):
            col = (
                ee.ImageCollection(collection_id)
                .filterBounds(aoi)
                .filterDate(req.start_date.isoformat(), req.end_date.isoformat())
                .filter(ee.Filter.lt("CLOUD_COVER", req.cloud_max_pct))
                .select(LANDSAT_THERMAL_BANDS)
                .sort("system:time_start")
            )
            count = col.size().getInfo()
            logger.info(f"Found {count} scenes in {collection_id}")
            image_list = col.toList(count)
            for i in range(count):
                img = ee.Image(image_list.get(i))
                props = img.getInfo()["properties"]
                results.append(
                    {
                        "id": props.get("system:index", f"thermal_{i}"),
                        "datetime": datetime.utcfromtimestamp(
                            props["system:time_start"] / 1000
                        ).isoformat(),
                        "cloud_pct": props.get("CLOUD_COVER", -1),
                        "source": "GEE",
                        "collection": collection_id,
                    }
                )
        return results

    def export_tiles(
        self,
        req: DownloadRequest,
        *,
        max_images: int | None = None,
        drive_folder: str = "SENTINEL_tiles",
    ) -> list[str]:
        """Export Sentinel-2 tiles as GeoTIFFs to Google Drive.

        Returns a list of GEE task IDs.
        """
        ee = self._ee
        aoi = ee.Geometry.Rectangle(req.aoi.to_list())
        collection = (
            ee.ImageCollection(SENTINEL2_COLLECTION)
            .filterBounds(aoi)
            .filterDate(req.start_date.isoformat(), req.end_date.isoformat())
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", req.cloud_max_pct))
            .select(req.bands)
            .sort("system:time_start")
        )
        count = collection.size().getInfo()
        if max_images:
            count = min(count, max_images)

        task_ids: list[str] = []
        image_list = collection.toList(count)
        for i in range(count):
            img = ee.Image(image_list.get(i))
            props = img.getInfo()["properties"]
            scene_id = props.get("system:index", f"scene_{i}").replace("/", "_")

            task = ee.batch.Export.image.toDrive(
                image=img.toFloat(),
                description=f"S2_{scene_id}",
                folder=drive_folder,
                region=aoi,
                scale=req.resolution_m,
                maxPixels=1e9,
                crs="EPSG:4326",
                dimensions=f"{req.tile_size_px}x{req.tile_size_px}",
                fileFormat="GeoTIFF",
            )
            task.start()
            task_ids.append(task.id)
            logger.info(f"Started GEE export task: {task.id} ({scene_id})")

        return task_ids

    def wait_for_tasks(
        self, task_ids: list[str], poll_interval: int = 30, timeout: int = 7200
    ) -> dict[str, str]:
        """Poll GEE tasks until all complete or timeout."""
        import ee

        statuses: dict[str, str] = {tid: "UNKNOWN" for tid in task_ids}
        start = time.time()
        while time.time() - start < timeout:
            all_done = True
            for tid in task_ids:
                task = ee.batch.Task(tid)  # type: ignore[attr-defined]
                status = task.status()
                state = status.get("state", "UNKNOWN")
                statuses[tid] = state
                if state not in ("COMPLETED", "FAILED", "CANCELLED"):
                    all_done = False
            if all_done:
                break
            time.sleep(poll_interval)
        return statuses


# ---------------------------------------------------------------------------
# Planetary Computer STAC backend
# ---------------------------------------------------------------------------


class PlanetaryComputerDownloader:
    """Download Sentinel-2 tiles from Microsoft Planetary Computer via STAC."""

    STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    COLLECTION = "sentinel-2-l2a"

    def __init__(self) -> None:
        try:
            import planetary_computer as pc
            import pystac_client

            self._pc = pc
            self._client = pystac_client.Client.open(
                self.STAC_API_URL, modifier=pc.sign_inplace
            )
        except ImportError:
            raise ImportError(
                "Install planetary-computer and pystac-client: "
                "pip install planetary-computer pystac-client"
            )
        logger.info("Planetary Computer STAC client initialized")

    def search(self, req: DownloadRequest) -> list[dict[str, Any]]:
        """Search for Sentinel-2 items matching the request."""
        search = self._client.search(
            collections=[self.COLLECTION],
            bbox=req.aoi.to_list(),
            datetime=f"{req.start_date.isoformat()}/{req.end_date.isoformat()}",
            query={"eo:cloud_cover": {"lt": req.cloud_max_pct}},
            max_items=500,
        )
        items = list(search.items())
        logger.info(f"Planetary Computer: found {len(items)} Sentinel-2 items")

        results = []
        for item in items:
            results.append(
                {
                    "id": item.id,
                    "datetime": item.datetime.isoformat() if item.datetime else "",
                    "cloud_pct": item.properties.get("eo:cloud_cover", -1),
                    "source": "planetary_computer",
                    "bbox": list(item.bbox) if item.bbox else [],
                    "assets": list(item.assets.keys()),
                }
            )
        return results

    def download_tile(
        self,
        item_meta: dict[str, Any],
        req: DownloadRequest,
        *,
        output_dir: Path | None = None,
    ) -> Path:
        """Download and crop a single item to a 512x512 tile as numpy array.

        Falls back to rasterio windowed read for efficiency.
        """
        import pystac_client
        import rasterio
        from rasterio.windows import from_bounds

        out_dir = output_dir or req.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Re-fetch the item to get signed URLs
        item = self._client.get_collection(self.COLLECTION)  # type: ignore
        search = self._client.search(
            collections=[self.COLLECTION],
            ids=[item_meta["id"]],
        )
        items = list(search.items())
        if not items:
            raise ValueError(f"Item {item_meta['id']} not found on re-query")
        item = items[0]

        band_map = {
            "B02": "B2", "B03": "B3", "B04": "B4", "B05": "B5",
            "B06": "B6", "B07": "B7", "B08": "B8", "B8A": "B8A",
            "B11": "B11", "B12": "B12",
        }

        arrays: list[np.ndarray] = []
        for asset_key, sentinel_band in band_map.items():
            if sentinel_band not in req.bands:
                continue
            href = item.assets[asset_key].href
            with rasterio.open(href) as src:
                window = from_bounds(
                    req.aoi.west, req.aoi.south, req.aoi.east, req.aoi.north,
                    transform=src.transform,
                )
                data = src.read(
                    1,
                    window=window,
                    out_shape=(req.tile_size_px, req.tile_size_px),
                )
                arrays.append(data)

        tile = np.stack(arrays, axis=-1)  # (H, W, C)
        out_path = out_dir / f"{item_meta['id']}.npy"
        np.save(out_path, tile)
        logger.info(f"Saved tile {out_path.name}: shape={tile.shape}")
        return out_path

    def download_all(
        self,
        req: DownloadRequest,
        *,
        max_items: int | None = None,
    ) -> list[Path]:
        """Search and download all matching tiles."""
        items = self.search(req)
        if max_items:
            items = items[:max_items]

        paths: list[Path] = []
        progress = make_progress()
        with progress:
            task = progress.add_task("Downloading S2 tiles", total=len(items))
            for meta in items:
                try:
                    p = self.download_tile(meta, req)
                    paths.append(p)
                except Exception as exc:
                    logger.warning(f"Failed to download {meta['id']}: {exc}")
                progress.advance(task)
        return paths


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------


def download_satellite(
    aoi: AOI | list[float],
    start_date: str | date,
    end_date: str | date,
    *,
    backend: Literal["gee", "planetary_computer"] = "gee",
    output_dir: str | Path = "data/satellite/raw",
    cloud_max_pct: float = 20.0,
    include_thermal: bool = True,
    max_images: int | None = None,
    gee_project: str | None = None,
) -> dict[str, Any]:
    """Unified entry point for satellite data download.

    Parameters
    ----------
    aoi:
        Bounding box as AOI or ``[west, south, east, north]``.
    start_date, end_date:
        Date range.
    backend:
        ``"gee"`` for Google Earth Engine, ``"planetary_computer"`` for
        Microsoft Planetary Computer.
    output_dir:
        Directory for downloaded tiles.
    cloud_max_pct:
        Maximum cloud cover percentage.
    include_thermal:
        Whether to also download Landsat thermal data (GEE only).
    max_images:
        Cap on number of images to download.
    gee_project:
        GEE project ID (for ee.Initialize).

    Returns
    -------
    dict with keys ``"s2_metadata"``, ``"thermal_metadata"``, ``"paths"``.
    """
    if isinstance(aoi, (list, tuple)):
        aoi = AOI(*aoi)
    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = date.fromisoformat(end_date)

    req = DownloadRequest(
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        cloud_max_pct=cloud_max_pct,
        output_dir=Path(output_dir),
        include_thermal=include_thermal,
    )
    req.output_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {"s2_metadata": [], "thermal_metadata": [], "paths": []}

    if backend == "gee":
        dl = GEEDownloader(project=gee_project)
        result["s2_metadata"] = dl.search_sentinel2(req)
        if include_thermal:
            result["thermal_metadata"] = dl.search_landsat_thermal(req)
        task_ids = dl.export_tiles(req, max_images=max_images)
        result["gee_task_ids"] = task_ids
    elif backend == "planetary_computer":
        dl_pc = PlanetaryComputerDownloader()
        result["s2_metadata"] = dl_pc.search(req)
        result["paths"] = dl_pc.download_all(req, max_items=max_images)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Save metadata
    meta_path = req.output_dir / "download_metadata.json"
    serializable = {
        k: (
            [str(p) for p in v] if k == "paths" else v
        )
        for k, v in result.items()
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info(f"Metadata saved to {meta_path}")

    return result
