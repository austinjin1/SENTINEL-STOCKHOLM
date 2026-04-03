"""
Satellite imagery preprocessing for SENTINEL.

Computes spectral water-quality indices, tiles imagery to 5.12 km grid cells,
resizes to 224x224x10 for ViT input, and maintains a rolling temporal buffer.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from sentinel.utils.config import load_config
from sentinel.utils.logging import get_logger, make_progress

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Band index mapping (order matches SENTINEL2_BANDS in download.py)
# ---------------------------------------------------------------------------

BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
BAND_IDX = {name: i for i, name in enumerate(BAND_NAMES)}

# Convenience accessors
_B2 = BAND_IDX["B2"]    # Blue  (490 nm)
_B3 = BAND_IDX["B3"]    # Green (560 nm)
_B4 = BAND_IDX["B4"]    # Red   (665 nm)
_B5 = BAND_IDX["B5"]    # Red-edge 1 (705 nm)
_B6 = BAND_IDX["B6"]    # Red-edge 2 (740 nm)
_B7 = BAND_IDX["B7"]    # Red-edge 3 (783 nm)
_B8 = BAND_IDX["B8"]    # NIR   (842 nm)
_B8A = BAND_IDX["B8A"]  # Narrow NIR (865 nm)
_B11 = BAND_IDX["B11"]  # SWIR-1 (1610 nm)
_B12 = BAND_IDX["B12"]  # SWIR-2 (2190 nm)


# ---------------------------------------------------------------------------
# Spectral index computation
# ---------------------------------------------------------------------------


def _safe_divide(a: np.ndarray, b: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Element-wise division with zero-denominator protection."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(b) > 1e-10, a / b, fill)
    return result.astype(np.float32)


def compute_ndci(tile: np.ndarray) -> np.ndarray:
    """Normalized Difference Chlorophyll Index.

    NDCI = (B5 - B4) / (B5 + B4)
    Mishra & Mishra (2012).
    """
    return _safe_divide(
        tile[..., _B5] - tile[..., _B4],
        tile[..., _B5] + tile[..., _B4],
    )


def compute_fai(tile: np.ndarray) -> np.ndarray:
    """Floating Algae Index.

    FAI = B8 - [B4 + (B11 - B4) * (842 - 665) / (1610 - 665)]
    Hu (2009).
    """
    lam_red, lam_nir, lam_swir = 665.0, 842.0, 1610.0
    baseline = tile[..., _B4] + (tile[..., _B11] - tile[..., _B4]) * (
        (lam_nir - lam_red) / (lam_swir - lam_red)
    )
    return (tile[..., _B8] - baseline).astype(np.float32)


def compute_ndti(tile: np.ndarray) -> np.ndarray:
    """Normalized Difference Turbidity Index.

    NDTI = (B4 - B3) / (B4 + B3)
    Lacaux et al. (2007).
    """
    return _safe_divide(
        tile[..., _B4] - tile[..., _B3],
        tile[..., _B4] + tile[..., _B3],
    )


def compute_mndwi(tile: np.ndarray) -> np.ndarray:
    """Modified Normalized Difference Water Index.

    MNDWI = (B3 - B11) / (B3 + B11)
    Xu (2006).
    """
    return _safe_divide(
        tile[..., _B3] - tile[..., _B11],
        tile[..., _B3] + tile[..., _B11],
    )


def compute_oil_index(tile: np.ndarray) -> np.ndarray:
    """Oil Index based on SWIR contrast.

    OI = (B12 - B11) / (B12 + B11)
    Proxy for hydrocarbon surface films.
    """
    return _safe_divide(
        tile[..., _B12] - tile[..., _B11],
        tile[..., _B12] + tile[..., _B11],
    )


def compute_all_indices(tile: np.ndarray) -> dict[str, np.ndarray]:
    """Compute all five spectral indices for a tile.

    Parameters
    ----------
    tile:
        Array of shape ``(H, W, 10)`` with reflectance values (float or
        uint16 surface reflectance).

    Returns
    -------
    dict mapping index name to ``(H, W)`` float32 array.
    """
    tile_f = tile.astype(np.float32)
    return {
        "NDCI": compute_ndci(tile_f),
        "FAI": compute_fai(tile_f),
        "NDTI": compute_ndti(tile_f),
        "MNDWI": compute_mndwi(tile_f),
        "OilIndex": compute_oil_index(tile_f),
    }


# ---------------------------------------------------------------------------
# Tile specification and indexing
# ---------------------------------------------------------------------------


@dataclass
class TileSpec:
    """A 5.12 km x 5.12 km tile in the SENTINEL grid.

    Indexed by its centroid latitude/longitude and linked to a HUC-8
    watershed identifier.
    """

    centroid_lat: float
    centroid_lon: float
    huc8_id: str = ""
    tile_id: str = ""

    def __post_init__(self) -> None:
        if not self.tile_id:
            self.tile_id = f"{self.centroid_lat:.4f}_{self.centroid_lon:.4f}"


def generate_tile_grid(
    west: float, south: float, east: float, north: float,
    tile_size_km: float = 5.12,
) -> list[TileSpec]:
    """Generate a grid of TileSpecs covering the given bounding box.

    Parameters
    ----------
    west, south, east, north:
        WGS-84 bounding box.
    tile_size_km:
        Tile side length in kilometres (default 5.12 km).
    """
    # Approximate degrees per km at mid-latitude
    mid_lat = (south + north) / 2.0
    deg_per_km_lat = 1.0 / 111.0
    deg_per_km_lon = 1.0 / (111.0 * np.cos(np.radians(mid_lat)))

    step_lat = tile_size_km * deg_per_km_lat
    step_lon = tile_size_km * deg_per_km_lon

    tiles: list[TileSpec] = []
    lat = south + step_lat / 2
    while lat < north:
        lon = west + step_lon / 2
        while lon < east:
            tiles.append(TileSpec(centroid_lat=round(lat, 6), centroid_lon=round(lon, 6)))
            lon += step_lon
        lat += step_lat

    logger.info(f"Generated {len(tiles)} tile specs for AOI")
    return tiles


# ---------------------------------------------------------------------------
# Resize for ViT input
# ---------------------------------------------------------------------------


def resize_tile(tile: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Resize a tile from (H, W, C) to (target_size, target_size, C).

    Uses bilinear interpolation via scipy to avoid heavy CV library deps.
    """
    from scipy.ndimage import zoom

    if tile.ndim != 3:
        raise ValueError(f"Expected 3-D tile, got shape {tile.shape}")

    h, w, c = tile.shape
    if h == target_size and w == target_size:
        return tile.astype(np.float32)

    zoom_factors = (target_size / h, target_size / w, 1.0)
    resized = zoom(tile.astype(np.float32), zoom_factors, order=1)
    return resized.astype(np.float32)


# ---------------------------------------------------------------------------
# Rolling temporal buffer
# ---------------------------------------------------------------------------


@dataclass
class TemporalBuffer:
    """Maintains a rolling buffer of the last *N* cloud-free acquisitions
    per tile.

    Data is stored on disk as ``.npy`` files and an index JSON keeps track
    of ordering and metadata.
    """

    buffer_dir: Path
    max_acquisitions: int = 10
    _index: dict[str, list[dict[str, Any]]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        index_path = self.buffer_dir / "buffer_index.json"
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                self._index = json.load(f)

    def add(
        self,
        tile_id: str,
        tile_data: np.ndarray,
        acquisition_date: str,
        cloud_pct: float,
    ) -> None:
        """Add an acquisition to the buffer for a tile."""
        entries = self._index.setdefault(tile_id, [])

        # Check for duplicate
        if any(e["date"] == acquisition_date for e in entries):
            logger.debug(f"Tile {tile_id} already has {acquisition_date}, skipping")
            return

        # Save array
        fname = f"{tile_id}_{acquisition_date}.npy"
        np.save(self.buffer_dir / fname, tile_data)

        entries.append(
            {"date": acquisition_date, "cloud_pct": cloud_pct, "file": fname}
        )
        entries.sort(key=lambda e: e["date"])

        # Evict oldest if over limit
        while len(entries) > self.max_acquisitions:
            removed = entries.pop(0)
            old_path = self.buffer_dir / removed["file"]
            if old_path.exists():
                old_path.unlink()
            logger.debug(f"Evicted {removed['file']} from buffer")

        self._save_index()

    def get(self, tile_id: str) -> list[np.ndarray]:
        """Return arrays for all buffered acquisitions (oldest first)."""
        entries = self._index.get(tile_id, [])
        arrays = []
        for e in entries:
            path = self.buffer_dir / e["file"]
            if path.exists():
                arrays.append(np.load(path))
        return arrays

    def get_temporal_stack(self, tile_id: str, target_size: int = 224) -> np.ndarray:
        """Return a ``(T, 224, 224, 10)`` stack for the tile, resizing if needed."""
        arrays = self.get(tile_id)
        if not arrays:
            raise ValueError(f"No acquisitions buffered for tile {tile_id}")
        resized = [resize_tile(a, target_size) for a in arrays]
        return np.stack(resized, axis=0)

    def _save_index(self) -> None:
        with open(self.buffer_dir / "buffer_index.json", "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2)


# ---------------------------------------------------------------------------
# High-level preprocessing pipeline
# ---------------------------------------------------------------------------


def preprocess_tile(
    raw_tile: np.ndarray,
    *,
    target_size: int = 224,
    compute_indices: bool = True,
) -> dict[str, np.ndarray]:
    """Full preprocessing for a single raw tile.

    Parameters
    ----------
    raw_tile:
        Array of shape ``(512, 512, 10)`` with surface reflectance.
    target_size:
        Output spatial size for ViT input.
    compute_indices:
        Whether to compute spectral indices on the original-resolution tile.

    Returns
    -------
    dict with keys:
        - ``"tile_224"``: resized tile ``(224, 224, 10)``, float32
        - ``"indices"``: dict of spectral index arrays (original resolution)
    """
    result: dict[str, Any] = {}

    # Scale uint16 reflectance to [0, 1] if needed
    if raw_tile.dtype in (np.uint16, np.int16):
        raw_tile = raw_tile.astype(np.float32) / 10_000.0

    result["tile_224"] = resize_tile(raw_tile, target_size)

    if compute_indices:
        result["indices"] = compute_all_indices(raw_tile)

    return result


def preprocess_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    target_size: int = 224,
    buffer_size: int = 10,
) -> Path:
    """Batch-preprocess all ``.npy`` tiles in a directory.

    Resized tiles are saved to *output_dir*, and a temporal buffer is
    maintained alongside them.

    Returns the output directory path.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_files = sorted(input_dir.glob("*.npy"))
    if not tile_files:
        logger.warning(f"No .npy tiles found in {input_dir}")
        return output_dir

    buffer = TemporalBuffer(
        buffer_dir=output_dir / "temporal_buffer",
        max_acquisitions=buffer_size,
    )

    progress = make_progress()
    with progress:
        task = progress.add_task("Preprocessing tiles", total=len(tile_files))
        for tf in tile_files:
            raw = np.load(tf)
            result = preprocess_tile(raw, target_size=target_size)

            # Save resized tile
            out_path = output_dir / f"processed_{tf.stem}.npy"
            np.save(out_path, result["tile_224"])

            # Save indices
            for idx_name, idx_arr in result.get("indices", {}).items():
                idx_path = output_dir / f"{tf.stem}_{idx_name}.npy"
                np.save(idx_path, idx_arr)

            # Parse date from filename if possible (format: ..._YYYYMMDD.npy)
            stem = tf.stem
            date_str = stem.split("_")[-1] if "_" in stem else "unknown"

            buffer.add(
                tile_id=stem.rsplit("_", 1)[0] if "_" in stem else stem,
                tile_data=result["tile_224"],
                acquisition_date=date_str,
                cloud_pct=0.0,
            )

            progress.advance(task)

    logger.info(f"Preprocessed {len(tile_files)} tiles -> {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# Water-pixel extraction for HydroViT MAE pretraining
# ---------------------------------------------------------------------------


def extract_water_pixels(
    tile: np.ndarray,
    water_mask: np.ndarray,
    patch_size: int = 16,
) -> list[np.ndarray]:
    """Extract water-only patches from a tile using a JRC binary mask.

    Patches where more than 75 % of pixels are classified as water are
    retained for HydroViT masked-autoencoder pretraining.

    Parameters
    ----------
    tile:
        Array of shape ``(H, W, C)`` with spectral data.
    water_mask:
        Binary array of shape ``(H, W)`` where 1 = water, 0 = non-water.
        Will be resized to match *tile* spatial dimensions if necessary.
    patch_size:
        Side length of square patches to extract.

    Returns
    -------
    List of ``(patch_size, patch_size, C)`` arrays where >75 % of pixels
    are water.
    """
    from scipy.ndimage import zoom as scipy_zoom

    h, w = tile.shape[:2]

    # Resize water mask to tile dimensions if needed
    mh, mw = water_mask.shape[:2]
    if (mh, mw) != (h, w):
        mask = scipy_zoom(
            water_mask.astype(np.float32),
            (h / mh, w / mw),
            order=0,  # nearest-neighbour for binary mask
        )
        mask = (mask > 0.5).astype(np.uint8)
    else:
        mask = water_mask.astype(np.uint8)

    patches: list[np.ndarray] = []
    for row_start in range(0, h - patch_size + 1, patch_size):
        for col_start in range(0, w - patch_size + 1, patch_size):
            mask_patch = mask[
                row_start : row_start + patch_size,
                col_start : col_start + patch_size,
            ]
            water_frac = mask_patch.sum() / (patch_size * patch_size)
            if water_frac > 0.75:
                tile_patch = tile[
                    row_start : row_start + patch_size,
                    col_start : col_start + patch_size,
                    :,
                ]
                patches.append(tile_patch.copy())

    logger.info(
        f"Extracted {len(patches)} water patches "
        f"(patch_size={patch_size}) from tile {tile.shape}"
    )
    return patches


# ---------------------------------------------------------------------------
# Multi-resolution alignment (S2 10 m + S3 300 m)
# ---------------------------------------------------------------------------


class MultiResolutionAligner:
    """Align Sentinel-2 (10 m) and Sentinel-3 (300 m) tiles to a common grid.

    S2 provides high spatial detail; S3 provides daily temporal coverage and
    additional OLCI bands useful for water-quality inference.
    """

    def align(
        self,
        s2_tile: np.ndarray,
        s3_tile: np.ndarray,
        s2_geo: dict[str, Any],
        s3_geo: dict[str, Any],
    ) -> dict[str, Any]:
        """Produce aligned representations of an S2/S3 tile pair.

        Parameters
        ----------
        s2_tile:
            Sentinel-2 tile of shape ``(512, 512, C_s2)`` at 10 m.
        s3_tile:
            Sentinel-3 OLCI tile of shape ``(512, 512, C_s3)`` at 300 m.
        s2_geo:
            Georeference dict with at least ``"crs"`` and ``"transform"``
            (affine parameters) for the S2 tile.
        s3_geo:
            Georeference dict for the S3 tile.

        Returns
        -------
        dict with:
            - ``"s2_resampled"``: S2 tile resized to 512 x 512 (identity if
              already that size).
            - ``"s3_upsampled"``: S3 tile bilinearly upsampled to match S2
              spatial grid at 10 m resolution.
            - ``"scale_ratio"``: spatial scale ratio (S3_res / S2_res).
            - ``"s2_geo"``, ``"s3_geo"``: original georeference metadata.
        """
        from scipy.ndimage import zoom as scipy_zoom

        # Ensure float
        s2 = s2_tile.astype(np.float32)
        s3 = s3_tile.astype(np.float32)

        # S3 is 300 m vs S2 10 m => ratio 30x
        s2_res = s2_geo.get("resolution_m", 10.0)
        s3_res = s3_geo.get("resolution_m", 300.0)
        scale_ratio = s3_res / s2_res  # typically 30

        # Upsample S3 to the S2 spatial grid dimensions
        target_h, target_w = s2.shape[0], s2.shape[1]
        s3_h, s3_w = s3.shape[0], s3.shape[1]
        zoom_h = target_h / s3_h
        zoom_w = target_w / s3_w
        s3_upsampled = scipy_zoom(s3, (zoom_h, zoom_w, 1.0), order=1)

        logger.info(
            f"Aligned S2 {s2.shape} + S3 {s3.shape} -> "
            f"S3 upsampled to {s3_upsampled.shape} (scale_ratio={scale_ratio})"
        )

        return {
            "s2_resampled": s2,
            "s3_upsampled": s3_upsampled.astype(np.float32),
            "scale_ratio": scale_ratio,
            "s2_geo": s2_geo,
            "s3_geo": s3_geo,
        }


# ---------------------------------------------------------------------------
# In-situ co-registration
# ---------------------------------------------------------------------------


class InSituCoRegistration:
    """Match satellite tiles with in-situ monitoring station observations.

    Given a catalogue of stations (with locations and timestamps), find
    stations that fall within a spatial/temporal tolerance of each satellite
    tile acquisition.
    """

    def match(
        self,
        satellite_tiles: list[dict[str, Any]],
        station_catalog: list[dict[str, Any]],
        *,
        temporal_tol_hours: float = 3.0,
        spatial_tol_m: float = 500.0,
    ) -> list[dict[str, Any]]:
        """Find co-located (tile, station) pairs.

        Parameters
        ----------
        satellite_tiles:
            Each dict must contain ``"path"`` (str or Path), ``"datetime"``
            (ISO string or :class:`~datetime.datetime`), ``"centroid_lat"``,
            ``"centroid_lon"``.
        station_catalog:
            Each dict must contain ``"station_id"``, ``"lat"``, ``"lon"``,
            ``"datetime"`` (ISO string or datetime), and ``"data"`` (dict of
            measured parameters).
        temporal_tol_hours:
            Maximum time difference in hours.
        spatial_tol_m:
            Maximum distance in metres between tile centroid and station.

        Returns
        -------
        List of matched records, each containing ``"tile_path"``,
        ``"station_data"``, and ``"time_offset_hours"``.
        """
        from datetime import timedelta

        temporal_tol = timedelta(hours=temporal_tol_hours)
        matches: list[dict[str, Any]] = []

        for tile_info in satellite_tiles:
            tile_dt = (
                tile_info["datetime"]
                if isinstance(tile_info["datetime"], datetime)
                else datetime.fromisoformat(tile_info["datetime"])
            )
            tile_lat = tile_info["centroid_lat"]
            tile_lon = tile_info["centroid_lon"]

            for station in station_catalog:
                station_dt = (
                    station["datetime"]
                    if isinstance(station["datetime"], datetime)
                    else datetime.fromisoformat(station["datetime"])
                )

                # Temporal check
                dt_diff = abs((tile_dt - station_dt).total_seconds())
                if dt_diff > temporal_tol.total_seconds():
                    continue

                # Spatial check (Haversine approximation)
                dist_m = self._haversine_m(
                    tile_lat, tile_lon, station["lat"], station["lon"]
                )
                if dist_m > spatial_tol_m:
                    continue

                matches.append(
                    {
                        "tile_path": str(tile_info["path"]),
                        "station_data": station["data"],
                        "station_id": station["station_id"],
                        "time_offset_hours": dt_diff / 3600.0,
                        "distance_m": dist_m,
                    }
                )

        logger.info(
            f"Co-registration: {len(matches)} matches from "
            f"{len(satellite_tiles)} tiles x {len(station_catalog)} stations"
        )
        return matches

    @staticmethod
    def _haversine_m(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Great-circle distance in metres between two WGS-84 points."""
        R = 6_371_000.0  # Earth radius in metres
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)
        a = (
            np.sin(dphi / 2) ** 2
            + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
        )
        return float(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


# ---------------------------------------------------------------------------
# HydroViT 16-parameter label structure
# ---------------------------------------------------------------------------

HYDROVIT_PARAMETERS: list[str] = [
    "chl_a",
    "turbidity",
    "secchi_depth",
    "cdom",
    "tss",
    "total_nitrogen",
    "total_phosphorus",
    "dissolved_oxygen",
    "ammonia",
    "nitrate",
    "ph",
    "water_temp",
    "phycocyanin",
    "oil_probability",
    "acdom",
    "pollution_anomaly_index",
]


@dataclass
class HydroViTLabel:
    """16-parameter water-quality label for HydroViT supervised training.

    All fields are ``Optional[float]``; ``None`` indicates the parameter was
    not measured for this sample.
    """

    chl_a: float | None = None
    turbidity: float | None = None
    secchi_depth: float | None = None
    cdom: float | None = None
    tss: float | None = None
    total_nitrogen: float | None = None
    total_phosphorus: float | None = None
    dissolved_oxygen: float | None = None
    ammonia: float | None = None
    nitrate: float | None = None
    ph: float | None = None
    water_temp: float | None = None
    phycocyanin: float | None = None
    oil_probability: float | None = None
    acdom: float | None = None
    pollution_anomaly_index: float | None = None

    def to_vector(self) -> np.ndarray:
        """Return a length-16 float32 vector; missing values are ``NaN``."""
        return np.array(
            [
                getattr(self, p) if getattr(self, p) is not None else np.nan
                for p in HYDROVIT_PARAMETERS
            ],
            dtype=np.float32,
        )

    def available_mask(self) -> np.ndarray:
        """Boolean mask indicating which of the 16 parameters are present."""
        return np.array(
            [getattr(self, p) is not None for p in HYDROVIT_PARAMETERS],
            dtype=bool,
        )


def create_training_pair(
    tile: np.ndarray,
    matched_station_data: dict[str, Any],
) -> tuple[np.ndarray, HydroViTLabel]:
    """Build a (tile, label) training pair from a satellite tile and
    co-registered in-situ measurements.

    Parameters
    ----------
    tile:
        Preprocessed tile array, typically ``(224, 224, C)``.
    matched_station_data:
        Dict of measured water-quality parameters keyed by parameter name
        (must be a subset of :data:`HYDROVIT_PARAMETERS`).

    Returns
    -------
    Tuple of ``(tile, label)`` where *label* is a :class:`HydroViTLabel`
    with measured values filled in and unmeasured values as ``None``.
    """
    label_kwargs: dict[str, float | None] = {}
    for param in HYDROVIT_PARAMETERS:
        val = matched_station_data.get(param)
        label_kwargs[param] = float(val) if val is not None else None

    label = HydroViTLabel(**label_kwargs)

    n_available = int(label.available_mask().sum())
    logger.debug(
        f"Training pair: tile {tile.shape}, "
        f"{n_available}/{len(HYDROVIT_PARAMETERS)} params available"
    )
    return tile, label


# ---------------------------------------------------------------------------
# Pollution Anomaly Index (PAI)
# ---------------------------------------------------------------------------


def compute_pollution_anomaly_index(
    current_params: dict[str, float],
    baseline_stats: dict[str, dict[str, float]],
) -> float:
    """Compute the Pollution Anomaly Index (PAI).

    PAI is the Mahalanobis distance of the current parameter vector from a
    site-specific seasonal baseline, treating each parameter as independent
    (diagonal covariance).

    Parameters
    ----------
    current_params:
        Dict mapping parameter names to their current measured values.
    baseline_stats:
        Dict mapping parameter names to ``{"mean": ..., "std": ...}``
        representing the site-specific seasonal baseline.  Only parameters
        present in *both* ``current_params`` and ``baseline_stats`` are
        used.

    Returns
    -------
    PAI value (float >= 0).  Higher values indicate greater deviation from
    the baseline.
    """
    shared_keys = sorted(
        set(current_params.keys()) & set(baseline_stats.keys())
    )
    if not shared_keys:
        logger.warning("No shared parameters for PAI computation; returning 0")
        return 0.0

    z_scores: list[float] = []
    for key in shared_keys:
        mean = baseline_stats[key]["mean"]
        std = baseline_stats[key]["std"]
        if std < 1e-12:
            # Avoid division by zero; treat as zero deviation
            z_scores.append(0.0)
        else:
            z_scores.append((current_params[key] - mean) / std)

    # Mahalanobis distance with diagonal covariance = L2 norm of z-scores
    pai = float(np.sqrt(np.sum(np.array(z_scores) ** 2)))
    return pai
