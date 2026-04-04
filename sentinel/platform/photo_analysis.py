"""
SENTINEL photo-based water quality assessment.

Users upload a photograph of a water body and the module estimates visible
water quality indicators using a vision model (HydroViT or ResNet backbone).
Results can be cross-referenced against satellite-derived values for the
same location.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Turbidity classification
# ---------------------------------------------------------------------------


class TurbidityClass(str, Enum):
    """Visual turbidity classification from a photo."""

    CLEAR = "clear"
    SLIGHT = "slight"
    MODERATE = "moderate"
    TURBID = "turbid"
    OPAQUE = "opaque"


# ---------------------------------------------------------------------------
# Photo analysis result
# ---------------------------------------------------------------------------


@dataclass
class PhotoAnalysisResult:
    """Output of photo-based water quality assessment.

    All probability / percentage fields are in ``[0, 1]`` except
    ``algal_coverage`` which is ``[0, 100]`` and ``water_color_index``
    which follows the Forel-Ule scale (1--21).
    """

    turbidity_class: TurbidityClass = TurbidityClass.CLEAR
    turbidity_confidence: float = 0.0
    algal_coverage: float = 0.0          # 0-100 %
    algal_confidence: float = 0.0
    color_anomaly: bool = False
    color_anomaly_confidence: float = 0.0
    oil_sheen: float = 0.0              # probability 0-1
    oil_sheen_confidence: float = 0.0
    foam_presence: float = 0.0          # probability 0-1
    foam_confidence: float = 0.0
    water_color_index: int = 1          # Forel-Ule 1-21
    water_color_confidence: float = 0.0
    latitude: float = 0.0
    longitude: float = 0.0
    model_version: str = ""
    raw_logits: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Vision model wrapper
# ---------------------------------------------------------------------------


def _load_image_as_tensor(image: Any) -> "np.ndarray":
    """Load an image from a file path, PIL Image, or numpy array.

    Returns a float32 numpy array of shape ``(H, W, 3)`` with values in
    ``[0, 1]``.
    """
    if isinstance(image, (str, Path)):
        try:
            from PIL import Image as PILImage
        except ImportError as exc:
            raise ImportError(
                "Pillow is required for loading images from file paths. "
                "Install with: pip install Pillow"
            ) from exc
        img = PILImage.open(str(image)).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr

    # PIL Image
    try:
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            return arr
    except ImportError:
        pass

    if isinstance(image, np.ndarray):
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        return image

    raise TypeError(
        f"Unsupported image type: {type(image)}. "
        "Pass a file path, PIL Image, or numpy array."
    )


class PhotoWaterAnalyzer:
    """Analyze water quality from ground-level photographs.

    Uses a fine-tuned vision backbone (HydroViT or ResNet) to estimate
    visible water quality parameters.  If no trained model weights are
    available, the analyzer falls back to a heuristic color-space analysis.

    Parameters
    ----------
    model_path:
        Path to a serialized model checkpoint.  If *None*, heuristic
        mode is used.
    device:
        PyTorch device string (e.g. ``"cpu"``, ``"cuda:0"``).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.model: Any = None
        self.model_version = "heuristic-v1"

        if model_path is not None and Path(model_path).exists():
            self._load_model(Path(model_path))

    def _load_model(self, path: Path) -> None:
        """Load a trained PyTorch model from *path*."""
        try:
            import torch

            self.model = torch.load(str(path), map_location=self.device)
            if hasattr(self.model, "eval"):
                self.model.eval()
            self.model_version = f"checkpoint-{path.stem}"
            logger.info(f"Loaded photo analysis model from {path}")
        except Exception as exc:
            logger.warning(f"Could not load model from {path}: {exc}")
            self.model = None

    # ---- heuristic fallback ------------------------------------------------

    @staticmethod
    def _heuristic_analysis(arr: np.ndarray) -> dict[str, Any]:
        """Simple color-space heuristic when no trained model is available.

        Operates on an RGB float32 array of shape ``(H, W, 3)``.
        """
        mean_rgb = arr.mean(axis=(0, 1))  # (3,)
        r, g, b = float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])

        # Green dominance -> algal
        green_excess = g - 0.5 * (r + b)
        algal_coverage = float(np.clip(green_excess * 200.0, 0, 100))

        # Brown/yellow -> turbid
        brown_score = r - b
        if brown_score > 0.15:
            turbidity = TurbidityClass.TURBID
        elif brown_score > 0.07:
            turbidity = TurbidityClass.MODERATE
        elif brown_score > 0.02:
            turbidity = TurbidityClass.SLIGHT
        else:
            turbidity = TurbidityClass.CLEAR

        # Rainbow-sheen detection (high color variance in patches)
        patch_std = float(arr.std(axis=(0, 1)).mean())
        oil_sheen = float(np.clip(patch_std * 3.0 - 0.2, 0, 1))

        # White patches -> foam
        brightness = arr.mean(axis=2)
        foam_frac = float((brightness > 0.9).mean())
        foam_presence = float(np.clip(foam_frac * 10.0, 0, 1))

        # Color anomaly: deviation from expected blue-green water palette
        expected_b_minus_r = 0.05
        color_dev = abs((b - r) - expected_b_minus_r)
        color_anomaly = color_dev > 0.15

        # Forel-Ule index: rough mapping from blue-to-brown spectrum
        ratio = (b - r + 0.5) * 21.0
        fu_index = int(np.clip(22 - ratio * 21, 1, 21))

        return {
            "turbidity_class": turbidity,
            "algal_coverage": algal_coverage,
            "color_anomaly": color_anomaly,
            "oil_sheen": oil_sheen,
            "foam_presence": foam_presence,
            "water_color_index": fu_index,
        }

    # ---- main entry point --------------------------------------------------

    def analyze_photo(
        self,
        image: Any,
        lat: float,
        lon: float,
    ) -> PhotoAnalysisResult:
        """Analyze a photo of a water body.

        Parameters
        ----------
        image:
            File path (str/Path), PIL Image, or numpy array.
        lat, lon:
            GPS coordinates of the photo location.

        Returns
        -------
        :class:`PhotoAnalysisResult` with estimated parameters and
        confidence scores.
        """
        arr = _load_image_as_tensor(image)

        if self.model is not None:
            result = self._run_model_inference(arr)
        else:
            result = self._heuristic_analysis(arr)

        # Heuristic confidence is capped at 0.5 to signal uncertainty
        conf = 0.5 if self.model is None else 0.85

        return PhotoAnalysisResult(
            turbidity_class=result["turbidity_class"],
            turbidity_confidence=conf,
            algal_coverage=result["algal_coverage"],
            algal_confidence=conf,
            color_anomaly=result["color_anomaly"],
            color_anomaly_confidence=conf,
            oil_sheen=result["oil_sheen"],
            oil_sheen_confidence=conf,
            foam_presence=result["foam_presence"],
            foam_confidence=conf,
            water_color_index=result["water_color_index"],
            water_color_confidence=conf,
            latitude=lat,
            longitude=lon,
            model_version=self.model_version,
        )

    def _run_model_inference(self, arr: np.ndarray) -> dict[str, Any]:
        """Run the loaded PyTorch model on a preprocessed image array."""
        try:
            import torch

            tensor = (
                torch.from_numpy(arr)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.device)
            )

            with torch.no_grad():
                output = self.model(tensor)

            # Expected output dict or tuple; fall back to heuristic if
            # the model output format is unexpected.
            if isinstance(output, dict):
                return {
                    "turbidity_class": TurbidityClass(
                        output.get("turbidity_class", "clear")
                    ),
                    "algal_coverage": float(output.get("algal_coverage", 0.0)),
                    "color_anomaly": bool(output.get("color_anomaly", False)),
                    "oil_sheen": float(output.get("oil_sheen", 0.0)),
                    "foam_presence": float(output.get("foam_presence", 0.0)),
                    "water_color_index": int(output.get("water_color_index", 1)),
                }

        except Exception as exc:
            logger.warning(f"Model inference failed, falling back to heuristic: {exc}")

        return self._heuristic_analysis(arr)


# ---------------------------------------------------------------------------
# Cross-reference with remote sensing
# ---------------------------------------------------------------------------


def cross_reference_with_remote_sensing(
    photo_result: PhotoAnalysisResult,
    lat: float,
    lon: float,
    satellite_catalog: dict[str, Any],
) -> dict[str, Any]:
    """Compare photo-based estimates with satellite-derived values.

    Parameters
    ----------
    photo_result:
        Output from :meth:`PhotoWaterAnalyzer.analyze_photo`.
    lat, lon:
        GPS coordinates (should match the photo location).
    satellite_catalog:
        Dictionary keyed by parameter name with satellite-derived values.
        Expected keys include ``"turbidity_ntu"``, ``"chlorophyll_a_ug_l"``,
        ``"water_color_index"``.

    Returns
    -------
    Dictionary with agreement metrics for each comparable parameter.
    """
    agreements: dict[str, Any] = {}

    # Turbidity
    sat_turb = satellite_catalog.get("turbidity_ntu")
    if sat_turb is not None:
        # Map turbidity class to rough NTU midpoint for comparison
        class_to_ntu = {
            TurbidityClass.CLEAR: 2.0,
            TurbidityClass.SLIGHT: 10.0,
            TurbidityClass.MODERATE: 50.0,
            TurbidityClass.TURBID: 200.0,
            TurbidityClass.OPAQUE: 1000.0,
        }
        photo_ntu = class_to_ntu.get(photo_result.turbidity_class, 50.0)
        ratio = min(photo_ntu, sat_turb) / max(photo_ntu, sat_turb, 0.1)
        agreements["turbidity"] = {
            "photo_estimate_ntu": photo_ntu,
            "satellite_ntu": sat_turb,
            "agreement_ratio": round(ratio, 3),
        }

    # Chlorophyll / algal coverage
    sat_chl = satellite_catalog.get("chlorophyll_a_ug_l")
    if sat_chl is not None:
        # Rough mapping: 100% coverage ~ 300 ug/L chl-a
        photo_chl_equiv = photo_result.algal_coverage * 3.0
        if max(photo_chl_equiv, sat_chl) > 0:
            ratio = min(photo_chl_equiv, sat_chl) / max(
                photo_chl_equiv, sat_chl, 0.1
            )
        else:
            ratio = 1.0
        agreements["chlorophyll_a"] = {
            "photo_algal_pct": photo_result.algal_coverage,
            "photo_chl_equiv_ug_l": round(photo_chl_equiv, 1),
            "satellite_chl_ug_l": sat_chl,
            "agreement_ratio": round(ratio, 3),
        }

    # Forel-Ule color index
    sat_fu = satellite_catalog.get("water_color_index")
    if sat_fu is not None:
        diff = abs(photo_result.water_color_index - sat_fu)
        agreements["water_color_index"] = {
            "photo_fu": photo_result.water_color_index,
            "satellite_fu": sat_fu,
            "absolute_difference": diff,
            "agree_within_3": diff <= 3,
        }

    agreements["location"] = {"latitude": lat, "longitude": lon}

    return agreements


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------


class PhotoDataset:
    """Dataset for training the water photo analysis model.

    Wraps a directory of labeled water body images.  Each sample yields
    an image tensor and a label dictionary.

    Parameters
    ----------
    image_dir:
        Root directory containing sub-folders per label or a manifest CSV.
    manifest_csv:
        Optional CSV with columns ``filename, turbidity_class,
        algal_coverage, color_anomaly, oil_sheen, foam_presence,
        water_color_index``.
    transform:
        Optional callable applied to each image array before returning.
    """

    def __init__(
        self,
        image_dir: str | Path,
        manifest_csv: str | Path | None = None,
        transform: Any = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.samples: list[dict[str, Any]] = []

        if manifest_csv is not None:
            self._load_manifest(Path(manifest_csv))
        else:
            self._discover_images()

    def _load_manifest(self, csv_path: Path) -> None:
        """Load labeled samples from a CSV manifest."""
        import csv

        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                filepath = self.image_dir / row["filename"]
                if filepath.exists():
                    self.samples.append({
                        "filepath": filepath,
                        "turbidity_class": row.get("turbidity_class", "clear"),
                        "algal_coverage": float(row.get("algal_coverage", 0)),
                        "color_anomaly": row.get("color_anomaly", "false").lower() == "true",
                        "oil_sheen": float(row.get("oil_sheen", 0)),
                        "foam_presence": float(row.get("foam_presence", 0)),
                        "water_color_index": int(row.get("water_color_index", 1)),
                    })

        logger.info(f"Loaded {len(self.samples)} samples from {csv_path}")

    def _discover_images(self) -> None:
        """Auto-discover images when no manifest is provided."""
        extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        for fp in sorted(self.image_dir.rglob("*")):
            if fp.suffix.lower() in extensions:
                self.samples.append({
                    "filepath": fp,
                    "turbidity_class": "unknown",
                    "algal_coverage": 0.0,
                    "color_anomaly": False,
                    "oil_sheen": 0.0,
                    "foam_presence": 0.0,
                    "water_color_index": 1,
                })
        logger.info(
            f"Discovered {len(self.samples)} images in {self.image_dir}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, dict[str, Any]]:
        sample = self.samples[idx]
        arr = _load_image_as_tensor(sample["filepath"])

        if self.transform is not None:
            arr = self.transform(arr)

        labels = {k: v for k, v in sample.items() if k != "filepath"}
        return arr, labels
