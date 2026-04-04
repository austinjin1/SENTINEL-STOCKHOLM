"""Multimodal Conformal Anomaly Detection.

Distribution-free anomaly detection with coverage guarantees for
heterogeneous environmental monitoring data.  Conformal prediction
provides the guarantee:

.. math::

    P(\\text{true state} \\in \\text{prediction set}) \\geq 1 - \\alpha

regardless of the underlying data distribution, requiring only the
assumption of exchangeability (or approximate exchangeability within
stationary regimes).

The module provides:

- Geometry-aware non-conformity scores for Euclidean, simplex, and
  image feature spaces.
- A calibrated anomaly detector with exact finite-sample coverage.
- CUSUM-based change point detection to partition the data stream
  into approximately stationary regimes where conformal guarantees hold.
- A multimodal ensemble that combines per-modality detectors with
  multiple testing correction.

References
----------
[1] Vovk, V. et al. (2005). "Algorithmic Learning in a Random World."
    Springer.
[2] Lei, J. et al. (2018). "Distribution-Free Predictive Inference for
    Regression." JASA.
[3] Bates, S. et al. (2021). "Distribution-Free, Risk-Controlling
    Prediction Sets." JASA.
[4] Laxhammar, R. & Falkman, G. (2014). "Online Learning and Sequential
    Anomaly Detection in Trajectories." IEEE TPAMI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Space types (mirrors causal_discovery.py but kept local for independence)
# ---------------------------------------------------------------------------


class SpaceType(Enum):
    """Geometric space type for non-conformity scoring."""

    EUCLIDEAN = auto()
    SIMPLEX = auto()
    IMAGE_FEATURE = auto()


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AnomalyPrediction:
    """Result of conformal anomaly detection for a single observation."""

    is_anomaly: bool
    p_value: float
    nonconformity_score: float
    threshold: float
    prediction_set_radius: float  # radius of the conformal prediction set


@dataclass
class ChangePoint:
    """A detected change point in the data stream."""

    index: int
    cusum_statistic: float
    direction: str  # "increase" or "decrease"


# ---------------------------------------------------------------------------
# Geometry-aware non-conformity scores
# ---------------------------------------------------------------------------


class GeometryAwareNonconformityScore:
    """Non-conformity scores meaningful across heterogeneous spaces.

    Defines appropriate distance-based non-conformity measures for each
    geometric space type:

    - **Euclidean**: Mahalanobis distance from the calibration centroid.
    - **Simplex**: Aitchison distance (CLR-space Euclidean distance).
    - **Image feature**: Learned feature distance (cosine dissimilarity).

    The non-conformity score quantifies how "different" a new observation
    is from the calibration set, in a geometry-appropriate way.
    """

    def __init__(self) -> None:
        # Cached calibration statistics (set during calibration)
        self._calibration_mean: Optional[torch.Tensor] = None
        self._calibration_cov_inv: Optional[torch.Tensor] = None
        self._calibration_scores: Optional[torch.Tensor] = None
        self._space_type: SpaceType = SpaceType.EUCLIDEAN

    @staticmethod
    def _clr_transform(x: torch.Tensor) -> torch.Tensor:
        """CLR transform for compositional data."""
        eps = 1e-10
        log_x = torch.log(x.clamp(min=eps))
        return log_x - log_x.mean(dim=-1, keepdim=True)

    def _mahalanobis_score(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        cov_inv: torch.Tensor,
    ) -> torch.Tensor:
        """Mahalanobis distance from mean under inverse covariance.

        .. math::

            d_M(x, \\mu) = \\sqrt{(x - \\mu)^T \\Sigma^{-1} (x - \\mu)}

        Args:
            x: ``(N, D)`` observations.
            mean: ``(D,)`` centroid.
            cov_inv: ``(D, D)`` inverse covariance matrix.

        Returns:
            Distances ``(N,)``.
        """
        diff = x - mean.unsqueeze(0)  # (N, D)
        left = diff @ cov_inv  # (N, D)
        dist_sq = (left * diff).sum(dim=-1)  # (N,)
        return torch.sqrt(dist_sq.clamp(min=0.0))

    def _aitchison_score(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        cov_inv: torch.Tensor,
    ) -> torch.Tensor:
        """Aitchison distance: Mahalanobis in CLR space."""
        x_clr = self._clr_transform(x)
        mean_clr = self._clr_transform(mean.unsqueeze(0)).squeeze(0)
        return self._mahalanobis_score(x_clr, mean_clr, cov_inv)

    def _image_feature_score(
        self, x: torch.Tensor, mean: torch.Tensor
    ) -> torch.Tensor:
        """Cosine dissimilarity in feature space.

        .. math::

            d(x, \\mu) = 1 - \\frac{x \\cdot \\mu}{\\|x\\| \\|\\mu\\|}
        """
        x_norm = F.normalize(x, dim=-1)
        mean_norm = F.normalize(mean.unsqueeze(0), dim=-1)
        cos_sim = (x_norm * mean_norm).sum(dim=-1)
        return 1.0 - cos_sim

    def fit(
        self,
        calibration_data: torch.Tensor,
        space_type: SpaceType = SpaceType.EUCLIDEAN,
        ridge: float = 1e-4,
    ) -> None:
        """Fit calibration statistics from a calibration set.

        Args:
            calibration_data: ``(N, D)`` calibration observations.
            space_type: Geometric space type.
            ridge: Regularization for covariance inversion.
        """
        self._space_type = space_type

        if space_type == SpaceType.SIMPLEX:
            data = self._clr_transform(calibration_data)
        else:
            data = calibration_data

        self._calibration_mean = calibration_data.mean(dim=0)

        if space_type != SpaceType.IMAGE_FEATURE:
            # Compute regularized inverse covariance
            cov = data.T @ data / data.size(0) - data.mean(0).unsqueeze(1) @ data.mean(0).unsqueeze(0)
            D = cov.size(0)
            cov_reg = cov + ridge * torch.eye(D, device=cov.device)
            self._calibration_cov_inv = torch.linalg.inv(cov_reg)
        else:
            self._calibration_cov_inv = None

        # Pre-compute calibration scores for the calibration set itself
        self._calibration_scores = self.compute_score(
            calibration_data, space_type=space_type
        )

    def compute_score(
        self,
        observation: torch.Tensor,
        space_type: Optional[SpaceType] = None,
    ) -> torch.Tensor:
        """Compute non-conformity score for one or more observations.

        Args:
            observation: ``(N, D)`` or ``(D,)`` observations.
            space_type: Override space type (defaults to fitted type).

        Returns:
            Non-conformity scores ``(N,)`` or scalar.
        """
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        st = space_type or self._space_type
        mean = self._calibration_mean
        assert mean is not None, "Must call fit() before compute_score()"

        if st == SpaceType.EUCLIDEAN:
            scores = self._mahalanobis_score(
                observation, mean, self._calibration_cov_inv
            )
        elif st == SpaceType.SIMPLEX:
            # Need CLR-space covariance
            scores = self._aitchison_score(
                observation, mean, self._calibration_cov_inv
            )
        elif st == SpaceType.IMAGE_FEATURE:
            scores = self._image_feature_score(observation, mean)
        else:
            raise ValueError(f"Unknown space type: {st}")

        return scores.squeeze(0) if squeeze else scores


# ---------------------------------------------------------------------------
# Conformal anomaly detector
# ---------------------------------------------------------------------------


class ConformalAnomalyDetector(nn.Module):
    """Conformal anomaly detector with distribution-free coverage guarantee.

    After calibration on held-out exchangeable data, the detector provides
    the finite-sample guarantee:

    .. math::

        P(\\alpha_n \\geq \\alpha_{\\text{new}}) \\geq 1 - \\alpha

    where :math:`\\alpha_n` is the non-conformity score of the new
    observation and the threshold is the :math:`\\lceil (1-\\alpha)(n+1)\\rceil / n`
    quantile of the calibration scores.

    Args:
        scorer: Geometry-aware non-conformity scorer.
        alpha: Target miscoverage rate (e.g., 0.05 for 95% coverage).
    """

    def __init__(
        self,
        scorer: Optional[GeometryAwareNonconformityScore] = None,
        alpha: float = 0.05,
    ) -> None:
        super().__init__()
        self.scorer = scorer or GeometryAwareNonconformityScore()
        self.alpha = alpha
        self.register_buffer(
            "_threshold", torch.tensor(float("inf"))
        )
        self.register_buffer(
            "_calibration_scores", torch.tensor([])
        )

    def calibrate(
        self,
        calibration_data: torch.Tensor,
        alpha: Optional[float] = None,
        space_type: SpaceType = SpaceType.EUCLIDEAN,
    ) -> float:
        """Calibrate the detector on held-out data.

        Sets the detection threshold at the :math:`\\lceil (1-\\alpha)(n+1) \\rceil / n`
        quantile of calibration non-conformity scores, which guarantees
        finite-sample coverage.

        Args:
            calibration_data: ``(N, D)`` calibration observations
                (assumed exchangeable with future test data).
            alpha: Miscoverage rate. Defaults to ``self.alpha``.
            space_type: Geometric space type for scoring.

        Returns:
            The calibrated threshold.
        """
        if alpha is not None:
            self.alpha = alpha

        # Fit scorer and get calibration scores
        self.scorer.fit(calibration_data, space_type=space_type)
        scores = self.scorer.compute_score(calibration_data, space_type)
        self._calibration_scores = scores

        # Conformal quantile: ceil((1-alpha)(n+1)) / n
        n = scores.size(0)
        quantile_level = math.ceil((1 - self.alpha) * (n + 1)) / n
        quantile_level = min(quantile_level, 1.0)

        threshold = torch.quantile(scores, quantile_level)
        self._threshold = threshold

        logger.info(
            f"Calibrated at alpha={self.alpha:.3f}: "
            f"threshold={threshold.item():.4f}, "
            f"n_calibration={n}"
        )
        return threshold.item()

    def predict(
        self,
        observation: torch.Tensor,
        space_type: Optional[SpaceType] = None,
    ) -> AnomalyPrediction:
        """Predict whether an observation is anomalous.

        Args:
            observation: ``(D,)`` single observation.
            space_type: Override space type.

        Returns:
            :class:`AnomalyPrediction` with anomaly flag, p-value,
            and prediction set radius.
        """
        score = self.scorer.compute_score(observation, space_type)
        score_val = score.item() if score.dim() == 0 else score[0].item()

        # Compute conformal p-value
        cal_scores = self._calibration_scores
        n = cal_scores.size(0)
        # p-value = (#{calibration scores >= test score} + 1) / (n + 1)
        n_geq = (cal_scores >= score_val).sum().item()
        p_value = (n_geq + 1) / (n + 1)

        threshold_val = self._threshold.item()
        is_anomaly = score_val > threshold_val

        return AnomalyPrediction(
            is_anomaly=is_anomaly,
            p_value=p_value,
            nonconformity_score=score_val,
            threshold=threshold_val,
            prediction_set_radius=threshold_val,
        )

    def forward(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch forward: compute anomaly flags and p-values.

        Args:
            observations: ``(N, D)`` batch of observations.

        Returns:
            Tuple of ``(is_anomaly, p_values)``, both ``(N,)``.
        """
        scores = self.scorer.compute_score(observations)
        threshold = self._threshold
        is_anomaly = scores > threshold

        # Vectorized p-values
        cal = self._calibration_scores  # (M,)
        n = cal.size(0)
        # (N, M) comparison
        n_geq = (cal.unsqueeze(0) >= scores.unsqueeze(1)).sum(dim=1).float()
        p_values = (n_geq + 1) / (n + 1)

        return is_anomaly, p_values


# ---------------------------------------------------------------------------
# Change point detector (CUSUM)
# ---------------------------------------------------------------------------


class ChangePointDetector:
    """CUSUM-based change point detector for data stream segmentation.

    Partitions the data stream into approximately stationary segments so
    that conformal guarantees (which require exchangeability) hold within
    each regime.

    Uses the cumulative sum (CUSUM) algorithm:

    .. math::

        S_t = \\max(0, S_{t-1} + (x_t - \\mu_0) - k)

    where :math:`\\mu_0` is the in-control mean and :math:`k` is the
    allowance parameter.  A change point is declared when :math:`S_t > h`
    (the decision threshold).

    Args:
        allowance: CUSUM allowance parameter :math:`k`.
        threshold: Decision threshold :math:`h`.
        warmup: Number of initial observations to estimate :math:`\\mu_0`.
    """

    def __init__(
        self,
        allowance: float = 0.5,
        threshold: float = 5.0,
        warmup: int = 50,
    ) -> None:
        self.allowance = allowance
        self.threshold = threshold
        self.warmup = warmup

    def detect(self, data: torch.Tensor) -> List[ChangePoint]:
        """Detect change points in a univariate or multivariate stream.

        For multivariate data, the L2 norm of each observation is used
        as the scalar monitoring statistic.

        Args:
            data: ``(T,)`` or ``(T, D)`` time series.

        Returns:
            List of detected :class:`ChangePoint` objects.
        """
        if data.dim() == 2:
            # Use L2 norm for multivariate
            x = torch.norm(data, dim=-1)
        else:
            x = data

        T = x.size(0)
        if T <= self.warmup:
            return []

        # Estimate in-control parameters from warmup period
        mu0 = x[: self.warmup].mean().item()
        sigma0 = x[: self.warmup].std().item()
        if sigma0 < 1e-8:
            sigma0 = 1.0

        # Standardize
        x_std = (x - mu0) / sigma0

        # Two-sided CUSUM
        s_pos = 0.0
        s_neg = 0.0
        change_points: List[ChangePoint] = []

        for t in range(self.warmup, T):
            s_pos = max(0.0, s_pos + x_std[t].item() - self.allowance)
            s_neg = max(0.0, s_neg - x_std[t].item() - self.allowance)

            if s_pos > self.threshold:
                change_points.append(
                    ChangePoint(
                        index=t,
                        cusum_statistic=s_pos,
                        direction="increase",
                    )
                )
                # Reset after detection
                s_pos = 0.0
                s_neg = 0.0

            elif s_neg > self.threshold:
                change_points.append(
                    ChangePoint(
                        index=t,
                        cusum_statistic=s_neg,
                        direction="decrease",
                    )
                )
                s_pos = 0.0
                s_neg = 0.0

        return change_points

    def segment(self, data: torch.Tensor) -> List[Tuple[int, int]]:
        """Segment data into stationary regimes.

        Args:
            data: ``(T,)`` or ``(T, D)`` time series.

        Returns:
            List of ``(start, end)`` index pairs for each segment.
        """
        cps = self.detect(data)
        T = data.size(0)

        boundaries = [0] + [cp.index for cp in cps] + [T]
        segments = []
        for i in range(len(boundaries) - 1):
            segments.append((boundaries[i], boundaries[i + 1]))
        return segments


# ---------------------------------------------------------------------------
# Multimodal conformal ensemble
# ---------------------------------------------------------------------------


class MultimodalConformalEnsemble(nn.Module):
    """Combines per-modality conformal detectors with multiple testing correction.

    Each modality has its own :class:`ConformalAnomalyDetector`.  At test
    time, the ensemble combines their p-values using either Bonferroni
    or Benjamini-Hochberg correction.

    - **Bonferroni**: conservative; rejects if any modality p-value
      < alpha / K (controls FWER).
    - **Benjamini-Hochberg**: less conservative; controls FDR.

    Args:
        modality_names: Names of the modalities.
        alpha: Overall significance level.
        correction: ``"bonferroni"`` or ``"bh"`` (Benjamini-Hochberg).
    """

    def __init__(
        self,
        modality_names: List[str],
        alpha: float = 0.05,
        correction: str = "bonferroni",
    ) -> None:
        super().__init__()
        if correction not in ("bonferroni", "bh"):
            raise ValueError(
                f"correction must be 'bonferroni' or 'bh', got '{correction}'"
            )
        self.modality_names = modality_names
        self.alpha = alpha
        self.correction = correction
        self.detectors: Dict[str, ConformalAnomalyDetector] = {}

    def add_detector(
        self,
        modality: str,
        detector: ConformalAnomalyDetector,
    ) -> None:
        """Register a per-modality detector.

        Args:
            modality: Modality name (must be in ``modality_names``).
            detector: Calibrated conformal anomaly detector.
        """
        if modality not in self.modality_names:
            raise ValueError(
                f"Unknown modality '{modality}'. "
                f"Registered: {self.modality_names}"
            )
        self.detectors[modality] = detector

    def calibrate_all(
        self,
        calibration_data: Dict[str, torch.Tensor],
        space_types: Dict[str, SpaceType],
    ) -> None:
        """Calibrate all modality detectors.

        Args:
            calibration_data: Mapping from modality name to calibration
                data tensor.
            space_types: Mapping from modality name to space type.
        """
        for name in self.modality_names:
            if name not in calibration_data:
                logger.warning(f"No calibration data for modality '{name}'")
                continue
            scorer = GeometryAwareNonconformityScore()
            detector = ConformalAnomalyDetector(
                scorer=scorer, alpha=self.alpha
            )
            detector.calibrate(
                calibration_data[name],
                space_type=space_types.get(name, SpaceType.EUCLIDEAN),
            )
            self.detectors[name] = detector

    def _bonferroni_test(
        self, p_values: Dict[str, float]
    ) -> Tuple[bool, Dict[str, bool]]:
        """Bonferroni correction: reject if any p < alpha/K."""
        K = len(p_values)
        threshold = self.alpha / K
        per_modality = {name: pv < threshold for name, pv in p_values.items()}
        is_anomaly = any(per_modality.values())
        return is_anomaly, per_modality

    def _bh_test(
        self, p_values: Dict[str, float]
    ) -> Tuple[bool, Dict[str, bool]]:
        """Benjamini-Hochberg correction."""
        K = len(p_values)
        sorted_items = sorted(p_values.items(), key=lambda t: t[1])

        per_modality = {name: False for name in p_values}
        max_k = -1
        for rank, (name, pv) in enumerate(sorted_items, start=1):
            threshold = rank * self.alpha / K
            if pv <= threshold:
                max_k = rank

        if max_k > 0:
            for rank, (name, pv) in enumerate(sorted_items, start=1):
                if rank <= max_k:
                    per_modality[name] = True

        is_anomaly = any(per_modality.values())
        return is_anomaly, per_modality

    def predict(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> Tuple[bool, Dict[str, AnomalyPrediction]]:
        """Predict anomaly status using all available modalities.

        Args:
            observations: Mapping from modality name to observation
                tensor ``(D,)`` for each available modality.

        Returns:
            Tuple of ``(is_anomaly, per_modality_predictions)`` where
            ``is_anomaly`` accounts for multiple testing correction.
        """
        predictions: Dict[str, AnomalyPrediction] = {}
        p_values: Dict[str, float] = {}

        for name, obs in observations.items():
            if name not in self.detectors:
                logger.warning(f"No detector for modality '{name}', skipping")
                continue
            pred = self.detectors[name].predict(obs)
            predictions[name] = pred
            p_values[name] = pred.p_value

        if not p_values:
            return False, predictions

        if self.correction == "bonferroni":
            is_anomaly, per_modality_flags = self._bonferroni_test(p_values)
        else:
            is_anomaly, per_modality_flags = self._bh_test(p_values)

        # Update predictions with corrected anomaly flags
        for name, flag in per_modality_flags.items():
            if name in predictions:
                predictions[name] = AnomalyPrediction(
                    is_anomaly=flag,
                    p_value=predictions[name].p_value,
                    nonconformity_score=predictions[name].nonconformity_score,
                    threshold=predictions[name].threshold,
                    prediction_set_radius=predictions[name].prediction_set_radius,
                )

        return is_anomaly, predictions

    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> Tuple[bool, Dict[str, AnomalyPrediction]]:
        """Alias for :meth:`predict`."""
        return self.predict(observations)
