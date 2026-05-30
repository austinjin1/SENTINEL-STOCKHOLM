"""WaterDroneNet: Physics-conditioned water quality prediction from drone imagery.

Fuses RGB+NIR drone imagery with cheap point sensors (temperature, TDS) to
predict a full panel of 12 water quality indicators.  Three architectural
novelties drive accuracy beyond purely data-driven baselines:

1. **FiLM sensor conditioning** -- temperature and TDS scalar readings
   modulate ViT patch features via Feature-wise Linear Modulation, so that
   visually identical scenes at different temperatures are interpreted
   correctly.

2. **Physics-residual prediction** -- each output target is predicted as
   ``physics_prior(inputs) + learned_residual``.  The prior carries the
   known physical relationship (e.g. DO saturation curve, TDS-conductivity
   proportionality) while the residual corrects for local deviations.

3. **Uncertainty-routed output** -- a per-target, per-sample trust score
   flags predictions as green / yellow / red, enabling downstream systems to
   decide whether to act on or re-query a given measurement.

References
----------
- Benson & Krause (1984) for the DO saturation curve.
- FiLM: Visual Reasoning with a General Conditioning Layer (Perez et al., 2018).
- Bayesian deep learning uncertainty via softplus-parameterised sigma
  (Nix & Weigend, 1994).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from sentinel.models.fusion.embedding_registry import SHARED_EMBEDDING_DIM
from sentinel.models.satellite_encoder.hydrovit_backbone import (
    VIT_EMBED_DIM,
    VIT_PATCH_SIZE,
    VIT_IMAGE_SIZE,
    adapt_patch_embed_weights,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of input channels: R, G, B, NIR.
DRONE_IN_CHANS: int = 4

#: Native ViT-Small embedding dimension (matches HydroViT).
DRONE_EMBED_DIM: int = VIT_EMBED_DIM  # 384

#: Spatial image size expected by the drone encoder.
DRONE_IMG_SIZE: int = VIT_IMAGE_SIZE  # 224

#: Shared embedding dimension used throughout SENTINEL.
FUSED_DIM: int = SHARED_EMBEDDING_DIM  # 256

#: Number of cheap scalar inputs: [temperature (°C), TDS (ppm)].
NUM_SCALAR_INPUTS: int = 2

#: Water quality targets predicted by WaterDroneNet.
TARGET_PARAMS: Tuple[str, ...] = (
    "dissolved_oxygen",           # mg/L
    "ph",                         # pH units
    "turbidity",                  # NTU
    "chlorophyll_a",              # μg/L
    "nitrate",                    # mg/L
    "phosphate",                  # mg/L
    "conductivity",               # μS/cm
    "temperature",                # °C  (self-prediction from scalar input)
    "total_suspended_solids",     # mg/L
    "biochemical_oxygen_demand",  # mg/L
    "ammonia",                    # mg/L
    "total_organic_carbon",       # mg/L
)
NUM_TARGETS: int = len(TARGET_PARAMS)

#: Trust-score thresholds for the three-tier flag system.
TRUST_GREEN_THRESHOLD: float = 0.7
TRUST_RED_THRESHOLD: float = 0.3


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TargetPrediction:
    """Probabilistic prediction for a single water quality target.

    Attributes:
        mu: Predicted mean value, shape ``[B]``.
        sigma: Predicted aleatoric uncertainty (std), shape ``[B]``.
        physics_prior_value: Value of the physics prior used for this
            prediction before the learned residual was added, shape ``[B]``.
            ``None`` for targets whose prior returns zero.
    """

    mu: torch.Tensor
    sigma: torch.Tensor
    physics_prior_value: torch.Tensor


@dataclass
class WaterDroneNetOutput:
    """Full structured output from a :class:`WaterDroneNet` forward pass.

    Attributes:
        predictions: Dict mapping each target name in :data:`TARGET_PARAMS`
            to its :class:`TargetPrediction`.
        trust_scores: Dict mapping each target name to a scalar trust
            score in ``[0, 1]``, shape ``[B]`` per entry.
        trust_flags: Dict mapping each target name to a list of ``B``
            human-readable flag strings: ``"green"`` / ``"yellow"`` /
            ``"red"``.
        fused_embedding: Shared 256-d embedding produced by the
            :class:`FiLMConditionedFusion` layer, shape ``[B, 256]``.
            Compatible with the SENTINEL embedding registry.
    """

    predictions: Dict[str, TargetPrediction]
    trust_scores: Dict[str, torch.Tensor]
    trust_flags: Dict[str, List[str]]
    fused_embedding: torch.Tensor


# ---------------------------------------------------------------------------
# 1. DroneVisionEncoder
# ---------------------------------------------------------------------------

class DroneVisionEncoder(nn.Module):
    """Lightweight ViT encoder for 4-channel (RGB + NIR) drone imagery.

    Uses timm's ``vit_small_patch16_224`` adapted from 3 to 4 input channels
    via cyclic weight tiling (same strategy as :class:`HydroViTBackbone`).
    Returns both per-patch tokens for the FiLM fusion layer and the CLS
    token as a compact scene descriptor.

    Optionally loads HydroViT pretrained weights, keeping the patch embedding
    adaptation for the fourth NIR channel.

    Args:
        in_chans: Number of input channels.  Default ``4`` (RGB + NIR).
        img_size: Spatial resolution.  Default ``224``.
        pretrained: If ``True`` and *hydrovit_checkpoint_path* is ``None``,
            load ImageNet ViT-S/16 weights via timm with band adaptation.
        hydrovit_checkpoint_path: Optional path to a HydroViT checkpoint.
            When supplied, the patch-embedding adaptation is applied and
            weights are loaded with ``strict=False`` so extra/missing keys
            (e.g. the MAE decoder, spectral embed) are silently skipped.
        freeze_backbone: Freeze all ViT weights and only train new heads.
    """

    def __init__(
        self,
        in_chans: int = DRONE_IN_CHANS,
        img_size: int = DRONE_IMG_SIZE,
        pretrained: bool = True,
        hydrovit_checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = DRONE_EMBED_DIM

        # Build ViT-Small backbone via timm (no classification head).
        self.vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            in_chans=in_chans,
            img_size=img_size,
            num_classes=0,
        )

        # Load pretrained weights.
        if hydrovit_checkpoint_path is not None:
            self._load_hydrovit_checkpoint(hydrovit_checkpoint_path, in_chans)
        elif pretrained:
            self._load_imagenet_pretrained(in_chans)

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        # Number of patch tokens = (img_size / patch_size)^2.
        self.num_patches: int = (img_size // VIT_PATCH_SIZE) ** 2

    # ------------------------------------------------------------------
    # Weight loading helpers
    # ------------------------------------------------------------------

    def _load_imagenet_pretrained(self, in_chans: int) -> None:
        """Load timm ImageNet ViT-S weights with 4-channel adaptation."""
        pretrained_model = timm.create_model(
            "vit_small_patch16_224", pretrained=True, in_chans=3
        )
        state_dict = pretrained_model.state_dict()
        del pretrained_model

        patch_key = "patch_embed.proj.weight"
        if patch_key in state_dict and state_dict[patch_key].shape[1] != in_chans:
            logger.info(
                "DroneVisionEncoder: adapting patch embed from %d to %d channels",
                state_dict[patch_key].shape[1],
                in_chans,
            )
            state_dict[patch_key] = adapt_patch_embed_weights(
                state_dict[patch_key], in_chans
            )

        missing, unexpected = self.vit.load_state_dict(state_dict, strict=False)
        if missing:
            logger.debug("DroneVisionEncoder missing keys: %s", missing)
        if unexpected:
            logger.debug("DroneVisionEncoder unexpected keys: %s", unexpected)

    def _load_hydrovit_checkpoint(self, path: str, in_chans: int) -> None:
        """Load HydroViT checkpoint, adapting patch embedding as needed."""
        raw = torch.load(path, map_location="cpu", weights_only=True)
        # Support various checkpoint conventions.
        if isinstance(raw, dict):
            state_dict = raw.get("model", raw.get("state_dict", raw))
        else:
            state_dict = raw

        # Strip a leading "vit." prefix if the checkpoint was saved from
        # HydroViTBackbone (which wraps timm's vit as self.vit).
        stripped = {}
        for k, v in state_dict.items():
            stripped[k[4:] if k.startswith("vit.") else k] = v
        state_dict = stripped

        patch_key = "patch_embed.proj.weight"
        if patch_key in state_dict and state_dict[patch_key].shape[1] != in_chans:
            logger.info(
                "DroneVisionEncoder: adapting HydroViT patch embed from %d to %d channels",
                state_dict[patch_key].shape[1],
                in_chans,
            )
            state_dict[patch_key] = adapt_patch_embed_weights(
                state_dict[patch_key], in_chans
            )

        missing, unexpected = self.vit.load_state_dict(state_dict, strict=False)
        logger.info(
            "Loaded HydroViT checkpoint from %s (missing=%d, unexpected=%d)",
            path,
            len(missing),
            len(unexpected),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode drone imagery into patch tokens and CLS token.

        Args:
            images: Drone imagery tensor, shape ``[B, 4, 224, 224]``.

        Returns:
            Tuple of:
            - **patch_tokens**: Per-patch embeddings, shape ``[B, N, 384]``
              where N = (224/16)² = 196.
            - **cls_token**: Global scene embedding, shape ``[B, 384]``.
        """
        # Patch embedding.
        tokens = self.vit.patch_embed(images)               # [B, N, 384]
        B, N, D = tokens.shape

        # Prepend CLS token.
        cls = self.vit.cls_token.expand(B, -1, -1)          # [B, 1, 384]
        tokens = torch.cat([cls, tokens], dim=1)             # [B, N+1, 384]

        # Add positional embedding.
        tokens = tokens + self.vit.pos_embed[:, : N + 1, :]

        # Apply dropout (timm's pos_drop).
        tokens = self.vit.pos_drop(tokens)

        # Apply norm_pre if present (some timm configs).
        if hasattr(self.vit, "norm_pre"):
            tokens = self.vit.norm_pre(tokens)

        # Transformer blocks.
        for block in self.vit.blocks:
            tokens = block(tokens)
        tokens = self.vit.norm(tokens)

        cls_token = tokens[:, 0, :]        # [B, 384]
        patch_tokens = tokens[:, 1:, :]    # [B, N, 384]

        return patch_tokens, cls_token


# ---------------------------------------------------------------------------
# 2. ScalarEncoder (FiLM parameter generator)
# ---------------------------------------------------------------------------

class ScalarEncoder(nn.Module):
    """Encodes cheap scalar sensors into FiLM modulation parameters.

    Maps the 2-dimensional scalar input ``[temperature, TDS]`` to a pair of
    FiLM vectors ``(gamma, beta)`` each of dimension ``DRONE_EMBED_DIM``.
    These are used to modulate vision patch features so that identical
    spectral signatures are interpreted differently across temperature and
    salinity regimes.

    Architecture::

        [B, 2]
          → Linear(2, 64) → GELU
          → Linear(64, 128) → GELU
          → [gamma_head: Linear(128, 384)]  → gamma [B, 384]
          → [beta_head:  Linear(128, 384)]  → beta  [B, 384]

    Args:
        scalar_dim: Number of scalar inputs.  Default 2 (temperature, TDS).
        hidden_dim_1: Width of the first hidden layer.
        hidden_dim_2: Width of the second hidden layer.
        out_dim: Output dimensionality matching the vision encoder.
    """

    def __init__(
        self,
        scalar_dim: int = NUM_SCALAR_INPUTS,
        hidden_dim_1: int = 64,
        hidden_dim_2: int = 128,
        out_dim: int = DRONE_EMBED_DIM,
    ) -> None:
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(scalar_dim, hidden_dim_1),
            nn.GELU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.GELU(),
        )
        self.gamma_head = nn.Linear(hidden_dim_2, out_dim)
        self.beta_head = nn.Linear(hidden_dim_2, out_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Initialise gamma to 1 (identity FiLM) and beta to 0.
        nn.init.ones_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.bias)

    def forward(
        self, scalars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute FiLM modulation parameters from scalar sensor readings.

        Args:
            scalars: Shape ``[B, 2]`` — ``[:, 0]`` is temperature in °C,
                ``[:, 1]`` is TDS in ppm.

        Returns:
            Tuple of:
            - **gamma**: FiLM scale parameter, shape ``[B, 384]``.
            - **beta**: FiLM shift parameter, shape ``[B, 384]``.
        """
        h = self.shared_mlp(scalars)
        gamma = self.gamma_head(h)   # [B, 384]
        beta = self.beta_head(h)     # [B, 384]
        return gamma, beta


# ---------------------------------------------------------------------------
# 3. FiLMConditionedFusion
# ---------------------------------------------------------------------------

class FiLMConditionedFusion(nn.Module):
    """Fuses FiLM-conditioned patch tokens with scalar context.

    Two-stage fusion:

    1. **FiLM modulation** -- scale and shift each patch token by the
       sensor-derived ``(gamma, beta)`` vectors.
    2. **Cross-attention** -- a single learned scalar query attends over
       the modulated patch sequence to pool into a fixed-size embedding.
    3. **Projection** -- a linear layer maps the 384-d attended embedding
       to the 256-d SENTINEL shared embedding space.

    Args:
        vision_dim: Dimension of the vision encoder output.  Default 384.
        out_dim: Dimension of the fused embedding.  Default 256.
        num_heads: Number of attention heads in the cross-attention.
        dropout: Dropout probability applied after attention.
    """

    def __init__(
        self,
        vision_dim: int = DRONE_EMBED_DIM,
        out_dim: int = FUSED_DIM,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vision_dim = vision_dim
        self.out_dim = out_dim

        # Scalar query: a learned vector that selects relevant patch features.
        self.scalar_query = nn.Parameter(torch.randn(1, 1, vision_dim) * 0.02)

        # Cross-attention: query = scalar_query, key/value = modulated patches.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=vision_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(vision_dim)

        # Feed-forward after attention.
        self.ffn = nn.Sequential(
            nn.Linear(vision_dim, vision_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(vision_dim * 2, vision_dim),
        )
        self.ffn_norm = nn.LayerNorm(vision_dim)

        # Project to shared SENTINEL embedding space.
        self.projection = nn.Linear(vision_dim, out_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.projection.weight, nonlinearity="relu")
        nn.init.zeros_(self.projection.bias)
        for module in self.ffn.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Produce a fused 256-d embedding from patches and FiLM params.

        Args:
            patch_tokens: Vision patch tokens, shape ``[B, N, 384]``.
            gamma: FiLM scale, shape ``[B, 384]``.
            beta: FiLM shift, shape ``[B, 384]``.

        Returns:
            fused_embedding: Shape ``[B, 256]``.
        """
        # --- Stage 1: FiLM modulation ---
        # Broadcast gamma/beta across the patch dimension.
        modulated = gamma.unsqueeze(1) * patch_tokens + beta.unsqueeze(1)  # [B, N, 384]

        # --- Stage 2: Cross-attention over modulated patches ---
        B = patch_tokens.shape[0]
        query = self.scalar_query.expand(B, -1, -1)      # [B, 1, 384]

        attn_out, _ = self.cross_attn(
            query=query,
            key=modulated,
            value=modulated,
        )  # [B, 1, 384]

        # Pre-norm residual connection on the query.
        query_out = self.attn_norm(query + attn_out)     # [B, 1, 384]

        # Feed-forward with residual.
        ffn_out = self.ffn_norm(query_out + self.ffn(query_out))  # [B, 1, 384]
        pooled = ffn_out.squeeze(1)                               # [B, 384]

        # --- Stage 3: Project to shared space ---
        fused = self.projection(pooled)                           # [B, 256]
        return fused


# ---------------------------------------------------------------------------
# 4. Physics priors
# ---------------------------------------------------------------------------

def _physics_prior_dissolved_oxygen(
    temperature: torch.Tensor,
    tds: torch.Tensor,
) -> torch.Tensor:
    """DO saturation prior from Benson & Krause (1984).

    DO_sat(T) = 14.62 - 0.3898*T + 0.006969*T² - 5.897e-5*T³

    In well-oxygenated water, actual DO ≈ DO_sat.  We use this as a
    strong prior that the residual corrects downward for hypoxic conditions.

    Args:
        temperature: Water temperature in Celsius, shape ``[B]``.
        tds: TDS in ppm (unused here but kept for signature consistency).

    Returns:
        DO saturation estimate in mg/L, shape ``[B]``.
    """
    T = temperature
    return 14.62 - 0.3898 * T + 0.006969 * T ** 2 - 5.897e-5 * T ** 3


def _physics_prior_turbidity(
    images: torch.Tensor,
    temperature: torch.Tensor,
    tds: torch.Tensor,
) -> torch.Tensor:
    """Band-ratio turbidity prior from NIR/Red reflectance.

    Turbidity scales with suspended particles, which increase NIR
    backscatter relative to red.  We approximate it from the spatial mean
    of the NIR (channel index 3) and Red (channel index 0) bands.

    Prior: turbidity_proxy = max(NIR_mean / (Red_mean + ε), 0) * scale

    Args:
        images: Input image tensor, shape ``[B, 4, H, W]``.
        temperature: Unused (kept for uniform signature).
        tds: Unused (kept for uniform signature).

    Returns:
        Turbidity proxy in notional NTU, shape ``[B]``.
    """
    red_mean = images[:, 0, :, :].mean(dim=(-2, -1))    # [B]
    nir_mean = images[:, 3, :, :].mean(dim=(-2, -1))    # [B]
    eps = 1e-6
    ratio = nir_mean / (red_mean + eps)
    # Empirical scale: ratio ~1 maps to ~5 NTU, clamped to reasonable range.
    prior = torch.clamp(ratio * 5.0, min=0.0, max=500.0)
    return prior


def _physics_prior_chlorophyll_a(
    images: torch.Tensor,
    temperature: torch.Tensor,
    tds: torch.Tensor,
) -> torch.Tensor:
    """OC4-style chlorophyll-a band ratio prior from Green/Red reflectance.

    Chlorophyll-a absorbs red strongly and reflects green.  The Green/Red
    ratio is a proxy commonly used in remote sensing.

    Args:
        images: Input image tensor, shape ``[B, 4, H, W]``.
        temperature: Unused (kept for uniform signature).
        tds: Unused (kept for uniform signature).

    Returns:
        Chlorophyll-a proxy in notional μg/L, shape ``[B]``.
    """
    green_mean = images[:, 1, :, :].mean(dim=(-2, -1))  # [B]
    red_mean = images[:, 0, :, :].mean(dim=(-2, -1))    # [B]
    eps = 1e-6
    ratio = green_mean / (red_mean + eps)
    # Empirical scale: ratio ~1.2 ≈ 5 μg/L, ratio ~2.0 ≈ 30 μg/L.
    prior = torch.clamp((ratio - 1.0) * 25.0, min=0.0, max=500.0)
    return prior


def _physics_prior_ph(
    temperature: torch.Tensor,
    tds: torch.Tensor,
) -> torch.Tensor:
    """Nernstian pH estimate from temperature and TDS proxy.

    For natural freshwaters pH is weakly correlated with ion concentration
    (TDS) via the carbonate equilibrium.  We encode a mild prior that:
    - Low TDS → slightly acidic (soft water, pH ≈ 6.5).
    - High TDS → near neutral / slightly alkaline (pH ≈ 7.8).
    - Temperature shifts via the Nernstian term dE/dT ≈ +0.003 pH/°C.

    Args:
        temperature: Water temperature in Celsius, shape ``[B]``.
        tds: TDS in ppm, shape ``[B]``.

    Returns:
        pH prior, shape ``[B]``.
    """
    # Empirical sigmoid mapping TDS → pH with clamping.
    # Anchored: TDS=50 → 6.8, TDS=500 → 7.5
    tds_norm = torch.clamp(tds / 500.0, min=0.01, max=10.0)
    ph_tds = 6.5 + 1.0 * (tds_norm / (tds_norm + 1.0))
    # Nernstian temperature correction (relative to 25 °C reference).
    ph_temp_correction = 0.003 * (25.0 - temperature)
    return ph_tds + ph_temp_correction


def _physics_prior_nitrate(
    temperature: torch.Tensor,
    tds: torch.Tensor,
) -> torch.Tensor:
    """Weak nitrate prior from TDS (very approximate).

    Nitrate contributes to TDS; in agricultural watersheds ~10-20 mg/L
    NO3 per 100 ppm TDS is a rough rule of thumb, but this prior carries
    high uncertainty and the residual network is expected to dominate.

    Args:
        temperature: Unused (kept for uniform signature).
        tds: TDS in ppm, shape ``[B]``.

    Returns:
        Nitrate prior in mg/L, shape ``[B]``.
    """
    # Very weak: ~ 1.5% of TDS, clamped to [0, 50] mg/L.
    prior = torch.clamp(tds * 0.015, min=0.0, max=50.0)
    return prior


def _physics_prior_conductivity(
    temperature: torch.Tensor,
    tds: torch.Tensor,
) -> torch.Tensor:
    """Strong conductivity prior from TDS via the TDS–conductivity relation.

    For most natural freshwaters: TDS (mg/L) ≈ k × Conductivity (μS/cm),
    where k ≈ 0.65 (range 0.5–0.9 depending on ionic composition).

    Inverting: Conductivity ≈ TDS / 0.65.

    Args:
        temperature: Unused (kept for uniform signature).
        tds: TDS in ppm, shape ``[B]``.

    Returns:
        Conductivity estimate in μS/cm, shape ``[B]``.
    """
    return tds / 0.65


# ---------------------------------------------------------------------------
# 5. PhysicsResidualHead
# ---------------------------------------------------------------------------

@dataclass
class HeadConfig:
    """Configuration for a single :class:`PhysicsResidualHead`.

    Attributes:
        target_name: Human-readable target identifier (matches
            :data:`TARGET_PARAMS`).
        uses_image: Whether the physics prior needs the image tensor.
        residual_hidden_dim: Width of the residual MLP.
    """

    target_name: str
    uses_image: bool = False
    residual_hidden_dim: int = 128


class PhysicsResidualHead(nn.Module):
    """Prediction head combining a physics prior with a learned residual.

    Each head predicts::

        mu    = physics_prior(inputs) + mu_residual(fused_embedding)
        sigma = softplus(log_sigma(fused_embedding))

    The physics prior is a pure Python/PyTorch function injected at
    construction time.  The residual network is a 3-layer MLP.

    Args:
        target_name: Identifier for this target (informational).
        fused_dim: Dimensionality of the fused embedding.  Default 256.
        residual_hidden_dim: Width of the residual MLP.
        uses_image: Whether the physics prior requires the image tensor.
    """

    def __init__(
        self,
        target_name: str,
        fused_dim: int = FUSED_DIM,
        residual_hidden_dim: int = 128,
        uses_image: bool = False,
    ) -> None:
        super().__init__()
        self.target_name = target_name
        self.uses_image = uses_image

        self.residual_net = nn.Sequential(
            nn.Linear(fused_dim, residual_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(residual_hidden_dim, residual_hidden_dim // 2),
            nn.GELU(),
        )
        # Separate heads for mean residual and log-sigma.
        self.mu_head = nn.Linear(residual_hidden_dim // 2, 1)
        self.log_sigma_head = nn.Linear(residual_hidden_dim // 2, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Small initial log_sigma → moderate initial uncertainty.
        nn.init.constant_(self.log_sigma_head.bias, -1.0)

    def forward(
        self,
        fused_embedding: torch.Tensor,
        physics_prior_value: torch.Tensor,
    ) -> TargetPrediction:
        """Predict mu and sigma for this target.

        Args:
            fused_embedding: Shape ``[B, 256]``.
            physics_prior_value: Physics prior scalar, shape ``[B]``.
                Pass a zero tensor for targets without a meaningful prior.

        Returns:
            :class:`TargetPrediction` with ``mu``, ``sigma``, and
            ``physics_prior_value``.
        """
        h = self.residual_net(fused_embedding)      # [B, hidden//2]

        mu_residual = self.mu_head(h).squeeze(-1)   # [B]
        log_sigma = self.log_sigma_head(h).squeeze(-1)  # [B]

        mu = physics_prior_value + mu_residual
        sigma = F.softplus(log_sigma) + 1e-6        # strictly positive

        return TargetPrediction(
            mu=mu,
            sigma=sigma,
            physics_prior_value=physics_prior_value,
        )


# ---------------------------------------------------------------------------
# 6. UncertaintyRouter
# ---------------------------------------------------------------------------

def _trust_flag(score: torch.Tensor) -> List[str]:
    """Convert a batch of trust scores to flag strings.

    Args:
        score: Trust scores in ``[0, 1]``, shape ``[B]``.

    Returns:
        List of ``B`` strings: ``"green"``, ``"yellow"``, or ``"red"``.
    """
    flags: List[str] = []
    for s in score.detach().cpu().tolist():
        if s >= TRUST_GREEN_THRESHOLD:
            flags.append("green")
        elif s >= TRUST_RED_THRESHOLD:
            flags.append("yellow")
        else:
            flags.append("red")
    return flags


class UncertaintyRouter(nn.Module):
    """Per-target trust scoring from prediction uncertainty and embedding.

    For each target, a small MLP maps the tuple
    ``[fused_embedding, mu, sigma]`` to a scalar trust score in ``[0, 1]``.
    A score above :data:`TRUST_GREEN_THRESHOLD` means the prediction is
    reliable; below :data:`TRUST_RED_THRESHOLD` means it should be
    disregarded without additional measurements.

    Architecture (shared weights across targets, conditioned on target idx)::

        [fused_embedding (256), mu (1), sigma (1)]  →  [258]
            → Linear(258, 64) → GELU
            → Linear(64, 32)  → GELU
            → Linear(32, 1)   → Sigmoid
            → trust_score [B]

    Note: Each target gets its own independent MLP so that different targets
    can learn different relationships between embedding uncertainty and
    trustworthiness.

    Args:
        fused_dim: Dimensionality of the fused embedding.
        num_targets: Number of water quality targets.
        hidden_dim: Width of the trust MLP hidden layers.
    """

    def __init__(
        self,
        fused_dim: int = FUSED_DIM,
        num_targets: int = NUM_TARGETS,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        input_dim = fused_dim + 2  # embedding + mu + sigma

        # One MLP per target, stored as a ModuleList for proper registration.
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(num_targets)
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Bias last layers toward moderate trust at initialisation.
        for mlp in self.mlps:
            last_linear = mlp[-1]
            nn.init.constant_(last_linear.bias, 0.0)

    def forward(
        self,
        fused_embedding: torch.Tensor,
        predictions: Dict[str, TargetPrediction],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[str]]]:
        """Compute trust scores and flags for all targets.

        Args:
            fused_embedding: Shape ``[B, 256]``.
            predictions: Dict mapping target name → :class:`TargetPrediction`.

        Returns:
            Tuple of:
            - **trust_scores**: Dict mapping target name → tensor ``[B]``
              in ``[0, 1]``.
            - **trust_flags**: Dict mapping target name → list of ``B``
              flag strings.
        """
        trust_scores: Dict[str, torch.Tensor] = {}
        trust_flags: Dict[str, List[str]] = {}

        for target_idx, target_name in enumerate(TARGET_PARAMS):
            pred = predictions[target_name]
            # Concatenate fused embedding with mu and sigma.
            x = torch.cat(
                [
                    fused_embedding,
                    pred.mu.unsqueeze(-1),
                    pred.sigma.unsqueeze(-1),
                ],
                dim=-1,
            )  # [B, 258]

            logit = self.mlps[target_idx](x).squeeze(-1)   # [B]
            score = torch.sigmoid(logit)                    # [B]

            trust_scores[target_name] = score
            trust_flags[target_name] = _trust_flag(score)

        return trust_scores, trust_flags


# ---------------------------------------------------------------------------
# 7. WaterDroneNet (main model)
# ---------------------------------------------------------------------------

class WaterDroneNet(nn.Module):
    """Physics-conditioned water quality prediction from drone + sensor data.

    WaterDroneNet ingests RGB+NIR drone imagery together with cheap scalar
    sensor readings (temperature and TDS) and predicts a panel of 12 water
    quality parameters, each with a calibrated uncertainty estimate and a
    per-sample trust flag.

    The three novelties over a plain regression network:

    1. **FiLM sensor conditioning** -- temperature and TDS modulate the ViT
       patch features before fusion, so the model knows *which water quality
       regime* it is looking at before interpreting colour cues.
    2. **Physics-residual heads** -- every prediction is anchored to a known
       physical equation; the network only has to learn the *deviation* from
       that anchor, which dramatically reduces the function class and improves
       extrapolation.
    3. **Uncertainty routing** -- a per-target gate scores each prediction's
       reliability, enabling downstream decision-making without manual
       threshold tuning.

    Usage::

        model = WaterDroneNet()
        output = model(images, scalars)
        do_mean  = output.predictions["dissolved_oxygen"].mu
        do_sigma = output.predictions["dissolved_oxygen"].sigma
        do_trust = output.trust_flags["dissolved_oxygen"]  # ["green", ...]

    Args:
        pretrained_vision: Load ImageNet weights for the vision encoder.
        hydrovit_checkpoint_path: Optional HydroViT checkpoint for the
            vision encoder (takes priority over *pretrained_vision*).
        freeze_vision_backbone: Freeze ViT weights during training.
        vision_dropout: Dropout in the FiLM fusion cross-attention.
    """

    def __init__(
        self,
        pretrained_vision: bool = True,
        hydrovit_checkpoint_path: Optional[str] = None,
        freeze_vision_backbone: bool = False,
        vision_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # --- Sub-modules ---
        self.vision_encoder = DroneVisionEncoder(
            in_chans=DRONE_IN_CHANS,
            img_size=DRONE_IMG_SIZE,
            pretrained=pretrained_vision,
            hydrovit_checkpoint_path=hydrovit_checkpoint_path,
            freeze_backbone=freeze_vision_backbone,
        )
        self.scalar_encoder = ScalarEncoder(
            scalar_dim=NUM_SCALAR_INPUTS,
            out_dim=DRONE_EMBED_DIM,
        )
        self.fusion = FiLMConditionedFusion(
            vision_dim=DRONE_EMBED_DIM,
            out_dim=FUSED_DIM,
            num_heads=8,
            dropout=vision_dropout,
        )
        self.uncertainty_router = UncertaintyRouter(
            fused_dim=FUSED_DIM,
            num_targets=NUM_TARGETS,
        )

        # Build one PhysicsResidualHead per target.
        self.prediction_heads = nn.ModuleDict({
            name: PhysicsResidualHead(
                target_name=name,
                fused_dim=FUSED_DIM,
                residual_hidden_dim=128,
                uses_image=(name in {"turbidity", "chlorophyll_a"}),
            )
            for name in TARGET_PARAMS
        })

    # ------------------------------------------------------------------
    # Physics prior dispatch
    # ------------------------------------------------------------------

    def _compute_physics_priors(
        self,
        images: torch.Tensor,
        scalars: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all physics prior values for the current batch.

        Args:
            images: Shape ``[B, 4, 224, 224]``.
            scalars: Shape ``[B, 2]`` — ``[:, 0]`` temperature, ``[:, 1]`` TDS.

        Returns:
            Dict mapping each target name to a prior tensor of shape ``[B]``.
        """
        temperature = scalars[:, 0]   # [B]
        tds = scalars[:, 1]           # [B]
        B = scalars.shape[0]
        device = scalars.device
        zeros = torch.zeros(B, device=device)

        priors: Dict[str, torch.Tensor] = {}

        priors["dissolved_oxygen"] = _physics_prior_dissolved_oxygen(
            temperature, tds
        )
        priors["ph"] = _physics_prior_ph(temperature, tds)
        priors["turbidity"] = _physics_prior_turbidity(images, temperature, tds)
        priors["chlorophyll_a"] = _physics_prior_chlorophyll_a(
            images, temperature, tds
        )
        priors["nitrate"] = _physics_prior_nitrate(temperature, tds)

        # Phosphate: no strong physics prior; use zero so residual dominates.
        priors["phosphate"] = zeros.clone()

        priors["conductivity"] = _physics_prior_conductivity(temperature, tds)

        # Temperature self-prediction: prior = sensor reading itself.
        priors["temperature"] = temperature.clone()

        # Remaining targets: zero prior (residual dominates).
        for name in (
            "total_suspended_solids",
            "biochemical_oxygen_demand",
            "ammonia",
            "total_organic_carbon",
        ):
            priors[name] = zeros.clone()

        return priors

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        scalars: torch.Tensor,
    ) -> WaterDroneNetOutput:
        """Run a full forward pass and return all predictions and trust flags.

        Args:
            images: RGB+NIR drone imagery, shape ``[B, 4, 224, 224]``.
                Channel order: ``[R, G, B, NIR]``.  Values should be
                float32 in the range ``[0, 1]`` (or normalised equivalently).
            scalars: Sensor readings, shape ``[B, 2]``.
                ``[:, 0]`` is temperature in °C, ``[:, 1]`` is TDS in ppm.

        Returns:
            :class:`WaterDroneNetOutput` with predictions, trust scores,
            trust flags, and the 256-d fused embedding.
        """
        # --- 1. Vision encoding ---
        patch_tokens, _cls_token = self.vision_encoder(images)  # [B, N, 384], [B, 384]

        # --- 2. Scalar → FiLM parameters ---
        gamma, beta = self.scalar_encoder(scalars)  # [B, 384] each

        # --- 3. FiLM-conditioned cross-attention fusion ---
        fused_embedding = self.fusion(patch_tokens, gamma, beta)  # [B, 256]

        # --- 4. Physics priors ---
        physics_priors = self._compute_physics_priors(images, scalars)

        # --- 5. Physics-residual prediction heads ---
        predictions: Dict[str, TargetPrediction] = {}
        for name in TARGET_PARAMS:
            pred = self.prediction_heads[name](
                fused_embedding=fused_embedding,
                physics_prior_value=physics_priors[name],
            )
            predictions[name] = pred

        # --- 6. Uncertainty routing ---
        trust_scores, trust_flags = self.uncertainty_router(
            fused_embedding=fused_embedding,
            predictions=predictions,
        )

        return WaterDroneNetOutput(
            predictions=predictions,
            trust_scores=trust_scores,
            trust_flags=trust_flags,
            fused_embedding=fused_embedding,
        )

    # ------------------------------------------------------------------
    # Convenience / diagnostics
    # ------------------------------------------------------------------

    def predict_dict(
        self,
        images: torch.Tensor,
        scalars: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Run forward and return a flat dict keyed by target name.

        Convenience wrapper for notebooks and evaluation scripts.

        Args:
            images: Shape ``[B, 4, 224, 224]``.
            scalars: Shape ``[B, 2]``.

        Returns:
            Dict with keys equal to :data:`TARGET_PARAMS`, each mapping
            to a nested dict with ``"mu"``, ``"sigma"``, ``"prior"``,
            and ``"trust"`` tensors of shape ``[B]``.
        """
        with torch.no_grad():
            output = self.forward(images, scalars)

        result: Dict[str, Dict[str, torch.Tensor]] = {}
        for name in TARGET_PARAMS:
            pred = output.predictions[name]
            result[name] = {
                "mu": pred.mu,
                "sigma": pred.sigma,
                "prior": pred.physics_prior_value,
                "trust": output.trust_scores[name],
            }
        return result

    def num_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
