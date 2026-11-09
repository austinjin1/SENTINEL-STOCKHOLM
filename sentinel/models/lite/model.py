# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Lite student: timm ReLU backbone (MobileNetV4-Conv-S / EfficientNet-Lite0) + light FPN decoder
producing per-pixel μ and log σ² maps (stride 4) for the 4 optical targets, plus an optional per-pixel
spectral branch (1×1 convs on log band ratios) that can be pretrained on GLORIA camera-band spectra.

Model input: (N, 4, H, W) float, bands R, G, B, NIR, already normalised by :func:`normalise_input`.
Model output: (N, 2T, H/4, W/4): channels [0, T) = μ (standardised log10 target), [T, 2T) = log σ².
De-standardise with the ``y_mean`` / ``y_std`` buffers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .data import TARGETS

BACKBONES = {"mnv4s": "mobilenetv4_conv_small.e2400_r224_in1k", "effl0": "efficientnet_lite0.ra_in1k"}


def _cbr(cin, cout, k=1, groups=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, groups=groups, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def _dws(c):
    return nn.Sequential(_cbr(c, c, 3, groups=c), _cbr(c, c, 1))


class SpectralBranch(nn.Module):
    """Per-pixel MLP on [log(R/G), log(B/G), log(NIR/G), log G] of the *raw-DN-like* input (x in [0, 1]).
    BatchNorm on the 4 features lets the same weights run on GLORIA Rrs (pretraining) and NAIP DN."""

    def __init__(self, n_out: int, hidden: int = 32, eps: float = 1.0 / 255.0):
        super().__init__()
        self.eps = eps
        self.norm = nn.BatchNorm2d(4)
        self.mlp = nn.Sequential(nn.Conv2d(4, hidden, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden, hidden, 1),
                                 nn.ReLU(inplace=True), nn.Conv2d(hidden, n_out, 1))

    @staticmethod
    def log_features(x: torch.Tensor, eps: float) -> torch.Tensor:
        lx = torch.log(x.clamp_min(0) + eps)
        r, g, b, n = lx[:, 0:1], lx[:, 1:2], lx[:, 2:3], lx[:, 3:4]
        return torch.cat([r - g, b - g, n - g, g], 1)

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(self.log_features(x01, self.eps)))


class LiteStudent(nn.Module):
    def __init__(self, backbone: str = "mnv4s", *, pretrained: bool = True, dec_ch: int = 48,
                 n_targets: int = len(TARGETS), spectral: bool = False, in_chans: int = 4, film: bool = False,
                 film_source: str = "frame"):
        super().__init__()
        import timm

        self.backbone_name = backbone
        self.encoder = timm.create_model(BACKBONES.get(backbone, backbone), features_only=True, in_chans=in_chans,
                                         pretrained=pretrained, out_indices=(1, 2, 3, 4))
        chs = self.encoder.feature_info.channels()
        self.lateral = nn.ModuleList([_cbr(c, dec_ch) for c in chs])
        self.smooth = _dws(dec_ch)
        self.head = nn.Sequential(_dws(dec_ch), nn.Conv2d(dec_ch, 2 * n_targets, 1))
        self.n_targets = n_targets
        self.spectral = SpectralBranch(2 * n_targets) if spectral else None
        # frame-statistics conditioning (FiLM on the decoder): per-band mean/std of the [0,1] input over the frame,
        # computable on the drone per frame (NAIP vendor colour balance varies per state-year)
        self.film = nn.Sequential(nn.Linear(2 * in_chans, 32), nn.ReLU(inplace=True), nn.Linear(32, 2 * dec_ch)) \
            if film else None
        self.film_source = film_source       # 'frame': stats of this input; 'scene': supplied (DOQQ / flight stats)
        if self.film is not None:
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)
        self.register_buffer("y_mean", torch.zeros(n_targets))
        self.register_buffer("y_std", torch.ones(n_targets))
        # input statistics for 'raw' mode (x = DN/255 standardised); stored for export
        self.register_buffer("x_mean", torch.zeros(in_chans))
        self.register_buffer("x_std", torch.ones(in_chans))
        self.input_mode = "raw"

    @property
    def needs_x01(self) -> bool:
        return self.spectral is not None or self.film is not None

    def forward(self, x: torch.Tensor, x01: torch.Tensor | None = None, cond: torch.Tensor | None = None) -> torch.Tensor:
        feats = self.encoder(x)
        top = self.lateral[-1](feats[-1])
        for f, lat in zip(reversed(feats[:-1]), reversed(list(self.lateral[:-1]))):
            top = Fn.interpolate(top, size=f.shape[-2:], mode="nearest") + lat(f)
        top = self.smooth(top)
        if self.film is not None:
            if self.film_source == "scene":
                if cond is None:
                    raise ValueError("scene FiLM needs cond (N, 8): per-band scene centre/255 and scale/255")
                st = cond
            else:
                if x01 is None:
                    raise ValueError("FiLM needs the [0,1] input x01")
                st = torch.cat([x01.mean((2, 3)), x01.std((2, 3))], 1)
            g, b = self.film(st).chunk(2, 1)
            top = top * (1 + g[:, :, None, None]) + b[:, :, None, None]
        out = self.head(top)
        if self.spectral is not None:
            if x01 is None:
                raise ValueError("spectral branch needs the [0,1] input x01")
            s = self.spectral(Fn.avg_pool2d(x01, 4))
            out = out + Fn.interpolate(s, size=out.shape[-2:], mode="nearest")
        mu, lv = out[:, : self.n_targets], out[:, self.n_targets:].clamp(-8.0, 6.0)
        return torch.cat([mu, lv], 1)


class ExportWrapper(nn.Module):
    """Single-input module for ONNX: takes DN/255 in [0, 1] (raw mode) or the scene-normalised input (scene mode)
    and returns de-standardised log10 μ and log σ (log10 units) maps. ``rgb_only`` reproduces the training-time
    NIR ablation exactly (train.normalise_input): model-input NIR = 0 and spectral-branch NIR = G (review W10)."""

    def __init__(self, model: LiteStudent, scene_mode: bool = False, rgb_only: bool = False):
        super().__init__()
        self.m = model
        self.scene_mode = scene_mode
        self.rgb_only = rgb_only

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        if self.scene_mode:
            if self.m.needs_x01:
                raise ValueError("scene mode export does not support the spectral branch / FiLM")
            x = x01                       # caller supplies scene-normalised input
            if self.rgb_only:
                x = torch.cat([x[:, :3], torch.zeros_like(x[:, 3:4])], 1)
            out = self.m(x)
        else:
            x = (x01 - self.m.x_mean[None, :, None, None]) / self.m.x_std[None, :, None, None]
            if self.rgb_only:
                x = torch.cat([x[:, :3], torch.zeros_like(x[:, 3:4])], 1)
                x01 = torch.cat([x01[:, :3], x01[:, 1:2]], 1)
            out = self.m(x, x01 if self.m.needs_x01 else None)
        T = self.m.n_targets
        mu = out[:, :T] * self.m.y_std[None, :, None, None] + self.m.y_mean[None, :, None, None]
        log_sigma = 0.5 * out[:, T:] + torch.log(self.m.y_std)[None, :, None, None]
        return torch.cat([mu, log_sigma], 1)


def count_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))


def count_macs(m: nn.Module, size: int = 512, in_chans: int = 4) -> float:
    """MACs per tile via torch.utils.flop_counter (conv/matmul only; FLOPs / 2)."""
    from torch.utils.flop_counter import FlopCounterMode

    m = m.eval()
    x = torch.zeros(1, in_chans, size, size)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(x)
    return fc.get_total_flops() / 2.0
