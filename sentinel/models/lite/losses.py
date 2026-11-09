# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Losses for SENTINEL-Lite (research 08 §e.6): pooled in-situ Gaussian (β-)NLL and the cross-scale
distillation NLL of S2-cell block means against teacher (μ_T, σ_T² + σ_scale²), trust-weighted."""

from __future__ import annotations

import torch


def pool_maps(out: torch.Tensor, weights: torch.Tensor, n_targets: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted chip-level μ and variance. ``out`` (N, 2T, h, w), ``weights`` (N, h, w) summing to 1."""
    w = weights.unsqueeze(1)
    mu = (out[:, :n_targets] * w).sum((2, 3))
    var = (torch.exp(out[:, n_targets:]) * w).sum((2, 3))
    return mu, var


def pool_maps_median(out: torch.Tensor, weights: torch.Tensor, n_targets: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted-median chip μ (robust to a minority of corrupted pixels, e.g. glint) and the weighted-mean variance.
    Inference-time alternative to :func:`pool_maps` (handoff idea "median pooling")."""
    N, _, h, w_ = out.shape
    mu = out[:, :n_targets].reshape(N, n_targets, h * w_)
    wt = weights.reshape(N, 1, h * w_).expand(-1, n_targets, -1)
    srt, idx = torch.sort(mu, dim=-1)
    cw = torch.cumsum(torch.gather(wt, -1, idx), -1)
    k = (cw < 0.5 * cw[..., -1:]).sum(-1, keepdim=True).clamp_max(h * w_ - 1)
    med = torch.gather(srt, -1, k).squeeze(-1)
    var = (torch.exp(out[:, n_targets:]) * weights.unsqueeze(1)).sum((2, 3))
    return med, var


def gaussian_nll(mu, var, y, mask, beta: float = 0.5) -> torch.Tensor:
    """β-NLL (Seitzer et al. 2022) averaged over observed entries; mask (N, T) bool."""
    var = var.clamp_min(1e-4)
    nll = 0.5 * (torch.log(var) + (y - mu) ** 2 / var)
    if beta > 0:
        nll = nll * var.detach() ** beta
    m = mask.float()
    # mean per target over observed chips, then mean over targets present in the batch
    per_t = (nll * m).sum(0) / m.sum(0).clamp_min(1.0)
    present = (m.sum(0) > 0).float()
    return (per_t * present).sum() / present.sum().clamp_min(1.0)


def distill_nll(out: torch.Tensor, cell_map: torch.Tensor, water_w: torch.Tensor, t_mu: torch.Tensor,
                t_var: torch.Tensor, t_w: torch.Tensor, n_targets: int, sigma_scale2: float = 0.01) -> torch.Tensor:
    """Block-averaged distillation loss.

    out      (N, 2T, h, w) student maps (standardised targets)
    cell_map (N, h, w) long: index of the S2 10-m cell each output pixel falls in (0..K-1), -1 = none
    water_w  (N, h, w) water weight of each output pixel (0 = land)
    t_mu, t_var (N, K, T) teacher mean / variance in the student's standardised units (NaN = no teacher)
    t_w      (N, K, T) trust weight per cell and target (0 = unusable: cloud/glint/adjacency flagged)
    """
    N, _, h, w = out.shape
    K = t_mu.shape[1]
    mu = out[:, :n_targets].permute(0, 2, 3, 1).reshape(N, h * w, n_targets)
    cm = cell_map.reshape(N, h * w)
    ww = water_w.reshape(N, h * w) * (cm >= 0).float()
    idx = cm.clamp_min(0)
    num = torch.zeros(N, K, n_targets, device=out.device).scatter_add_(
        1, idx.unsqueeze(-1).expand(-1, -1, n_targets), mu * ww.unsqueeze(-1))
    den = torch.zeros(N, K, device=out.device).scatter_add_(1, idx, ww)
    block = num / den.clamp_min(1e-6).unsqueeze(-1)
    valid = (den > 0.5).unsqueeze(-1) & torch.isfinite(t_mu) & torch.isfinite(t_var) & (t_w > 0)
    tv = torch.nan_to_num(t_var, nan=1.0) + sigma_scale2
    nll = 0.5 * (torch.log(tv) + (block - torch.nan_to_num(t_mu)) ** 2 / tv)
    wt = t_w * valid.float()
    return (nll * wt).sum() / wt.sum().clamp_min(1.0)
