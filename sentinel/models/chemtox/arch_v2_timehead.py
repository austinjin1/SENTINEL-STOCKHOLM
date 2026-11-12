"""ChemTox-TimeHead: a learned monotone time law bolted onto a frozen tabular model.

New module, no project row read. The cheap experiment that separates *mechanism* from
*representation*: keep the GBM exactly as it is, have it predict the **incipient** (long-duration
asymptotic) value, and learn a small per-row rate head that maps that asymptote to any requested
duration through the **same** law as TED (:mod:`arch_v2_timelaw`).

If TimeHead captures most of TED's T6 advantage then the expensive graph trunk is not what is
working and the write-up says so; if it does not, TED's trunk has earned its GPU-hours. Either way
the project learns something for about 1 GPU-hour, which is why amendment §5.7 orders it before TED.

Governing text in precedence order: ``chemtox_arch_v2_lead_rulings.md`` (sha256 ``553be413…``),
``chemtox_arch_v2_addendum_01.md``, the stamped amendment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from sentinel.models.chemtox import arch_v2_timelaw as L

__all__ = ["TimeHeadError", "TimeHeadConfig", "RateMLP", "TimeHead", "incipient_target"]


class TimeHeadError(ValueError):
    """A pre-registered rule about TimeHead was violated."""


@dataclass
class TimeHeadConfig:
    n_features: int
    hidden: int = 64
    depth: int = 2
    dropout: float = 0.0
    b_decades: float = L.DEFAULT_B_DECADES
    anchor: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def n_parameters_budget(self) -> int:
        """The design budget: a rate head of under 50k parameters (amendment §5.7)."""
        return 50_000


class RateMLP(nn.Module):
    """The <50k-parameter rate head: descriptors + species + conditions -> ``log10 k``."""

    def __init__(self, cfg: TimeHeadConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = cfg.n_features
        for _ in range(max(1, cfg.depth)):
            layers += [nn.Linear(d, cfg.hidden), nn.ReLU()]
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            d = cfg.hidden
        self.body = nn.Sequential(*layers)
        self.rate = L.RateHead(d, b_decades=cfg.b_decades, anchor=cfg.anchor, hidden=cfg.hidden)

    def forward(self, x: Tensor, log_kow: Tensor, log10_mw: Tensor) -> Tensor:
        return self.rate.rate(self.body(x), log_kow, log10_mw)


class TimeHead(nn.Module):
    """``log10 LC50(t) = y_inf_frozen - log10(1 - exp(-k(x) t))``.

    ``y_inf_frozen`` comes from the **frozen** tabular model and is passed in, not learned here;
    only ``k`` is learned. That is the whole point: the representation is held fixed so the
    mechanism is the only thing varying.
    """

    def __init__(self, cfg: TimeHeadConfig) -> None:
        super().__init__()
        if cfg.n_features <= 0:
            raise TimeHeadError("n_features must be positive")
        self.cfg = cfg
        self.head = RateMLP(cfg)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: Tensor, y_inf_frozen: Tensor, t: Tensor, log_kow: Tensor,
                log10_mw: Tensor) -> dict:
        if x.ndim != 2:
            raise TimeHeadError(f"x must be (n, n_features); got {tuple(x.shape)}")
        y = y_inf_frozen.reshape(-1)
        tt = t.reshape(-1)
        if not (y.shape[0] == tt.shape[0] == x.shape[0]):
            raise TimeHeadError("x, y_inf_frozen and t must agree on the row count")
        if y.requires_grad:
            raise TimeHeadError(
                "y_inf_frozen carries a gradient: the tabular model is FROZEN in this design, so "
                "detach it before calling TimeHead")
        k = self.head(x, log_kow, log10_mw)
        with L.fp32_block():
            pred = y.to(torch.float32) + L.saturation_term(k, tt)
        return {"pred": pred, "k": k, "y_inf": y}


def incipient_target(y: Sequence[float], duration_h: Sequence[float], *,
                     long_duration_h: float = 96.0) -> np.ndarray:
    """The subset mask the frozen model is trained on: rows at or beyond ``long_duration_h``.

    The frozen GBM predicts the *incipient* value, so it is fitted on the longest durations
    available inside the training window — on T6 that is exactly 96 h, since the AUX-LEAK ruling
    caps every T6 training row at 96 h.
    """
    yy = np.asarray(list(y), dtype=float)
    tt = np.asarray(list(duration_h), dtype=float)
    if yy.size != tt.size:
        raise TimeHeadError("y and duration_h must have the same length")
    mask = np.isfinite(tt) & (tt >= long_duration_h)
    if not mask.any():
        raise TimeHeadError(
            f"no training row at or beyond {long_duration_h} h: the incipient target is undefined")
    return mask
