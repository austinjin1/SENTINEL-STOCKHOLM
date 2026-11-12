"""ChemTox-TED: the time-and-endpoint-decomposed multitask network, and its ablations A1-A6.

New module. No project row is read: :class:`TED` takes already-built tensors, and the caller obtains
rows through ``arch_v2_guard.load_training_pool`` / ``load_auxiliary_rows``. The time law and the
endpoint ordering live in :mod:`sentinel.models.chemtox.arch_v2_timelaw`, so the same arithmetic —
and the same tests — back TED and TimeHead.

Governing text in precedence order: ``chemtox_arch_v2_lead_rulings.md`` (sha256 ``553be413…``),
``chemtox_arch_v2_addendum_01.md``, the stamped amendment (§7.2 for the ablations).

The one design claim
--------------------
**Duration and endpoint identity never enter the trunk.** They act only through the composition
layer, which is the restriction a GBM cannot be given. :meth:`TED.forward` therefore takes the
trunk features and the duration separately and :func:`assert_no_time_in_trunk` fails the run if a
duration- or endpoint-derived column reaches the trunk — except under ablation **A3**, which adds
them back deliberately as a leak test.

The ablations, each a constructor flag
--------------------------------------
======  ====================================================================================
**A1**  ``mechanism=False``: the trunk with free-form duration conditioning (TRIDENT-shaped),
        no composition layer. *Is the mechanism doing anything?*
**A2**  ``b_decades=0.0``: the pure anchored law, no learned residual. The physics-only floor.
**A3**  ``leak_time_into_trunk=True``: duration and endpoint fed to the trunk as well.
**A4**  ``multitask=False``: acute-only; the chronic and censored heads are not trained.
**A5**  ``bcf_head=True``: the bioconcentration head (admissible; the H4 report landed).
**A6**  ``q10=True``: a temperature term with a **free** prior; no literature constant is assumed.
======  ====================================================================================

**Attribution rule (binding).** If A1 scores within the block's governing MDE of the full model, the
win belongs to the trunk and the data, the mechanistic framing is dropped from the claim and the
paper, and that is reported in the headline table. :data:`ABLATIONS` carries the list so a runner
cannot quietly omit one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from sentinel.models.chemtox import arch_v2_timelaw as L

__all__ = ["TedError", "ABLATIONS", "TIME_DERIVED_PREFIXES", "assert_no_time_in_trunk",
           "TedConfig", "TED", "heteroscedastic_nll", "censored_nll", "loss_terms"]


class TedError(ValueError):
    """A pre-registered rule about TED was violated."""


#: The required ablation set. A claim for TED needs A1 and A2 present (amendment §7.2).
ABLATIONS = ("A1", "A2", "A3", "A4", "A5", "A6")

#: Any trunk column whose name starts with one of these is duration- or endpoint-derived and is
#: inadmissible in the trunk unless ablation A3 is in force.
TIME_DERIVED_PREFIXES = ("duration", "dur_", "obs_duration", "exposure_duration", "endpoint",
                         "endpoint_kind", "band", "duration_band", "log_duration", "log10_dur")


def assert_no_time_in_trunk(columns: Sequence[str], *, allow: bool = False) -> None:
    """The design claim, checked mechanically rather than promised.

    ``allow=True`` is ablation A3 and is the only way a time- or endpoint-derived column may reach
    the trunk.
    """
    hits = sorted(c for c in columns
                  if str(c).lower().startswith(TIME_DERIVED_PREFIXES))
    if hits and not allow:
        raise TedError(
            f"duration/endpoint-derived columns {hits} reached the TED trunk. They may act ONLY "
            f"through the composition layer; feeding them to the trunk is ablation A3 and must be "
            f"requested explicitly (leak_time_into_trunk=True).")
    if allow and not hits:
        raise TedError(
            "ablation A3 was requested but no duration/endpoint column is present in the trunk "
            "features, so the leak test would measure nothing")


@dataclass
class TedConfig:
    """One TED configuration. Every field is a pre-registered dimension of the search."""

    n_features: int
    latent_dim: int = 256
    hidden: int = 256
    depth: int = 2
    dropout: float = 0.0
    b_decades: float = L.DEFAULT_B_DECADES
    anchor: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # ablation switches
    mechanism: bool = True              # A1 off => False
    multitask: bool = True              # A4 off => False
    leak_time_into_trunk: bool = False  # A3 on  => True
    bcf_head: bool = False              # A5 on  => True
    q10: bool = False                   # A6 on  => True

    def ablation_name(self) -> str | None:
        if not self.mechanism:
            return "A1"
        if self.b_decades == 0.0:
            return "A2"
        if self.leak_time_into_trunk:
            return "A3"
        if not self.multitask:
            return "A4"
        if self.bcf_head:
            return "A5"
        if self.q10:
            return "A6"
        return None


class TED(nn.Module):
    """Trunk -> latent ``z``; ``z`` + duration -> the composition layer -> every endpoint.

    The trunk is a plain MLP over the concatenated descriptor / fingerprint / embedding block. The
    graph encoder of the full design is a drop-in replacement for the trunk and is not required for
    any property tested here, so it is supplied by the caller through ``trunk`` when wanted.
    """

    def __init__(self, cfg: TedConfig, *, trunk: nn.Module | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.n_features <= 0:
            raise TedError("n_features must be positive")
        if trunk is not None:
            self.trunk = trunk
        else:
            layers: list[nn.Module] = []
            d = cfg.n_features + (1 if cfg.mechanism is False else 0)  # A1 conditions on log t
            for _ in range(max(1, cfg.depth)):
                layers += [nn.Linear(d, cfg.hidden), nn.ReLU()]
                if cfg.dropout > 0:
                    layers.append(nn.Dropout(cfg.dropout))
                d = cfg.hidden
            layers.append(nn.Linear(d, cfg.latent_dim))
            self.trunk = nn.Sequential(*layers)

        if cfg.mechanism:
            self.head = L.AcuteChronicHead(cfg.latent_dim, b_decades=cfg.b_decades,
                                           anchor=cfg.anchor)
        else:
            # A1: free-form duration conditioning, TRIDENT-shaped, no composition layer
            self.head = None
            self.direct = nn.Linear(cfg.latent_dim, 2)      # acute, chronic
        self.log_sigma = nn.Linear(cfg.latent_dim, 1)       # heteroscedastic sigma (target H6)
        self.bcf = nn.Linear(cfg.latent_dim, 1) if cfg.bcf_head else None
        self.q10_raw = nn.Parameter(torch.zeros(1)) if cfg.q10 else None

    # ---------------------------------------------------------------- forward
    def forward(self, x: Tensor, t: Tensor, log_kow: Tensor, log10_mw: Tensor, *,
                temperature_c: Tensor | None = None) -> dict:
        if x.ndim != 2:
            raise TedError(f"x must be (n, n_features); got {tuple(x.shape)}")
        tt = t.reshape(-1)
        if tt.shape[0] != x.shape[0]:
            raise TedError("t must have one entry per row of x")
        if self.cfg.mechanism:
            z = self.trunk(x)
            out = self.head(z, tt, log_kow, log10_mw)
        else:
            # A1 gets log duration as a trunk input, which is exactly what it is for
            z = self.trunk(torch.cat([x, torch.log10(tt.clamp(min=1e-6)).reshape(-1, 1)], dim=1))
            direct = self.direct(z)
            out = {"acute": direct[:, 0], "chronic": direct[:, 1],
                   "log10_acr": torch.clamp(direct[:, 0] - direct[:, 1], min=0.0),
                   "k": torch.full_like(direct[:, 0], float("nan")),
                   "k_chronic": torch.full_like(direct[:, 0], float("nan"))}
        out["z"] = z
        out["log_sigma"] = self.log_sigma(z).reshape(-1)
        if self.bcf is not None:
            out["log10_bcf"] = self.bcf(z).reshape(-1)
        if self.q10_raw is not None:
            if temperature_c is None:
                raise TedError("ablation A6 needs temperature_c")
            # a LEARNED Q10 with a free prior: no literature constant is assumed (amendment §7.2)
            q10 = torch.nn.functional.softplus(self.q10_raw) + 1.0
            shift = torch.log10(q10) * (temperature_c.reshape(-1) - 20.0) / 10.0
            out["acute"] = out["acute"] - shift
            out["chronic"] = out["chronic"] - shift
            out["q10"] = q10
        if not self.cfg.multitask:
            out.pop("chronic", None)        # A4: acute only
        return out

    def ordering_holds(self, out: Mapping[str, Tensor]) -> bool:
        """``EC10 <= EC50``. True by construction when the mechanism is on; checked under A1."""
        if "chronic" not in out:
            return True
        return bool(torch.all(out["chronic"] <= out["acute"] + 1e-5))


# ------------------------------------------------------------------------------------- losses

def heteroscedastic_nll(pred: Tensor, target: Tensor, log_sigma: Tensor) -> Tensor:
    """Gaussian NLL with a per-row sigma (target H6). ``log_sigma`` is clamped for stability."""
    ls = torch.clamp(log_sigma, min=-6.0, max=3.0)
    inv = torch.exp(-2.0 * ls)
    return torch.mean(0.5 * inv * (pred - target) ** 2 + ls)


def censored_nll(pred: Tensor, lower: Tensor, upper: Tensor, log_sigma: Tensor) -> Tensor:
    """Tobit term for the interval-censored acute rows (target H2).

    ``upper = inf`` is right censoring, ``lower = -inf`` left censoring; a row with
    ``lower == upper`` contributes the ordinary Gaussian term.
    """
    ls = torch.clamp(log_sigma, min=-6.0, max=3.0)
    sigma = torch.exp(ls)
    exact = torch.isfinite(lower) & torch.isfinite(upper) & (lower == upper)
    loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
    if exact.any():
        loss = loss + heteroscedastic_nll(pred[exact], lower[exact], ls[exact]) * exact.float().mean()
    cen = ~exact
    if cen.any():
        normal = torch.distributions.Normal(pred[cen], sigma[cen])
        hi = torch.where(torch.isfinite(upper[cen]), upper[cen],
                         torch.full_like(pred[cen], 1e6))
        lo = torch.where(torch.isfinite(lower[cen]), lower[cen],
                         torch.full_like(pred[cen], -1e6))
        mass = (normal.cdf(hi) - normal.cdf(lo)).clamp(min=1e-12)
        loss = loss + (-torch.log(mass)).mean() * cen.float().mean()
    return loss


def loss_terms(out: Mapping[str, Tensor], targets: Mapping[str, Tensor],
               weights: Mapping[str, float] | None = None) -> dict:
    """Assemble the pre-declared loss. Weights are swept over two settings (amendment §7.2)."""
    w = {"acute": 1.0, "chronic": 0.5, "censored": 0.5, "bcf": 0.25}
    w.update(weights or {})
    terms: dict[str, Tensor] = {}
    if "acute" in targets:
        terms["acute"] = w["acute"] * heteroscedastic_nll(
            out["acute"], targets["acute"], out["log_sigma"])
    if "chronic" in targets and "chronic" in out:
        terms["chronic"] = w["chronic"] * heteroscedastic_nll(
            out["chronic"], targets["chronic"], out["log_sigma"])
    if "censored_lower" in targets and "censored_upper" in targets:
        terms["censored"] = w["censored"] * censored_nll(
            out["acute"], targets["censored_lower"], targets["censored_upper"], out["log_sigma"])
    if "log10_bcf" in targets and "log10_bcf" in out:
        terms["bcf"] = w["bcf"] * torch.mean((out["log10_bcf"] - targets["log10_bcf"]) ** 2)
    if not terms:
        raise TedError("no loss term applies: targets carried none of the declared keys")
    terms["total"] = torch.stack(list(terms.values())).sum()
    return terms
