"""ChemTox-MMoE: L3's published multi-endpoint mixture-of-experts, as a COMPARATOR.

New module, no project row read. Wu et al. 2026 (*J. Chem. Inf. Model.* 66(20):13302-13312, DOI
10.1021/acs.jcim.6c02487) is the strongest published multi-endpoint aquatic-toxicity architecture,
so amendment §4.3 builds it as a **comparator and ensemble member, never a novelty claim**: a
shared fused representation, per-task gates, an EC50 >= EC10 ordering constraint and
heteroscedastic uncertainty.

Under ruling **F2** this arm is one of the two that condition (a) may be read against — whichever of
``mmoe`` and ``gpboost_published`` has the lower pool-validation score — so a weak re-implementation
would quietly weaken the comparison. Its §4.6 check is therefore pre-registered: it must beat
``lightgbm_noemb`` on pool validation, and a miss is reported as a failed reproduction and
disclosed beside the result, but never removes the arm from (a).

The ordering here is imposed in **output space**, as L3 does it, which is the difference from TED:
TED derives the ordering from one latent plus a non-negative ACR. :func:`ordered_outputs` makes
L3's version explicit so the contrast is visible in code, not only in prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

__all__ = ["MMoEError", "TASKS", "MMoEConfig", "MMoE", "ordered_outputs"]


class MMoEError(ValueError):
    """A pre-registered rule about the MMoE comparator was violated."""


#: The endpoint families this project has. L3 uses six fish/crustacean/algae EC50+EC10 tasks; ours
#: are the endpoint families of the stamped amendment's targets H1 and H3.
TASKS = ("acute", "chronic")


@dataclass
class MMoEConfig:
    n_features: int
    n_experts: int = 4
    expert_hidden: int = 128
    expert_out: int = 64
    tower_hidden: int = 64
    dropout: float = 0.0
    tasks: tuple[str, ...] = TASKS
    enforce_ordering: bool = True

    def __post_init__(self) -> None:
        if self.n_features <= 0 or self.n_experts <= 0:
            raise MMoEError("n_features and n_experts must be positive")
        if not self.tasks:
            raise MMoEError("at least one task is needed")
        if self.enforce_ordering and tuple(self.tasks[:2]) != ("acute", "chronic"):
            raise MMoEError(
                "the ordering constraint assumes tasks[0] is the acute and tasks[1] the chronic "
                "endpoint; reorder or set enforce_ordering=False")


def ordered_outputs(acute: Tensor, chronic_raw: Tensor) -> tuple[Tensor, Tensor]:
    """L3's constraint, in OUTPUT space: ``chronic = acute - softplus(raw)``, so chronic <= acute.

    Contrast with TED, where the ordering falls out of one shared latent and a non-negative ACR.
    """
    gap = nn.functional.softplus(chronic_raw)
    return acute, acute - gap


class MMoE(nn.Module):
    """Shared experts, per-task gates, per-task towers, heteroscedastic sigma per task."""

    def __init__(self, cfg: MMoEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(cfg.n_features, cfg.expert_hidden), nn.ReLU(),
                          nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
                          nn.Linear(cfg.expert_hidden, cfg.expert_out), nn.ReLU())
            for _ in range(cfg.n_experts)])
        self.gates = nn.ModuleDict({
            t: nn.Linear(cfg.n_features, cfg.n_experts) for t in cfg.tasks})
        self.towers = nn.ModuleDict({
            t: nn.Sequential(nn.Linear(cfg.expert_out, cfg.tower_hidden), nn.ReLU(),
                             nn.Linear(cfg.tower_hidden, 2))      # value, log_sigma
            for t in cfg.tasks})

    def forward(self, x: Tensor) -> dict:
        if x.ndim != 2 or x.shape[1] != self.cfg.n_features:
            raise MMoEError(f"x must be (n, {self.cfg.n_features}); got {tuple(x.shape)}")
        stacked = torch.stack([e(x) for e in self.experts], dim=1)      # (n, n_experts, expert_out)
        out: dict = {"gate_weights": {}}
        raw: dict[str, Tensor] = {}
        for task in self.cfg.tasks:
            w = torch.softmax(self.gates[task](x), dim=1)                # (n, n_experts)
            fused = torch.einsum("ne,neo->no", w, stacked)
            head = self.towers[task](fused)
            raw[task] = head[:, 0]
            out[f"log_sigma_{task}"] = head[:, 1]
            out["gate_weights"][task] = w
        if self.cfg.enforce_ordering:
            acute, chronic = ordered_outputs(raw["acute"], raw["chronic"])
            out["acute"], out["chronic"] = acute, chronic
        else:
            out.update(raw)
        return out

    def ordering_holds(self, out: dict) -> bool:
        if "chronic" not in out or "acute" not in out:
            return True
        return bool(torch.all(out["chronic"] <= out["acute"] + 1e-5))
