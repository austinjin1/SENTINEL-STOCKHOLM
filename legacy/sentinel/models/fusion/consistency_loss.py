"""Cross-modal self-supervised consistency loss.

When multiple modalities observe the same water quality state at the
same time and location, their projected embeddings should be similar.
This module provides a contrastive alignment signal that can be used as
an auxiliary training loss alongside the supervised task losses.

Positive pairs: embeddings from *different* modalities at the same
location-time.  Negative pairs: embeddings from *different* locations
(or sufficiently different times).

The loss is an InfoNCE-style contrastive loss operating on the
cosine-similarity matrix of cross-modal embedding pairs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalConsistencyLoss(nn.Module):
    """Contrastive alignment loss for cross-modal embeddings.

    Given a batch of projected embeddings from multiple modalities --
    each tagged with a location id -- the loss encourages embeddings
    from different modalities at the *same* location to be similar
    (positive pairs) and embeddings from *different* locations to be
    dissimilar (negative pairs).

    Args:
        temperature: Temperature parameter for the InfoNCE softmax.
            Lower = sharper distribution.  Default 0.07 (CLIP-style).
        learn_temperature: If ``True``, make the temperature a learnable
            parameter (in log-space).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learn_temperature: bool = True,
    ) -> None:
        super().__init__()
        if learn_temperature:
            self.log_temperature = nn.Parameter(
                torch.tensor(temperature).log()
            )
        else:
            self.register_buffer(
                "log_temperature",
                torch.tensor(temperature).log(),
            )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
        location_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the cross-modal consistency loss.

        Args:
            embeddings: Mapping from modality id to projected embedding
                tensor of shape ``[B, D]``.  Only modalities with actual
                data should be included (not no-data tokens).
            location_ids: Integer tensor of shape ``[B]`` identifying the
                location for each sample in the batch.  Samples with the
                same location id are treated as co-located observations.

        Returns:
            Scalar loss tensor.  Returns ``0.0`` (with grad) if fewer
            than 2 modalities have data or the batch has no positive
            pairs.
        """
        modality_keys = list(embeddings.keys())

        if len(modality_keys) < 2:
            # Need at least 2 modalities to form cross-modal pairs.
            device = next(iter(embeddings.values())).device
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Collect all embeddings into a single matrix and track which
        # modality each came from.
        all_embeddings: List[torch.Tensor] = []
        all_locations: List[torch.Tensor] = []
        all_modalities: List[int] = []

        for mod_idx, mid in enumerate(modality_keys):
            emb = embeddings[mid]
            B = emb.shape[0]
            all_embeddings.append(emb)
            all_locations.append(location_ids)
            all_modalities.extend([mod_idx] * B)

        # [N, D] where N = sum of batch sizes across modalities.
        all_emb = torch.cat(all_embeddings, dim=0)
        all_loc = torch.cat(all_locations, dim=0)  # [N]
        all_mod = torch.tensor(
            all_modalities, device=all_emb.device, dtype=torch.long
        )  # [N]
        N = all_emb.shape[0]

        if N < 2:
            return torch.tensor(0.0, device=all_emb.device, requires_grad=True)

        # L2-normalize for cosine similarity.
        all_emb_norm = F.normalize(all_emb, dim=-1)

        # Cosine similarity matrix [N, N].
        sim = torch.matmul(all_emb_norm, all_emb_norm.t()) / self.temperature

        # Positive mask: same location, different modality.
        loc_match = all_loc.unsqueeze(0) == all_loc.unsqueeze(1)  # [N, N]
        mod_diff = all_mod.unsqueeze(0) != all_mod.unsqueeze(1)    # [N, N]
        positive_mask = loc_match & mod_diff  # [N, N]

        # Check that there are positive pairs.
        if not positive_mask.any():
            return torch.tensor(0.0, device=all_emb.device, requires_grad=True)

        # Self-mask: exclude diagonal.
        self_mask = ~torch.eye(N, dtype=torch.bool, device=all_emb.device)

        # InfoNCE: for each anchor i with at least one positive j,
        # loss_i = -log( sum_pos exp(sim_ij) / sum_all exp(sim_ik) )
        # We use the multi-positive formulation.
        exp_sim = torch.exp(sim) * self_mask.float()
        denominator = exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [N, 1]

        # Log-softmax per row, then average over positive entries.
        log_prob = sim - torch.log(denominator)

        # Mask to rows that have at least one positive.
        has_positive = positive_mask.any(dim=1)  # [N]
        if not has_positive.any():
            return torch.tensor(0.0, device=all_emb.device, requires_grad=True)

        # Average log-prob of positive pairs.
        pos_log_prob = (log_prob * positive_mask.float()).sum(dim=1)  # [N]
        num_positives = positive_mask.float().sum(dim=1).clamp(min=1.0)  # [N]
        per_anchor_loss = -pos_log_prob / num_positives  # [N]

        loss = per_anchor_loss[has_positive].mean()
        return loss
