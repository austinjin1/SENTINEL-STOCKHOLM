"""Stream Network Graph Neural Network for watershed-level contamination propagation.

Adds spatial topology reasoning to SENTINEL.  Instead of treating each
monitoring site independently, the Stream GNN models the directed stream
network so that contamination signals propagate downstream with
physically-grounded travel-time delays.

Nodes = SENTINEL monitoring sites (identified by NHDPlusV2 COMID).
Edges = stream reaches connecting sites, directed downstream.
Edge attributes = travel time (hours), stream order, drainage area ratio,
                  reach distance (km).

Architecture
------------
1. **DirectionalMessagePassing** -- edge-conditioned message passing with
   separate upstream and downstream message functions, travel-time decay
   weighting, and GATv2-style attention.
2. **StreamNetworkGNN** -- stacks multiple DirectionalMessagePassing layers
   with residual connections and layer normalization.
3. **StreamEncoder** -- wraps the GNN with SENTINEL-compatible interface,
   outputting graph-enriched embeddings in the shared 256-d fusion space.
4. **ContaminationPropagator** -- given upstream anomaly detections, predicts
   downstream contamination probability and expected arrival time.
5. **build_stream_graph** -- utility to construct a PyG ``Data`` object from
   site/reach metadata (NHDPlusV2 topology).

The module gracefully falls back to pure-PyTorch sparse operations when
``torch_geometric`` is not installed.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detect torch_geometric availability
# ---------------------------------------------------------------------------
_HAS_PYG = False
try:
    import torch_geometric  # noqa: F401
    from torch_geometric.data import Data
    from torch_geometric.nn import MessagePassing as PyGMessagePassing
    from torch_geometric.utils import add_self_loops, softmax as pyg_softmax

    _HAS_PYG = True
    logger.info("torch_geometric detected -- using native PyG layers.")
except ImportError:
    logger.info(
        "torch_geometric not available -- using pure-PyTorch sparse fallback."
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SHARED_EMBEDDING_DIM: int = 256
NUM_EDGE_FEATURES: int = 4  # travel_time, stream_order, drainage_area_ratio, distance_km


# ============================================================================
# Pure-PyTorch sparse fallback helpers
# ============================================================================

def _sparse_message_pass(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> torch.Tensor:
    """Weighted neighbourhood aggregation using sparse matmul.

    Args:
        x: Node features ``[N, D]``.
        edge_index: COO edge indices ``[2, E]``.
        edge_weight: Per-edge scalar weights ``[E]``.

    Returns:
        Aggregated messages ``[N, D]``.
    """
    N = x.size(0)
    src, dst = edge_index  # src -> dst (downstream direction)
    adj = torch.sparse_coo_tensor(
        torch.stack([dst, src]),
        edge_weight,
        size=(N, N),
        device=x.device,
    ).coalesce()
    return torch.sparse.mm(adj, x)


def _scatter_softmax(
    logits: torch.Tensor,
    index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Softmax over variable-size groups defined by *index*.

    Args:
        logits: Unnormalized scores ``[E]``.
        index: Target node indices ``[E]`` -- softmax groups.
        num_nodes: Total number of nodes (for scatter sizing).

    Returns:
        Softmax-normalized scores ``[E]``.
    """
    # Numerically stable: subtract per-group max.
    max_vals = torch.full((num_nodes,), -1e9, device=logits.device, dtype=logits.dtype)
    max_vals.scatter_reduce_(0, index, logits, reduce="amax", include_self=False)
    logits = logits - max_vals[index]
    exp_logits = logits.exp()
    sum_exp = torch.zeros(num_nodes, device=logits.device, dtype=logits.dtype)
    sum_exp.scatter_add_(0, index, exp_logits)
    return exp_logits / (sum_exp[index] + 1e-12)


# ============================================================================
# Lightweight PyG-style Data container (fallback)
# ============================================================================

if not _HAS_PYG:

    @dataclass
    class Data:  # type: ignore[no-redef]
        """Minimal graph container mirroring ``torch_geometric.data.Data``.

        Only implements fields used by this module so that the rest of the
        codebase is agnostic to whether PyG is installed.
        """

        x: Optional[torch.Tensor] = None
        edge_index: Optional[torch.Tensor] = None
        edge_attr: Optional[torch.Tensor] = None
        num_nodes: Optional[int] = None
        # Extra metadata carried for convenience.
        site_ids: Optional[List[str]] = None
        comids: Optional[List[int]] = None

        def to(self, device: torch.device) -> "Data":
            """Move all tensor fields to *device*."""
            fields = {}
            for attr in ("x", "edge_index", "edge_attr"):
                val = getattr(self, attr)
                if val is not None:
                    fields[attr] = val.to(device)
                else:
                    fields[attr] = val
            fields["num_nodes"] = self.num_nodes
            fields["site_ids"] = self.site_ids
            fields["comids"] = self.comids
            return Data(**fields)


# ============================================================================
# DirectionalMessagePassing
# ============================================================================

class DirectionalMessagePassing(nn.Module):
    """Edge-conditioned, directional message-passing layer.

    Implements separate upstream and downstream message functions with
    travel-time exponential decay and GATv2-style attention:

    .. math::

        \\alpha_{ij} = \\text{softmax}_j\\bigl(
            \\mathbf{a}^\\top \\text{LeakyReLU}(
                \\mathbf{W}[\\mathbf{h}_i \\| \\mathbf{h}_j \\| \\mathbf{e}_{ij}]
            )\\bigr)

    Messages are multiplied by a travel-time decay:

    .. math::

        w_{ij} = \\exp(-t_{ij} / \\tau)

    where :math:`t_{ij}` is the travel time in hours and :math:`\\tau` is a
    learnable time constant.

    Args:
        in_dim: Input node feature dimension.
        out_dim: Output node feature dimension.
        edge_dim: Edge feature dimension.
        heads: Number of attention heads.
        dropout: Dropout applied to attention weights.
        negative_slope: LeakyReLU slope for attention logits.
    """

    def __init__(
        self,
        in_dim: int = SHARED_EMBEDDING_DIM,
        out_dim: int = SHARED_EMBEDDING_DIM,
        edge_dim: int = NUM_EDGE_FEATURES,
        heads: int = 4,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.heads = heads
        self.head_dim = out_dim // heads
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        self.negative_slope = negative_slope
        self.dropout = dropout

        # --- Downstream (src -> dst) message pathway ---
        self.W_down = nn.Linear(in_dim, out_dim, bias=False)
        self.W_down_nbr = nn.Linear(in_dim, out_dim, bias=False)
        self.W_down_edge = nn.Linear(edge_dim, out_dim, bias=False)
        self.attn_down = nn.Parameter(torch.empty(heads, self.head_dim))

        # --- Upstream (dst -> src) message pathway ---
        self.W_up = nn.Linear(in_dim, out_dim, bias=False)
        self.W_up_nbr = nn.Linear(in_dim, out_dim, bias=False)
        self.W_up_edge = nn.Linear(edge_dim, out_dim, bias=False)
        self.attn_up = nn.Parameter(torch.empty(heads, self.head_dim))

        # --- Travel-time decay ---
        # Learnable log-tau so tau = exp(log_tau) stays positive.
        self.log_tau = nn.Parameter(torch.tensor(math.log(6.0)))  # init ~6 h

        # --- Output ---
        self.out_proj = nn.Linear(2 * out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.W_down.weight)
        nn.init.xavier_uniform_(self.W_down_nbr.weight)
        nn.init.xavier_uniform_(self.W_down_edge.weight)
        nn.init.xavier_uniform_(self.W_up.weight)
        nn.init.xavier_uniform_(self.W_up_nbr.weight)
        nn.init.xavier_uniform_(self.W_up_edge.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.xavier_uniform_(self.attn_down.unsqueeze(0))
        nn.init.xavier_uniform_(self.attn_up.unsqueeze(0))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_travel_time_decay(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Compute exp(-travel_time / tau) from edge attributes.

        Travel time is assumed to be the first feature in *edge_attr*.

        Args:
            edge_attr: ``[E, edge_dim]`` -- first column is travel time (hours).

        Returns:
            Decay weights ``[E]`` in ``(0, 1]``.
        """
        tau = self.log_tau.exp().clamp(min=0.1)
        travel_time = edge_attr[:, 0]  # hours
        return torch.exp(-travel_time / tau)

    def _directional_attention(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        W_self: nn.Linear,
        W_nbr: nn.Linear,
        W_edge: nn.Linear,
        attn_vec: torch.Tensor,
        decay: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention-weighted, edge-conditioned, decay-modulated messages.

        Args:
            x: Node features ``[N, in_dim]``.
            edge_index: ``[2, E]`` -- row 0 = source, row 1 = target.
            edge_attr: ``[E, edge_dim]``.
            W_self, W_nbr, W_edge: Linear projections.
            attn_vec: Attention parameters ``[heads, head_dim]``.
            decay: Per-edge travel-time decay ``[E]``.

        Returns:
            Aggregated messages at each target node, ``[N, out_dim]``.
        """
        N = x.size(0)
        src, dst = edge_index
        H, D = self.heads, self.head_dim

        # Project source (neighbour) and target (self) nodes plus edge features.
        h_self = W_self(x)  # [N, out_dim]
        h_nbr = W_nbr(x)   # [N, out_dim]
        h_edge = W_edge(edge_attr)  # [E, out_dim]

        # GATv2 attention: apply activation *before* dot product.
        # For each edge (src -> dst): combine dst (self), src (nbr), edge.
        msg = h_self[dst] + h_nbr[src] + h_edge  # [E, out_dim]
        msg = F.leaky_relu(msg, negative_slope=self.negative_slope)

        # Reshape for multi-head attention.
        msg_heads = msg.view(-1, H, D)  # [E, H, D]
        attn_logits = (msg_heads * attn_vec.unsqueeze(0)).sum(dim=-1)  # [E, H]

        # Incorporate travel-time decay as additive bias in log-space.
        decay_bias = decay.log().unsqueeze(-1)  # [E, 1]
        attn_logits = attn_logits + decay_bias

        # Softmax per destination node, per head.
        if _HAS_PYG:
            attn_weights = pyg_softmax(attn_logits, dst, num_nodes=N)  # [E, H]
        else:
            # Flatten heads into separate softmax calls.
            attn_list = []
            for h in range(H):
                attn_list.append(_scatter_softmax(attn_logits[:, h], dst, N))
            attn_weights = torch.stack(attn_list, dim=-1)  # [E, H]

        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Weighted message values: use projected neighbour features.
        vals = h_nbr[src].view(-1, H, D)  # [E, H, D]
        weighted = vals * attn_weights.unsqueeze(-1)  # [E, H, D]

        # Scatter-add to destination nodes.
        out = torch.zeros(N, H, D, device=x.device, dtype=x.dtype)
        out.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand_as(weighted), weighted)
        return out.reshape(N, -1)  # [N, out_dim]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Apply directional message passing.

        Both upstream->downstream and downstream->upstream messages are
        computed and concatenated before projection.

        Args:
            x: Node embeddings ``[N, in_dim]``.
            edge_index: Directed edges ``[2, E]`` (pointing downstream).
            edge_attr: Edge features ``[E, edge_dim]``.

        Returns:
            Updated node embeddings ``[N, out_dim]``.
        """
        decay = self._compute_travel_time_decay(edge_attr)

        # Downstream messages: information flows from upstream to downstream.
        # edge_index already points src(upstream) -> dst(downstream).
        msg_down = self._directional_attention(
            x, edge_index, edge_attr,
            self.W_down, self.W_down_nbr, self.W_down_edge,
            self.attn_down, decay,
        )

        # Upstream messages: reverse the edge direction.
        edge_index_rev = edge_index.flip(0)
        msg_up = self._directional_attention(
            x, edge_index_rev, edge_attr,
            self.W_up, self.W_up_nbr, self.W_up_edge,
            self.attn_up, decay,
        )

        # Combine bidirectional messages.
        combined = torch.cat([msg_down, msg_up], dim=-1)  # [N, 2*out_dim]
        out = self.out_proj(combined)  # [N, out_dim]
        out = self.drop(out)

        # Residual connection + layer norm.
        if x.size(-1) == self.out_dim:
            out = self.norm(out + x)
        else:
            out = self.norm(out)

        return out


# ============================================================================
# StreamNetworkGNN
# ============================================================================

class StreamNetworkGNN(nn.Module):
    """Multi-layer directional GNN for stream network topology.

    Stacks :class:`DirectionalMessagePassing` layers with residual connections
    to propagate information along the stream network.  Operates on a single
    graph (one watershed) with variable numbers of nodes and edges.

    The module is *stateless* with respect to graph structure: the topology is
    passed as input tensors so that different watersheds can share the same
    model weights.

    Args:
        node_dim: Input (and output) node embedding dimension.
        edge_dim: Edge feature dimension.
        num_layers: Number of message-passing layers.
        heads: Attention heads per layer.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        node_dim: int = SHARED_EMBEDDING_DIM,
        edge_dim: int = NUM_EDGE_FEATURES,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            DirectionalMessagePassing(
                in_dim=node_dim,
                out_dim=node_dim,
                edge_dim=edge_dim,
                heads=heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm for stable output distribution.
        self.output_norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the full GNN stack.

        Supports both single-graph and batched-graph inputs.  When
        ``x`` has shape ``[B, N, D]`` (batched with uniform graph size),
        the method internally iterates over the batch dimension.  When
        ``x`` has shape ``[N_total, D]`` (PyG-style batched graph), set
        *batch* to the PyG batch vector.

        Args:
            x: Node embeddings.
                - ``[N, D]`` for a single graph or PyG-style batch.
                - ``[B, N, D]`` for a padded batch of identical graphs.
            edge_index: ``[2, E]`` directed edges (downstream).
            edge_attr: ``[E, edge_dim]`` edge features.
            batch: Optional PyG batch vector ``[N_total]`` mapping each
                node to its graph index.  Only needed when ``x`` is 2-D
                and represents multiple graphs.

        Returns:
            Updated node embeddings with the same shape as *x*.
        """
        if x.dim() == 3:
            # Batched mode: iterate over batch dimension.
            B, N, D = x.shape
            outputs = []
            for b in range(B):
                h = x[b]  # [N, D]
                for layer in self.layers:
                    h = layer(h, edge_index, edge_attr)
                outputs.append(h)
            return self.output_norm(torch.stack(outputs, dim=0))  # [B, N, D]
        else:
            # Single graph / PyG-style batch.
            h = x
            for layer in self.layers:
                h = layer(h, edge_index, edge_attr)
            return self.output_norm(h)


# ============================================================================
# StreamEncoder -- SENTINEL-compatible wrapper
# ============================================================================

class StreamEncoder(nn.Module):
    """SENTINEL-compatible stream network encoder.

    Wraps :class:`StreamNetworkGNN` to accept raw per-site embeddings from
    any upstream encoder (sensor, satellite, etc.) and return
    graph-enriched embeddings in the shared 256-d fusion space.

    Usage::

        encoder = StreamEncoder()

        # site_embeddings: from any SENTINEL encoder, [B, N_sites, 256]
        # graph: Data object from build_stream_graph()
        enriched = encoder(site_embeddings, graph)
        # enriched['embedding'] shape: [B, N_sites, 256]

    Args:
        input_dim: Dimension of incoming per-site embeddings.
        shared_embed_dim: SENTINEL shared embedding dimension.
        num_layers: GNN depth.
        heads: Attention heads per GNN layer.
        edge_dim: Edge feature dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = SHARED_EMBEDDING_DIM,
        shared_embed_dim: int = SHARED_EMBEDDING_DIM,
        num_layers: int = 3,
        heads: int = 4,
        edge_dim: int = NUM_EDGE_FEATURES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.shared_embed_dim = shared_embed_dim

        # Optional input projection if input_dim != shared_embed_dim.
        if input_dim != shared_embed_dim:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, shared_embed_dim),
                nn.GELU(),
                nn.LayerNorm(shared_embed_dim),
            )
        else:
            self.input_proj = nn.Identity()

        self.gnn = StreamNetworkGNN(
            node_dim=shared_embed_dim,
            edge_dim=edge_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )

        # Output projection: blend GNN output with original embedding.
        self.gate = nn.Sequential(
            nn.Linear(2 * shared_embed_dim, shared_embed_dim),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(shared_embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier init for gate projection."""
        for m in self.gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        site_embeddings: torch.Tensor,
        graph: Data,
    ) -> Dict[str, torch.Tensor]:
        """Enrich per-site embeddings with stream network context.

        Args:
            site_embeddings: Raw site embeddings from an upstream encoder.
                Shape ``[B, N_sites, D]`` or ``[N_sites, D]``.
            graph: Stream network graph with ``edge_index`` and ``edge_attr``.

        Returns:
            Dict with:
                ``'embedding'``: Graph-enriched embeddings, same shape as
                    input but projected to ``shared_embed_dim``.
                ``'gnn_output'``: Raw GNN output before gating.
        """
        squeezed = False
        if site_embeddings.dim() == 2:
            site_embeddings = site_embeddings.unsqueeze(0)
            squeezed = True

        h = self.input_proj(site_embeddings)  # [B, N, shared_embed_dim]

        edge_index = graph.edge_index.to(h.device)
        edge_attr = graph.edge_attr.to(h.device)

        gnn_out = self.gnn(h, edge_index, edge_attr)  # [B, N, shared_embed_dim]

        # Gated residual: allow the model to learn how much spatial context
        # to blend with the original per-site embedding.
        gate_input = torch.cat([h, gnn_out], dim=-1)  # [B, N, 2*D]
        g = self.gate(gate_input)  # [B, N, D], values in (0, 1)
        enriched = g * gnn_out + (1 - g) * h  # [B, N, D]
        enriched = self.output_norm(enriched)

        if squeezed:
            enriched = enriched.squeeze(0)
            gnn_out = gnn_out.squeeze(0)

        return {
            "embedding": enriched,
            "fusion_embedding": enriched,
            "gnn_output": gnn_out,
        }


# ============================================================================
# ContaminationPropagator
# ============================================================================

@dataclass
class PropagationPrediction:
    """Per-site contamination propagation predictions.

    Attributes:
        contamination_prob: Probability of contamination at each downstream
            site, shape ``[B, N]`` or ``[N]``.
        expected_arrival_hours: Expected arrival time of contamination at
            each site (hours from detection), shape ``[B, N]`` or ``[N]``.
        source_attribution: Soft assignment of each site's predicted
            contamination to upstream sources, shape ``[B, N, N]`` or
            ``[N, N]``.
    """

    contamination_prob: torch.Tensor
    expected_arrival_hours: torch.Tensor
    source_attribution: torch.Tensor


class ContaminationPropagator(nn.Module):
    """Predict downstream contamination propagation from upstream anomalies.

    Given per-site anomaly scores and the stream network graph, predicts:
    1. Contamination probability at each downstream site.
    2. Expected arrival time of the contamination front.
    3. Source attribution (which upstream site is the likely origin).

    The propagation model combines learned GNN message passing with
    physics-informed travel-time priors.

    Args:
        embed_dim: Node embedding dimension.
        edge_dim: Edge feature dimension.
        hidden_dim: Hidden layer dimension for prediction heads.
        num_gnn_layers: GNN layers for propagation reasoning.
    """

    def __init__(
        self,
        embed_dim: int = SHARED_EMBEDDING_DIM,
        edge_dim: int = NUM_EDGE_FEATURES,
        hidden_dim: int = 128,
        num_gnn_layers: int = 2,
    ) -> None:
        super().__init__()

        # Encode anomaly scores into the embedding space.
        self.anomaly_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        # GNN for propagation reasoning.
        self.gnn = StreamNetworkGNN(
            node_dim=embed_dim,
            edge_dim=edge_dim,
            num_layers=num_gnn_layers,
            heads=4,
            dropout=0.1,
        )

        # Combine site embedding with propagation context.
        self.combiner = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Contamination probability head.
        self.prob_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Expected arrival time head (outputs positive hours).
        self.arrival_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # ensures positive output
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.anomaly_encoder, self.combiner,
                       self.prob_head, self.arrival_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        site_embeddings: torch.Tensor,
        anomaly_scores: torch.Tensor,
        graph: Data,
    ) -> PropagationPrediction:
        """Predict contamination propagation across the stream network.

        Args:
            site_embeddings: Per-site embeddings ``[B, N, D]`` or ``[N, D]``.
            anomaly_scores: Per-site anomaly scores ``[B, N]`` or ``[N]``.
                Values in ``[0, 1]`` where 1 = definite anomaly.
            graph: Stream network ``Data`` object.

        Returns:
            :class:`PropagationPrediction` with per-site predictions.
        """
        squeezed = False
        if site_embeddings.dim() == 2:
            site_embeddings = site_embeddings.unsqueeze(0)
            anomaly_scores = anomaly_scores.unsqueeze(0)
            squeezed = True

        B, N, D = site_embeddings.shape
        device = site_embeddings.device

        # Encode anomaly scores.
        anomaly_emb = self.anomaly_encoder(anomaly_scores.unsqueeze(-1))  # [B, N, D]

        # Add anomaly information to site embeddings.
        combined_input = site_embeddings + anomaly_emb  # [B, N, D]

        # Propagate through GNN.
        edge_index = graph.edge_index.to(device)
        edge_attr = graph.edge_attr.to(device)
        propagated = self.gnn(combined_input, edge_index, edge_attr)  # [B, N, D]

        # Combine original and propagated representations.
        combined = torch.cat([site_embeddings, propagated], dim=-1)  # [B, N, 2D]
        hidden = self.combiner(combined)  # [B, N, hidden_dim]

        # Predict contamination probability and arrival time.
        contam_prob = torch.sigmoid(self.prob_head(hidden).squeeze(-1))  # [B, N]
        arrival_hours = self.arrival_head(hidden).squeeze(-1)  # [B, N]

        # Source attribution via attention between all node pairs.
        # Use dot-product similarity between propagated embeddings.
        prop_norm = F.normalize(propagated, dim=-1)  # [B, N, D]
        source_attn = torch.bmm(prop_norm, prop_norm.transpose(1, 2))  # [B, N, N]
        # Mask to only consider upstream sources (nonzero anomaly scores).
        anomaly_mask = (anomaly_scores > 0.1).float().unsqueeze(1)  # [B, 1, N]
        source_attn = source_attn * anomaly_mask
        source_attn = F.softmax(source_attn, dim=-1)  # [B, N, N]

        if squeezed:
            contam_prob = contam_prob.squeeze(0)
            arrival_hours = arrival_hours.squeeze(0)
            source_attn = source_attn.squeeze(0)

        return PropagationPrediction(
            contamination_prob=contam_prob,
            expected_arrival_hours=arrival_hours,
            source_attribution=source_attn,
        )


# ============================================================================
# Graph construction utilities
# ============================================================================

def build_stream_graph(
    sites: List[Dict[str, Union[str, float, int]]],
    reaches: List[Dict[str, Union[int, float]]],
) -> Data:
    """Build a PyG-compatible directed graph from NHDPlusV2 site/reach data.

    Constructs a directed acyclic graph where edges point downstream
    (from upstream monitoring site to downstream monitoring site).

    Args:
        sites: List of site descriptors, each containing:
            - ``site_id`` (str): Unique site identifier.
            - ``lat`` (float): Latitude.
            - ``lon`` (float): Longitude.
            - ``comid`` (int): NHDPlusV2 COMID for the reach containing
              this site.
        reaches: List of reach descriptors, each containing:
            - ``from_comid`` (int): Upstream COMID.
            - ``to_comid`` (int): Downstream COMID.
            - ``travel_time_hours`` (float): Estimated travel time in hours.
            - ``stream_order`` (int): Strahler stream order (1-8).
            - ``drainage_area_km2`` (float): Contributing drainage area.

    Returns:
        :class:`Data` object with:
            - ``edge_index``: ``[2, E]`` long tensor, directed downstream.
            - ``edge_attr``: ``[E, 4]`` float tensor with columns
              ``[travel_time, stream_order, drainage_area_ratio, distance_km]``.
            - ``num_nodes``: Number of monitoring sites.
            - ``site_ids``: List of site ID strings.
            - ``comids``: List of COMID integers.

    Raises:
        ValueError: If a reach references a COMID not present in *sites*.

    Example::

        sites = [
            {"site_id": "USGS-01", "lat": 39.0, "lon": -77.0, "comid": 1001},
            {"site_id": "USGS-02", "lat": 38.9, "lon": -76.9, "comid": 1002},
            {"site_id": "USGS-03", "lat": 38.8, "lon": -76.8, "comid": 1003},
        ]
        reaches = [
            {"from_comid": 1001, "to_comid": 1002,
             "travel_time_hours": 2.5, "stream_order": 3,
             "drainage_area_km2": 150.0},
            {"from_comid": 1002, "to_comid": 1003,
             "travel_time_hours": 1.8, "stream_order": 4,
             "drainage_area_km2": 320.0},
        ]
        graph = build_stream_graph(sites, reaches)
    """
    # Map COMID -> node index.
    comid_to_idx: Dict[int, int] = {}
    site_ids: List[str] = []
    comids: List[int] = []
    for i, site in enumerate(sites):
        comid = int(site["comid"])
        comid_to_idx[comid] = i
        site_ids.append(str(site["site_id"]))
        comids.append(comid)

    N = len(sites)

    # Compute max drainage area for ratio normalization.
    max_drainage = max(
        (float(r.get("drainage_area_km2", 0)) for r in reaches),
        default=1.0,
    )
    if max_drainage <= 0:
        max_drainage = 1.0

    # Build edge lists.
    src_list: List[int] = []
    dst_list: List[int] = []
    edge_features: List[List[float]] = []

    for reach in reaches:
        from_comid = int(reach["from_comid"])
        to_comid = int(reach["to_comid"])

        if from_comid not in comid_to_idx:
            warnings.warn(
                f"Reach from_comid={from_comid} not found in sites; skipping."
            )
            continue
        if to_comid not in comid_to_idx:
            warnings.warn(
                f"Reach to_comid={to_comid} not found in sites; skipping."
            )
            continue

        src_idx = comid_to_idx[from_comid]
        dst_idx = comid_to_idx[to_comid]

        travel_time = float(reach.get("travel_time_hours", 1.0))
        stream_order = float(reach.get("stream_order", 1))
        drainage_area = float(reach.get("drainage_area_km2", 0.0))
        drainage_ratio = drainage_area / max_drainage

        # Use real distance if available, otherwise approximate from travel time.
        distance_km = float(reach.get("distance_km", travel_time * 3.6))

        src_list.append(src_idx)
        dst_list.append(dst_idx)
        edge_features.append([
            travel_time,
            stream_order,
            drainage_ratio,
            distance_km,
        ])

    if len(src_list) == 0:
        # No edges: return isolated nodes.
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, NUM_EDGE_FEATURES, dtype=torch.float32)
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)

    return Data(
        x=None,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=N,
        site_ids=site_ids,
        comids=comids,
    )


def build_example_watershed() -> Tuple[Data, List[Dict]]:
    """Build a small example watershed for testing and demonstration.

    Creates a simple 5-site stream network::

        Site A (headwater, order 1)
           \\
            --> Site C (confluence, order 3) --> Site D (order 4) --> Site E (outlet, order 5)
           /
        Site B (headwater, order 2)

    Returns:
        Tuple of (graph Data, list of site dicts).
    """
    sites = [
        {"site_id": "HEAD-A", "lat": 39.10, "lon": -77.20, "comid": 101},
        {"site_id": "HEAD-B", "lat": 39.05, "lon": -77.15, "comid": 102},
        {"site_id": "CONF-C", "lat": 39.00, "lon": -77.10, "comid": 103},
        {"site_id": "MID-D",  "lat": 38.95, "lon": -77.05, "comid": 104},
        {"site_id": "OUT-E",  "lat": 38.90, "lon": -77.00, "comid": 105},
    ]
    reaches = [
        {"from_comid": 101, "to_comid": 103,
         "travel_time_hours": 3.0, "stream_order": 1,
         "drainage_area_km2": 50.0},
        {"from_comid": 102, "to_comid": 103,
         "travel_time_hours": 2.0, "stream_order": 2,
         "drainage_area_km2": 120.0},
        {"from_comid": 103, "to_comid": 104,
         "travel_time_hours": 1.5, "stream_order": 3,
         "drainage_area_km2": 200.0},
        {"from_comid": 104, "to_comid": 105,
         "travel_time_hours": 1.0, "stream_order": 4,
         "drainage_area_km2": 350.0},
    ]
    graph = build_stream_graph(sites, reaches)
    return graph, sites
