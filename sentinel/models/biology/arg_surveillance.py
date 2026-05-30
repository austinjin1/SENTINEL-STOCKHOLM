"""Antibiotic resistance gene (ARG) surveillance from 16S community composition.

Phase 3.4 of SENTINEL 2.0.  Extends MicroBiomeNet to predict ARG
abundance and environmental resistance burden from microbial community
composition (16S rRNA OTU tables or MicroBiomeNet embeddings).

Target ARGs (WHO high-priority resistance genes)
-------------------------------------------------
1. **mcr-1**   -- colistin resistance (last-resort antibiotic)
2. **blaNDM**  -- carbapenem resistance (critical priority)
3. **vanA**    -- vancomycin resistance (critical priority)
4. **mecA**    -- methicillin resistance (MRSA marker)
5. **tetM**    -- tetracycline resistance (widespread)
6. **sul1**    -- sulfonamide resistance (environmental indicator)
7. **qnrS**   -- quinolone resistance (emerging)
8. **ermB**    -- macrolide resistance (common)

Architecture overview
---------------------
::

    16S OTU table (B, 5000) --CLR--> OTU encoder
                                       |
                            OR         v
    MicroBiomeNet embedding (B, 256) --+--> shared backbone (256->512->256)
                                              |
                      +--------+--------------+--------+--------+
                      v        v              v        v        v
                   mcr-1    blaNDM   ...   qnrS     ermB   burden
                   head      head          head      head    head
                      |        |              |        |        |
                      v        v              v        v        v
              log-abundance predictions       ARGBurdenIndex (0-100)

    TemporalARGTracker: monitors ARG prediction sequences for emerging
    resistance and novel ARG combinations.

Uncertainty estimation uses MC-dropout (20 stochastic passes) following
Gal & Ghahramani (2016).

Loss: MSE on log-abundance + pairwise ranking loss (preserves relative
ARG ordering across samples).
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Constants
# ============================================================================

NUM_OTUS: int = 5000
EMBEDDING_DIM: int = 256
BACKBONE_HIDDEN: int = 512
BACKBONE_OUT: int = 256
ARG_HEAD_HIDDEN: int = 64
MC_DROPOUT_SAMPLES: int = 20
DROPOUT_P: float = 0.2

# ---------------------------------------------------------------------------
# Target ARGs
# ---------------------------------------------------------------------------

ARG_NAMES: Tuple[str, ...] = (
    "mcr-1",    # colistin resistance
    "blaNDM",   # carbapenem resistance
    "vanA",     # vancomycin resistance
    "mecA",     # methicillin resistance
    "tetM",     # tetracycline resistance
    "sul1",     # sulfonamide resistance
    "qnrS",     # quinolone resistance
    "ermB",     # macrolide resistance
)
NUM_ARGS: int = len(ARG_NAMES)

ARG_INDEX_TO_NAME: Dict[int, str] = {i: name for i, name in enumerate(ARG_NAMES)}
ARG_NAME_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(ARG_NAMES)}


# ---------------------------------------------------------------------------
# WHO priority classification for burden weighting
# ---------------------------------------------------------------------------

class WHOPriority(enum.IntEnum):
    """WHO antimicrobial resistance priority levels."""

    CRITICAL = 3
    HIGH = 2
    MEDIUM = 1


@dataclass(frozen=True)
class ARGMetadata:
    """Metadata for a target antibiotic resistance gene."""

    index: int
    gene_name: str
    antibiotic_class: str
    resistance_target: str
    who_priority: WHOPriority
    burden_weight: float
    clinical_note: str


ARG_CATALOG: Tuple[ARGMetadata, ...] = (
    ARGMetadata(
        index=0,
        gene_name="mcr-1",
        antibiotic_class="Polymyxins",
        resistance_target="Colistin",
        who_priority=WHOPriority.CRITICAL,
        burden_weight=1.0,
        clinical_note="Last-resort antibiotic; plasmid-mediated, high horizontal transfer risk",
    ),
    ARGMetadata(
        index=1,
        gene_name="blaNDM",
        antibiotic_class="Carbapenems",
        resistance_target="Carbapenems (meropenem, imipenem)",
        who_priority=WHOPriority.CRITICAL,
        burden_weight=1.0,
        clinical_note="New Delhi metallo-beta-lactamase; confers pan-resistance to beta-lactams",
    ),
    ARGMetadata(
        index=2,
        gene_name="vanA",
        antibiotic_class="Glycopeptides",
        resistance_target="Vancomycin",
        who_priority=WHOPriority.CRITICAL,
        burden_weight=1.0,
        clinical_note="VRE marker; transferable operon, risk of VRSA emergence",
    ),
    ARGMetadata(
        index=3,
        gene_name="mecA",
        antibiotic_class="Beta-lactams",
        resistance_target="Methicillin (oxacillin)",
        who_priority=WHOPriority.HIGH,
        burden_weight=0.8,
        clinical_note="MRSA marker; SCCmec-encoded, widespread in clinical and community settings",
    ),
    ARGMetadata(
        index=4,
        gene_name="tetM",
        antibiotic_class="Tetracyclines",
        resistance_target="Tetracycline",
        who_priority=WHOPriority.MEDIUM,
        burden_weight=0.5,
        clinical_note="Ribosomal protection protein; ubiquitous in environmental and clinical isolates",
    ),
    ARGMetadata(
        index=5,
        gene_name="sul1",
        antibiotic_class="Sulfonamides",
        resistance_target="Sulfonamides",
        who_priority=WHOPriority.MEDIUM,
        burden_weight=0.4,
        clinical_note="Class 1 integron marker; sentinel for anthropogenic resistance pollution",
    ),
    ARGMetadata(
        index=6,
        gene_name="qnrS",
        antibiotic_class="Quinolones",
        resistance_target="Fluoroquinolones",
        who_priority=WHOPriority.HIGH,
        burden_weight=0.7,
        clinical_note="Plasmid-mediated quinolone resistance; emerging in aquatic environments",
    ),
    ARGMetadata(
        index=7,
        gene_name="ermB",
        antibiotic_class="Macrolides",
        resistance_target="Erythromycin (macrolides)",
        who_priority=WHOPriority.MEDIUM,
        burden_weight=0.5,
        clinical_note="23S rRNA methyltransferase; common in Firmicutes, cross-resistance to MLSB",
    ),
)

# Default burden weights tensor (registered as buffer in ARGBurdenIndex)
_DEFAULT_BURDEN_WEIGHTS: List[float] = [arg.burden_weight for arg in ARG_CATALOG]

# ---------------------------------------------------------------------------
# ARG burden risk thresholds
# ---------------------------------------------------------------------------

BURDEN_THRESHOLDS: Dict[str, float] = {
    "low": 20.0,
    "moderate": 40.0,
    "high": 60.0,
    "critical": 80.0,
}


class BurdenAlertLevel(enum.IntEnum):
    """Environmental ARG burden alert levels."""

    LOW = 0          # burden < 20
    MODERATE = 1     # 20 <= burden < 40
    HIGH = 2         # 40 <= burden < 60
    CRITICAL = 3     # 60 <= burden < 80
    EMERGENCY = 4    # burden >= 80


# ============================================================================
# Output dataclasses
# ============================================================================

@dataclass
class ARGPredictionOutput:
    """Output of the ARGPredictor model.

    All abundance tensors are in **log-abundance** space (natural log of
    copies-per-16S-copy, with pseudocount offset).

    Attributes
    ----------
    log_abundance : Tensor[B, 8]
        Point predictions of log-abundance for each ARG.
    burden_score : Tensor[B]
        Aggregate environmental ARG burden index (0--100).
    burden_alert : Tensor[B]
        Integer alert level per site (see :class:`BurdenAlertLevel`).
    mc_mean : Tensor[B, 8] or None
        MC-dropout posterior mean (populated when ``use_mc_dropout=True``).
    mc_std : Tensor[B, 8] or None
        MC-dropout posterior std (populated when ``use_mc_dropout=True``).
    mc_burden_mean : Tensor[B] or None
        MC-dropout burden score mean.
    mc_burden_std : Tensor[B] or None
        MC-dropout burden score std.
    attention_weights : Tensor[B, 5000] or None
        OTU attention weights from :class:`CommunityToARGMapper`
        (populated only when raw OTU input is used).
    """

    log_abundance: torch.Tensor
    burden_score: torch.Tensor
    burden_alert: torch.Tensor

    mc_mean: Optional[torch.Tensor] = None
    mc_std: Optional[torch.Tensor] = None
    mc_burden_mean: Optional[torch.Tensor] = None
    mc_burden_std: Optional[torch.Tensor] = None

    attention_weights: Optional[torch.Tensor] = None


@dataclass
class TemporalARGAlert:
    """Alert from the TemporalARGTracker.

    Attributes
    ----------
    alert_type : str
        One of ``"emerging"``, ``"novel_combination"``, ``"sustained_high"``.
    arg_names : List[str]
        ARG(s) involved in the alert.
    trend_slope : float
        Estimated rate of change (log-abundance per time step).
    confidence : float
        Confidence in the alert (0--1).
    message : str
        Human-readable alert description.
    """

    alert_type: str
    arg_names: List[str]
    trend_slope: float
    confidence: float
    message: str


# ============================================================================
# Sub-module 1: CommunityToARGMapper
# ============================================================================

class CommunityToARGMapper(nn.Module):
    """Map 16S community composition to ARG-relevant features.

    Uses attention over OTU features weighted by known associations
    between microbial taxa and ARG carriage.  The attention mechanism
    learns to focus on OTUs belonging to taxa with high ARG-carrying
    potential (e.g. *Enterobacteriaceae*, *Acinetobacter*,
    *Enterococcus*), enabling transfer learning from metagenomics
    databases (CARD, ResFinder).

    Architecture::

        raw_OTU (B, 5000)
            --> CLR transform
            --> Linear projection (5000 -> 256)
            --> GELU + Dropout
            --> Attention gating (OTU-level weights via known ARG carriers)
            --> Weighted aggregation -> (B, 256)

    Parameters
    ----------
    num_otus : int
        Number of OTU features (default 5000).
    embed_dim : int
        Output embedding dimension (default 256).
    dropout : float
        Dropout probability (default 0.2).
    """

    def __init__(
        self,
        num_otus: int = NUM_OTUS,
        embed_dim: int = EMBEDDING_DIM,
        dropout: float = DROPOUT_P,
    ) -> None:
        super().__init__()
        self.num_otus = num_otus
        self.embed_dim = embed_dim

        # OTU projection: maps CLR-transformed OTU vector to embedding space
        self.otu_projection = nn.Sequential(
            nn.Linear(num_otus, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Attention gate: learns OTU-level importance scores for ARG prediction
        # This acts as a soft prior that can incorporate known ARG-taxa associations
        self.attention_gate = nn.Sequential(
            nn.Linear(num_otus, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_otus),
        )

        # Learnable prior bias for known ARG-carrying taxa
        # Initialized to zero; can be loaded from metagenomics databases
        self.taxa_prior = nn.Parameter(torch.zeros(num_otus))

        # Final projection to match embedding dim
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming initialisation for Linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def load_taxa_prior(self, prior_weights: torch.Tensor) -> None:
        """Load pre-computed ARG-taxa association weights.

        These weights come from metagenomics databases (CARD, ResFinder)
        and encode the known probability of each OTU belonging to a
        taxon that carries ARGs.

        Parameters
        ----------
        prior_weights : Tensor[num_otus]
            Non-negative weights indicating ARG-carrying likelihood
            for each OTU.  Will be sigmoid-compressed internally.
        """
        if prior_weights.shape != (self.num_otus,):
            raise ValueError(
                f"Prior weights shape {prior_weights.shape} does not match "
                f"expected ({self.num_otus},)"
            )
        with torch.no_grad():
            self.taxa_prior.copy_(prior_weights)

    def forward(
        self,
        otu_abundances: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map raw OTU abundances to an ARG-relevant embedding.

        Parameters
        ----------
        otu_abundances : Tensor[B, num_otus]
            CLR-transformed OTU abundance table.

        Returns
        -------
        embedding : Tensor[B, embed_dim]
            ARG-relevant community embedding.
        attention_weights : Tensor[B, num_otus]
            Per-OTU attention weights (for interpretability).
        """
        B = otu_abundances.shape[0]

        # Compute attention weights over OTUs
        attn_logits = self.attention_gate(otu_abundances)  # (B, num_otus)

        # Add learnable taxa prior (known ARG carriers get upweighted)
        attn_logits = attn_logits + self.taxa_prior.unsqueeze(0)

        # Softmax to get normalized attention weights
        attention_weights = F.softmax(attn_logits, dim=-1)  # (B, num_otus)

        # Attention-weighted OTU values
        weighted_otus = otu_abundances * attention_weights  # (B, num_otus)

        # Project to embedding space
        embedding = self.otu_projection(weighted_otus)  # (B, embed_dim)

        # Final projection with residual-like LayerNorm
        embedding = self.output_projection(embedding)  # (B, embed_dim)

        return embedding, attention_weights


# ============================================================================
# Sub-module 2: ARGBurdenIndex
# ============================================================================

class ARGBurdenIndex(nn.Module):
    """Compute aggregate environmental ARG burden score.

    A weighted sum of individual ARG log-abundances, mapped to a 0--100
    risk scale.  Weights reflect WHO priority classification:

    - Critical (mcr-1, blaNDM, vanA): weight 1.0
    - High (mecA, qnrS):             weight 0.7--0.8
    - Medium (tetM, sul1, ermB):      weight 0.4--0.5

    The raw weighted sum is passed through a learned mapping
    (2-layer MLP) to produce the final 0--100 score, allowing the
    model to learn non-linear risk interactions (e.g. co-occurrence
    of multiple critical ARGs amplifies risk).

    Parameters
    ----------
    num_args : int
        Number of target ARGs (default 8).
    hidden_dim : int
        Hidden dimension for the risk mapping MLP (default 32).
    """

    def __init__(
        self,
        num_args: int = NUM_ARGS,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.num_args = num_args

        # WHO priority-based weights (frozen by default, trainable if needed)
        self.register_buffer(
            "burden_weights",
            torch.tensor(_DEFAULT_BURDEN_WEIGHTS, dtype=torch.float32),
        )

        # Learned risk mapper: weighted sum + raw abundances -> scalar score
        # Input: weighted_sum (1) + individual abundances (num_args)
        self.risk_mapper = nn.Sequential(
            nn.Linear(1 + num_args, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        log_abundance: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the burden score and alert level.

        Parameters
        ----------
        log_abundance : Tensor[B, num_args]
            Predicted log-abundance for each ARG.

        Returns
        -------
        burden_score : Tensor[B]
            Environmental ARG burden index in [0, 100].
        alert_level : Tensor[B]
            Integer alert level (see :class:`BurdenAlertLevel`).
        """
        # Weighted sum using WHO priority weights
        weighted_sum = (log_abundance * self.burden_weights.unsqueeze(0)).sum(
            dim=-1, keepdim=True
        )  # (B, 1)

        # Concatenate weighted sum with individual abundances for non-linear mapping
        mapper_input = torch.cat(
            [weighted_sum, log_abundance], dim=-1
        )  # (B, 1+num_args)

        # Map to raw score and clamp to [0, 100]
        raw_score = self.risk_mapper(mapper_input).squeeze(-1)  # (B,)
        burden_score = torch.clamp(torch.sigmoid(raw_score) * 100.0, 0.0, 100.0)

        # Determine alert level based on thresholds
        alert_level = torch.zeros_like(burden_score, dtype=torch.long)
        alert_level[burden_score >= BURDEN_THRESHOLDS["low"]] = BurdenAlertLevel.MODERATE
        alert_level[burden_score >= BURDEN_THRESHOLDS["moderate"]] = BurdenAlertLevel.HIGH
        alert_level[burden_score >= BURDEN_THRESHOLDS["high"]] = BurdenAlertLevel.CRITICAL
        alert_level[burden_score >= BURDEN_THRESHOLDS["critical"]] = BurdenAlertLevel.EMERGENCY

        return burden_score, alert_level


# ============================================================================
# Sub-module 3: TemporalARGTracker
# ============================================================================

class TemporalARGTracker(nn.Module):
    """Track ARG abundance trends over time and detect emerging resistance.

    Consumes a temporal sequence of ARG predictions and detects:

    1. **Emerging resistance**: rapid sustained increase in any single ARG
       (slope above threshold over a sliding window).
    2. **Novel ARG combinations**: co-elevation of multiple ARGs that
       historically did not co-occur (detected via learned co-occurrence
       embeddings).
    3. **Sustained high burden**: burden score above critical threshold
       for consecutive time steps.

    Architecture::

        ARG sequence (B, T, 8) --> GRU (8 -> 64, bidirectional)
                                     |
                                     +--> trend head   (128 -> 8)  per-ARG slope
                                     +--> combo head   (128 -> 28) pairwise co-elevation
                                     +--> alert head   (128 -> 3)  alert type logits

    Parameters
    ----------
    num_args : int
        Number of ARGs being tracked (default 8).
    hidden_dim : int
        GRU hidden dimension (default 64).
    num_layers : int
        Number of GRU layers (default 2).
    dropout : float
        Dropout probability (default 0.2).
    emerging_slope_threshold : float
        Minimum slope (log-abundance/timestep) to trigger an emerging
        resistance alert (default 0.3).
    sustained_window : int
        Number of consecutive time steps above critical burden to trigger
        a sustained-high alert (default 5).
    """

    # Number of pairwise ARG combinations: C(8, 2) = 28
    NUM_PAIRS: int = NUM_ARGS * (NUM_ARGS - 1) // 2

    def __init__(
        self,
        num_args: int = NUM_ARGS,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = DROPOUT_P,
        emerging_slope_threshold: float = 0.3,
        sustained_window: int = 5,
    ) -> None:
        super().__init__()
        self.num_args = num_args
        self.hidden_dim = hidden_dim
        self.emerging_slope_threshold = emerging_slope_threshold
        self.sustained_window = sustained_window

        # Bidirectional GRU for temporal encoding
        self.gru = nn.GRU(
            input_size=num_args,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        gru_out_dim = hidden_dim * 2  # bidirectional

        # Per-ARG trend estimation head: predicts slope of each ARG
        self.trend_head = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_args),
        )

        # Pairwise co-elevation detection head
        self.combo_head = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, self.NUM_PAIRS),
        )

        # Alert classification head: {emerging, novel_combination, sustained_high}
        self.alert_head = nn.Sequential(
            nn.Linear(gru_out_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),
        )

        # Historical co-occurrence baseline (learnable, initialized from data)
        self.register_buffer(
            "baseline_co_occurrence",
            torch.zeros(self.NUM_PAIRS),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _pair_indices(num_args: int) -> List[Tuple[int, int]]:
        """Generate ordered pairs of ARG indices for co-occurrence analysis."""
        pairs = []
        for i in range(num_args):
            for j in range(i + 1, num_args):
                pairs.append((i, j))
        return pairs

    def forward(
        self,
        arg_sequence: torch.Tensor,
        burden_sequence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Analyze temporal ARG trends.

        Parameters
        ----------
        arg_sequence : Tensor[B, T, num_args]
            Sequence of ARG log-abundance predictions over T time steps.
        burden_sequence : Tensor[B, T] or None
            Optional sequence of burden scores (for sustained-high detection).

        Returns
        -------
        dict with keys:
            ``trend_slopes`` : Tensor[B, num_args]
                Estimated rate of change for each ARG.
            ``co_elevation_scores`` : Tensor[B, NUM_PAIRS]
                Pairwise co-elevation scores (sigmoid-activated).
            ``alert_logits`` : Tensor[B, 3]
                Logits for {emerging, novel_combination, sustained_high}.
            ``alert_probs`` : Tensor[B, 3]
                Softmax alert probabilities.
            ``emerging_flags`` : Tensor[B, num_args]
                Boolean flags for ARGs with slope above threshold.
            ``sustained_high_flag`` : Tensor[B]
                Boolean flag for sustained high burden (if burden_sequence
                is provided).
        """
        B, T, _ = arg_sequence.shape

        # GRU encoding
        gru_out, _ = self.gru(arg_sequence)  # (B, T, 2*hidden)

        # Use the final time step's representation
        final_repr = gru_out[:, -1, :]  # (B, 2*hidden)

        # Trend estimation
        trend_slopes = self.trend_head(final_repr)  # (B, num_args)

        # Co-elevation detection
        co_elevation_logits = self.combo_head(final_repr)  # (B, NUM_PAIRS)
        # Subtract baseline co-occurrence to detect *novel* co-elevation
        co_elevation_scores = torch.sigmoid(
            co_elevation_logits - self.baseline_co_occurrence.unsqueeze(0)
        )

        # Alert classification
        alert_logits = self.alert_head(final_repr)  # (B, 3)
        alert_probs = F.softmax(alert_logits, dim=-1)

        # Rule-based emerging resistance flags
        emerging_flags = trend_slopes > self.emerging_slope_threshold  # (B, num_args)

        # Sustained high burden detection
        sustained_high_flag = torch.zeros(B, dtype=torch.bool, device=arg_sequence.device)
        if burden_sequence is not None and T >= self.sustained_window:
            # Check if burden exceeds critical threshold for sustained_window
            # consecutive steps (from the end of the sequence)
            critical = BURDEN_THRESHOLDS["high"]
            recent_burden = burden_sequence[:, -self.sustained_window:]  # (B, W)
            sustained_high_flag = (recent_burden >= critical).all(dim=-1)  # (B,)

        return {
            "trend_slopes": trend_slopes,
            "co_elevation_scores": co_elevation_scores,
            "alert_logits": alert_logits,
            "alert_probs": alert_probs,
            "emerging_flags": emerging_flags,
            "sustained_high_flag": sustained_high_flag,
        }

    def generate_alerts(
        self,
        trend_output: Dict[str, torch.Tensor],
    ) -> List[List[TemporalARGAlert]]:
        """Convert model output to human-readable alerts.

        Parameters
        ----------
        trend_output : dict
            Output from :meth:`forward`.

        Returns
        -------
        List[List[TemporalARGAlert]]
            Per-sample list of active alerts.
        """
        B = trend_output["trend_slopes"].shape[0]
        all_alerts: List[List[TemporalARGAlert]] = []
        pairs = self._pair_indices(self.num_args)

        for b in range(B):
            sample_alerts: List[TemporalARGAlert] = []

            # Emerging resistance alerts
            emerging = trend_output["emerging_flags"][b]  # (num_args,)
            for arg_idx in range(self.num_args):
                if emerging[arg_idx].item():
                    slope = trend_output["trend_slopes"][b, arg_idx].item()
                    name = ARG_INDEX_TO_NAME[arg_idx]
                    conf = min(slope / (self.emerging_slope_threshold * 3.0), 1.0)
                    sample_alerts.append(TemporalARGAlert(
                        alert_type="emerging",
                        arg_names=[name],
                        trend_slope=slope,
                        confidence=conf,
                        message=(
                            f"Emerging resistance detected: {name} increasing at "
                            f"{slope:.3f} log-abundance/timestep"
                        ),
                    ))

            # Novel combination alerts
            co_scores = trend_output["co_elevation_scores"][b]  # (NUM_PAIRS,)
            for pair_idx, (i, j) in enumerate(pairs):
                if co_scores[pair_idx].item() > 0.8:
                    names = [ARG_INDEX_TO_NAME[i], ARG_INDEX_TO_NAME[j]]
                    sample_alerts.append(TemporalARGAlert(
                        alert_type="novel_combination",
                        arg_names=names,
                        trend_slope=0.0,
                        confidence=co_scores[pair_idx].item(),
                        message=(
                            f"Novel ARG co-elevation detected: "
                            f"{names[0]} + {names[1]} "
                            f"(score={co_scores[pair_idx].item():.2f})"
                        ),
                    ))

            # Sustained high burden
            if trend_output["sustained_high_flag"][b].item():
                sample_alerts.append(TemporalARGAlert(
                    alert_type="sustained_high",
                    arg_names=list(ARG_NAMES),
                    trend_slope=0.0,
                    confidence=0.95,
                    message=(
                        f"Sustained high ARG burden: critical threshold exceeded "
                        f"for >= {self.sustained_window} consecutive time steps"
                    ),
                ))

            all_alerts.append(sample_alerts)

        return all_alerts


# ============================================================================
# Main model: ARGPredictor
# ============================================================================

class ARGPredictor(nn.Module):
    """Predict antibiotic resistance gene abundance from 16S community data.

    Accepts either raw 16S OTU abundances (CLR-transformed, 5000-dim)
    or pre-computed MicroBiomeNet embeddings (256-dim).  Produces
    per-ARG log-abundance predictions and an aggregate burden score
    with MC-dropout uncertainty estimation.

    Architecture::

        Input (OTU or embedding)
            --> CommunityToARGMapper (if OTU input)
            --> Shared backbone: Linear(256, 512) -> GELU -> Dropout
                              -> Linear(512, 256) -> GELU -> Dropout
            --> 8 per-ARG heads: Linear(256, 64) -> GELU -> Dropout
                              -> Linear(64, 1)
            --> ARGBurdenIndex: weighted aggregation -> risk score (0-100)

    Parameters
    ----------
    num_otus : int
        Number of OTU features for raw input mode (default 5000).
    embedding_dim : int
        MicroBiomeNet embedding dimension (default 256).
    backbone_hidden : int
        Hidden dimension of the shared backbone (default 512).
    backbone_out : int
        Output dimension of the shared backbone (default 256).
    arg_head_hidden : int
        Hidden dimension per ARG prediction head (default 64).
    num_args : int
        Number of target ARGs (default 8).
    dropout : float
        Dropout probability for MC-dropout (default 0.2).
    mc_samples : int
        Number of stochastic forward passes (default 20).
    """

    def __init__(
        self,
        num_otus: int = NUM_OTUS,
        embedding_dim: int = EMBEDDING_DIM,
        backbone_hidden: int = BACKBONE_HIDDEN,
        backbone_out: int = BACKBONE_OUT,
        arg_head_hidden: int = ARG_HEAD_HIDDEN,
        num_args: int = NUM_ARGS,
        dropout: float = DROPOUT_P,
        mc_samples: int = MC_DROPOUT_SAMPLES,
    ) -> None:
        super().__init__()
        self.num_otus = num_otus
        self.embedding_dim = embedding_dim
        self.num_args = num_args
        self.dropout_p = dropout
        self.mc_samples = mc_samples

        # OTU-to-embedding mapper (used when raw OTU input is provided)
        self.community_mapper = CommunityToARGMapper(
            num_otus=num_otus,
            embed_dim=embedding_dim,
            dropout=dropout,
        )

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(embedding_dim, backbone_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(backbone_hidden, backbone_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Per-ARG prediction heads
        self.arg_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(backbone_out, arg_head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(arg_head_hidden, 1),
            )
            for _ in range(num_args)
        ])

        # Burden index calculator
        self.burden_index = ARGBurdenIndex(num_args=num_args)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Kaiming initialisation for all Linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def _enable_dropout(self) -> None:
        """Set all Dropout modules to train mode (stochastic)."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def _disable_dropout(self) -> None:
        """Set all Dropout modules back to eval mode (deterministic)."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.eval()

    # ------------------------------------------------------------------
    def _single_forward(
        self,
        embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single deterministic forward pass through backbone + heads.

        Parameters
        ----------
        embedding : Tensor[B, embedding_dim]
            Community embedding (from MicroBiomeNet or CommunityToARGMapper).

        Returns
        -------
        log_abundance : Tensor[B, num_args]
        burden_score : Tensor[B]
        alert_level : Tensor[B]
        """
        h = self.backbone(embedding)  # (B, backbone_out)

        # Per-ARG predictions
        arg_preds = [head(h) for head in self.arg_heads]  # list of (B, 1)
        log_abundance = torch.cat(arg_preds, dim=-1)      # (B, num_args)

        # Burden score
        burden_score, alert_level = self.burden_index(log_abundance)

        return log_abundance, burden_score, alert_level

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        input_type: str = "embedding",
        use_mc_dropout: bool = False,
    ) -> ARGPredictionOutput:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Either CLR-transformed OTU abundances ``(B, 5000)`` or
            MicroBiomeNet embeddings ``(B, 256)``.
        input_type : str
            ``"otu"`` for raw OTU input, ``"embedding"`` for pre-computed
            MicroBiomeNet embeddings (default ``"embedding"``).
        use_mc_dropout : bool
            If True, perform ``mc_samples`` stochastic forward passes
            and return posterior mean/std.

        Returns
        -------
        ARGPredictionOutput
        """
        attention_weights: Optional[torch.Tensor] = None

        # Map OTU input to embedding if needed
        if input_type == "otu":
            embedding, attention_weights = self.community_mapper(x)
        elif input_type == "embedding":
            embedding = x
        else:
            raise ValueError(
                f"Unknown input_type '{input_type}'. "
                f"Expected 'otu' or 'embedding'."
            )

        if not use_mc_dropout:
            log_abundance, burden_score, alert_level = self._single_forward(
                embedding
            )
            return ARGPredictionOutput(
                log_abundance=log_abundance,
                burden_score=burden_score,
                burden_alert=alert_level,
                attention_weights=attention_weights,
            )

        # MC-dropout: multiple stochastic forward passes
        was_training = self.training
        self._enable_dropout()

        mc_abundances: List[torch.Tensor] = []
        mc_burdens: List[torch.Tensor] = []

        for _ in range(self.mc_samples):
            log_ab, burden, _ = self._single_forward(embedding)
            mc_abundances.append(log_ab)
            mc_burdens.append(burden)

        if not was_training:
            self._disable_dropout()

        # Stack and compute statistics
        mc_stack = torch.stack(mc_abundances, dim=0)   # (S, B, num_args)
        mc_burden_stack = torch.stack(mc_burdens, dim=0)  # (S, B)

        mc_mean = mc_stack.mean(dim=0)         # (B, num_args)
        mc_std = mc_stack.std(dim=0)           # (B, num_args)
        mc_burden_mean = mc_burden_stack.mean(dim=0)  # (B,)
        mc_burden_std = mc_burden_stack.std(dim=0)    # (B,)

        # Use MC mean as point prediction for consistency
        _, alert_level = self.burden_index(mc_mean)

        return ARGPredictionOutput(
            log_abundance=mc_mean,
            burden_score=mc_burden_mean,
            burden_alert=alert_level,
            mc_mean=mc_mean,
            mc_std=mc_std,
            mc_burden_mean=mc_burden_mean,
            mc_burden_std=mc_burden_std,
            attention_weights=attention_weights,
        )


# ============================================================================
# Loss function
# ============================================================================

class ARGSurveillanceLoss(nn.Module):
    """Combined loss for ARG abundance prediction.

    Combines:

    1. **MSE loss** on log-abundance predictions (primary regression target).
    2. **Pairwise ranking loss** that preserves the relative ordering of
       ARG abundances across samples.  This encourages the model to learn
       the correct rank structure even when absolute abundance
       calibration is uncertain.

    The total loss is::

        L = mse_weight * MSE(pred, target)
          + rank_weight * RankingLoss(pred, target)

    Parameters
    ----------
    mse_weight : float
        Weight for the MSE component (default 1.0).
    rank_weight : float
        Weight for the ranking loss component (default 0.1).
    rank_margin : float
        Margin for the ranking loss (default 0.1).
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        rank_weight: float = 0.1,
        rank_margin: float = 0.1,
    ) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.rank_weight = rank_weight
        self.rank_margin = rank_margin

    def _pairwise_ranking_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise ranking loss across ARGs within each sample.

        For each pair of ARGs (i, j) where target_i > target_j, we
        penalise the model if pred_i - pred_j < margin.

        Parameters
        ----------
        pred : Tensor[B, A]
        target : Tensor[B, A]

        Returns
        -------
        Tensor (scalar)
        """
        B, A = pred.shape

        # Pairwise differences: (B, A, 1) - (B, 1, A) = (B, A, A)
        target_diff = target.unsqueeze(2) - target.unsqueeze(1)  # (B, A, A)
        pred_diff = pred.unsqueeze(2) - pred.unsqueeze(1)        # (B, A, A)

        # Only penalise pairs where target_i > target_j (upper triangle of
        # the target_diff > 0 region)
        valid_mask = (target_diff > 0).float()

        # Hinge-style ranking loss:
        # loss = max(0, margin - (pred_i - pred_j)) when target_i > target_j
        rank_loss = F.relu(self.rank_margin - pred_diff)  # (B, A, A)
        rank_loss = rank_loss * valid_mask

        # Average over valid pairs
        num_valid = valid_mask.sum()
        if num_valid > 0:
            return rank_loss.sum() / num_valid
        return torch.tensor(0.0, device=pred.device)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined ARG surveillance loss.

        Parameters
        ----------
        pred : Tensor[B, num_args]
            Predicted log-abundances.
        target : Tensor[B, num_args]
            Ground truth log-abundances.

        Returns
        -------
        dict with keys:
            ``total`` : combined loss (scalar).
            ``mse``   : MSE component (scalar).
            ``rank``  : ranking loss component (scalar).
        """
        mse = F.mse_loss(pred, target)
        rank = self._pairwise_ranking_loss(pred, target)

        total = self.mse_weight * mse + self.rank_weight * rank

        return {
            "total": total,
            "mse": mse,
            "rank": rank,
        }


# ============================================================================
# Convenience: full ARG surveillance pipeline
# ============================================================================

class ARGSurveillancePipeline(nn.Module):
    """End-to-end ARG surveillance pipeline.

    Wraps :class:`ARGPredictor` and :class:`TemporalARGTracker` into a
    single module that can:

    1. Predict ARG abundances from a single community sample.
    2. Track temporal trends over a sequence of samples.
    3. Generate human-readable alerts for emerging resistance.

    Parameters
    ----------
    predictor_kwargs : dict or None
        Keyword arguments for :class:`ARGPredictor`.
    tracker_kwargs : dict or None
        Keyword arguments for :class:`TemporalARGTracker`.
    """

    def __init__(
        self,
        predictor_kwargs: Optional[Dict[str, Any]] = None,
        tracker_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.predictor = ARGPredictor(**(predictor_kwargs or {}))
        self.tracker = TemporalARGTracker(**(tracker_kwargs or {}))

    def predict_single(
        self,
        x: torch.Tensor,
        input_type: str = "embedding",
        use_mc_dropout: bool = False,
    ) -> ARGPredictionOutput:
        """Predict ARG abundances for a single time point.

        Parameters
        ----------
        x : Tensor
            OTU abundances ``(B, 5000)`` or embedding ``(B, 256)``.
        input_type : str
            ``"otu"`` or ``"embedding"``.
        use_mc_dropout : bool
            Enable MC-dropout uncertainty.

        Returns
        -------
        ARGPredictionOutput
        """
        return self.predictor(x, input_type=input_type, use_mc_dropout=use_mc_dropout)

    def track_temporal(
        self,
        arg_sequence: torch.Tensor,
        burden_sequence: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[List[TemporalARGAlert]]]:
        """Analyse temporal ARG trends and generate alerts.

        Parameters
        ----------
        arg_sequence : Tensor[B, T, num_args]
            Sequence of ARG log-abundance predictions.
        burden_sequence : Tensor[B, T] or None
            Sequence of burden scores.

        Returns
        -------
        trend_output : dict
            Raw temporal analysis tensors.
        alerts : List[List[TemporalARGAlert]]
            Per-sample list of human-readable alerts.
        """
        trend_output = self.tracker(arg_sequence, burden_sequence)
        alerts = self.tracker.generate_alerts(trend_output)
        return trend_output, alerts

    def forward(
        self,
        x: torch.Tensor,
        input_type: str = "embedding",
        use_mc_dropout: bool = False,
    ) -> ARGPredictionOutput:
        """Alias for :meth:`predict_single` for compatibility with
        standard PyTorch training loops.
        """
        return self.predict_single(x, input_type=input_type, use_mc_dropout=use_mc_dropout)
