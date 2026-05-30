"""Environmental Metatranscriptomic Pathogen Surveillance (Phase 3.8).

Extends ToxiGene from controlled-exposure transcriptomics to
environmental metatranscriptomics: instead of analysing gene
expression in lab-exposed fish, this module processes environmental
RNA-seq features (from wastewater, surface water, or sediment
samples) to detect known and novel pathogen signatures.

Architecture overview
---------------------

**EnvironmentalRNAEncoder**::

    Raw RNA-seq features (B, num_genes)
        --> CLR / log-CPM normalisation layer
        --> Gene-level attention with functional annotation prior
        --> Compressed representation (B, 256)

**PathogenSignatureDB**::

    Curated gene expression signatures for known pathogens:
    - SARS-CoV-2: RdRp, N, S, E, ORF1ab markers
    - Avian influenza (H5N1): HA, NA, M, NP segments
    - Vibrio spp.: ctxA, tcpA, hlyA virulence genes
    - Cryptosporidium: 18S rRNA, COWP, GP60 markers

**MetatranscriptomicSurveillance**::

    EnvironmentalRNAEncoder output (B, 256)
    + SENTINEL embedding (B, 256)
        --> Pathogen detection heads (known pathogens)
        --> Abundance estimation heads (quantitative)
        --> Temporal trend analysis
        --> MC-dropout uncertainty

**NovelPathogenDetector**::

    Environmental RNA embedding (B, 256)
        --> Autoencoder reconstruction
        --> Anomaly scoring (reconstruction error)
        --> Isolation forest-style density estimation
        --> Novelty alert with confidence
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Constants
# ============================================================================

SHARED_EMBEDDING_DIM: int = 256
MC_DROPOUT_SAMPLES: int = 20
DROPOUT_P: float = 0.2

# Number of environmental RNA-seq gene features
NUM_GENES: int = 10000

# Compressed gene representation dim
GENE_ENCODING_DIM: int = 256

# Autoencoder latent dim for anomaly detection
ANOMALY_LATENT_DIM: int = 64

# Functional gene categories for attention prior
NUM_GENE_CATEGORIES: int = 8
GENE_CATEGORY_NAMES: Tuple[str, ...] = (
    "viral_replication",
    "virulence_factors",
    "antimicrobial_resistance",
    "stress_response",
    "metabolic_pathways",
    "ribosomal_markers",
    "mobile_genetic_elements",
    "hypothetical_unknown",
)


# ---------------------------------------------------------------------------
# Target pathogens
# ---------------------------------------------------------------------------

class PathogenCategory(IntEnum):
    """Known pathogen categories for surveillance."""
    SARS_COV2 = 0
    AVIAN_INFLUENZA = 1
    VIBRIO = 2
    CRYPTOSPORIDIUM = 3
    NOROVIRUS = 4
    LEGIONELLA = 5
    GIARDIA = 6
    CAMPYLOBACTER = 7


NUM_PATHOGENS: int = len(PathogenCategory)

PATHOGEN_NAMES: Tuple[str, ...] = (
    "SARS-CoV-2",
    "Avian Influenza (H5Nx)",
    "Vibrio spp.",
    "Cryptosporidium spp.",
    "Norovirus",
    "Legionella pneumophila",
    "Giardia lamblia",
    "Campylobacter spp.",
)

PATHOGEN_INDEX_TO_NAME: Dict[int, str] = {
    i: name for i, name in enumerate(PATHOGEN_NAMES)
}


@dataclass(frozen=True)
class PathogenSignature:
    """Known gene expression signature for a pathogen.

    Attributes
    ----------
    index : int
        Pathogen category index.
    name : str
        Pathogen common name.
    marker_genes : Tuple[str, ...]
        Key marker gene names for this pathogen.
    num_markers : int
        Number of marker genes in the curated signature.
    detection_threshold : float
        Minimum signature score for positive detection.
    clinical_significance : str
        Clinical relevance note.
    """

    index: int
    name: str
    marker_genes: Tuple[str, ...]
    num_markers: int
    detection_threshold: float
    clinical_significance: str


PATHOGEN_SIGNATURES: Tuple[PathogenSignature, ...] = (
    PathogenSignature(
        index=0,
        name="SARS-CoV-2",
        marker_genes=("RdRp", "N_gene", "S_gene", "E_gene", "ORF1ab",
                       "ORF3a", "ORF7a", "ORF8"),
        num_markers=8,
        detection_threshold=0.7,
        clinical_significance="COVID-19 pandemic surveillance; wastewater epidemiology gold standard",
    ),
    PathogenSignature(
        index=1,
        name="Avian Influenza (H5Nx)",
        marker_genes=("HA_H5", "NA_N1", "M_segment", "NP", "NS1",
                       "PB2", "PA", "PB1"),
        num_markers=8,
        detection_threshold=0.6,
        clinical_significance="Pandemic preparedness; zoonotic spillover risk at wildlife-water interface",
    ),
    PathogenSignature(
        index=2,
        name="Vibrio spp.",
        marker_genes=("ctxA", "tcpA", "hlyA", "toxR", "ompU",
                       "vpsL", "hapA"),
        num_markers=7,
        detection_threshold=0.5,
        clinical_significance="Cholera and wound infections; climate-sensitive waterborne pathogen",
    ),
    PathogenSignature(
        index=3,
        name="Cryptosporidium spp.",
        marker_genes=("18S_rRNA", "COWP", "GP60", "HSP70", "actin",
                       "DHFR"),
        num_markers=6,
        detection_threshold=0.5,
        clinical_significance="Chlorine-resistant protozoan; major drinking water threat",
    ),
    PathogenSignature(
        index=4,
        name="Norovirus",
        marker_genes=("VP1_capsid", "RdRp_noro", "VPg", "protease",
                       "VP2"),
        num_markers=5,
        detection_threshold=0.6,
        clinical_significance="Leading cause of gastroenteritis; extremely low infectious dose",
    ),
    PathogenSignature(
        index=5,
        name="Legionella pneumophila",
        marker_genes=("mip", "dotA", "icmT", "pilE", "flaA",
                       "lspA"),
        num_markers=6,
        detection_threshold=0.5,
        clinical_significance="Legionnaires' disease; aerosolised waterborne transmission",
    ),
    PathogenSignature(
        index=6,
        name="Giardia lamblia",
        marker_genes=("gdh", "bg_beta_giardin", "tpi", "SSU_rRNA",
                       "ef1a"),
        num_markers=5,
        detection_threshold=0.5,
        clinical_significance="Most common intestinal parasite worldwide; cyst-forming",
    ),
    PathogenSignature(
        index=7,
        name="Campylobacter spp.",
        marker_genes=("cadF", "ciaB", "cdtB", "flaA_camp",
                       "hipO", "mapA"),
        num_markers=6,
        detection_threshold=0.5,
        clinical_significance="Leading bacterial cause of gastroenteritis; poultry/water reservoir",
    ),
)


# ---------------------------------------------------------------------------
# Alert levels
# ---------------------------------------------------------------------------

class PathogenAlertLevel(IntEnum):
    """Alert levels for pathogen detection."""
    NOT_DETECTED = 0     # score < detection threshold
    LOW_LEVEL = 1        # threshold <= score < 0.5 above threshold
    MODERATE = 2         # moderate detection level
    HIGH_LEVEL = 3       # strong detection signal
    CRITICAL = 4         # very high, confirmed multi-marker detection

PATHOGEN_ALERT_OFFSETS: Dict[str, float] = {
    "low": 0.0,
    "moderate": 0.15,
    "high": 0.30,
    "critical": 0.50,
}


# ============================================================================
# Output dataclasses
# ============================================================================

@dataclass
class PathogenDetectionOutput:
    """Output of the MetatranscriptomicSurveillance model.

    Attributes
    ----------
    detection_scores : Tensor[B, P]
        Detection scores for each pathogen (0-1).
    detection_flags : Tensor[B, P]
        Boolean: score > detection_threshold for each pathogen.
    log_abundance : Tensor[B, P]
        Estimated log10 gene copies/L for detected pathogens.
    alert_levels : Tensor[B, P]
        Integer alert level per pathogen.
    mc_detection_mean : Tensor[B, P] or None
        MC-dropout posterior mean of detection score.
    mc_detection_std : Tensor[B, P] or None
        MC-dropout posterior std of detection score.
    mc_abundance_mean : Tensor[B, P] or None
        MC-dropout posterior mean of log-abundance.
    mc_abundance_std : Tensor[B, P] or None
        MC-dropout posterior std of log-abundance.
    """

    detection_scores: torch.Tensor
    detection_flags: torch.Tensor
    log_abundance: torch.Tensor
    alert_levels: torch.Tensor

    mc_detection_mean: Optional[torch.Tensor] = None
    mc_detection_std: Optional[torch.Tensor] = None
    mc_abundance_mean: Optional[torch.Tensor] = None
    mc_abundance_std: Optional[torch.Tensor] = None


@dataclass
class NovelPathogenAlert:
    """Alert from the NovelPathogenDetector.

    Attributes
    ----------
    anomaly_score : float
        Anomaly score (higher = more anomalous).
    reconstruction_error : float
        Autoencoder reconstruction error.
    is_novel : bool
        Whether the sample is flagged as containing a novel pathogen.
    confidence : float
        Confidence in the novelty assessment.
    message : str
        Human-readable alert description.
    """

    anomaly_score: float
    reconstruction_error: float
    is_novel: bool
    confidence: float
    message: str


@dataclass
class SurveillanceOutput:
    """Combined output of the full surveillance pipeline.

    Attributes
    ----------
    known_pathogens : PathogenDetectionOutput
        Detection results for known pathogens.
    novel_anomaly_scores : Tensor[B]
        Anomaly scores for novel pathogen detection.
    novel_reconstruction_error : Tensor[B]
        Autoencoder reconstruction error.
    novel_flags : Tensor[B]
        Boolean flags for novel pathogen alerts.
    rna_embedding : Tensor[B, 256]
        Learned environmental RNA embedding.
    """

    known_pathogens: PathogenDetectionOutput
    novel_anomaly_scores: torch.Tensor
    novel_reconstruction_error: torch.Tensor
    novel_flags: torch.Tensor
    rna_embedding: torch.Tensor


# ============================================================================
# Sub-module 1: EnvironmentalRNAEncoder
# ============================================================================

class EnvironmentalRNAEncoder(nn.Module):
    """Encode raw environmental RNA-seq features into a compact embedding.

    Processes high-dimensional gene expression profiles from
    environmental metatranscriptomic sequencing (wastewater, surface
    water, sediment) into a 256-d embedding suitable for downstream
    pathogen detection and community characterisation.

    Architecture::

        Raw gene features (B, num_genes)
            --> Log-CPM normalisation
            --> Gene attention with functional prior
                (upweights viral, virulence, AMR genes)
            --> Compressed MLP: 10000 -> 2048 -> 512 -> 256
            --> LayerNorm + Dropout
            --> output (B, 256)

    Parameters
    ----------
    num_genes : int
        Number of gene features (default 10000).
    output_dim : int
        Output embedding dimension (default 256).
    dropout : float
        Dropout probability (default 0.2).
    """

    def __init__(
        self,
        num_genes: int = NUM_GENES,
        output_dim: int = GENE_ENCODING_DIM,
        dropout: float = DROPOUT_P,
    ) -> None:
        super().__init__()
        self.num_genes = num_genes
        self.output_dim = output_dim

        # Log-CPM normalisation layer (learnable scale + shift)
        self.norm_scale = nn.Parameter(torch.ones(num_genes))
        self.norm_shift = nn.Parameter(torch.zeros(num_genes))

        # Gene-level attention for feature selection
        self.gene_attention = nn.Sequential(
            nn.Linear(num_genes, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_genes),
        )

        # Learnable functional prior: upweights pathogen-relevant genes
        # This is initialised to zero and loaded from annotation databases
        self.functional_prior = nn.Parameter(torch.zeros(num_genes))

        # Compression MLP
        self.encoder = nn.Sequential(
            nn.Linear(num_genes, 2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(2048),
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def load_functional_prior(
        self,
        prior_weights: torch.Tensor,
    ) -> None:
        """Load gene-level functional annotation weights.

        Higher weights for genes involved in viral replication,
        virulence, antimicrobial resistance, etc.

        Parameters
        ----------
        prior_weights : Tensor[num_genes]
            Non-negative weights.
        """
        if prior_weights.shape != (self.num_genes,):
            raise ValueError(
                f"Prior weights shape {prior_weights.shape} does not "
                f"match expected ({self.num_genes},)"
            )
        with torch.no_grad():
            self.functional_prior.copy_(prior_weights)

    def forward(
        self,
        gene_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode environmental RNA-seq features.

        Parameters
        ----------
        gene_features : Tensor[B, num_genes]
            Raw gene expression counts or log-CPM values.

        Returns
        -------
        embedding : Tensor[B, 256]
            Compressed gene expression embedding.
        attention_weights : Tensor[B, num_genes]
            Per-gene attention weights for interpretability.
        """
        # Learnable normalisation
        x = gene_features * self.norm_scale.unsqueeze(0) + self.norm_shift.unsqueeze(0)

        # Gene attention with functional prior
        attn_logits = self.gene_attention(x)  # [B, num_genes]
        attn_logits = attn_logits + self.functional_prior.unsqueeze(0)
        attention_weights = F.softmax(attn_logits, dim=-1)  # [B, num_genes]

        # Attention-weighted features
        weighted = x * attention_weights

        # Compress to embedding
        embedding = self.encoder(weighted)  # [B, 256]

        return embedding, attention_weights


# ============================================================================
# Sub-module 2: PathogenSignatureDB
# ============================================================================

class PathogenSignatureDB(nn.Module):
    """Database of known pathogen gene expression signatures.

    Stores learnable signature vectors for each known pathogen.
    These vectors encode the expected pattern of marker gene
    expression when a pathogen is present.  Detection is performed
    by computing cosine similarity between the environmental RNA
    embedding and each pathogen signature.

    Parameters
    ----------
    embedding_dim : int
        Embedding dimension (default 256).
    num_pathogens : int
        Number of known pathogens (default 8).
    num_markers_per_pathogen : int
        Maximum marker genes per pathogen signature (default 10).
    """

    def __init__(
        self,
        embedding_dim: int = GENE_ENCODING_DIM,
        num_pathogens: int = NUM_PATHOGENS,
        num_markers_per_pathogen: int = 10,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_pathogens = num_pathogens
        self.num_markers = num_markers_per_pathogen

        # Learnable signature embeddings
        self.signature_embeddings = nn.Embedding(num_pathogens, embedding_dim)
        self.signature_norm = nn.LayerNorm(embedding_dim)

        # Per-pathogen detection bias (adjusts for prevalence priors)
        self.detection_bias = nn.Parameter(torch.zeros(num_pathogens))

        # Marker gene weight matrix: encodes which features are most
        # informative for each pathogen
        self.marker_weights = nn.Parameter(
            torch.randn(num_pathogens, embedding_dim) * 0.02
        )

        # Detection thresholds (from PathogenSignature catalog)
        thresholds = torch.tensor(
            [sig.detection_threshold for sig in PATHOGEN_SIGNATURES],
            dtype=torch.float32,
        )
        self.register_buffer("detection_thresholds", thresholds)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.signature_embeddings.weight, mean=0.0, std=0.02)

    def get_signatures(self) -> torch.Tensor:
        """Get all pathogen signature embeddings.

        Returns
        -------
        Tensor[P, 256]
            Normalised pathogen signature vectors.
        """
        idx = torch.arange(
            self.num_pathogens,
            device=self.signature_embeddings.weight.device,
        )
        sigs = self.signature_embeddings(idx)
        return self.signature_norm(sigs)

    def compute_similarity(
        self,
        rna_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine similarity between RNA embedding and signatures.

        Parameters
        ----------
        rna_embedding : Tensor[B, 256]
            Environmental RNA embedding.

        Returns
        -------
        Tensor[B, P]
            Cosine similarity scores for each pathogen.
        """
        sigs = self.get_signatures()  # [P, 256]

        # Apply marker weights to both query and signatures
        weighted_query = rna_embedding.unsqueeze(1) * self.marker_weights.unsqueeze(0)
        # [B, P, 256]

        weighted_sigs = sigs * self.marker_weights  # [P, 256]

        # Cosine similarity per pathogen
        query_norm = F.normalize(weighted_query, dim=-1)
        sig_norm = F.normalize(weighted_sigs, dim=-1).unsqueeze(0)

        similarity = (query_norm * sig_norm).sum(dim=-1)  # [B, P]
        similarity = similarity + self.detection_bias.unsqueeze(0)

        return similarity


# ============================================================================
# Sub-module 3: Known Pathogen Detection Heads
# ============================================================================

class PathogenDetectionHeads(nn.Module):
    """Per-pathogen detection and quantification heads.

    Takes the RNA embedding and pathogen signature similarity to
    produce calibrated detection scores and abundance estimates.

    Parameters
    ----------
    embedding_dim : int
        RNA embedding dimension (default 256).
    num_pathogens : int
        Number of known pathogens (default 8).
    hidden_dim : int
        Hidden layer size (default 128).
    dropout : float
        Dropout probability (default 0.2).
    """

    def __init__(
        self,
        embedding_dim: int = GENE_ENCODING_DIM,
        num_pathogens: int = NUM_PATHOGENS,
        hidden_dim: int = 128,
        dropout: float = DROPOUT_P,
    ) -> None:
        super().__init__()
        self.num_pathogens = num_pathogens

        # Combined input: RNA embedding + similarity scores
        input_dim = embedding_dim + num_pathogens

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Per-pathogen detection score head
        self.detection_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
            for _ in range(num_pathogens)
        ])

        # Per-pathogen abundance estimation head
        self.abundance_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
            for _ in range(num_pathogens)
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        rna_embedding: torch.Tensor,
        similarity_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict detection scores and abundance for each pathogen.

        Parameters
        ----------
        rna_embedding : Tensor[B, 256]
            Environmental RNA embedding.
        similarity_scores : Tensor[B, P]
            Cosine similarity from PathogenSignatureDB.

        Returns
        -------
        detection_scores : Tensor[B, P]
            Calibrated detection scores in [0, 1].
        log_abundance : Tensor[B, P]
            Estimated log10 abundance (gene copies/L).
        """
        combined = torch.cat([rna_embedding, similarity_scores], dim=-1)
        shared_feat = self.shared(combined)  # [B, hidden]

        det_list: List[torch.Tensor] = []
        abund_list: List[torch.Tensor] = []

        for p in range(self.num_pathogens):
            det = torch.sigmoid(
                self.detection_heads[p](shared_feat).squeeze(-1)
            )
            abund = self.abundance_heads[p](shared_feat).squeeze(-1)
            det_list.append(det)
            abund_list.append(abund)

        detection_scores = torch.stack(det_list, dim=-1)    # [B, P]
        log_abundance = torch.stack(abund_list, dim=-1)     # [B, P]

        return detection_scores, log_abundance


# ============================================================================
# Sub-module 4: NovelPathogenDetector
# ============================================================================

class NovelPathogenDetector(nn.Module):
    """Anomaly detection for novel pathogen signatures.

    Uses an autoencoder to learn the distribution of known
    environmental RNA-seq patterns.  Samples with high
    reconstruction error represent novel gene expression
    signatures that do not match any known pathogen -- these
    may indicate emerging or previously uncharacterised pathogens.

    Architecture::

        RNA embedding (B, 256)
            --> Encoder: 256 -> 128 -> 64 (latent)
            --> Decoder: 64 -> 128 -> 256 (reconstruction)
            --> Anomaly score = reconstruction error
            --> Density estimator: learned threshold

    Parameters
    ----------
    embedding_dim : int
        Input RNA embedding dimension (default 256).
    latent_dim : int
        Autoencoder latent dimension (default 64).
    dropout : float
        Dropout probability (default 0.2).
    anomaly_threshold : float
        Initial anomaly score threshold (default 2.0).
        Will be calibrated on training data.
    """

    def __init__(
        self,
        embedding_dim: int = GENE_ENCODING_DIM,
        latent_dim: int = ANOMALY_LATENT_DIM,
        dropout: float = DROPOUT_P,
        anomaly_threshold: float = 2.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.anomaly_threshold = anomaly_threshold

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim),
        )

        # Learned anomaly scoring: maps reconstruction error to calibrated score
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(embedding_dim + latent_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Running statistics for adaptive thresholding
        self.register_buffer("running_mean_error", torch.tensor(0.0))
        self.register_buffer("running_std_error", torch.tensor(1.0))
        self.register_buffer("num_samples_seen", torch.tensor(0, dtype=torch.long))

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def update_statistics(
        self,
        reconstruction_error: torch.Tensor,
    ) -> None:
        """Update running error statistics for adaptive thresholding.

        Parameters
        ----------
        reconstruction_error : Tensor[B]
            Per-sample reconstruction errors.
        """
        with torch.no_grad():
            batch_mean = reconstruction_error.mean()
            batch_var = reconstruction_error.var()
            n = self.num_samples_seen.float()
            b = float(reconstruction_error.size(0))

            # Welford's online algorithm for running mean/variance
            new_n = n + b
            delta = batch_mean - self.running_mean_error
            self.running_mean_error += delta * b / new_n

            # Update variance approximation
            m2 = self.running_std_error ** 2 * n + batch_var * b + delta ** 2 * n * b / new_n
            self.running_std_error = torch.sqrt(m2 / new_n + 1e-8)

            self.num_samples_seen += int(b)

    def forward(
        self,
        rna_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Detect novel pathogen signatures via anomaly detection.

        Parameters
        ----------
        rna_embedding : Tensor[B, 256]
            Environmental RNA embedding.

        Returns
        -------
        anomaly_scores : Tensor[B]
            Calibrated anomaly scores (higher = more novel).
        reconstruction_error : Tensor[B]
            Raw reconstruction error per sample.
        novel_flags : Tensor[B]
            Boolean flags for samples exceeding anomaly threshold.
        """
        # Encode to latent space
        latent = self.encoder(rna_embedding)  # [B, latent_dim]

        # Reconstruct
        reconstruction = self.decoder(latent)  # [B, 256]

        # Reconstruction error (per-sample MSE)
        error = (rna_embedding - reconstruction).pow(2).mean(dim=-1)  # [B]

        # Learned anomaly scoring
        scorer_input = torch.cat([
            rna_embedding - reconstruction,  # residual signal
            latent,
        ], dim=-1)  # [B, 256+64]
        anomaly_scores = self.anomaly_scorer(scorer_input).squeeze(-1)  # [B]
        anomaly_scores = F.relu(anomaly_scores)  # non-negative

        # Adaptive thresholding using running statistics
        if self.num_samples_seen > 100:
            threshold = (
                self.running_mean_error
                + self.anomaly_threshold * self.running_std_error
            )
        else:
            threshold = torch.tensor(
                self.anomaly_threshold,
                device=rna_embedding.device,
            )

        novel_flags = error > threshold

        # Update statistics during training
        if self.training:
            self.update_statistics(error)

        return anomaly_scores, error, novel_flags


# ============================================================================
# Sub-module 5: Environmental Context Fusion
# ============================================================================

class EnvironmentalContextFusion(nn.Module):
    """Fuse RNA embedding with SENTINEL environmental embedding.

    Combines the metatranscriptomic signal with broader environmental
    context to improve pathogen detection accuracy.  Environmental
    conditions (temperature, season, upstream land use) are known
    predictors of pathogen presence and should modulate detection
    sensitivity.

    Parameters
    ----------
    rna_dim : int
        RNA embedding dimension (default 256).
    env_dim : int
        SENTINEL embedding dimension (default 256).
    output_dim : int
        Output dimension (default 256).
    dropout : float
        Dropout probability (default 0.2).
    """

    def __init__(
        self,
        rna_dim: int = GENE_ENCODING_DIM,
        env_dim: int = SHARED_EMBEDDING_DIM,
        output_dim: int = 256,
        dropout: float = DROPOUT_P,
    ) -> None:
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(rna_dim + env_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

        # Gating mechanism: learn how much environmental context to use
        self.gate = nn.Sequential(
            nn.Linear(rna_dim + env_dim, output_dim),
            nn.Sigmoid(),
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
        rna_embedding: torch.Tensor,
        sentinel_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse RNA and environmental embeddings.

        Parameters
        ----------
        rna_embedding : Tensor[B, 256]
        sentinel_embedding : Tensor[B, 256]

        Returns
        -------
        Tensor[B, 256]
            Fused embedding.
        """
        combined = torch.cat([rna_embedding, sentinel_embedding], dim=-1)
        fused = self.fusion(combined)
        gate = self.gate(combined)
        return fused * gate


# ============================================================================
# Main Model 1: MetatranscriptomicSurveillance
# ============================================================================

class MetatranscriptomicSurveillance(nn.Module):
    """Environmental metatranscriptomic pathogen surveillance system.

    Extends ToxiGene from controlled-exposure transcriptomics to
    environmental metatranscriptomics.  Processes environmental
    RNA-seq features to detect known pathogen signatures and estimate
    pathogen abundance with uncertainty quantification.

    Parameters
    ----------
    num_genes : int
        Number of gene features (default 10000).
    embedding_dim : int
        SENTINEL embedding dimension (default 256).
    num_pathogens : int
        Number of known pathogen targets (default 8).
    dropout : float
        Dropout probability (default 0.2).
    mc_samples : int
        Number of MC-dropout forward passes (default 20).

    Example
    -------
    >>> model = MetatranscriptomicSurveillance()
    >>> genes = torch.randn(4, 10000)
    >>> sentinel = torch.randn(4, 256)
    >>> output = model(genes, sentinel)
    >>> output.detection_scores.shape
    torch.Size([4, 8])
    """

    def __init__(
        self,
        num_genes: int = NUM_GENES,
        embedding_dim: int = SHARED_EMBEDDING_DIM,
        num_pathogens: int = NUM_PATHOGENS,
        dropout: float = DROPOUT_P,
        mc_samples: int = MC_DROPOUT_SAMPLES,
    ) -> None:
        super().__init__()
        self.num_genes = num_genes
        self.embedding_dim = embedding_dim
        self.num_pathogens = num_pathogens
        self.mc_samples = mc_samples
        self.dropout_p = dropout

        # Sub-modules
        self.rna_encoder = EnvironmentalRNAEncoder(
            num_genes=num_genes,
            output_dim=GENE_ENCODING_DIM,
            dropout=dropout,
        )

        self.context_fusion = EnvironmentalContextFusion(
            rna_dim=GENE_ENCODING_DIM,
            env_dim=embedding_dim,
            output_dim=256,
            dropout=dropout,
        )

        self.signature_db = PathogenSignatureDB(
            embedding_dim=GENE_ENCODING_DIM,
            num_pathogens=num_pathogens,
        )

        self.detection_heads = PathogenDetectionHeads(
            embedding_dim=256,
            num_pathogens=num_pathogens,
            hidden_dim=128,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # MC-dropout helpers
    # ------------------------------------------------------------------

    def _enable_dropout(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def _disable_dropout(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.eval()

    # ------------------------------------------------------------------
    # Alert level computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_alert_levels(
        detection_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Assign alert levels based on detection scores.

        Uses per-pathogen thresholds from the signature catalog.

        Parameters
        ----------
        detection_scores : Tensor[B, P]

        Returns
        -------
        Tensor[B, P] of long
        """
        B, P = detection_scores.shape
        alerts = torch.zeros(B, P, dtype=torch.long, device=detection_scores.device)

        for p in range(min(P, len(PATHOGEN_SIGNATURES))):
            threshold = PATHOGEN_SIGNATURES[p].detection_threshold
            scores = detection_scores[:, p]

            alerts[:, p] = torch.where(
                scores >= threshold + PATHOGEN_ALERT_OFFSETS["critical"],
                torch.tensor(PathogenAlertLevel.CRITICAL, device=scores.device),
                alerts[:, p],
            )
            mask_high = (scores >= threshold + PATHOGEN_ALERT_OFFSETS["high"]) & (
                scores < threshold + PATHOGEN_ALERT_OFFSETS["critical"]
            )
            alerts[:, p] = torch.where(
                mask_high,
                torch.tensor(PathogenAlertLevel.HIGH_LEVEL, device=scores.device),
                alerts[:, p],
            )
            mask_mod = (scores >= threshold + PATHOGEN_ALERT_OFFSETS["moderate"]) & (
                scores < threshold + PATHOGEN_ALERT_OFFSETS["high"]
            )
            alerts[:, p] = torch.where(
                mask_mod,
                torch.tensor(PathogenAlertLevel.MODERATE, device=scores.device),
                alerts[:, p],
            )
            mask_low = (scores >= threshold) & (
                scores < threshold + PATHOGEN_ALERT_OFFSETS["moderate"]
            )
            alerts[:, p] = torch.where(
                mask_low,
                torch.tensor(PathogenAlertLevel.LOW_LEVEL, device=scores.device),
                alerts[:, p],
            )

        return alerts

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def _single_forward(
        self,
        gene_features: torch.Tensor,
        sentinel_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single deterministic forward pass.

        Returns
        -------
        detection_scores, log_abundance, rna_embedding
        """
        # Encode RNA features
        rna_emb, _ = self.rna_encoder(gene_features)

        # Fuse with environmental context
        fused = self.context_fusion(rna_emb, sentinel_embedding)

        # Compute signature similarity
        similarity = self.signature_db.compute_similarity(rna_emb)

        # Detect and quantify pathogens
        detection_scores, log_abundance = self.detection_heads(
            fused, similarity,
        )

        return detection_scores, log_abundance, rna_emb

    def forward(
        self,
        gene_features: torch.Tensor,
        sentinel_embedding: torch.Tensor,
        use_mc_dropout: bool = False,
    ) -> PathogenDetectionOutput:
        """Forward pass with optional MC-dropout uncertainty.

        Parameters
        ----------
        gene_features : Tensor[B, num_genes]
            Environmental RNA-seq gene expression features.
        sentinel_embedding : Tensor[B, 256]
            SENTINEL environmental embedding.
        use_mc_dropout : bool
            If True and in eval mode, run mc_samples stochastic
            forward passes.

        Returns
        -------
        PathogenDetectionOutput
        """
        if not use_mc_dropout or self.training:
            det_scores, log_abund, rna_emb = self._single_forward(
                gene_features, sentinel_embedding,
            )

            # Detection flags based on per-pathogen thresholds
            thresholds = self.signature_db.detection_thresholds.unsqueeze(0)
            det_flags = det_scores >= thresholds

            return PathogenDetectionOutput(
                detection_scores=det_scores,
                detection_flags=det_flags,
                log_abundance=log_abund,
                alert_levels=self._compute_alert_levels(det_scores),
            )

        # --- MC-dropout inference ---
        self._enable_dropout()

        mc_det: List[torch.Tensor] = []
        mc_abund: List[torch.Tensor] = []

        with torch.no_grad():
            for _ in range(self.mc_samples):
                det_scores, log_abund, rna_emb = self._single_forward(
                    gene_features, sentinel_embedding,
                )
                mc_det.append(det_scores)
                mc_abund.append(log_abund)

        self._disable_dropout()

        det_stack = torch.stack(mc_det, dim=0)      # [N, B, P]
        abund_stack = torch.stack(mc_abund, dim=0)  # [N, B, P]

        det_mean = det_stack.mean(dim=0)
        det_std = det_stack.std(dim=0)
        abund_mean = abund_stack.mean(dim=0)
        abund_std = abund_stack.std(dim=0)

        thresholds = self.signature_db.detection_thresholds.unsqueeze(0)
        det_flags = det_mean >= thresholds

        return PathogenDetectionOutput(
            detection_scores=det_mean,
            detection_flags=det_flags,
            log_abundance=abund_mean,
            alert_levels=self._compute_alert_levels(det_mean),
            mc_detection_mean=det_mean,
            mc_detection_std=det_std,
            mc_abundance_mean=abund_mean,
            mc_abundance_std=abund_std,
        )

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_loss(
        output: PathogenDetectionOutput,
        targets: Dict[str, torch.Tensor],
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute multi-task training loss.

        Parameters
        ----------
        output : PathogenDetectionOutput
            Model predictions.
        targets : dict
            Expected keys (all optional, missing keys are skipped):

            - ``"detection"`` : Tensor[B, P] -- binary detection labels.
            - ``"log_abundance"`` : Tensor[B, P] -- log10 abundance.

        loss_weights : dict, optional
            Per-task weighting factors.

        Returns
        -------
        total_loss : scalar Tensor
        per_task : Dict[str, Tensor]
        """
        if loss_weights is None:
            loss_weights = {
                "detection": 1.0,
                "abundance": 0.5,
            }

        losses: Dict[str, torch.Tensor] = {}
        device = output.detection_scores.device

        if "detection" in targets:
            losses["detection"] = F.binary_cross_entropy(
                output.detection_scores,
                targets["detection"].float(),
            )

        if "log_abundance" in targets:
            losses["abundance"] = F.mse_loss(
                output.log_abundance,
                targets["log_abundance"].float(),
            )

        total = torch.tensor(0.0, device=device)
        for key, loss_val in losses.items():
            w = loss_weights.get(key, 1.0)
            total = total + w * loss_val

        return total, losses


# ============================================================================
# Main Model 2: Full Surveillance Pipeline
# ============================================================================

class EnvironmentalSurveillancePipeline(nn.Module):
    """End-to-end environmental metatranscriptomic surveillance pipeline.

    Combines known pathogen detection (MetatranscriptomicSurveillance)
    with novel pathogen anomaly detection (NovelPathogenDetector) into
    a single module.

    Parameters
    ----------
    num_genes : int
        Number of gene features (default 10000).
    embedding_dim : int
        SENTINEL embedding dimension (default 256).
    num_pathogens : int
        Number of known pathogen targets (default 8).
    dropout : float
        Dropout probability (default 0.2).
    mc_samples : int
        Number of MC-dropout forward passes (default 20).
    anomaly_threshold : float
        Anomaly score threshold for novel detection (default 2.0).

    Example
    -------
    >>> pipeline = EnvironmentalSurveillancePipeline()
    >>> genes = torch.randn(4, 10000)
    >>> sentinel = torch.randn(4, 256)
    >>> output = pipeline(genes, sentinel)
    >>> output.known_pathogens.detection_scores.shape
    torch.Size([4, 8])
    >>> output.novel_anomaly_scores.shape
    torch.Size([4])
    """

    def __init__(
        self,
        num_genes: int = NUM_GENES,
        embedding_dim: int = SHARED_EMBEDDING_DIM,
        num_pathogens: int = NUM_PATHOGENS,
        dropout: float = DROPOUT_P,
        mc_samples: int = MC_DROPOUT_SAMPLES,
        anomaly_threshold: float = 2.0,
    ) -> None:
        super().__init__()

        self.surveillance = MetatranscriptomicSurveillance(
            num_genes=num_genes,
            embedding_dim=embedding_dim,
            num_pathogens=num_pathogens,
            dropout=dropout,
            mc_samples=mc_samples,
        )

        self.novel_detector = NovelPathogenDetector(
            embedding_dim=GENE_ENCODING_DIM,
            latent_dim=ANOMALY_LATENT_DIM,
            dropout=dropout,
            anomaly_threshold=anomaly_threshold,
        )

    def forward(
        self,
        gene_features: torch.Tensor,
        sentinel_embedding: torch.Tensor,
        use_mc_dropout: bool = False,
    ) -> SurveillanceOutput:
        """Run full surveillance pipeline.

        Parameters
        ----------
        gene_features : Tensor[B, num_genes]
            Environmental RNA-seq gene expression features.
        sentinel_embedding : Tensor[B, 256]
            SENTINEL environmental embedding.
        use_mc_dropout : bool
            Enable MC-dropout uncertainty estimation.

        Returns
        -------
        SurveillanceOutput
            Combined known and novel pathogen detection results.
        """
        # Known pathogen detection
        known_output = self.surveillance(
            gene_features, sentinel_embedding,
            use_mc_dropout=use_mc_dropout,
        )

        # Get RNA embedding for novel detection
        rna_emb, _ = self.surveillance.rna_encoder(gene_features)

        # Novel pathogen detection
        anomaly_scores, recon_error, novel_flags = self.novel_detector(rna_emb)

        return SurveillanceOutput(
            known_pathogens=known_output,
            novel_anomaly_scores=anomaly_scores,
            novel_reconstruction_error=recon_error,
            novel_flags=novel_flags,
            rna_embedding=rna_emb,
        )

    def compute_loss(
        self,
        output: SurveillanceOutput,
        targets: Dict[str, torch.Tensor],
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute combined loss for the full pipeline.

        Parameters
        ----------
        output : SurveillanceOutput
            Pipeline output.
        targets : dict
            Expected keys (all optional):

            - ``"detection"`` : Tensor[B, P] -- binary detection labels.
            - ``"log_abundance"`` : Tensor[B, P] -- log10 abundance.

        loss_weights : dict, optional
            Per-task weighting factors.

        Returns
        -------
        total_loss : scalar Tensor
        per_task : Dict[str, Tensor]
        """
        if loss_weights is None:
            loss_weights = {
                "detection": 1.0,
                "abundance": 0.5,
                "reconstruction": 0.3,
            }

        losses: Dict[str, torch.Tensor] = {}
        device = output.novel_anomaly_scores.device

        # Known pathogen losses
        known_total, known_losses = MetatranscriptomicSurveillance.compute_loss(
            output.known_pathogens, targets,
        )
        losses.update(known_losses)

        # Autoencoder reconstruction loss (train the anomaly detector)
        losses["reconstruction"] = output.novel_reconstruction_error.mean()

        total = torch.tensor(0.0, device=device)
        for key, loss_val in losses.items():
            w = loss_weights.get(key, 1.0)
            total = total + w * loss_val

        return total, losses

    def generate_alerts(
        self,
        output: SurveillanceOutput,
    ) -> List[List[str]]:
        """Generate human-readable alerts from surveillance output.

        Parameters
        ----------
        output : SurveillanceOutput
            Pipeline output.

        Returns
        -------
        List[List[str]]
            Per-sample list of alert messages.
        """
        B = output.known_pathogens.detection_scores.size(0)
        all_alerts: List[List[str]] = []

        for b in range(B):
            sample_alerts: List[str] = []

            # Known pathogen alerts
            for p in range(self.surveillance.num_pathogens):
                alert_level = output.known_pathogens.alert_levels[b, p].item()
                if alert_level >= PathogenAlertLevel.LOW_LEVEL:
                    name = PATHOGEN_INDEX_TO_NAME.get(p, f"Pathogen_{p}")
                    score = output.known_pathogens.detection_scores[b, p].item()
                    abund = output.known_pathogens.log_abundance[b, p].item()
                    level_name = PathogenAlertLevel(alert_level).name
                    sample_alerts.append(
                        f"[{level_name}] {name}: detection_score={score:.3f}, "
                        f"est. log10(copies/L)={abund:.2f}"
                    )

            # Novel pathogen alert
            if output.novel_flags[b].item():
                anomaly = output.novel_anomaly_scores[b].item()
                recon = output.novel_reconstruction_error[b].item()
                sample_alerts.append(
                    f"[NOVEL] Unknown pathogen signature detected: "
                    f"anomaly_score={anomaly:.3f}, reconstruction_error={recon:.4f}"
                )

            all_alerts.append(sample_alerts)

        return all_alerts
