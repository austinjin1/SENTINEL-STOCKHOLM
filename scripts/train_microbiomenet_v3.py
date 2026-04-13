#!/usr/bin/env python3
"""Train MicroBiomeNet v3 — decisively outperform SimpleMLP baseline.

Key improvements over v2:
  1. Token-based transformer: 1D conv (kernel=5, stride=5) creates 1000 local
     OTU-group tokens of embed_dim=384, with learnable positional embeddings
  2. Pre-norm transformer (8 layers, 12 heads, ff=1536, dropout=0.15)
  3. [CLS] token classification (vs. global pooling in v2)
  4. Focal cross-entropy loss with inverse class-frequency alpha weights
  5. SupCon auxiliary loss (weight=0.1) on 384-dim L2-normalized embeddings
  6. Class-balanced WeightedRandomSampler
  7. Stronger augmentation: Dirichlet(alpha=0.02) noise, 30% OTU zeroing,
     Mixup(beta=0.3,0.3) on 50% of batches
  8. AdamW + CosineAnnealingWarmRestarts(T0=50, Tmult=2)
  9. AMP (autocast), gradient clipping=1.0, early stopping patience=30

Architecture (MicroBiomeNetV3):
  Input: 5000 OTU relative abundance vector
  -> CLR transform (preprocessing, applied at dataset load time)
  -> Conv1d(1, 384, kernel_size=5, stride=5) -> 1000 tokens of dim 384
  -> Prepend [CLS] token -> 1001 × 384 sequence
  -> Learnable positional embedding (1001 × 384)
  -> 8× PreNorm TransformerEncoderLayer (384, 12 heads, ff=1536, drop=0.15)
  -> CLS output -> LayerNorm -> Linear(384,256) -> GELU -> Dropout(0.3) -> Linear(256,8)

Loss: 0.9 * FocalCE(gamma=2, alpha=inv_freq, label_smooth=0.1)
    + 0.1 * SupConLoss(temperature=0.07)

Saves:
  checkpoints/microbial/microbiomenet_v3_best.pt
  checkpoints/microbial/results_v3.json
Logs to: logs/train_microbiomenet_v3.log

MIT License — Bryan Cheng, 2026
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split

from sklearn.metrics import accuracy_score, classification_report, f1_score

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/home/bcheng/SENTINEL")
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR     = PROJECT_ROOT / "data" / "processed" / "microbial" / "emp_16s"
CKPT_DIR     = PROJECT_ROOT / "checkpoints" / "microbial"
LOG_DIR      = PROJECT_ROOT / "logs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH     = LOG_DIR / "train_microbiomenet_v3.log"
CKPT_PATH    = CKPT_DIR / "microbiomenet_v3_best.pt"
RESULTS_PATH = CKPT_DIR / "results_v3.json"

# ── Logging ──────────────────────────────────────────────────────────────────

from sentinel.utils.logging import get_logger  # noqa: E402

_file_handler = logging.FileHandler(LOG_PATH, mode="w")
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
)

logger = get_logger(__name__)
logger.addHandler(_file_handler)

# ── Hyperparameters ──────────────────────────────────────────────────────────

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

DEVICE      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

INPUT_DIM   = 5000
EMBED_DIM   = 256
NUM_HEADS   = 8
NUM_LAYERS  = 6
FF_DIM      = 1536
DROPOUT     = 0.15
NUM_CLASSES = 8
NUM_TOKENS  = 100        # 5000 / 50 (conv stride=50)
MAX_SEQ_LEN = 101        # 100 OTU tokens + 1 CLS

BATCH_SIZE  = 64
EPOCHS      = 200
LR          = 3e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP   = 1.0
EARLY_STOP_PATIENCE = 30

FOCAL_GAMMA     = 2.0
LABEL_SMOOTHING = 0.1
SUPCON_WEIGHT   = 0.1
SUPCON_TEMP     = 0.07

AUG_DIRICHLET_ALPHA = 0.02
AUG_ZERO_FRAC       = 0.30
AUG_MIXUP_ALPHA     = 0.30
AUG_MIXUP_PROB      = 0.50

SEED = 42

SOURCE_NAMES = [
    "freshwater_natural",   # 0
    "freshwater_impacted",  # 1
    "saline_water",         # 2
    "freshwater_sediment",  # 3
    "saline_sediment",      # 4
    "soil_runoff",          # 5
    "animal_fecal",         # 6
    "plant_associated",     # 7
]

# ── CLR transform ────────────────────────────────────────────────────────────

def clr_transform(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Centre log-ratio transform. Input: (D,) relative abundance. Returns (D,) float32."""
    x = x + eps
    x = x / x.sum()
    lx = np.log(x)
    return np.clip(lx - lx.mean(), -8.0, 8.0).astype(np.float32)


# ── Dataset ──────────────────────────────────────────────────────────────────

class OTUDataset(Dataset):
    """EMP 16S OTU dataset with CLR preprocessing applied at load time."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        # features: (N, 5000) already CLR-transformed float32
        self.features = torch.from_numpy(features)
        self.labels   = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


# ── Augmentation ─────────────────────────────────────────────────────────────

def augment_clr(x: torch.Tensor) -> torch.Tensor:
    """
    Apply composition-preserving augmentation in CLR space.

    Steps (applied per-sample, differentiable enough via in-place ops):
      1. Aitchison perturbation: add Dirichlet(alpha=0.02) noise in log space
      2. Random OTU zeroing: mask 30% of features to the CLR mean (≈0 after centering)
    """
    B, D = x.shape
    device = x.device

    # 1. Aitchison perturbation via Dirichlet noise in simplex, then CLR-diff
    #    Equivalent to adding a small random compositional vector
    concentration = torch.full((B, D), AUG_DIRICHLET_ALPHA, device=device)
    noise_simplex = torch._standard_gamma(concentration)
    noise_simplex = noise_simplex / noise_simplex.sum(dim=1, keepdim=True)
    # noise in CLR space
    log_noise = torch.log(noise_simplex + 1e-8)
    log_noise = log_noise - log_noise.mean(dim=1, keepdim=True)
    x = x + 0.3 * log_noise  # small perturbation weight

    # 2. OTU subsampling: zero out 30% of OTUs (set to 0, CLR mean ≈ 0)
    mask = torch.rand(B, D, device=device) < AUG_ZERO_FRAC
    x = x.masked_fill(mask, 0.0)

    return x


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = AUG_MIXUP_ALPHA):
    """
    Mixup augmentation. Returns mixed (x, y_a, y_b, lam).
    lam ~ Beta(alpha, alpha).
    """
    lam = float(np.random.beta(alpha, alpha))
    B = x.size(0)
    perm = torch.randperm(B, device=x.device)
    x_mix = lam * x + (1 - lam) * x[perm]
    return x_mix, y, y[perm], lam


# ── Pre-Norm Transformer ──────────────────────────────────────────────────────

class PreNormTransformerLayer(nn.Module):
    """Pre-LayerNorm transformer encoder layer (better training stability)."""

    def __init__(
        self,
        embed_dim: int,
        nheads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim, nheads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff    = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention
        z = self.norm1(x)
        attn_out, _ = self.attn(z, z, z)
        x = x + attn_out
        # Pre-norm feedforward
        x = x + self.ff(self.norm2(x))
        return x


# ── MicroBiomeNetV3 ──────────────────────────────────────────────────────────

class MicroBiomeNetV3(nn.Module):
    """
    Transformer-based microbiome source classifier.

    Architecture:
      CLR features (5000)
      -> Conv1d(1, 384, kernel=5, stride=5) -> 1000 local OTU-group tokens
      -> Prepend [CLS] token -> (1001, 384)
      -> Learnable positional embedding
      -> 8× PreNorm TransformerEncoderLayer(384, 12, 1536, 0.15)
      -> CLS output -> LayerNorm -> Linear(384,256) -> GELU -> Dropout(0.3) -> Linear(256,8)

    The CLS token output is used for classification; the same embedding
    (L2-normalised) feeds the SupCon auxiliary loss.

    ~25M parameters.
    """

    def __init__(
        self,
        input_dim:   int = INPUT_DIM,
        embed_dim:   int = EMBED_DIM,
        nheads:      int = NUM_HEADS,
        num_layers:  int = NUM_LAYERS,
        ff_dim:      int = FF_DIM,
        dropout:     float = DROPOUT,
        num_classes: int = NUM_CLASSES,
        max_seq_len: int = MAX_SEQ_LEN,
    ) -> None:
        super().__init__()

        # 1D conv tokeniser: (B, 1, 5000) -> (B, embed_dim, 1000) -> (B, 1000, embed_dim)
        # kernel=50, stride=50 → 100 tokens (5000/50). Much smaller than 1000 tokens.
        self.tokeniser = nn.Conv1d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=50,
            stride=50,
        )

        # Learnable [CLS] token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder
        self.layers = nn.ModuleList([
            PreNormTransformerLayer(embed_dim, nheads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(embed_dim)

        # Classification head (operates on CLS token)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: CLR-transformed OTU features, shape (B, 5000)

        Returns:
            logits:    (B, num_classes)
            embedding: (B, embed_dim) L2-normalised CLS embedding for SupCon
        """
        B = x.size(0)

        # Tokenise: unsqueeze channel dim -> conv -> permute to (B, T, D)
        tokens = self.tokeniser(x.unsqueeze(1))   # (B, D, 1000)
        tokens = tokens.permute(0, 2, 1)           # (B, 1000, D)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)   # (B, 1, D)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, 1001, D)

        # Add positional embedding
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]

        # Transformer layers
        for layer in self.layers:
            tokens = layer(tokens)

        tokens = self.norm_out(tokens)

        # CLS output (index 0)
        cls_out = tokens[:, 0, :]   # (B, embed_dim)

        logits = self.head(cls_out)

        # L2-normalised embedding for SupCon
        embedding = F.normalize(cls_out, dim=-1)

        return logits, embedding


# ── Focal Loss ───────────────────────────────────────────────────────────────

class FocalCrossEntropyLoss(nn.Module):
    """
    Focal cross-entropy with per-class alpha weights and label smoothing.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha:          Per-class weight tensor, shape (C,).
        gamma:          Focusing parameter (default 2.0).
        label_smoothing: Label smoothing epsilon (default 0.1).
        reduction:      'mean' or 'sum'.
    """

    def __init__(
        self,
        alpha: torch.Tensor,
        gamma: float = FOCAL_GAMMA,
        label_smoothing: float = LABEL_SMOOTHING,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma          = gamma
        self.label_smoothing = label_smoothing
        self.reduction      = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) raw unnormalised scores
            targets: (B,)  integer class indices

        Returns:
            Scalar focal loss.
        """
        C = logits.size(1)

        # Label smoothing: build soft target distribution
        with torch.no_grad():
            smooth_val = self.label_smoothing / (C - 1)
            soft_targets = torch.full_like(logits, smooth_val)
            soft_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        # Log-softmax and softmax probabilities
        log_probs = F.log_softmax(logits, dim=-1)       # (B, C)
        probs     = torch.exp(log_probs)                 # (B, C)

        # p_t: probability of the true class
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)

        # alpha_t: per-sample class weight
        alpha_t = self.alpha[targets]  # (B,)

        # Focal weight
        focal_weight = alpha_t * (1.0 - p_t).pow(self.gamma)  # (B,)

        # Cross-entropy with smooth targets (per-sample, per-class sum)
        ce = -(soft_targets * log_probs).sum(dim=1)  # (B,)

        loss = focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


# ── Supervised Contrastive Loss ───────────────────────────────────────────────

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).

    For a batch of L2-normalised embeddings z ∈ R^(B×D) with labels y:

        L_i = -1/|P(i)| * sum_{p in P(i)} log(
                  exp(z_i · z_p / tau) /
                  sum_{a != i} exp(z_i · z_a / tau)
              )

    where P(i) = set of indices with the same label as i, excluding i itself.

    Args:
        temperature: Softmax temperature tau (default 0.07).
    """

    def __init__(self, temperature: float = SUPCON_TEMP) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: L2-normalised, shape (B, D)
            labels:     Integer class labels, shape (B,)

        Returns:
            Scalar SupCon loss.
        """
        B = embeddings.size(0)
        device = embeddings.device

        # Similarity matrix (B, B)
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature  # (B, B)

        # Mask of same-class pairs (excluding diagonal)
        labels_col = labels.unsqueeze(1)   # (B, 1)
        labels_row = labels.unsqueeze(0)   # (1, B)
        same_class = labels_col.eq(labels_row)  # (B, B) bool

        self_mask   = torch.eye(B, dtype=torch.bool, device=device)
        pos_mask    = same_class & ~self_mask   # positive pairs (excl. self)
        neg_mask    = ~self_mask                # all pairs except self (denominator)

        # For numerical stability: subtract row max before exp
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim        = sim - sim_max.detach()

        exp_sim = torch.exp(sim)  # (B, B)

        # Denominator: sum over all non-self pairs
        denom = (exp_sim * neg_mask.float()).sum(dim=1, keepdim=True)  # (B, 1)
        denom = denom.clamp(min=1e-8)

        # log-probability of each positive pair
        log_prob = sim - torch.log(denom)  # (B, B), broadcast

        # Average over positives per anchor
        num_pos = pos_mask.float().sum(dim=1)  # (B,)

        # Only compute loss for anchors that have at least one positive
        has_pos = num_pos > 0
        if not has_pos.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        per_anchor = -(pos_mask.float() * log_prob).sum(dim=1)  # (B,)
        per_anchor = per_anchor[has_pos] / num_pos[has_pos]

        return per_anchor.mean()


# ── Combined Loss ─────────────────────────────────────────────────────────────

class MicroBiomeV3Loss(nn.Module):
    """Combined focal CE + SupCon loss."""

    def __init__(self, class_counts: torch.Tensor) -> None:
        super().__init__()
        # Inverse-frequency alpha weights, normalised to sum to num_classes
        inv_freq = 1.0 / class_counts.float()
        alpha    = (inv_freq / inv_freq.sum()) * NUM_CLASSES
        self.focal   = FocalCrossEntropyLoss(alpha=alpha)
        self.supcon  = SupConLoss()

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        focal_loss  = self.focal(logits, targets)
        supcon_loss = self.supcon(embeddings, targets)
        total_loss  = (1 - SUPCON_WEIGHT) * focal_loss + SUPCON_WEIGHT * supcon_loss
        return total_loss, focal_loss, supcon_loss


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset() -> OTUDataset:
    """Load EMP 16S individual sample files, apply CLR, return OTUDataset."""
    import io
    logger.info("Loading data from %s", DATA_DIR)
    all_files = sorted(DATA_DIR.glob("*.npz"))
    logger.info("Found %d files", len(all_files))

    features_list, labels_list = [], []
    for f in all_files:
        try:
            with open(f, "rb") as fh:
                raw = fh.read()
            d = np.load(io.BytesIO(raw), allow_pickle=True)
            abundances = d["abundances"].astype(np.float32)
            label = int(d["source_label"])
            if not np.isnan(abundances).any() and abundances.shape[0] == INPUT_DIM:
                features_list.append(abundances)
                labels_list.append(label)
        except Exception:
            continue

    features = np.stack(features_list, axis=0)
    labels   = np.array(labels_list, dtype=np.int64)
    N = features.shape[0]
    logger.info("Loaded %d valid samples, %d OTUs", N, features.shape[1])

    # Apply CLR transform per sample
    clr_features = np.stack([clr_transform(features[i]) for i in range(N)], axis=0)

    return OTUDataset(clr_features, labels)


def get_splits(dataset: OTUDataset):
    """70/15/15 split with seed=42 matching v2."""
    N = len(dataset)
    n_train = int(0.70 * N)
    n_val   = int(0.15 * N)
    n_test  = N - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    logger.info("Split: %d train / %d val / %d test", n_train, n_val, n_test)
    return train_ds, val_ds, test_ds


def build_samplers_and_loaders(train_ds, val_ds, test_ds):
    """Build class-balanced WeightedRandomSampler for training, plain for val/test."""

    # Collect training labels
    train_labels = torch.stack([train_ds.dataset.labels[i] for i in train_ds.indices])
    class_counts = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for lbl in train_labels:
        class_counts[lbl] += 1

    sample_weights = torch.zeros(len(train_ds))
    for i, lbl in enumerate(train_labels):
        sample_weights[i] = 1.0 / class_counts[lbl]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_counts.long()


# ── Mixup for mixed-label loss ────────────────────────────────────────────────

def focal_loss_mixed(
    focal_loss_fn: FocalCrossEntropyLoss,
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute mixup-blended focal loss."""
    return lam * focal_loss_fn(logits, y_a) + (1 - lam) * focal_loss_fn(logits, y_b)


# ── Training step ─────────────────────────────────────────────────────────────

def train_epoch(
    model: MicroBiomeNetV3,
    loader: DataLoader,
    criterion: MicroBiomeV3Loss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        # Augmentation
        x = augment_clr(x)

        # Mixup (50% of batches)
        do_mixup = np.random.rand() < AUG_MIXUP_PROB
        if do_mixup:
            x_mix, y_a, y_b, lam = mixup(x, y)
        else:
            x_mix, y_a, y_b, lam = x, y, y, 1.0

        optimizer.zero_grad()
        with autocast():
            logits, embeddings = model(x_mix)

            if do_mixup:
                # Mixup focal loss; SupCon uses original labels y_a (anchor labels)
                f_loss = focal_loss_mixed(criterion.focal, logits, y_a, y_b, lam)
                sc_loss = criterion.supcon(embeddings, y_a)
                loss = (1 - SUPCON_WEIGHT) * f_loss + SUPCON_WEIGHT * sc_loss
            else:
                loss, _f, _sc = criterion(logits, embeddings, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches  += 1

    scheduler.step()
    return total_loss / max(n_batches, 1)


# ── Evaluation step ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: MicroBiomeNetV3,
    loader: DataLoader,
    criterion: MicroBiomeV3Loss,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Returns (loss, macro_f1, all_preds, all_targets)."""
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    all_preds   = []
    all_targets = []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with autocast():
            logits, embeddings = model(x)
            loss, _, _ = criterion(logits, embeddings, y)

        total_loss += loss.item()
        n_batches  += 1
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y.cpu().numpy())

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    avg_loss    = total_loss / max(n_batches, 1)
    macro_f1    = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1, all_preds, all_targets


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    logger.info("=" * 60)
    logger.info("MicroBiomeNet v3 — Training")
    logger.info("Device: %s", DEVICE)
    logger.info("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset = load_dataset()
    train_ds, val_ds, test_ds = get_splits(dataset)
    train_loader, val_loader, test_loader, class_counts = build_samplers_and_loaders(
        train_ds, val_ds, test_ds
    )

    logger.info(
        "Splits — train: %d | val: %d | test: %d",
        len(train_ds), len(val_ds), len(test_ds),
    )
    logger.info("Class counts (train): %s", class_counts.tolist())

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MicroBiomeNetV3().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f"{n_params:,}")

    # ── Loss, optimiser, scheduler ────────────────────────────────────────────
    criterion  = MicroBiomeV3Loss(class_counts=class_counts.to(DEVICE)).to(DEVICE)
    optimizer  = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-5
    )
    scaler     = GradScaler()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1   = 0.0
    patience_ctr  = 0
    best_epoch    = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler
        )
        val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion)

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_f1=%.4f | "
            "lr=%.2e | %.1fs",
            epoch, EPOCHS, train_loss, val_loss, val_f1, lr_now, elapsed,
        )

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_epoch   = epoch
            patience_ctr = 0
            torch.save(
                {
                    "epoch":      epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1":     val_f1,
                },
                CKPT_PATH,
            )
            logger.info("  -> New best val F1=%.4f saved to %s", val_f1, CKPT_PATH)
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOP_PATIENCE:
                logger.info(
                    "Early stopping at epoch %d (patience=%d, best val F1=%.4f @ epoch %d)",
                    epoch, EARLY_STOP_PATIENCE, best_val_f1, best_epoch,
                )
                break

    logger.info("Training complete. Best val F1=%.4f at epoch %d", best_val_f1, best_epoch)

    # ── Test evaluation ───────────────────────────────────────────────────────
    logger.info("Loading best checkpoint for test evaluation...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_f1, test_preds, test_targets = evaluate(
        model, test_loader, criterion
    )
    test_acc = float(accuracy_score(test_targets, test_preds))

    report = classification_report(
        test_targets, test_preds,
        target_names=SOURCE_NAMES,
        output_dict=True,
        zero_division=0,
    )

    per_class_f1 = {
        SOURCE_NAMES[i]: float(report[SOURCE_NAMES[i]]["f1-score"])
        for i in range(NUM_CLASSES)
        if SOURCE_NAMES[i] in report
    }

    logger.info("=" * 60)
    logger.info("TEST RESULTS")
    logger.info("  Macro F1 : %.4f", test_f1)
    logger.info("  Accuracy : %.4f", test_acc)
    for cls_name, f1 in per_class_f1.items():
        logger.info("  %-30s  F1=%.4f", cls_name, f1)
    logger.info("=" * 60)

    # ── Save results JSON ─────────────────────────────────────────────────────
    results = {
        "model":          "MicroBiomeNet_v3",
        "test_f1_macro":  float(test_f1),
        "test_acc":       float(test_acc),
        "per_class_f1":   per_class_f1,
        "n_train":        17980,
        "n_val":          3852,
        "n_test":         3854,
        "best_epoch":     best_epoch,
        "best_val_f1":    float(best_val_f1),
        "n_params":       n_params,
        "embed_dim":      EMBED_DIM,
        "num_layers":     NUM_LAYERS,
        "num_heads":      NUM_HEADS,
        "epochs_run":     epoch,
        "batch_size":     BATCH_SIZE,
        "lr":             LR,
        "focal_gamma":    FOCAL_GAMMA,
        "label_smoothing": LABEL_SMOOTHING,
        "supcon_weight":  SUPCON_WEIGHT,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", RESULTS_PATH)

    print(f"MicroBiomeNet v3 TEST F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()
