"""Post-hoc calibration module for the Digital Biosentinel.

Implements temperature scaling for probability calibration, reliability
diagram plotting, and the Expected Calibration Error (ECE) metric.

After training the dose-response model, predicted probabilities (e.g.
P(mortality)) may be systematically over- or under-confident.  Temperature
scaling learns a single scalar T > 0 such that ``sigmoid(logit / T)``
is well-calibrated on a held-out validation set.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class TemperatureScaler(nn.Module):
    """Learnable temperature-scaling module.

    Wraps a single positive temperature parameter that is optimised
    on a held-out calibration set *after* the main model is fully
    trained.  The base model's weights are **not** touched.

    Parameters
    ----------
    initial_temperature : float
        Starting value for *T*.  1.0 means no change to logits.
    """

    def __init__(self, initial_temperature: float = 1.5) -> None:
        super().__init__()
        # Store log(T) so that T = exp(log_T) is always positive.
        self.log_temperature = nn.Parameter(
            torch.tensor(initial_temperature).log()
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature.

        Parameters
        ----------
        logits : Tensor[B] or Tensor[B, C]
            Raw (pre-sigmoid / pre-softmax) logits.

        Returns
        -------
        Tensor
            Scaled logits: ``logits / T``.
        """
        return logits / self.temperature

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 200,
        verbose: bool = False,
    ) -> float:
        """Fit the temperature on a held-out calibration set.

        Uses L-BFGS to minimise the negative log-likelihood (NLL) of
        the calibration labels under the temperature-scaled predictions.

        Parameters
        ----------
        logits : Tensor[N]
            Pre-sigmoid logits from the model (gathered on calibration set).
        labels : Tensor[N]
            Binary ground-truth labels (0 or 1).
        lr : float
            Learning rate for L-BFGS.
        max_iter : int
            Maximum L-BFGS iterations.
        verbose : bool
            Print loss at each iteration.

        Returns
        -------
        float
            Final temperature value.
        """
        logits = logits.detach().float()
        labels = labels.detach().float()

        optimizer = torch.optim.LBFGS(
            [self.log_temperature], lr=lr, max_iter=max_iter,
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled = self.forward(logits)
            loss = F.binary_cross_entropy_with_logits(scaled, labels)
            loss.backward()
            if verbose:
                print(
                    f"  T={self.temperature.item():.4f}  "
                    f"NLL={loss.item():.6f}"
                )
            return loss

        optimizer.step(closure)
        return self.temperature.item()

    def predict_calibrated(self, logits: torch.Tensor) -> torch.Tensor:
        """Return calibrated probabilities (sigmoid of scaled logits)."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(logits))


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------

def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> Tuple[float, List[float], List[float], List[int]]:
    """Compute the Expected Calibration Error (ECE).

    ECE = sum_b (|B_b| / N) * |accuracy(B_b) - confidence(B_b)|

    Parameters
    ----------
    probs : Tensor[N]
        Predicted probabilities in [0, 1].
    labels : Tensor[N]
        Binary ground-truth labels.
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    ece : float
        The ECE value.
    bin_accuracies : List[float]
        Per-bin observed accuracy.
    bin_confidences : List[float]
        Per-bin mean predicted confidence.
    bin_counts : List[int]
        Number of samples per bin.
    """
    probs = probs.detach().cpu().float()
    labels = labels.detach().cpu().float()

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
    bin_accuracies: List[float] = []
    bin_confidences: List[float] = []
    bin_counts: List[int] = []
    ece = 0.0
    N = len(probs)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i].item(), bin_boundaries[i + 1].item()
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)

        count = mask.sum().item()
        bin_counts.append(int(count))

        if count == 0:
            bin_accuracies.append(0.0)
            bin_confidences.append((lo + hi) / 2.0)
            continue

        bin_acc = labels[mask].mean().item()
        bin_conf = probs[mask].mean().item()
        bin_accuracies.append(bin_acc)
        bin_confidences.append(bin_conf)
        ece += (count / N) * abs(bin_acc - bin_conf)

    return ece, bin_accuracies, bin_confidences, bin_counts


# ---------------------------------------------------------------------------
# Reliability Diagram
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None,
) -> Optional[object]:
    """Plot a reliability (calibration) diagram.

    Parameters
    ----------
    probs : Tensor[N]
        Predicted probabilities.
    labels : Tensor[N]
        Binary ground-truth labels.
    n_bins : int
        Number of bins.
    title : str
        Plot title.
    save_path : str, optional
        If provided, save the figure to this path (PNG).

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure object, or None if matplotlib is unavailable.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting reliability diagrams. "
            "Install with: pip install matplotlib"
        )

    ece, bin_accs, bin_confs, bin_counts = expected_calibration_error(
        probs, labels, n_bins=n_bins,
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]},
    )

    # --- Top: reliability diagram ---
    bin_width = 1.0 / n_bins
    bin_centers = [(i + 0.5) * bin_width for i in range(n_bins)]

    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    # Gap bars (miscalibration)
    for bc, ba, bconf in zip(bin_centers, bin_accs, bin_confs):
        color = "#4CAF50" if abs(ba - bconf) < 0.05 else "#FF5722"
        ax1.bar(
            bc, ba, width=bin_width * 0.8, color=color,
            edgecolor="black", linewidth=0.5, alpha=0.8,
        )

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Observed Frequency (Accuracy)")
    ax1.set_title(f"{title}  (ECE = {ece:.4f})")
    ax1.legend(loc="upper left")

    # --- Bottom: histogram of predictions ---
    ax2.bar(
        bin_centers, bin_counts, width=bin_width * 0.8,
        color="#2196F3", edgecolor="black", linewidth=0.5, alpha=0.7,
    )
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Distribution")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
