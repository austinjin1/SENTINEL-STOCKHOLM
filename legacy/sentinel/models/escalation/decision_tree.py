"""Interpretable decision-tree extraction from a trained escalation policy.

After training the neural policy via PPO, this module distils the learned
behaviour into a shallow decision tree (max depth 6) that can be printed,
visualised, and deployed on resource-constrained systems without a GPU or
neural-network runtime.

The resulting tree serves as the *SENTINEL Simplified Escalation Protocol*
— a transparent rule set that water-quality agencies can audit, modify,
and deploy independently.
"""

from __future__ import annotations

import json
import logging
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, export_text

from sentinel.models.escalation.environment import (
    FUSED_DIM,
    NUM_ACTIONS,
    NUM_MODALITIES,
    NUM_TIERS,
    STATE_DIM,
)

logger = logging.getLogger(__name__)

# Human-readable feature names for the observation vector
FEATURE_NAMES: List[str] = (
    [f"fused_{i:03d}" for i in range(FUSED_DIM)]
    + ["anomaly_sensor", "anomaly_satellite", "anomaly_microbial", "anomaly_molecular"]
    + ["tier_0", "tier_1", "tier_2", "tier_3"]
    + ["time_since_escalation", "historical_event_rate"]
)

ACTION_NAMES: List[str] = [
    "maintain",
    "escalate_+1",
    "escalate_+2",
    "de-escalate_-1",
]


# ---------------------------------------------------------------------------
# Dataset collection
# ---------------------------------------------------------------------------

@dataclass
class PolicyDataset:
    """State-action pairs collected by rolling out a trained policy.

    Attributes:
        states: Array of shape ``(N, STATE_DIM)``.
        actions: Array of shape ``(N,)`` with integer action labels.
        values: Array of shape ``(N,)`` with value estimates (optional).
    """

    states: np.ndarray
    actions: np.ndarray
    values: Optional[np.ndarray] = None


def collect_policy_dataset(
    model: Any,
    env: Any,
    n_episodes: int = 500,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> PolicyDataset:
    """Roll out *model* in *env* and record (state, action) pairs.

    Parameters
    ----------
    model : stable_baselines3.PPO or any object with ``predict(obs)``
        Trained policy.
    env : gymnasium.Env
        Environment instance (will be ``reset()`` repeatedly).
    n_episodes : int
        Number of episodes to collect.
    deterministic : bool
        If *True*, use the greedy (argmax) policy.
    seed : int, optional
        Random seed for the environment.

    Returns
    -------
    PolicyDataset
    """
    all_states: List[np.ndarray] = []
    all_actions: List[int] = []
    all_values: List[float] = []

    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        done = False
        while not done:
            action, _info = model.predict(obs, deterministic=deterministic)
            action = int(action)

            all_states.append(obs.copy())
            all_actions.append(action)

            # Optionally record value estimate
            if hasattr(model, "policy") and hasattr(model.policy, "predict_values"):
                import torch

                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                obs_t = obs_t.to(model.device)
                val = model.policy.predict_values(obs_t)
                all_values.append(float(val.item()))

            obs, _reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated

    states = np.array(all_states, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.int64)
    values = np.array(all_values, dtype=np.float32) if all_values else None

    logger.info(
        "Collected %d state-action pairs over %d episodes.", len(actions), n_episodes,
    )
    return PolicyDataset(states=states, actions=actions, values=values)


# ---------------------------------------------------------------------------
# Decision tree fitting
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """Result of extracting an interpretable tree from the neural policy.

    Attributes:
        tree: Fitted scikit-learn decision tree.
        accuracy: Agreement rate with the neural policy on training data.
        report: Scikit-learn classification report (per-action metrics).
        important_features: Top-k feature importances.
        text_rules: Human-readable text representation of the tree.
    """

    tree: DecisionTreeClassifier
    accuracy: float
    report: str
    important_features: List[Tuple[str, float]]
    text_rules: str


def extract_decision_tree(
    dataset: PolicyDataset,
    max_depth: int = 6,
    min_samples_leaf: int = 50,
    test_fraction: float = 0.2,
    seed: int = 42,
    top_k_features: int = 15,
) -> ExtractionResult:
    """Fit an interpretable decision tree to approximate the neural policy.

    Parameters
    ----------
    dataset : PolicyDataset
        Collected rollout data.
    max_depth : int
        Maximum tree depth (shallow = more interpretable).
    min_samples_leaf : int
        Minimum samples per leaf to prevent overfitting.
    test_fraction : float
        Held-out fraction for reporting accuracy.
    seed : int
        Random seed for train/test split and tree fitting.
    top_k_features : int
        Number of top features to report by importance.

    Returns
    -------
    ExtractionResult
    """
    rng = np.random.default_rng(seed)
    N = len(dataset.actions)
    indices = rng.permutation(N)
    split = int(N * (1.0 - test_fraction))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train = dataset.states[train_idx]
    y_train = dataset.actions[train_idx]
    X_test = dataset.states[test_idx]
    y_test = dataset.actions[test_idx]

    # Fit tree
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        class_weight="balanced",
    )
    tree.fit(X_train, y_train)

    # Evaluate
    y_pred_train = tree.predict(X_train)
    y_pred_test = tree.predict(X_test)

    train_acc = float(accuracy_score(y_train, y_pred_train))
    test_acc = float(accuracy_score(y_test, y_pred_test))

    report = classification_report(
        y_test, y_pred_test,
        target_names=ACTION_NAMES,
        zero_division=0,
    )

    logger.info(
        "Decision tree extracted: train_acc=%.3f  test_acc=%.3f  "
        "depth=%d  leaves=%d",
        train_acc, test_acc, tree.get_depth(), tree.get_n_leaves(),
    )

    # Feature importances
    importances = tree.feature_importances_
    feat_names = FEATURE_NAMES if len(FEATURE_NAMES) == dataset.states.shape[1] else [
        f"feat_{i}" for i in range(dataset.states.shape[1])
    ]
    ranked = sorted(
        zip(feat_names, importances), key=lambda x: x[1], reverse=True,
    )
    top_features = ranked[:top_k_features]

    # Text rules
    text_rules = export_text(
        tree,
        feature_names=feat_names,
        max_depth=max_depth,
    )

    return ExtractionResult(
        tree=tree,
        accuracy=test_acc,
        report=report,
        important_features=top_features,
        text_rules=text_rules,
    )


# ---------------------------------------------------------------------------
# Protocol export
# ---------------------------------------------------------------------------

def format_protocol(result: ExtractionResult) -> str:
    """Format the extraction result as a deployable escalation protocol.

    Returns a human-readable text document suitable for printing or
    inclusion in operational documentation.
    """
    lines = [
        "=" * 70,
        "SENTINEL SIMPLIFIED ESCALATION PROTOCOL",
        "Extracted from trained neural policy via decision-tree distillation",
        "=" * 70,
        "",
        f"Tree depth:          {result.tree.get_depth()}",
        f"Number of leaves:    {result.tree.get_n_leaves()}",
        f"Policy agreement:    {result.accuracy:.1%}",
        "",
        "--- Top features driving escalation decisions ---",
    ]

    for feat_name, importance in result.important_features:
        lines.append(f"  {feat_name:.<40s} {importance:.4f}")

    lines += [
        "",
        "--- Per-action classification report ---",
        result.report,
        "",
        "--- Decision rules ---",
        result.text_rules,
        "",
        "=" * 70,
        "END PROTOCOL",
        "=" * 70,
    ]
    return "\n".join(lines)


def save_protocol(
    result: ExtractionResult,
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    """Save the protocol and fitted tree to disk.

    Writes:
        ``protocol.txt``   — human-readable protocol
        ``tree_model.json`` — serialised tree parameters
        ``feature_importances.json`` — ranked feature importances

    Returns
    -------
    dict mapping file type to absolute Path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Protocol text
    protocol_path = out / "protocol.txt"
    protocol_path.write_text(format_protocol(result), encoding="utf-8")

    # Feature importances
    fi_path = out / "feature_importances.json"
    fi_data = [
        {"feature": name, "importance": float(imp)}
        for name, imp in result.important_features
    ]
    fi_path.write_text(json.dumps(fi_data, indent=2), encoding="utf-8")

    # Tree parameters (sklearn export)
    import joblib

    tree_path = out / "tree_model.joblib"
    joblib.dump(result.tree, tree_path)

    paths = {
        "protocol": protocol_path.resolve(),
        "importances": fi_path.resolve(),
        "tree_model": tree_path.resolve(),
    }
    logger.info("Saved protocol artefacts to %s", out.resolve())
    return paths
