"""
Cascade Escalation Controller training pipeline for SENTINEL.

Trains a PPO policy via stable-baselines3 with curriculum learning on
simulated contamination events, then extracts an interpretable decision
tree protocol.

3 curriculum phases:
  Phase 1: Easy events (low difficulty, high event ratio)
  Phase 2: Mixed difficulty
  Phase 3: Hard events (high difficulty, realistic event ratio)

After training: extracts decision tree (max depth 6), saves protocol.txt,
feature_importances.json, and tree model.

Usage:
    python -m sentinel.training.train_escalation
    python -m sentinel.training.train_escalation --total-timesteps 1000000 --output-dir outputs/escalation/long_run
    python -m sentinel.training.train_escalation --load-agent outputs/escalation/checkpoints/agent.zip --extract-only
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from sentinel.models.escalation.curriculum import (
    CurriculumCallback,
    CurriculumPhase,
    CurriculumScheduler,
)
from sentinel.models.escalation.decision_tree import (
    ExtractionResult,
    collect_policy_dataset,
    extract_decision_tree,
    format_protocol,
    save_protocol,
)
from sentinel.models.escalation.environment import (
    CascadeEscalationEnv,
    STATE_DIM,
    NUM_TIERS,
)
from sentinel.models.escalation.model import CascadeEscalationController
from sentinel.models.escalation.policy import (
    DEFAULT_PPO_HYPERPARAMS,
    create_ppo_agent,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EscalationTrainConfig:
    """Configuration for escalation policy training."""

    # PPO
    total_timesteps: int = 500_000
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    gae_lambda: float = 0.95
    gamma: float = 0.99
    lr: float = 3e-4

    # Curriculum
    phase1_difficulty: tuple = (0.0, 0.3)
    phase1_event_ratio: float = 0.7
    phase1_fraction: float = 0.3
    phase2_difficulty: tuple = (0.2, 0.7)
    phase2_event_ratio: float = 0.5
    phase2_fraction: float = 0.4
    phase3_difficulty: tuple = (0.5, 1.0)
    phase3_event_ratio: float = 0.5
    phase3_fraction: float = 0.3

    # Protocol extraction
    extract_n_episodes: int = 500
    extract_max_depth: int = 6

    # Infrastructure
    output_dir: str = "outputs/escalation"
    device: str = "auto"
    seed: int = 42
    log_interval: int = 10
    use_wandb: bool = True
    wandb_project: str = "sentinel"
    wandb_run_name: str = "escalation-ppo"
    progress_bar: bool = True


# ---------------------------------------------------------------------------
# Curriculum phases builder
# ---------------------------------------------------------------------------


def build_curriculum_phases(config: EscalationTrainConfig) -> List[CurriculumPhase]:
    """Build the 3-phase curriculum from config."""
    return [
        CurriculumPhase(
            name="easy_events",
            difficulty_range=config.phase1_difficulty,
            event_ratio=config.phase1_event_ratio,
            duration_fraction=config.phase1_fraction,
        ),
        CurriculumPhase(
            name="mixed_difficulty",
            difficulty_range=config.phase2_difficulty,
            event_ratio=config.phase2_event_ratio,
            duration_fraction=config.phase2_fraction,
        ),
        CurriculumPhase(
            name="hard_events",
            difficulty_range=config.phase3_difficulty,
            event_ratio=config.phase3_event_ratio,
            duration_fraction=config.phase3_fraction,
        ),
    ]


# ---------------------------------------------------------------------------
# Wandb callback
# ---------------------------------------------------------------------------


def _build_wandb_callback(config: EscalationTrainConfig):
    """Build a wandb logging callback if wandb is available."""
    if not config.use_wandb:
        return None
    try:
        from stable_baselines3.common.callbacks import BaseCallback
        import wandb

        class WandbCallback(BaseCallback):
            """Log SB3 training metrics to wandb."""

            def __init__(self, verbose: int = 0) -> None:
                super().__init__(verbose)
                self._run = None

            def _on_training_start(self) -> None:
                self._run = wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    config={
                        "total_timesteps": config.total_timesteps,
                        "lr": config.lr,
                        "n_steps": config.n_steps,
                        "batch_size": config.batch_size,
                        "seed": config.seed,
                    },
                    reinit=True,
                )

            def _on_step(self) -> bool:
                if self.n_calls % config.log_interval == 0 and self.locals.get("infos"):
                    infos = self.locals["infos"]
                    for info in infos:
                        if "episode" in info:
                            wandb.log({
                                "episode/reward": info["episode"]["r"],
                                "episode/length": info["episode"]["l"],
                                "timestep": self.num_timesteps,
                            })
                return True

            def _on_training_end(self) -> None:
                if self._run is not None:
                    self._run.finish()

        return WandbCallback()
    except ImportError:
        logger.warning("wandb or stable_baselines3 not available; skipping wandb callback")
        return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_agent(
    controller: CascadeEscalationController,
    n_episodes: int = 100,
    seed: int = 12345,
) -> Dict[str, float]:
    """Evaluate the trained agent on unseen episodes.

    Returns dict with mean_reward, mean_length, detection_rate,
    false_alarm_rate, mean_tier, cost_efficiency.
    """
    eval_env = CascadeEscalationEnv(seed=seed)
    rewards = []
    lengths = []
    detections = 0
    false_alarms = 0
    total_events = 0
    total_no_events = 0
    tier_sum = 0
    tier_count = 0

    for ep in range(n_episodes):
        obs, info = eval_env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0
        has_event = info.get("has_event", False)

        if has_event:
            total_events += 1
        else:
            total_no_events += 1

        detected = False

        while not done:
            action, _value = controller.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_length += 1
            tier_sum += action
            tier_count += 1

            if info.get("detected", False):
                detected = True

        rewards.append(ep_reward)
        lengths.append(ep_length)

        if has_event and detected:
            detections += 1
        if not has_event and detected:
            false_alarms += 1

    metrics = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "detection_rate": detections / max(total_events, 1),
        "false_alarm_rate": false_alarms / max(total_no_events, 1),
        "mean_tier": tier_sum / max(tier_count, 1),
    }
    return metrics


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_escalation(config: EscalationTrainConfig) -> Dict[str, Any]:
    """Run the full escalation training pipeline.

    1. Build environment and PPO agent with curriculum.
    2. Train for total_timesteps.
    3. Evaluate.
    4. Extract interpretable decision tree.
    5. Save all artifacts.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build curriculum phases
    phases = build_curriculum_phases(config)

    # Build PPO overrides from config
    ppo_kwargs = {
        "learning_rate": config.lr,
        "n_steps": config.n_steps,
        "batch_size": config.batch_size,
        "n_epochs": config.n_epochs,
        "clip_range": config.clip_range,
        "ent_coef": config.ent_coef,
        "vf_coef": config.vf_coef,
        "max_grad_norm": config.max_grad_norm,
        "gae_lambda": config.gae_lambda,
        "gamma": config.gamma,
    }

    # Create controller
    controller = CascadeEscalationController(
        seed=config.seed,
        device=config.device,
        curriculum_phases=phases,
        ppo_kwargs=ppo_kwargs,
    )

    # Build callbacks
    extra_callbacks = []
    wandb_cb = _build_wandb_callback(config)
    if wandb_cb is not None:
        extra_callbacks.append(wandb_cb)

    # Train
    logger.info(
        f"Training escalation policy: {config.total_timesteps} timesteps, "
        f"{len(phases)} curriculum phases"
    )
    train_summary = controller.train(
        total_timesteps=config.total_timesteps,
        log_interval=config.log_interval,
        progress_bar=config.progress_bar,
        extra_callbacks=extra_callbacks,
    )

    # Save agent
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    agent_path = controller.save(ckpt_dir / "agent")
    logger.info(f"Agent saved to {agent_path}")

    # Evaluate
    logger.info("Evaluating trained agent...")
    eval_metrics = evaluate_agent(controller, n_episodes=200, seed=config.seed + 999)
    logger.info(f"Evaluation: {json.dumps(eval_metrics, indent=2)}")

    eval_path = output_dir / "evaluation_metrics.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)

    # Extract interpretable decision tree
    logger.info(
        f"Extracting decision tree (depth={config.extract_max_depth}, "
        f"episodes={config.extract_n_episodes})..."
    )
    extraction_result = controller.extract_protocol(
        n_episodes=config.extract_n_episodes,
        max_depth=config.extract_max_depth,
        deterministic=True,
        seed=config.seed + 2000,
    )

    # Save protocol artifacts
    protocol_dir = output_dir / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = controller.save_protocol(protocol_dir)
    logger.info(f"Protocol artifacts saved to {protocol_dir}")

    # Print protocol summary
    protocol_text = controller.print_protocol()
    logger.info(f"\n{'='*60}\nExtracted Protocol:\n{'='*60}\n{protocol_text}")

    # Log feature importances
    if extraction_result.feature_importances is not None:
        fi = extraction_result.feature_importances
        top_features = sorted(
            fi.items(), key=lambda x: x[1], reverse=True,
        )[:10]
        logger.info("Top 10 feature importances:")
        for name, importance in top_features:
            logger.info(f"  {name}: {importance:.4f}")

    # Compile full summary
    summary = {
        "training": train_summary,
        "evaluation": eval_metrics,
        "extraction": {
            "accuracy": extraction_result.accuracy,
            "tree_depth": extraction_result.tree.get_depth(),
            "tree_leaves": extraction_result.tree.get_n_leaves(),
        },
        "artifacts": {k: str(v) for k, v in saved_paths.items()},
    }

    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Full summary saved to {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# Extract-only mode
# ---------------------------------------------------------------------------


def extract_only(
    config: EscalationTrainConfig,
    agent_path: str,
) -> Dict[str, Any]:
    """Load a trained agent and extract protocol without retraining."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    controller = CascadeEscalationController(
        seed=config.seed,
        device=config.device,
    )
    controller.load(agent_path)
    logger.info(f"Loaded agent from {agent_path}")

    # Evaluate
    eval_metrics = evaluate_agent(controller, n_episodes=200, seed=config.seed + 999)
    logger.info(f"Evaluation: {json.dumps(eval_metrics, indent=2)}")

    # Extract protocol
    extraction_result = controller.extract_protocol(
        n_episodes=config.extract_n_episodes,
        max_depth=config.extract_max_depth,
        deterministic=True,
        seed=config.seed + 2000,
    )

    protocol_dir = output_dir / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = controller.save_protocol(protocol_dir)

    protocol_text = controller.print_protocol()
    logger.info(f"\n{'='*60}\nExtracted Protocol:\n{'='*60}\n{protocol_text}")

    return {
        "evaluation": eval_metrics,
        "extraction": {
            "accuracy": extraction_result.accuracy,
            "tree_depth": extraction_result.tree.get_depth(),
            "tree_leaves": extraction_result.tree.get_n_leaves(),
        },
        "artifacts": {k: str(v) for k, v in saved_paths.items()},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SENTINEL Cascade Escalation Controller training",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/escalation")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    # Curriculum
    parser.add_argument("--phase1-fraction", type=float, default=0.3)
    parser.add_argument("--phase2-fraction", type=float, default=0.4)
    parser.add_argument("--phase3-fraction", type=float, default=0.3)

    # Extraction
    parser.add_argument("--extract-episodes", type=int, default=500)
    parser.add_argument("--extract-max-depth", type=int, default=6)
    parser.add_argument("--extract-only", action="store_true",
                        help="Skip training, extract protocol from saved agent")
    parser.add_argument("--load-agent", type=str, default="",
                        help="Path to saved agent (for --extract-only)")

    # Logging
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="sentinel")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--no-progress", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = EscalationTrainConfig(
        output_dir=args.output_dir,
        total_timesteps=args.total_timesteps,
        lr=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        phase1_fraction=args.phase1_fraction,
        phase2_fraction=args.phase2_fraction,
        phase3_fraction=args.phase3_fraction,
        extract_n_episodes=args.extract_episodes,
        extract_max_depth=args.extract_max_depth,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        log_interval=args.log_interval,
        progress_bar=not args.no_progress,
    )

    if args.extract_only:
        if not args.load_agent:
            raise ValueError("--extract-only requires --load-agent")
        summary = extract_only(config, args.load_agent)
    else:
        summary = train_escalation(config)

    logger.info(f"Done. Summary:\n{json.dumps(summary, indent=2, default=str)}")


if __name__ == "__main__":
    main()
