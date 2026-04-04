"""Complete Cascade Escalation Controller for SENTINEL.

This module provides the top-level :class:`CascadeEscalationController`
that orchestrates training (PPO with curriculum learning), inference, and
interpretable protocol extraction.  It is the single entry point that
downstream SENTINEL components should use.

Example usage::

    from sentinel.models.escalation.model import CascadeEscalationController

    controller = CascadeEscalationController(seed=42)
    controller.train(total_timesteps=500_000)

    action, value = controller.predict(state_vector)
    protocol = controller.extract_protocol(n_episodes=500)
    tier_cfg = controller.get_tier_config(2)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from sentinel.models.escalation.curriculum import (
    CurriculumCallback,
    CurriculumPhase,
    CurriculumScheduler,
)
from sentinel.models.escalation.decision_tree import (
    ExtractionResult,
    PolicyDataset,
    collect_policy_dataset,
    extract_decision_tree,
    format_protocol,
    save_protocol,
)
from sentinel.models.escalation.environment import (
    MODALITY_IDS,
    NUM_ACTIONS,
    NUM_MODALITIES,
    NUM_TIERS,
    STATE_DIM,
    TIER_COMPUTE_COST,
    TIER_MODALITIES,
    CascadeEscalationEnv,
)
from sentinel.models.escalation.policy import (
    DEFAULT_PPO_HYPERPARAMS,
    EscalationPolicyNetwork,
    create_ppo_agent,
)

logger = logging.getLogger(__name__)


class CascadeEscalationController:
    """Top-level cascade escalation controller.

    Wraps the PPO policy, training environment, curriculum scheduler, and
    decision-tree extraction into a unified interface.

    Parameters
    ----------
    seed : int or None
        Global random seed for reproducibility.
    device : str
        Torch device (``"auto"``, ``"cpu"``, or ``"cuda"``).
    curriculum_phases : list[CurriculumPhase] or None
        Custom curriculum phases.  ``None`` uses the default 3-phase
        curriculum.
    ppo_kwargs : dict or None
        Overrides for :data:`DEFAULT_PPO_HYPERPARAMS`.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        device: str = "auto",
        curriculum_phases: Optional[List[CurriculumPhase]] = None,
        ppo_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.seed = seed
        self.device = device
        self._curriculum_phases = curriculum_phases
        self._ppo_overrides = ppo_kwargs or {}

        # Lazily initialised
        self._env: Optional[CascadeEscalationEnv] = None
        self._agent: Optional[Any] = None  # PPO
        self._scheduler: Optional[CurriculumScheduler] = None
        self._extraction_result: Optional[ExtractionResult] = None
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """Whether :meth:`train` has been called successfully."""
        return self._is_trained

    @property
    def env(self) -> CascadeEscalationEnv:
        """The training environment (created on first access)."""
        if self._env is None:
            self._env = CascadeEscalationEnv(seed=self.seed)
        return self._env

    @property
    def agent(self) -> Any:
        """The SB3 PPO agent (created on first access)."""
        if self._agent is None:
            self._agent = create_ppo_agent(
                env=self.env,
                seed=self.seed,
                device=self.device,
                **self._ppo_overrides,
            )
        return self._agent

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        total_timesteps: int = 500_000,
        log_interval: int = 10,
        progress_bar: bool = False,
        extra_callbacks: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Train the escalation policy with curriculum learning.

        Parameters
        ----------
        total_timesteps : int
            Total environment steps for training.
        log_interval : int
            How often (in rollouts) to log training stats.
        progress_bar : bool
            Whether to display a tqdm progress bar.
        extra_callbacks : list, optional
            Additional SB3 callbacks to attach.

        Returns
        -------
        dict
            Training summary with keys ``total_timesteps``, ``phases``,
            and ``final_reward_mean``.
        """
        # Build curriculum scheduler
        self._scheduler = CurriculumScheduler(
            total_timesteps=total_timesteps,
            phases=self._curriculum_phases,
        )
        curriculum_cb = CurriculumCallback(
            scheduler=self._scheduler,
            log_interval=log_interval,
            verbose=1,
        )

        callbacks = [curriculum_cb]
        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        logger.info(
            "Starting PPO training: %d timesteps, %d curriculum phases",
            total_timesteps,
            len(self._scheduler.phases),
        )

        agent = self.agent
        agent.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=log_interval,
            progress_bar=progress_bar,
        )

        self._is_trained = True

        summary = {
            "total_timesteps": total_timesteps,
            "phases": [p.name for p in self._scheduler.phases],
        }
        logger.info("Training complete. Summary: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        state: Union[np.ndarray, torch.Tensor],
        deterministic: bool = True,
    ) -> Tuple[int, float]:
        """Predict the escalation action for a single state.

        Parameters
        ----------
        state : ndarray or Tensor of shape ``(STATE_DIM,)``
            Current observation vector.
        deterministic : bool
            If *True*, use the greedy policy; otherwise sample.

        Returns
        -------
        action : int
            Chosen escalation action (0-3).
        value : float
            Estimated state value from the critic.

        Raises
        ------
        RuntimeError
            If the controller has not been trained or loaded.
        """
        if self._agent is None:
            raise RuntimeError(
                "No trained agent available. Call train() or load() first."
            )

        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()

        state = np.asarray(state, dtype=np.float32)
        if state.ndim == 1:
            state = state.reshape(1, -1)

        action, _states = self._agent.predict(state, deterministic=deterministic)
        action = int(action)

        # Get value estimate
        obs_tensor = torch.as_tensor(state, dtype=torch.float32).to(self._agent.device)
        with torch.no_grad():
            value = self._agent.policy.predict_values(obs_tensor)
        value = float(value.item())

        return action, value

    def predict_batch(
        self,
        states: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict actions for a batch of states.

        Parameters
        ----------
        states : ndarray of shape ``(B, STATE_DIM)``
        deterministic : bool

        Returns
        -------
        actions : ndarray[B] of int
        values  : ndarray[B] of float
        """
        if self._agent is None:
            raise RuntimeError(
                "No trained agent available. Call train() or load() first."
            )

        states = np.asarray(states, dtype=np.float32)
        actions, _states = self._agent.predict(states, deterministic=deterministic)

        obs_tensor = torch.as_tensor(states, dtype=torch.float32).to(self._agent.device)
        with torch.no_grad():
            values = self._agent.policy.predict_values(obs_tensor)
        values = values.cpu().numpy().flatten()

        return np.asarray(actions), values

    # ------------------------------------------------------------------
    # Protocol extraction
    # ------------------------------------------------------------------

    def extract_protocol(
        self,
        n_episodes: int = 500,
        max_depth: int = 6,
        deterministic: bool = True,
        seed: Optional[int] = None,
    ) -> ExtractionResult:
        """Extract an interpretable decision-tree escalation protocol.

        Rolls out the trained neural policy, collects state-action pairs,
        and fits a shallow decision tree to approximate the policy.

        Parameters
        ----------
        n_episodes : int
            Number of rollout episodes for data collection.
        max_depth : int
            Maximum depth of the fitted decision tree.
        deterministic : bool
            Use greedy policy during data collection.
        seed : int, optional
            Random seed for data collection and tree fitting.

        Returns
        -------
        ExtractionResult
            Contains the fitted tree, accuracy metrics, feature
            importances, and human-readable rules.
        """
        if not self._is_trained and self._agent is None:
            raise RuntimeError(
                "No trained agent available. Call train() or load() first."
            )

        eval_seed = seed if seed is not None else (self.seed or 0) + 1000
        eval_env = CascadeEscalationEnv(seed=eval_seed)

        logger.info(
            "Collecting %d episodes for decision-tree extraction...", n_episodes,
        )
        dataset = collect_policy_dataset(
            model=self._agent,
            env=eval_env,
            n_episodes=n_episodes,
            deterministic=deterministic,
            seed=eval_seed,
        )

        result = extract_decision_tree(
            dataset=dataset,
            max_depth=max_depth,
            seed=eval_seed,
        )

        self._extraction_result = result
        logger.info(
            "Protocol extracted: accuracy=%.1f%%, depth=%d, leaves=%d",
            result.accuracy * 100,
            result.tree.get_depth(),
            result.tree.get_n_leaves(),
        )
        return result

    def print_protocol(self) -> str:
        """Return the formatted protocol text.

        Raises :class:`RuntimeError` if :meth:`extract_protocol` has not
        been called.
        """
        if self._extraction_result is None:
            raise RuntimeError("Call extract_protocol() first.")
        return format_protocol(self._extraction_result)

    def save_protocol(self, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """Save protocol artefacts to *output_dir*.

        Writes ``protocol.txt``, ``tree_model.joblib``, and
        ``feature_importances.json``.
        """
        if self._extraction_result is None:
            raise RuntimeError("Call extract_protocol() first.")
        return save_protocol(self._extraction_result, output_dir)

    # ------------------------------------------------------------------
    # Tier configuration (static helper)
    # ------------------------------------------------------------------

    @staticmethod
    def get_tier_config(tier: int) -> Dict[str, Any]:
        """Return the modality configuration for a monitoring tier.

        Parameters
        ----------
        tier : int
            Tier index in ``[0, 3]``.

        Returns
        -------
        dict with keys ``tier``, ``modalities``, ``compute_cost``,
        ``description``.
        """
        return CascadeEscalationEnv.get_tier_config(tier)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> Path:
        """Save the trained PPO agent to *path*.

        The agent is saved using SB3's built-in serialisation, which
        stores the policy weights, optimizer state, and replay buffer.

        Returns the resolved save path.
        """
        if self._agent is None:
            raise RuntimeError("No agent to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._agent.save(str(path))
        logger.info("Agent saved to %s", path.resolve())
        return path.resolve()

    def load(self, path: Union[str, Path]) -> None:
        """Load a previously saved PPO agent from *path*.

        The environment must already be initialised (accessed via
        :attr:`env`).
        """
        from stable_baselines3 import PPO

        path = Path(path)
        if not path.exists() and not path.with_suffix(".zip").exists():
            raise FileNotFoundError(f"No saved agent at {path}")

        self._agent = PPO.load(str(path), env=self.env, device=self.device)
        self._is_trained = True
        logger.info("Agent loaded from %s", path.resolve())

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        return f"CascadeEscalationController(status={status}, device={self.device})"
