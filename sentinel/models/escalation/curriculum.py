"""Curriculum learning scheduler for the Cascade Escalation Controller.

Training proceeds through phases of increasing difficulty:

1. **Phase 1 — Easy events**: Large, fast-onset contamination that is
   obvious from sensor data alone.  Low false-alarm penalty.
2. **Phase 2 — Medium events**: Moderate magnitude, moderate ramp.
   False-alarm penalty starts increasing.
3. **Phase 3 — Hard events**: Small, slow-onset events that require
   multi-modal confirmation.  Full false-alarm penalty applied.

The scheduler exposes a callback compatible with stable-baselines3 that
adjusts the environment parameters in-place at each rollout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from sentinel.models.escalation.environment import (
    C_FALSE,
    CascadeEscalationEnv,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Curriculum phase definition
# ---------------------------------------------------------------------------

@dataclass
class CurriculumPhase:
    """Specification for a single curriculum phase.

    Attributes:
        name: Human-readable label.
        difficulty_range: ``(low, high)`` — event difficulty is sampled
            uniformly from this interval each episode.
        event_ratio: Fraction of episodes that contain a contamination event.
        false_alarm_penalty: Multiplier applied to the base ``C_FALSE``.
        duration_fraction: Fraction of total training timesteps allocated
            to this phase.
    """

    name: str
    difficulty_range: tuple[float, float]
    event_ratio: float
    false_alarm_penalty: float
    duration_fraction: float


# Default three-phase curriculum
DEFAULT_PHASES: List[CurriculumPhase] = [
    CurriculumPhase(
        name="easy",
        difficulty_range=(0.0, 0.2),
        event_ratio=0.7,
        false_alarm_penalty=0.3,
        duration_fraction=0.30,
    ),
    CurriculumPhase(
        name="medium",
        difficulty_range=(0.15, 0.55),
        event_ratio=0.5,
        false_alarm_penalty=0.6,
        duration_fraction=0.35,
    ),
    CurriculumPhase(
        name="hard",
        difficulty_range=(0.4, 1.0),
        event_ratio=0.4,
        false_alarm_penalty=1.0,
        duration_fraction=0.35,
    ),
]


# ---------------------------------------------------------------------------
# Curriculum Scheduler
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """Manages progression through curriculum phases.

    The scheduler tracks the global training step count and returns the
    appropriate phase parameters.  It also supports smooth interpolation
    between adjacent phases over a configurable transition window.

    Parameters
    ----------
    total_timesteps : int
        Total training budget.
    phases : list[CurriculumPhase]
        Ordered list of phases.  ``duration_fraction`` values should sum
        to approximately 1.0.
    transition_frac : float
        Fraction of each phase's duration used for linear interpolation
        with the next phase (softens the boundary).
    """

    def __init__(
        self,
        total_timesteps: int,
        phases: Optional[List[CurriculumPhase]] = None,
        transition_frac: float = 0.15,
    ) -> None:
        self.total_timesteps = total_timesteps
        self.phases = phases or list(DEFAULT_PHASES)
        self.transition_frac = transition_frac

        # Pre-compute phase boundaries (in timestep units)
        self._boundaries: List[int] = []
        cumulative = 0
        for phase in self.phases:
            cumulative += int(phase.duration_fraction * total_timesteps)
            self._boundaries.append(cumulative)
        # Ensure last boundary covers full budget
        self._boundaries[-1] = total_timesteps

        self._current_phase_idx: int = 0

    @property
    def current_phase(self) -> CurriculumPhase:
        return self.phases[self._current_phase_idx]

    def get_params(self, timestep: int) -> Dict[str, Any]:
        """Return interpolated curriculum parameters for *timestep*.

        Returns
        -------
        dict with keys:
            ``difficulty`` (float), ``event_ratio`` (float),
            ``false_alarm_penalty`` (float), ``phase_name`` (str),
            ``phase_idx`` (int), ``progress`` (float in [0,1]).
        """
        timestep = max(0, min(timestep, self.total_timesteps))

        # Identify current phase
        phase_idx = 0
        for i, boundary in enumerate(self._boundaries):
            if timestep < boundary:
                phase_idx = i
                break
        else:
            phase_idx = len(self.phases) - 1

        self._current_phase_idx = phase_idx
        phase = self.phases[phase_idx]

        # Compute progress within this phase [0, 1]
        phase_start = self._boundaries[phase_idx - 1] if phase_idx > 0 else 0
        phase_end = self._boundaries[phase_idx]
        phase_len = max(phase_end - phase_start, 1)
        local_progress = (timestep - phase_start) / phase_len

        # Smooth interpolation near phase transition
        next_idx = min(phase_idx + 1, len(self.phases) - 1)
        next_phase = self.phases[next_idx]
        transition_start = 1.0 - self.transition_frac

        if local_progress > transition_start and phase_idx < len(self.phases) - 1:
            blend = (local_progress - transition_start) / self.transition_frac
            blend = min(blend, 1.0)
        else:
            blend = 0.0

        difficulty = _lerp_range(
            phase.difficulty_range, next_phase.difficulty_range, blend,
        )
        event_ratio = _lerp(phase.event_ratio, next_phase.event_ratio, blend)
        fa_penalty = _lerp(
            phase.false_alarm_penalty, next_phase.false_alarm_penalty, blend,
        )

        return {
            "difficulty": difficulty,
            "event_ratio": event_ratio,
            "false_alarm_penalty": fa_penalty * C_FALSE,
            "phase_name": phase.name,
            "phase_idx": phase_idx,
            "progress": timestep / max(self.total_timesteps, 1),
        }


# ---------------------------------------------------------------------------
# SB3 callback
# ---------------------------------------------------------------------------

class CurriculumCallback(BaseCallback):
    """Stable-baselines3 callback that applies curriculum scheduling.

    Attach to ``PPO.learn(callback=CurriculumCallback(...))`` to
    automatically adjust the training environment at each rollout.

    Parameters
    ----------
    scheduler : CurriculumScheduler
        Pre-configured scheduler instance.
    log_interval : int
        Print phase transitions every *log_interval* rollouts.
    """

    def __init__(
        self,
        scheduler: CurriculumScheduler,
        log_interval: int = 10,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.scheduler = scheduler
        self.log_interval = log_interval
        self._last_phase_idx: int = -1
        self._rollout_count: int = 0

    def _on_rollout_start(self) -> None:
        """Called at the beginning of each rollout collection."""
        self._rollout_count += 1
        timestep = self.num_timesteps
        params = self.scheduler.get_params(timestep)

        # Update the environment(s) — works with both single env and
        # vectorised environments (DummyVecEnv / SubprocVecEnv).
        envs = self._get_underlying_envs()
        for env in envs:
            env._difficulty = params["difficulty"]
            env._event_ratio = params["event_ratio"]
            env._false_alarm_penalty = params["false_alarm_penalty"]

        # Log phase transitions
        if params["phase_idx"] != self._last_phase_idx:
            self._last_phase_idx = params["phase_idx"]
            logger.info(
                "Curriculum phase transition at step %d: %s "
                "(difficulty=%.2f, event_ratio=%.2f, fa_penalty=%.2f)",
                timestep,
                params["phase_name"],
                params["difficulty"],
                params["event_ratio"],
                params["false_alarm_penalty"],
            )

        # Periodic log
        if self._rollout_count % self.log_interval == 0 and self.verbose > 0:
            logger.info(
                "Curriculum [step %d / %.1f%%]: phase=%s difficulty=%.2f",
                timestep,
                params["progress"] * 100,
                params["phase_name"],
                params["difficulty"],
            )

    def _on_step(self) -> bool:
        """Called after each environment step.  Return True to continue."""
        return True

    def _get_underlying_envs(self) -> List[CascadeEscalationEnv]:
        """Extract :class:`CascadeEscalationEnv` instances from the
        (possibly vectorised) training environment."""
        training_env = self.training_env
        if training_env is None:
            return []

        envs: List[CascadeEscalationEnv] = []
        # stable-baselines3 VecEnv wraps envs in .envs attribute
        if hasattr(training_env, "envs"):
            for e in training_env.envs:
                inner = _unwrap(e)
                if isinstance(inner, CascadeEscalationEnv):
                    envs.append(inner)
        else:
            inner = _unwrap(training_env)
            if isinstance(inner, CascadeEscalationEnv):
                envs.append(inner)
        return envs


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _unwrap(env: Any) -> Any:
    """Recursively unwrap a gymnasium environment to the base env."""
    while hasattr(env, "env"):
        env = env.env
    return env


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between *a* and *b* by factor *t*."""
    return a + (b - a) * t


def _lerp_range(
    r1: tuple[float, float],
    r2: tuple[float, float],
    t: float,
) -> float:
    """Return the midpoint of the interpolated range between *r1* and *r2*."""
    lo = _lerp(r1[0], r2[0], t)
    hi = _lerp(r1[1], r2[1], t)
    return (lo + hi) / 2.0
