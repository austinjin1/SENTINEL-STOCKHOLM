"""Cascade Escalation Environment for SENTINEL.

Gymnasium-compatible MDP environment that simulates water quality monitoring
with tiered modality escalation.  The agent decides at each timestep which
monitoring tier to operate in; the environment provides observations drawn
from historical (or synthetic) sensor time-series and rewards the agent for
timely detection while penalising unnecessary compute.

Tier definitions
----------------
0 — Sensor only (continuous passive monitoring, lowest cost)
1 — Sensor + Satellite (triggered analysis, medium cost)
2 — Sensor + Satellite + Microbial query (medium-high cost)
3 — Full pipeline incl. Digital Biosentinel (highest cost)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_TIERS: int = 4
FUSED_DIM: int = 256
NUM_MODALITIES: int = 4  # satellite, sensor, microbial, molecular
STATE_DIM: int = FUSED_DIM + NUM_MODALITIES + NUM_TIERS + 1 + 1  # 266

TIER_MODALITIES: Dict[int, List[str]] = {
    0: ["sensor"],
    1: ["sensor", "satellite"],
    2: ["sensor", "satellite", "microbial"],
    3: ["sensor", "satellite", "microbial", "molecular"],
}

TIER_COMPUTE_COST: Dict[int, float] = {0: 0.1, 1: 0.3, 2: 0.6, 3: 1.0}

# Reward constants
R_DETECT: float = 10.0
R_EARLY: float = 0.5
C_FALSE: float = 5.0
C_COMPUTE: float = 0.1
R_MISS: float = 50.0

# Actions
ACTION_MAINTAIN: int = 0
ACTION_ESCALATE_ONE: int = 1
ACTION_ESCALATE_TWO: int = 2
ACTION_DEESCALATE_ONE: int = 3
NUM_ACTIONS: int = 4

# Episode length limits
MAX_EPISODE_STEPS: int = 200  # ~200 hours at hourly resolution


# ---------------------------------------------------------------------------
# Event and episode data structures
# ---------------------------------------------------------------------------

@dataclass
class ContaminationEvent:
    """Description of a single contamination episode in the historical record.

    Attributes:
        onset_step: Timestep index where the event *actually* starts.
        duration: How many timesteps the event lasts.
        magnitude: Peak anomaly magnitude in [0, 1] (1 = catastrophic).
        ramp_steps: Number of timesteps over which the signal ramps up
            from baseline to *magnitude*.  Slow ramps are harder.
        modality_signatures: Per-modality peak anomaly score at the event.
            Shape ``(NUM_MODALITIES,)``.
    """

    onset_step: int
    duration: int
    magnitude: float
    ramp_steps: int
    modality_signatures: np.ndarray  # (4,)


@dataclass
class EpisodeScenario:
    """Everything the environment needs to simulate one episode.

    Attributes:
        sensor_series: Baseline sensor-derived fused embedding per step.
            Shape ``(T, FUSED_DIM)``.
        event: ``None`` for a normal (non-event) episode.
        historical_event_rate: Average events-per-year at this site.
    """

    sensor_series: np.ndarray  # (T, FUSED_DIM)
    event: Optional[ContaminationEvent] = None
    historical_event_rate: float = 0.5


# ---------------------------------------------------------------------------
# Scenario generators (synthetic data for training)
# ---------------------------------------------------------------------------

def generate_normal_scenario(
    length: int = MAX_EPISODE_STEPS,
    rng: np.random.Generator | None = None,
) -> EpisodeScenario:
    """Generate a normal (non-event) episode with ambient sensor noise."""
    rng = rng or np.random.default_rng()
    sensor_series = rng.standard_normal((length, FUSED_DIM)).astype(np.float32) * 0.1
    return EpisodeScenario(
        sensor_series=sensor_series,
        event=None,
        historical_event_rate=rng.uniform(0.1, 2.0),
    )


def generate_event_scenario(
    length: int = MAX_EPISODE_STEPS,
    difficulty: float = 0.5,
    rng: np.random.Generator | None = None,
) -> EpisodeScenario:
    """Generate an event episode.

    Parameters
    ----------
    difficulty:
        Value in [0, 1].  0 = easy (large, fast onset); 1 = hard (small,
        slow onset).
    """
    rng = rng or np.random.default_rng()

    # Scale event parameters by difficulty
    magnitude = float(np.clip(1.0 - 0.7 * difficulty + rng.normal(0, 0.05), 0.15, 1.0))
    ramp_steps = int(np.clip(2 + difficulty * 40 + rng.normal(0, 2), 1, 60))
    duration = int(np.clip(ramp_steps + 20 + rng.normal(0, 5), ramp_steps + 5, length // 2))

    # Event cannot start too early (need lead-in) or too late (need room)
    earliest = max(10, int(length * 0.1))
    latest = max(earliest + 1, length - duration - 10)
    onset = rng.integers(earliest, latest + 1)

    # Per-modality signatures: sensor always responds; higher modalities
    # provide cleaner signal at higher tiers
    base_sig = rng.uniform(0.3, 1.0, size=(NUM_MODALITIES,)).astype(np.float32)
    base_sig[0] = max(base_sig[0], 0.5)  # sensor always has decent signal
    modality_signatures = base_sig * magnitude

    sensor_series = rng.standard_normal((length, FUSED_DIM)).astype(np.float32) * 0.1

    event = ContaminationEvent(
        onset_step=int(onset),
        duration=int(duration),
        magnitude=magnitude,
        ramp_steps=int(ramp_steps),
        modality_signatures=modality_signatures,
    )
    return EpisodeScenario(
        sensor_series=sensor_series,
        event=event,
        historical_event_rate=rng.uniform(0.5, 3.0),
    )


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------

class CascadeEscalationEnv(gym.Env):
    """Tiered monitoring escalation environment.

    Observation (Box, float32, shape ``(STATE_DIM,)``):
        [0:256]   — fused representation (augmented by event signal)
        [256:260] — per-modality anomaly scores
        [260:264] — current tier one-hot
        [264]     — normalised time since last escalation
        [265]     — historical event rate at this site

    Action (Discrete 4):
        0 = maintain, 1 = escalate +1, 2 = escalate +2, 3 = de-escalate -1

    Reward:
        See module-level constants ``R_DETECT``, ``R_EARLY``, etc.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenarios: Optional[List[EpisodeScenario]] = None,
        event_ratio: float = 0.5,
        difficulty: float = 0.5,
        false_alarm_penalty: float = C_FALSE,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32,
        )

        self._provided_scenarios = scenarios
        self._event_ratio = event_ratio
        self._difficulty = difficulty
        self._false_alarm_penalty = false_alarm_penalty
        self._rng = np.random.default_rng(seed)

        # Episode state (initialised in reset)
        self._scenario: Optional[EpisodeScenario] = None
        self._step: int = 0
        self._tier: int = 0
        self._steps_since_escalation: int = 0
        self._alerted: bool = False
        self._detected_step: Optional[int] = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset to a new episode, returning initial observation."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Allow per-reset overrides via options
        difficulty = self._difficulty
        event_ratio = self._event_ratio
        if options:
            difficulty = options.get("difficulty", difficulty)
            event_ratio = options.get("event_ratio", event_ratio)

        # Choose or generate scenario
        if self._provided_scenarios:
            idx = int(self._rng.integers(0, len(self._provided_scenarios)))
            self._scenario = self._provided_scenarios[idx]
        elif self._rng.random() < event_ratio:
            self._scenario = generate_event_scenario(
                difficulty=difficulty, rng=self._rng,
            )
        else:
            self._scenario = generate_normal_scenario(rng=self._rng)

        self._step = 0
        self._tier = 0
        self._steps_since_escalation = 0
        self._alerted = False
        self._detected_step = None

        obs = self._build_observation()
        info = self._build_info()
        return obs, info

    def step(
        self, action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take one step in the environment.

        Returns:
            observation, reward, terminated, truncated, info
        """
        assert self._scenario is not None, "Call reset() before step()."
        assert self.action_space.contains(action), f"Invalid action {action}"

        # --- Apply tier transition -------------------------------------------
        prev_tier = self._tier
        if action == ACTION_MAINTAIN:
            pass
        elif action == ACTION_ESCALATE_ONE:
            self._tier = min(self._tier + 1, NUM_TIERS - 1)
        elif action == ACTION_ESCALATE_TWO:
            self._tier = min(self._tier + 2, NUM_TIERS - 1)
        elif action == ACTION_DEESCALATE_ONE:
            self._tier = max(self._tier - 1, 0)

        if self._tier != prev_tier:
            self._steps_since_escalation = 0
        else:
            self._steps_since_escalation += 1

        self._step += 1

        # --- Compute reward --------------------------------------------------
        reward = self._compute_reward(prev_tier)

        # --- Check termination -----------------------------------------------
        T = len(self._scenario.sensor_series)
        terminated = False
        truncated = self._step >= T

        # If event episode and we passed the event window without detecting,
        # apply miss penalty and terminate.
        event = self._scenario.event
        if event is not None and not self._alerted:
            event_end = event.onset_step + event.duration
            if self._step >= event_end:
                reward -= R_MISS
                terminated = True

        obs = self._build_observation()
        info = self._build_info()
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> np.ndarray:
        """Construct the state vector for the current timestep."""
        scenario = self._scenario
        assert scenario is not None

        T = len(scenario.sensor_series)
        t = min(self._step, T - 1)

        # Base fused representation
        fused = scenario.sensor_series[t].copy()  # (256,)

        # Inject event signal into the fused representation if the event
        # is active *and* the current tier can see the relevant modalities.
        anomaly_scores = np.zeros(NUM_MODALITIES, dtype=np.float32)
        event = scenario.event
        if event is not None and event.onset_step <= self._step < event.onset_step + event.duration:
            progress = self._step - event.onset_step
            # Ramp factor: linear ramp up over ramp_steps
            ramp = float(np.clip(progress / max(event.ramp_steps, 1), 0.0, 1.0))
            for m_idx, modality in enumerate(["sensor", "satellite", "microbial", "molecular"]):
                if modality in TIER_MODALITIES[self._tier]:
                    signal = event.modality_signatures[m_idx] * ramp
                    anomaly_scores[m_idx] = signal
                    # Perturb fused representation to reflect the anomaly
                    offset = self._rng.standard_normal(FUSED_DIM).astype(np.float32)
                    fused += offset * signal * 0.3

        # Tier one-hot
        tier_onehot = np.zeros(NUM_TIERS, dtype=np.float32)
        tier_onehot[self._tier] = 1.0

        # Time since last escalation (normalised)
        time_since = np.array(
            [self._steps_since_escalation / MAX_EPISODE_STEPS], dtype=np.float32,
        )

        # Historical event rate
        hist_rate = np.array(
            [scenario.historical_event_rate], dtype=np.float32,
        )

        obs = np.concatenate([fused, anomaly_scores, tier_onehot, time_since, hist_rate])
        assert obs.shape == (STATE_DIM,), f"Expected {STATE_DIM}, got {obs.shape}"
        return obs

    def _compute_reward(self, prev_tier: int) -> float:
        """Compute the scalar reward for the current transition."""
        reward = 0.0
        event = self._scenario.event if self._scenario else None

        # Compute cost: proportional to current tier
        reward -= C_COMPUTE * self._tier

        in_event = (
            event is not None
            and event.onset_step <= self._step < event.onset_step + event.duration
        )

        # Detection logic: if agent is at tier >= 2 during event, we consider
        # it an alert (the higher modalities confirm contamination).
        if in_event and self._tier >= 2 and not self._alerted:
            self._alerted = True
            self._detected_step = self._step
            reward += R_DETECT
            # Early detection bonus
            lead_time = max(0, event.onset_step + event.duration - self._step)
            reward += R_EARLY * lead_time

        # False alarm: escalated to tier >= 2 when there is no event
        if not in_event and self._tier >= 2:
            reward -= self._false_alarm_penalty * 0.1  # per-step cost

        return reward

    def _build_info(self) -> Dict[str, Any]:
        """Build the info dict returned alongside observations."""
        event = self._scenario.event if self._scenario else None
        return {
            "step": self._step,
            "tier": self._tier,
            "has_event": event is not None,
            "event_onset": event.onset_step if event else None,
            "alerted": self._alerted,
            "detected_step": self._detected_step,
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def get_tier_config(tier: int) -> Dict[str, Any]:
        """Return the modality configuration for a given tier.

        Returns:
            Dict with ``tier``, ``modalities``, ``compute_cost``, and a
            human-readable ``description``.
        """
        if tier < 0 or tier >= NUM_TIERS:
            raise ValueError(f"Tier must be in [0, {NUM_TIERS - 1}], got {tier}")

        descriptions = {
            0: "Passive monitoring — sensor-only continuous inference",
            1: "Anomaly detected — sensor + satellite triggered analysis",
            2: "Multi-modal confirmation — sensor + satellite + microbial query",
            3: "Full characterisation — all modalities + Digital Biosentinel",
        }
        return {
            "tier": tier,
            "modalities": TIER_MODALITIES[tier],
            "compute_cost": TIER_COMPUTE_COST[tier],
            "description": descriptions[tier],
        }
