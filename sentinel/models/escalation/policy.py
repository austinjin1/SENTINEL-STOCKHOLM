"""RL policy network for the Cascade Escalation Controller.

Defines a two-headed MLP (actor-critic) that is compatible with
stable-baselines3's PPO implementation.  The actor head outputs logits
over the four escalation actions; the critic head outputs a scalar
state-value estimate.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from sentinel.models.escalation.environment import NUM_ACTIONS, STATE_DIM


# ---------------------------------------------------------------------------
# Custom feature extractor (shared trunk)
# ---------------------------------------------------------------------------

class EscalationFeaturesExtractor(BaseFeaturesExtractor):
    """Shared feature extractor for the escalation policy.

    Maps the raw state vector (267 dims) through a two-layer MLP that
    produces a 64-dimensional feature representation consumed by both
    the policy (actor) and value (critic) heads.

    Architecture::

        state (267) -> Linear(128) -> ReLU -> Linear(64) -> ReLU
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        features_dim: int = 64,
    ) -> None:
        super().__init__(observation_space, features_dim)
        obs_dim = int(observation_space.shape[0])  # type: ignore[index]
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, features_dim),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


# ---------------------------------------------------------------------------
# Standalone policy network (for inspection / export outside SB3)
# ---------------------------------------------------------------------------

class EscalationPolicyNetwork(nn.Module):
    """Standalone actor-critic MLP for the cascade escalation controller.

    This module is intentionally kept independent of stable-baselines3 so
    it can be serialised, inspected, and deployed without the SB3 runtime.

    Architecture::

        state (267) -> 128 (ReLU) -> 64 (ReLU) -> 4  (action logits)
                                                   \\-> 1  (state value)
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden1: int = 128,
        hidden2: int = 64,
        num_actions: int = NUM_ACTIONS,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
        )

        # Actor head — action logits
        self.actor_head = nn.Linear(hidden2, num_actions)

        # Critic head — scalar state value
        self.critic_head = nn.Linear(hidden2, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.shared:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(module.bias)
        # Smaller init for heads (standard PPO practice)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(
        self, state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        state : Tensor[B, state_dim]

        Returns
        -------
        action_logits : Tensor[B, num_actions]
        state_value   : Tensor[B, 1]
        """
        features = self.shared(state)
        return self.actor_head(features), self.critic_head(features)

    def predict_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, float]:
        """Select an action for a single state.

        Parameters
        ----------
        state : Tensor[state_dim] or Tensor[1, state_dim]
        deterministic : If *True*, pick the argmax; otherwise sample.

        Returns
        -------
        action : int
        value  : float  (state-value estimate)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            logits, value = self.forward(state)
            if deterministic:
                action = int(logits.argmax(dim=-1).item())
            else:
                probs = torch.softmax(logits, dim=-1)
                action = int(torch.multinomial(probs, 1).item())
        return action, float(value.item())


# ---------------------------------------------------------------------------
# SB3-compatible policy class
# ---------------------------------------------------------------------------

def make_escalation_policy_kwargs() -> Dict:
    """Return ``policy_kwargs`` for ``PPO(..., policy_kwargs=...)``.

    This plugs the custom feature extractor into SB3's ActorCriticPolicy
    while keeping the actor/critic net architectures defined by SB3
    (which adds its own heads on top of the extracted features).
    """
    return {
        "features_extractor_class": EscalationFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 64},
        "net_arch": [],  # no additional hidden layers after extractor
        "activation_fn": nn.ReLU,
    }


# ---------------------------------------------------------------------------
# PPO training helper
# ---------------------------------------------------------------------------

DEFAULT_PPO_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}


def create_ppo_agent(
    env: gym.Env,
    total_timesteps: int = 500_000,
    seed: Optional[int] = None,
    device: str = "auto",
    **override_hyperparams,
) -> PPO:
    """Instantiate a PPO agent with SENTINEL escalation defaults.

    Parameters
    ----------
    env : gym.Env
        A :class:`CascadeEscalationEnv` (or compatible wrapper).
    total_timesteps : int
        Intended training budget (stored for reference; call
        ``agent.learn(total_timesteps=...)`` to actually train).
    seed : int, optional
        Random seed for reproducibility.
    device : str
        ``"auto"``, ``"cpu"``, or ``"cuda"``.
    **override_hyperparams
        Any key in :data:`DEFAULT_PPO_HYPERPARAMS` can be overridden.

    Returns
    -------
    PPO
        Ready-to-train SB3 PPO agent.
    """
    hyperparams = {**DEFAULT_PPO_HYPERPARAMS, **override_hyperparams}
    policy_kwargs = make_escalation_policy_kwargs()

    agent = PPO(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        device=device,
        verbose=1,
        policy_kwargs=policy_kwargs,
        **hyperparams,
    )
    return agent
