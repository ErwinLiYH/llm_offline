from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from d3rlpy.dataset import InfiniteBuffer, ReplayBuffer, Signature, Transition


@dataclass(frozen=True)
class MinariTransitionEpisode:
    """Episode preserving Minari's final next observation."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: bool
    truncated: bool
    source_variant: str

    def __post_init__(self):
        observations = np.asarray(self.observations, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.float32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"actions must be rank 2, got shape={actions.shape}")
        if observations.ndim != 2:
            raise ValueError(
                f"observations must be rank 2, got shape={observations.shape}"
            )
        if rewards.ndim == 1:
            rewards = rewards.reshape(-1, 1)
        if rewards.ndim != 2 or rewards.shape[1:] != (1,):
            raise ValueError(f"rewards must have shape (T, 1), got {rewards.shape}")
        step_count = actions.shape[0]
        if observations.shape[0] != step_count + 1:
            raise ValueError(
                "Minari episode must contain T+1 observations for T actions: "
                f"observations={observations.shape[0]}, actions={step_count}"
            )
        if rewards.shape[0] != step_count:
            raise ValueError("rewards and actions must have the same length")
        if step_count < 1:
            raise ValueError("Episodes must contain at least one transition")
        if not np.all(np.isfinite(observations)):
            raise ValueError("Episode observations contain non-finite values")
        if not np.all(np.isfinite(actions)):
            raise ValueError("Episode actions contain non-finite values")
        if not np.all(np.isfinite(rewards)):
            raise ValueError("Episode rewards contain non-finite values")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(self, "terminated", bool(self.terminated))
        object.__setattr__(self, "truncated", bool(self.truncated))

    @property
    def observation_signature(self) -> Signature:
        return Signature(
            dtype=[self.observations.dtype],
            shape=[self.observations.shape[1:]],
        )

    @property
    def action_signature(self) -> Signature:
        return Signature(
            dtype=[self.actions.dtype],
            shape=[self.actions.shape[1:]],
        )

    @property
    def reward_signature(self) -> Signature:
        return Signature(
            dtype=[self.rewards.dtype],
            shape=[self.rewards.shape[1:]],
        )

    def size(self) -> int:
        return int(self.actions.shape[0])

    @property
    def transition_count(self) -> int:
        return self.size()

    def compute_return(self) -> float:
        return float(np.sum(self.rewards))

    def serialize(self) -> dict[str, Any]:
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "source_variant": self.source_variant,
        }

    @classmethod
    def deserialize(cls, serializedData: dict[str, Any]):
        return cls(**serializedData)

    def __len__(self) -> int:
        return self.transition_count


class MinariTransitionPicker:
    def __call__(self, episode: MinariTransitionEpisode, index: int) -> Transition:
        if not 0 <= index < episode.transition_count:
            raise IndexError(
                f"Transition index {index} out of range for {episode.transition_count} transitions"
            )
        final = index == episode.transition_count - 1
        next_action = (
            np.zeros_like(episode.actions[index])
            if final
            else episode.actions[index + 1]
        )
        return Transition(
            observation=episode.observations[index],
            action=episode.actions[index],
            reward=episode.rewards[index],
            next_observation=episode.observations[index + 1],
            next_action=next_action,
            terminal=float(final and episode.terminated),
            interval=1,
            rewards_to_go=episode.rewards[index:],
        )


def build_replay_buffer(episodes: list[MinariTransitionEpisode]) -> ReplayBuffer:
    if not episodes:
        raise ValueError("Cannot build a d3rlpy ReplayBuffer from zero episodes")
    return ReplayBuffer(
        InfiniteBuffer(),
        episodes=episodes,
        transition_picker=MinariTransitionPicker(),
    )
