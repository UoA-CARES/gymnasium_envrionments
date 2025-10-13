import abc
import logging
from functools import cached_property
from typing import List, Dict, Any

import cv2
import numpy as np
from util.configurations import GymEnvironmentConfig


class MARLEnvironment(metaclass=abc.ABCMeta):
    """
    Multi-Agent Reinforcement Learning Environment Base Class

    This class provides the interface for multi-agent environments where
    multiple agents interact simultaneously in a shared environment.
    """

    def __init__(self, config: GymEnvironmentConfig) -> None:
        logging.info(f"Training MARL with Task {config.task}")

        self.task = config.task
        self.state_std = config.state_std
        self.action_std = config.action_std

    def render(self):
        frame = self.grab_frame()
        cv2.imshow(f"MARL-{self.task}", frame)
        cv2.waitKey(10)

    def set_log_path(self, log_path: str, step_count: int) -> None:
        pass

    def get_multimodal_observation(self) -> Dict[str, Any]:
        # Default implementation, override if necessary
        return {}

    @abc.abstractmethod
    def get_overlay_info(self) -> Dict[str, Any]:
        raise NotImplementedError("Override this method")

    @cached_property
    @abc.abstractmethod
    def min_action_value(self) -> float:
        raise NotImplementedError("Override this method")

    @cached_property
    @abc.abstractmethod
    def max_action_value(self) -> float:
        raise NotImplementedError("Override this method")

    @cached_property
    @abc.abstractmethod
    def observation_space(self) -> Dict[str, Any]:
        """
        Returns observation space information for multi-agent environment.
        Should include per-agent observation shapes and global state info.
        """
        raise NotImplementedError("Override this method")

    @cached_property
    @abc.abstractmethod
    def action_num(self) -> int:
        """Number of possible actions per agent"""
        raise NotImplementedError("Override this method")

    @cached_property
    @abc.abstractmethod
    def n_agents(self) -> int:
        """Number of agents in the environment"""
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def sample_action(self) -> List[int]:
        """
        Sample random actions for all agents.
        Returns: List of actions, one per agent
        """
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def get_avail_agent_actions(self, agent_id: int) -> np.ndarray:
        """
        Get available actions for a specific agent.
        Args:
            agent_id: ID of the agent
        Returns:
            Binary array indicating available actions
        """
        raise NotImplementedError("Override this method")

    def get_avail_actions(self) -> List[np.ndarray]:
        """
        Get available actions for all agents.
        Returns: List of available action arrays, one per agent
        """
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    @abc.abstractmethod
    def get_obs(self) -> List[np.ndarray]:
        """
        Get observations for all agents.
        Returns: List of observations, one per agent
        """
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def get_state(self) -> np.ndarray:
        """
        Get global state (centralized view of environment).
        Returns: Global state vector
        """
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def set_seed(self, seed: int) -> None:
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def reset(self, training: bool = True) -> np.ndarray:
        """
        Reset environment and return initial global state.
        Returns: Initial global state
        """
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def _step(self, actions: List[int]) -> tuple:
        """
        Internal step function that executes actions for all agents.
        Args:
            actions: List of actions, one per agent
        Returns:
            Tuple of (next_state, reward, done, truncated, info)
        """
        raise NotImplementedError("Override this method")

    def _add_relative_noise(
        self, data: np.ndarray, rel_std: float, min_std: float = 1e-3
    ) -> np.ndarray:
        """
        Adds Gaussian noise proportional to the absolute value of each element.
        rel_std = fraction of magnitude to perturb (e.g., 0.02 = 2%)
        min_std = lower bound to prevent zero noise for small values
        """
        # Per-element scale (avoid zeros)
        sigma = np.maximum(np.abs(data) * rel_std, min_std)

        # Gaussian noise with proportional std
        noise = np.random.normal(0, sigma, size=data.shape)
        return data + noise

    def _add_action_noise(self, actions: List[int]) -> List[int]:
        """
        Add noise to discrete actions (for MARL environments).
        This is more complex than continuous action noise.
        """
        if self.action_std <= 0:
            return actions

        noisy_actions = []
        for agent_id, action in enumerate(actions):
            # For discrete actions, we could randomly change action with some probability
            if np.random.random() < self.action_std:
                # Get available actions and choose randomly
                avail_actions = self.get_avail_agent_actions(agent_id)
                avail_indices = np.nonzero(avail_actions)[0]
                if len(avail_indices) > 0:
                    action = np.random.choice(avail_indices)
            noisy_actions.append(action)

        return noisy_actions

    def _add_observation_noise(
        self, observations: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Add noise to observations for all agents."""
        if self.state_std <= 0:
            return observations

        noisy_observations = []
        for obs in observations:
            noisy_obs = self._add_relative_noise(obs, self.state_std)
            noisy_observations.append(noisy_obs)

        return noisy_observations

    def step(self, actions: List[int]) -> tuple:
        """
        Execute one step with actions from all agents.
        Args:
            actions: List of actions, one per agent
        Returns:
            Tuple of (next_state, reward, done, truncated, info)
        """
        # Add action noise (for discrete actions, this might be action randomization)
        noisy_actions = self._add_action_noise(actions)

        # Execute environment step
        state, reward, done, truncated, info = self._step(noisy_actions)

        # Add observation noise to global state if needed
        if self.state_std > 0:
            state = self._add_relative_noise(state, self.state_std)

        return state, reward, done, truncated, info

    @abc.abstractmethod
    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        raise NotImplementedError("Override this method")

    def get_episode_limit(self) -> int:
        """
        Get maximum episode length (common in MARL environments).
        Override if your environment has episode limits.
        """
        return 200  # Default value

    def get_total_actions(self) -> int:
        """Get total number of actions across all agents."""
        return self.action_num * self.n_agents

    def close(self) -> None:
        """Close the environment and clean up resources."""
        pass
