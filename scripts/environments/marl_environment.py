import abc
import logging
from functools import cached_property
from typing import Any

import cv2
import numpy as np
from environments.base_environment import BaseEnvironment
from util.configurations import GymEnvironmentConfig


class MARLEnvironment(BaseEnvironment):
    """
    Multi-Agent Reinforcement Learning Environment Base Class

    This class provides the interface for multi-agent environments where
    multiple agents interact simultaneously in a shared environment.
    """

    def __init__(self, config: GymEnvironmentConfig) -> None:
        super().__init__(config)

    def render(self):
        frame = self.grab_frame()
        cv2.imshow(f"{self.task}", frame)
        cv2.waitKey(10)

    @cached_property
    @abc.abstractmethod
    def min_action_value(self) -> list[np.ndarray]:
        raise NotImplementedError("Override this method")

    @cached_property
    @abc.abstractmethod
    def max_action_value(self) -> list[np.ndarray]:
        raise NotImplementedError("Override this method")

    @cached_property
    @abc.abstractmethod
    def observation_space(self) -> dict[str, Any]:
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

    @abc.abstractmethod
    def sample_action(self) -> list[Any]:
        """
        Sample random actions for all agents.
        Returns: List of actions, one per agent
        """
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def set_seed(self, seed: int) -> None:
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def reset(self, training: bool = True) -> dict[str, Any]:
        """
        Reset environment and return initial global state.
        Returns: Initial global state
        """
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def _step(self, actions: list[Any]) -> tuple:
        """
        Internal step function that executes actions for all agents.
        Args:
            actions: List of actions, one per agent
        Returns:
            Tuple of (next_state, reward, done, truncated, info)
        """
        raise NotImplementedError("Override this method")

    def step(self, action: list[Any]) -> tuple:
        """
        Execute one step with actions from all agents.
        Args:
            actions: List of actions, one per agent
        Returns:
            Tuple of (next_state, reward, done, truncated, info)
        """

        state, reward, done, truncated, info = self._step(action)

        return state, reward, done, truncated, info

    @abc.abstractmethod
    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        raise NotImplementedError("Override this method")
