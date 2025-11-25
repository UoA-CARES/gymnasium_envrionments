import abc
import logging
from functools import cached_property
from typing import Any

import cv2
import numpy as np
from util.configurations import GymEnvironmentConfig


class BaseEnvironment(metaclass=abc.ABCMeta):
    """
    Base Environment class for both single-agent and multi-agent environments.

    This class provides the common interface and functionality that both
    GymEnvironment and MARLEnvironment share, making it easier to handle
    both types uniformly in training loops and run scripts.
    """

    def __init__(self, config: GymEnvironmentConfig) -> None:
        logging.info(f"Training with Task {config.task}")

        self.task = config.task
        self.state_std = config.state_std
        self.action_std = config.action_std

    def render(self):
        frame = self.grab_frame()
        cv2.imshow(f"{self.task}", frame)
        cv2.waitKey(10)

    def set_log_path(self, log_path: str, step_count: int) -> None:
        pass

    def get_multimodal_observation(self) -> dict:
        # Default implementation, override if necessary
        return {}

    def get_available_actions(self) -> np.ndarray:
        return np.array([])

    @cached_property
    def num_agents(self) -> int:
        return 1

    @abc.abstractmethod
    def get_overlay_info(self) -> dict:
        raise NotImplementedError("Override this method")

    @cached_property
    @abc.abstractmethod
    def min_action_value(self) -> Any:
        raise NotImplementedError("Override this method")

    @cached_property
    @abc.abstractmethod
    def max_action_value(self) -> Any:
        raise NotImplementedError("Override this method")

    @cached_property
    @abc.abstractmethod
    def observation_space(self) -> Any:
        raise NotImplementedError("Override this method")

    @cached_property
    @abc.abstractmethod
    def action_num(self) -> int:
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def sample_action(self) -> Any:
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def set_seed(self, seed: int) -> None:
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def reset(self, training: bool = True) -> Any:
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def step(self, action: Any) -> tuple:
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        raise NotImplementedError("Override this method")
