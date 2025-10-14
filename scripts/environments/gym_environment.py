import abc
import logging
from functools import cached_property

import cv2
import numpy as np
from util.configurations import GymEnvironmentConfig


class GymEnvironment(metaclass=abc.ABCMeta):
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

    def get_available_actions(self) -> dict:
        return {}

    @abc.abstractmethod
    def get_overlay_info(self) -> dict:
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
    def observation_space(self) -> int:
        raise NotImplementedError("Override this method")

    @cached_property
    @abc.abstractmethod
    def action_num(self) -> int:
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def sample_action(self):
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def set_seed(self, seed: int) -> None:
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def reset(self, training: bool = True) -> np.ndarray:
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def _step(self, action):
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

    def step(self, action):
        # Apply action noise
        if self.action_std > 0:
            action = self._add_relative_noise(action, self.action_std)
            action = np.clip(action, self.min_action_value, self.max_action_value)

        # Execute environment step (existing logic)
        state, reward, done, truncated, info = self._step(action)

        # Apply observation noise
        if self.state_std > 0:
            state = self._add_relative_noise(state, self.state_std)

        return state, reward, done, truncated, info

    @abc.abstractmethod
    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        raise NotImplementedError("Override this method")
