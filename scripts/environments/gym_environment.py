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

    def step(self, action):
        # Apply action noise
        if self.action_std > 0:
            print(f"Applying action noise {action}")
            action = action + np.random.normal(0, self.action_std, size=action.shape)
            action = np.clip(action, self.min_action_value, self.max_action_value)
            print(f"Applied action noise {action}")

        # Execute environment step (existing logic)
        state, reward, done, truncated, info = self._step(action)

        # Apply observation noise
        if self.state_std > 0:
            print(f"Applying state noise {state}")
            state = state + np.random.normal(0, self.state_std, size=state.shape)
            print(f"Applied state noise {state}")

        return state, reward, done, truncated, info

    @abc.abstractmethod
    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        raise NotImplementedError("Override this method")
