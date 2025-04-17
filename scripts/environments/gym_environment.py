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

    def render(self):
        frame = self.grab_frame()
        cv2.imshow(f"{self.task}", frame)
        cv2.waitKey(10)

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
    def reset(self):
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def step(self, action):
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        raise NotImplementedError("Override this method")
