import abc
import logging
from functools import cached_property

import cv2
from util.configurations import GymEnvironmentConfig


class GymEnvironment(metaclass=abc.ABCMeta):
    def __init__(self, config: GymEnvironmentConfig) -> None:
        logging.info(f"Training with Task {config.task}")
        self.task = config.task

    def render(self):
        frame = self.grab_frame()
        cv2.imshow(f"{self.task}", frame)
        cv2.waitKey(10)

    @cached_property
    @abc.abstractmethod
    def min_action_value(self):
        raise NotImplementedError("Override this method")

    @cached_property
    @abc.abstractmethod
    def max_action_value(self):
        raise NotImplementedError("Override this method")

    @cached_property
    @abc.abstractmethod
    def observation_space(self):
        raise NotImplementedError("Override this method")

    @cached_property
    @abc.abstractmethod
    def action_num(self):
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def sample_action(self):
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def set_seed(self, seed):
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def step(self, action):
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def grab_frame(self, height=240, width=300):
        raise NotImplementedError("Override this method")
