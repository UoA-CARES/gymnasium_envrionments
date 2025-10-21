from functools import cached_property

import cv2
import numpy as np
from environments.gym_environment import GymEnvironment
from util.configurations import SpaceConfig
from rl_corrective_gym.environments.environment_factory import SpaceEnvironmentFactory


class CorrectiveTransferEnvironment(GymEnvironment):
    def __init__(self, config: SpaceConfig) -> None:
        super().__init__(config)
        factory = SpaceEnvironmentFactory()
        self.env = factory.create_environment(config)

    @cached_property
    def min_action_value(self) -> float:
        return self.env.min_action_value

    @cached_property
    def max_action_value(self) -> float:
        return self.env.max_action_value

    @cached_property
    def observation_space(self) -> int:
        return self.env.observation_space_shape

    @cached_property
    def action_num(self) -> int:
        return self.env.action_num

    def sample_action(self):
        return self.env.sample_action()

    def set_seed(self, seed: int) -> None:
        if hasattr(self.env, "set_seed"):
            self.env.set_seed(seed)

    def reset(self, training: bool = True):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        return self.env.grab_frame(height, width)

    def get_overlay_info(self) -> dict:
        if hasattr(self.env, "get_overlay_info"):
            return self.env.get_overlay_info()
        return {}
