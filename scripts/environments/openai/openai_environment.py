from functools import cached_property

import cv2
import gymnasium as gym
import numpy as np
from cares_reinforcement_learning.util import helpers as hlp
from environments.gym_environment import GymEnvironment
from gymnasium import spaces
from util.configurations import OpenAIConfig


class OpenAIEnvironment(GymEnvironment):
    def __init__(self, config: OpenAIConfig, seed: int) -> None:
        super().__init__(config, seed)

        self.env = gym.make(config.task, render_mode="rgb_array")
        self.set_seed(self.seed)

        # If Box space, we will apply action normalization - even if redundant
        self.apply_action_normalization = isinstance(self.env.action_space, spaces.Box)

    @cached_property
    def max_action_value(self) -> np.ndarray:
        return self.env.action_space.high

    @cached_property
    def min_action_value(self) -> np.ndarray:
        return self.env.action_space.low

    @cached_property
    def observation_space(self) -> int:
        return self.env.observation_space.shape[0]

    @cached_property
    def action_num(self) -> int:
        if isinstance(self.env.action_space, spaces.Box):
            action_num = self.env.action_space.shape[0]
        elif isinstance(self.env.action_space, spaces.Discrete):
            action_num = self.env.action_space.n
        else:
            raise ValueError(
                f"Unhandled action space type: {type(self.env.action_space)}"
            )
        return action_num

    def sample_action(self) -> int:
        action = self.env.action_space.sample()
        if self.apply_action_normalization:
            action = hlp.normalize(action, self.max_action_value, self.min_action_value)
        return action

    def set_seed(self, seed: int) -> None:
        _, _ = self.env.reset(seed=seed)
        # Note issues: https://github.com/rail-berkeley/softlearning/issues/75
        self.env.action_space.seed(seed)

    def reset(self, training: bool = True) -> np.ndarray:
        state, _ = self.env.reset()
        return state

    def _step(self, action: int) -> tuple:
        if self.apply_action_normalization:
            action = hlp.denormalize(
                action, self.max_action_value, self.min_action_value
            )

        state, reward, done, truncated, info = self.env.step(action)
        return state, reward, done, truncated, info

    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        frame = self.env.render()
        frame = cv2.resize(frame, (width, height))
        # Convert to BGR for use with OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def get_overlay_info(self) -> dict:
        # TODO: Add overlay information for gyms as needed
        return {}
