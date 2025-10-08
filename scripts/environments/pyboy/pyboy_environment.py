from functools import cached_property

import numpy as np
from environments.gym_environment import GymEnvironment
from util.configurations import PyBoyConfig

from pyboy_environment import suite


class PyboyEnvironment(GymEnvironment):
    def __init__(self, config: PyBoyConfig) -> None:
        super().__init__(config)

        self.env = suite.make(
            config.domain,
            config.task,
            config.act_freq,
            config.emulation_speed,
            config.headless,
        )

    @cached_property
    def min_action_value(self) -> float:
        return self.env.min_action_value

    @cached_property
    def max_action_value(self) -> float:
        return self.env.max_action_value

    @cached_property
    def observation_space(self) -> int:
        return self.env.observation_space

    @cached_property
    def action_num(self) -> int:
        return self.env.action_num

    def sample_action(self):
        return self.env.sample_action()

    def set_seed(self, seed: int) -> None:
        self.env.set_seed(seed)

    def reset(self, training: bool = True) -> np.ndarray:
        return self.env.reset(training=training)

    def _step(self, action: int) -> tuple:
        return self.env.step(action)

    def grab_frame(self, height=240, width=300) -> np.ndarray:
        return self.env.grab_frame(height, width)

    def get_overlay_info(self) -> dict:
        return self.env.get_overlay_info()

    def get_multimodal_observation(self) -> dict:
        # Default implementation, override if necessary
        # TODO Sam - create a get_multimodal_observation method in pyboy side and call here
        return {}
