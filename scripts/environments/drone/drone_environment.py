import os
import time
from functools import cached_property
import numpy as np
import cv2

from scripts.environments.gym_environment import GymEnvironment
from drone_gym import move_to_position
from util.configurations import GymEnvironmentConfig

class DroneEnvironment(GymEnvironment):
    def __init__(self, config: GymEnvironmentConfig, evaluation: bool = False) -> None:
        super().__init__(config)

        self.env = move_to_position.DroneNavigationTask()

    def reset(self, training: bool = True) -> np.ndarray:
        return self.env.reset()

    def sample_action(self) -> np.ndarray:
        return self.env.sample_action()

    def set_seed(self, seed: int) -> None:
        self.env.set_seed(seed)

    def get_overlay_info(self) -> dict:
        # TODO: Add overlay information for gyms as needed
        return self.env.get_overlay_info()

    def step(self, action):
        return self.env.step(action)

    @cached_property
    def max_action_value(self) -> float:
        return self.env.max_action_value

    @cached_property
    def min_action_value(self) -> float:
        return self.env.min_action_value

    @cached_property
    def observation_space(self) -> int:
        return self.env.generate_state_dict

    @cached_property
    def action_num(self) -> int:
        # Don't need discrete action num for drone control?
        return 1

    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        frame = self.env.render()
        frame = cv2.resize(frame, (width, height))
        # Convert to BGR for use with OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
