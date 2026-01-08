from functools import cached_property

import numpy as np
from drone_gym import (
    move_to_2d_position,
    move_to_random_2d_position,
    move_to_3d_position,
    move_to_random_3d_position,
)
from drone_gym.drone_sim import DroneSim
from drone_gym.drone import Drone
from environments.gym_environment import GymEnvironment
from util.configurations import DroneConfig


class DroneEnvironment(GymEnvironment):
    def __init__(self, config: DroneConfig, evaluation: bool = False) -> None:
        super().__init__(config)

        task_map = {
            "move_to_2d_position": move_to_2d_position.MoveToPosition,
            "move_to_random_2d_position": move_to_random_2d_position.MoveToRandomPosition,
            "move_to_3d_position": move_to_3d_position.MoveTo3DPosition,
            "move_to_random_3d_position": move_to_random_3d_position.MoveToRandom3DPosition,
        }

        # Instantiate the task
        self.env = task_map[config.task]()

    def reset(self, training: bool = True):
        return self.env.reset(training)

    def sample_action(self):
        return self.env.sample_action()

    def set_seed(self, seed: int) -> None:
        self.env.set_seed()

    def get_overlay_info(self) -> dict:
        return self.env.get_overlay_info()

    def _step(self, action):
        return self.env.step(action)

    @cached_property
    def max_action_value(self) -> np.ndarray:
        return self.env.max_action_value

    @cached_property
    def min_action_value(self) -> np.ndarray:
        return self.env.min_action_value

    @cached_property
    def observation_space(self) -> int:
        return self.env.observation_space

    @cached_property
    def action_num(self) -> int:
        return self.env.action_num

    def grab_frame(self, height=720, width=1280) -> np.ndarray:
        return self.env.grab_frame(height, width)
