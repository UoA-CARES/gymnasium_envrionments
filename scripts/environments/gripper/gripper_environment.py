from functools import cached_property

import cv2
import numpy as np
from environments.gym_environment import GymEnvironment
from gripper_gym.environments.environment_factory import EnvironmentFactory
from util.configurations import GripperConfig


class GripperEnvironment(GymEnvironment):
    def __init__(self, config: GripperConfig) -> None:
        super().__init__(config)

        factory = EnvironmentFactory()
        self.domain = config.domain
        self.task = config.task
        self.gripper_id = config.gripper_id

        self.env = factory.create_environment(self.domain, self.task, self.gripper_id)

    @cached_property
    def min_action_value(self) -> float:
        return self.env.min_action_value

    @cached_property
    def max_action_value(self) -> float:
        return self.env.max_action_value

    @cached_property
    def observation_space(self) -> int:
        observation_space = len(self.env.reset())
        return observation_space

    @cached_property
    def action_num(self) -> int:
        action_num = self.env.gripper.num_motors
        return action_num

    def sample_action(self):
        return self.env.sample_action()

    def set_seed(self, seed: int) -> None:
        if hasattr(self.env, "set_seed"):
            self.env.set_seed(seed)

    def reset(self, training: bool = True):
        return self.env.reset()

    def _step(self, action):
        return self.env.step(action)

    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        if hasattr(self.env, "render"):
            frame = self.env.render()
        elif hasattr(self.env, "grab_frame"):
            frame = self.env.grab_frame()
        else:
            return np.zeros((height, width, 3), dtype=np.uint8)

        if frame is not None:
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        return np.zeros((height, width, 3), dtype=np.uint8)

    def get_overlay_info(self) -> dict:
        if hasattr(self.env, "get_overlay_info"):
            return self.env.get_overlay_info()
        return {}
