import cv2

import logging

from dm_control import suite
import numpy as np

from functools import cached_property
from envrionments.GymEnvironment import GymEnvironment
from util.configurations import GymEnvironmentConfig


class DMCS(GymEnvironment):
    def __init__(self, config: GymEnvironmentConfig) -> None:
        super().__init__(config)
        logging.info(f"Training on Domain {config.domain}")

        self.domain = config.domain
        self.env = suite.load(self.domain, self.task)

    @cached_property
    def min_action_value(self) -> float:
        return self.env.action_spec().minimum[0]

    @cached_property
    def max_action_value(self) -> float:
        return self.env.action_spec().maximum[0]

    @cached_property
    def observation_space(self) -> int:
        time_step = self.env.reset()
        observation = np.hstack(
            list(time_step.observation.values())
        )  # # e.g. position, orientation, joint_angles
        return len(observation)

    @cached_property
    def action_num(self) -> int:
        return self.env.action_spec().shape[0]

    def set_seed(self, seed: int) -> None:
        self.env = suite.load(self.domain, self.task, task_kwargs={"random": seed})

    def reset(self) -> np.ndarray:
        time_step = self.env.reset()
        observation = np.hstack(
            list(time_step.observation.values())
        )  # # e.g. position, orientation, joint_angles
        return observation

    def step(self, action: int) -> tuple:
        time_step = self.env.step(action)
        state, reward, done = (
            np.hstack(list(time_step.observation.values())),
            time_step.reward,
            time_step.last(),
        )
        return (
            state,
            reward,
            done,
            False,
        )  # for consistency with open ai gym just add false for truncated

    def grab_frame(self, camera_id=0, height=240, width=300) -> np.ndarray:
        frame = self.env.physics.render(camera_id=camera_id, height=height, width=width)
        # Convert to BGR for use with OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
