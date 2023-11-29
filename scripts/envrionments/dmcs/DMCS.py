import logging

import cv2

from dm_control import suite

import numpy as np
from collections import deque

# from typing import override
from functools import cached_property

from util.configurations import GymEnvironmentConfig
from envrionments.GymEnvironment import GymEnvironment

class DMCS(GymEnvironment):
    def __init__(self, config: GymEnvironmentConfig) -> None:
        super().__init__(config)
        logging.info(f"Training on Domain {config.domain}")

        self.domain = config.domain
        self.env = suite.load(self.domain, self.task)

    @cached_property
    def min_action_value(self):
        return self.env.action_spec().minimum[0]
    
    @cached_property
    def max_action_value(self):
        return self.env.action_spec().maximum[0]

    @cached_property
    def observation_space(self):
        time_step = self.env.reset()
        observation = np.hstack(list(time_step.observation.values())) # # e.g. position, orientation, joint_angles
        return len(observation)
    
    @cached_property
    def action_num(self):
        return self.env.action_spec().shape[0]

    def set_seed(self, seed):
        self.env = suite.load(self.domain, self.task, task_kwargs={'random': seed})

    def reset(self):
        time_step = self.env.reset()
        observation = np.hstack(list(time_step.observation.values())) # # e.g. position, orientation, joint_angles
        return observation

    def step(self, action):
        time_step = self.env.step(action)
        state, reward, done = np.hstack(list(time_step.observation.values())), time_step.reward, time_step.last()
        return state, reward, done, False # for consistency with open ai gym just add false for truncated
    
    def grab_frame(self, camera_id=0, height=240, width=300):
        frame = self.env.physics.render(camera_id=camera_id, height=height, width=width)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to BGR for use with OpenCV
        return frame
