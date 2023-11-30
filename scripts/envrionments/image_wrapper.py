import logging
from collections import deque
from functools import cached_property

import numpy as np
from envrionments.gym_environment import GymEnvironment


class ImageWrapper:
    def __init__(self, gym: GymEnvironment, k=3):
        self.gym = gym

        self.k = k  # number of frames to be stacked
        self.frames_stacked = deque([], maxlen=k)

        self.frame_width = 84
        self.frame_height = 84
        logging.info("Image Observation is on")

    @cached_property
    def observation_space(self):
        return (9, self.frame_width, self.frame_height)

    @cached_property
    def action_num(self):
        return self.gym.action_num

    @cached_property
    def min_action_value(self):
        return self.gym.min_action_value

    @cached_property
    def max_action_value(self):
        return self.gym.max_action_value

    def set_seed(self, seed):
        self.gym.set_seed(seed)

    def grab_frame(self, height=240, width=300):
        return self.gym.grab_frame(height=height, width=width)

    def reset(self):
        _ = self.gym.reset()
        frame = self.grab_frame(height=self.frame_height, width=self.frame_width)
        frame = np.moveaxis(frame, -1, 0)
        for _ in range(self.k):
            self.frames_stacked.append(frame)
        stacked_frames = np.concatenate(list(self.frames_stacked), axis=0)
        return stacked_frames

    def step(self, action):
        _, reward, done, truncated = self.gym.step(action)
        frame = self.grab_frame(height=self.frame_height, width=self.frame_width)
        frame = np.moveaxis(frame, -1, 0)
        self.frames_stacked.append(frame)
        stacked_frames = np.concatenate(list(self.frames_stacked), axis=0)
        return stacked_frames, reward, done, truncated
