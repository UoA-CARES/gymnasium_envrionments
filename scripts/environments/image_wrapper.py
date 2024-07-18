import logging
from collections import deque
from functools import cached_property

import cv2
import numpy as np
from environments.gym_environment import GymEnvironment, GymEnvironmentConfig


class ImageWrapper:
    def __init__(self, config: GymEnvironmentConfig, gym: GymEnvironment):
        self.gym = gym

        self.grey_scale = bool(config.grey_scale)

        self.frames_to_stack = config.frames_to_stack
        self.frames_stacked = deque([], maxlen=self.frames_to_stack)

        self.frame_width = config.frame_width
        self.frame_height = config.frame_height
        logging.info("Image Observation is on")

    @cached_property
    def observation_space(self):
        channels = 1 if self.grey_scale else 3
        channels *= self.frames_to_stack
        return (channels, self.frame_width, self.frame_height)

    @cached_property
    def action_num(self):
        return self.gym.action_num

    @cached_property
    def min_action_value(self):
        return self.gym.min_action_value

    @cached_property
    def max_action_value(self):
        return self.gym.max_action_value

    def sample_action(self):
        return self.gym.sample_action()

    def set_seed(self, seed):
        self.gym.set_seed(seed)

    def grab_frame(self, height=240, width=300, grey_scale=False):
        frame = self.gym.grab_frame(height=height, width=width)
        if grey_scale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame.resize((height, width, 1))
        return frame

    def reset(self):
        _ = self.gym.reset()
        frame = self.grab_frame(
            height=self.frame_height, width=self.frame_width, grey_scale=self.grey_scale
        )
        frame = np.moveaxis(frame, -1, 0)
        for _ in range(self.frames_to_stack):
            self.frames_stacked.append(frame)
        stacked_frames = np.concatenate(list(self.frames_stacked), axis=0)
        return stacked_frames

    def step(self, action):
        _, reward, done, truncated = self.gym.step(action)
        frame = self.grab_frame(
            height=self.frame_height, width=self.frame_width, grey_scale=self.grey_scale
        )
        frame = np.moveaxis(frame, -1, 0)
        self.frames_stacked.append(frame)
        stacked_frames = np.concatenate(list(self.frames_stacked), axis=0)
        return stacked_frames, reward, done, truncated
