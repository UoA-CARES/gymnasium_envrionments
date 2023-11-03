import logging

import cv2

import gym
from gym import spaces

import numpy as np
from collections import deque

# from typing import override
from functools import cached_property

from util.configurations import GymEnvironmentConfig

class OpenAIGym:
    def __init__(self, config: GymEnvironmentConfig) -> None:
        logging.info(f"Training task {config.task}")
        self.env = gym.make(config.task, render_mode="rgb_array")

    @cached_property
    def max_action_value(self):
        return self.env.action_space.high[0]

    @cached_property
    def min_action_value(self):
        return self.env.action_space.low[0]

    @cached_property
    def observation_space(self):
        return self.env.observation_space.shape[0]
    
    @cached_property
    def action_num(self):
        if type(self.env.action_space) == spaces.Box:
            action_num = self.env.action_space.shape[0]
        elif type(self.env.action_space) == spaces.Discrete:
            action_num= self.env.action_space.n
        else:
            raise ValueError(f"Unhandled action space type: {type(self.env.action_space)}")
        return action_num

    def set_seed(self, seed):
        self.env.action_space.seed(seed)
        _, _ = self.env.reset(seed=seed)

    def reset(self):
        state, _ = self.env.reset()
        return state

    def step(self, action):
        state, reward, done, truncated, _ = self.env.step(action)
        return state, reward, done, truncated
    
    def grab_frame(self, height=240, width=300):
        frame = self.env.render()
        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to BGR for use with OpenCV
        return frame
    
class OpenAIGymImage(OpenAIGym):
    def __init__(self, config: GymEnvironmentConfig, k=3):
        self.k    = k  # number of frames to be stacked
        self.frames_stacked = deque([], maxlen=k)

        self.frame_width = 84
        self.frame_height = 84

        logging.info(f"Image Observation is on")
        super().__init__(config)

    # @override
    @property
    def observation_space(self):
        return (9, self.frame_width, self.frame_height)

    # @override
    def reset(self):
        _ = self.env.reset()
        frame = self.grab_frame(height=self.frame_height, width=self.frame_width)
        frame = np.moveaxis(frame, -1, 0)                    
        for _ in range(self.k):
            self.frames_stacked.append(frame)
        stacked_frames = np.concatenate(list(self.frames_stacked), axis=0)
        return stacked_frames

    # @override
    def step(self, action):
        _, reward, done, truncated, _ = self.env.step(action)
        frame = self.grab_frame(height=self.frame_height, width=self.frame_width)
        frame = np.moveaxis(frame, -1, 0)
        self.frames_stacked.append(frame)
        stacked_frames = np.concatenate(list(self.frames_stacked), axis=0)
        return stacked_frames, reward, done, truncated