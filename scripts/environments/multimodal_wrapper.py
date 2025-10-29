import logging
from collections import deque
from functools import cached_property

import cv2
import numpy as np
from environments.gym_environment import GymEnvironment, GymEnvironmentConfig


class MultiModalWrapper:
    def __init__(self, config: GymEnvironmentConfig, gym: GymEnvironment):
        self.gym = gym

        self.grey_scale = bool(config.grey_scale)

        self.frames_to_stack = config.frames_to_stack
        self.frames_stacked: deque[list[np.ndarray]] = deque(
            [], maxlen=self.frames_to_stack
        )

        self.frame_width = config.frame_width
        self.frame_height = config.frame_height
        logging.info("Multi-modal Observation is on")

    def set_log_path(self, log_path: str, step_count: int) -> None:
        self.gym.set_log_path(log_path, step_count)

    def get_overlay_info(self):
        return self.gym.get_overlay_info()

    def get_available_actions(self) -> dict:
        return self.gym.get_available_actions()

    @cached_property
    def observation_space(self):
        channels = 1 if self.grey_scale else 3
        channels *= self.frames_to_stack
        image_space = (channels, self.frame_width, self.frame_height)

        vector_space = self.gym.observation_space

        return {"image": image_space, "vector": vector_space}

    @cached_property
    def action_num(self) -> int:
        return self.gym.action_num

    @cached_property
    def min_action_value(self) -> float:
        return self.gym.min_action_value

    @cached_property
    def max_action_value(self) -> float:
        return self.gym.max_action_value

    def render(self):
        self.gym.render()

    def sample_action(self):
        return self.gym.sample_action()

    def set_seed(self, seed: int) -> None:
        self.gym.set_seed(seed)

    def grab_frame(self, height: int = 240, width: int = 300, grey_scale: bool = False):
        frame = self.gym.grab_frame(height=height, width=width)
        if grey_scale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame.resize((height, width, 1))
        return frame

    def reset(self, training: bool = True) -> dict[str, np.ndarray]:
        vector_state = self.gym.reset(training=training)
        frame = self.grab_frame(
            height=self.frame_height, width=self.frame_width, grey_scale=self.grey_scale
        )
        frame = np.moveaxis(frame, -1, 0)
        for _ in range(self.frames_to_stack):
            self.frames_stacked.append(frame)
        stacked_frames = np.concatenate(list(self.frames_stacked), axis=0)

        state = {"image": stacked_frames, "vector": vector_state}

        multi_modal = self.gym.get_multimodal_observation()

        state.update(multi_modal)

        return state

    def step(self, action):
        vector_state, reward, done, truncated, info = self.gym._step(action)
        frame = self.grab_frame(
            height=self.frame_height, width=self.frame_width, grey_scale=self.grey_scale
        )
        frame = np.moveaxis(frame, -1, 0)
        self.frames_stacked.append(frame)
        stacked_frames = np.concatenate(list(self.frames_stacked), axis=0)

        state = {"image": stacked_frames, "vector": vector_state}

        multi_modal = self.gym.get_multimodal_observation()

        state.update(multi_modal)

        return state, reward, done, truncated, info
