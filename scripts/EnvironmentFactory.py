import logging

import cv2

import gym
from gym import spaces

from dm_control import suite

import numpy as np
from collections import deque

# from typing import override
from functools import cached_property

from configurations import GymEnvironmentConfig
from envrionments.OpenAIGym import OpenAIGym, OpenAIGymImage
from envrionments.DMCS import DMCS, DMCSImage

class EnvironmentFactory:
    def __init__(self) -> None:
        pass

    def create_environment(self, config: GymEnvironmentConfig):
        logging.info(f"Training Environment: {config.gym}")
        if config.gym == 'dmcs':
            env = DMCSImage(config) if config.image_observation else DMCS(config)
        elif config.gym == "openai":
            env = OpenAIGymImage(config) if config.image_observation else OpenAIGym(config)
        else:
            raise ValueError(f"Unkown environment: {config.gym}")
        return env
            