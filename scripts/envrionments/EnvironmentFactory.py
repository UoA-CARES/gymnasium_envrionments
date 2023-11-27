import logging

import cv2

import gym
from gym import spaces

from dm_control import suite

import numpy as np
from collections import deque

# from typing import override
from functools import cached_property

from util.configurations import GymEnvironmentConfig
from envrionments.OpenAIGym import OpenAIGym, OpenAIGymImage
from envrionments.DMCS import DMCS, DMCSImage
from envrionments.pyboy.pokemon.PokemonEnvironment import PokemonEnvironment, PokemonImage
from envrionments.pyboy.mario.MarioEnvironment import MarioEnvironment, MarioImage

def create_pyboy_environment(config: GymEnvironmentConfig):
    if config.task == "pokemon":
        env = PokemonImage(config) if config.image_observation else PokemonEnvironment(config)
    elif config.task == "mario":
        env = MarioImage(config) if config.image_observation else MarioEnvironment(config)
    else:
        raise ValueError(f"Unkown pyboy environment: {config.task}")
    return env

class EnvironmentFactory:
    def __init__(self) -> None:
        pass

    def create_environment(self, config: GymEnvironmentConfig):
        logging.info(f"Training Environment: {config.gym}")
        if config.gym == 'dmcs':
            env = DMCSImage(config) if config.image_observation else DMCS(config)
        elif config.gym == "openai":
            env = OpenAIGymImage(config) if config.image_observation else OpenAIGym(config)
        elif config.gym == "pyboy":# TODO extend to other pyboy games...maybe another repo?
            env = create_pyboy_environment(config)
        else:
            raise ValueError(f"Unkown environment: {config.gym}")
        return env
            