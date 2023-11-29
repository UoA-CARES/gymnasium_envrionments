import logging

from util.configurations import GymEnvironmentConfig
from envrionments.openai.OpenAIGym import OpenAIGym 
from envrionments.dmcs.DMCS import DMCS

from envrionments.pyboy.pokemon.Pokemon import Pokemon
from envrionments.pyboy.mario.Mario import Mario

from envrionments.ImageWrapper import ImageWrapper

def create_pyboy_environment(config: GymEnvironmentConfig):
    if config.task == "pokemon":
        env = Pokemon(config)
    elif config.task == "mario":
        env = Mario(config)
    else:
        raise ValueError(f"Unkown pyboy environment: {config.task}")
    return env

class EnvironmentFactory:
    def __init__(self) -> None:
        pass

    def create_environment(self, config: GymEnvironmentConfig):
        logging.info(f"Training Environment: {config.gym}")
        if config.gym == 'dmcs':
            env = DMCS(config)
        elif config.gym == "openai":
            env = OpenAIGym(config)
        elif config.gym == "pyboy":# TODO extend to other pyboy games...maybe another repo?
            env = create_pyboy_environment(config)
        else:
            raise ValueError(f"Unkown environment: {config.gym}")
        return ImageWrapper(env) if config.image_observation else env
            