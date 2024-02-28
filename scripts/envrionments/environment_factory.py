import logging

from envrionments.dmcs.dmcs_environment import DMCSEnvironment
from envrionments.gym_environment import GymEnvironment
from envrionments.pyboy.pyboy_environment import PyboyEnvironment
from envrionments.image_wrapper import ImageWrapper
from envrionments.openai.openai_environment import OpenAIEnvrionment
from util.configurations import GymEnvironmentConfig


class EnvironmentFactory:
    def __init__(self) -> None:
        pass

    def create_environment(self, config: GymEnvironmentConfig) -> GymEnvironment:
        logging.info(f"Training Environment: {config.gym}")
        if config.gym == "dmcs":
            env = DMCSEnvironment(config)
        elif config.gym == "openai":
            env = OpenAIEnvrionment(config)
        elif config.gym == "pyboy":
            env = PyboyEnvironment(config)
        else:
            raise ValueError(f"Unkown environment: {config.gym}")
        return ImageWrapper(env) if bool(config.image_observation) else env
