import logging

from environments.dmcs.dmcs_environment import DMCSEnvironment
from environments.gym_environment import GymEnvironment
from environments.image_wrapper import ImageWrapper
from environments.openai.openai_environment import OpenAIEnvironment
from environments.pyboy.pyboy_environment import PyboyEnvironment
from util.configurations import GymEnvironmentConfig


class EnvironmentFactory:
    def __init__(self) -> None:
        pass

    def create_environment(
        self, config: GymEnvironmentConfig, image_observation
    ) -> GymEnvironment | ImageWrapper:
        logging.info(f"Training Environment: {config.gym}")

        if config.gym == "dmcs":
            env: GymEnvironment = DMCSEnvironment(config)
        elif config.gym == "openai":
            env: GymEnvironment = OpenAIEnvironment(config)
        elif config.gym == "pyboy":
            env: GymEnvironment = PyboyEnvironment(config)
        else:
            raise ValueError(f"Unkown environment: {config.gym}")
        return ImageWrapper(config, env) if bool(image_observation) else env
