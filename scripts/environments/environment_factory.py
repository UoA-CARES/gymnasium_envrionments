from environments.gym_environment import GymEnvironment
from environments.image_wrapper import ImageWrapper
from util.configurations import GymEnvironmentConfig

# Disable these as this is a deliberate use of dynamic imports
# pylint: disable=import-outside-toplevel


class EnvironmentFactory:
    def __init__(self) -> None:
        pass

    def create_environment(
        self, config: GymEnvironmentConfig, image_observation
    ) -> GymEnvironment | ImageWrapper:

        env: GymEnvironment | ImageWrapper
        if config.gym == "dmcs":
            from environments.dmcs.dmcs_environment import DMCSEnvironment

            env = DMCSEnvironment(config)
        elif config.gym == "openai":
            from environments.openai.openai_environment import OpenAIEnvironment

            env = OpenAIEnvironment(config)
        elif config.gym == "pyboy":
            from environments.pyboy.pyboy_environment import PyboyEnvironment

            env = PyboyEnvironment(config)
        elif config.gym == "space":
            from rl_corrective_gym.gym_env_setup.corrective_transfer_env import (
                CorrectiveTransferEnvironment,
            )

            env = CorrectiveTransferEnvironment(config)
        else:
            raise ValueError(f"Unkown environment: {config.gym}")

        return ImageWrapper(config, env) if bool(image_observation) else env
