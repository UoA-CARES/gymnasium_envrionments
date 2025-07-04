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
    ) -> tuple[GymEnvironment | ImageWrapper, GymEnvironment | ImageWrapper]:

        env: GymEnvironment | ImageWrapper
        eval_env: GymEnvironment | ImageWrapper
        if config.gym == "dmcs":
            from environments.dmcs.dmcs_environment import DMCSEnvironment

            env = DMCSEnvironment(config)
            eval_env = DMCSEnvironment(config)
        elif config.gym == "openai":
            from environments.openai.openai_environment import OpenAIEnvironment

            env = OpenAIEnvironment(config)
            eval_env = OpenAIEnvironment(config)
        elif config.gym == "pyboy":
            from environments.pyboy.pyboy_environment import PyboyEnvironment

            env = PyboyEnvironment(config)
            eval_env = PyboyEnvironment(config)
        elif config.gym == "pokeenv":
            from environments.showdown.showdown_environment import (
                ShowdownEnvironment,
            )

            env = ShowdownEnvironment(config, evaluation=False)
            eval_env = ShowdownEnvironment(config, evaluation=True)
        else:
            raise ValueError(f"Unkown environment: {config.gym}")

        env = ImageWrapper(config, env) if bool(image_observation) else env
        eval_env = (
            ImageWrapper(config, eval_env) if bool(image_observation) else eval_env
        )
        return env, eval_env
