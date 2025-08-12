from environments.gym_environment import GymEnvironment
from environments.image_wrapper import ImageWrapper
from util.configurations import GymEnvironmentConfig
import util.configurations as cfg

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
        if isinstance(config, cfg.DMCSConfig):
            from environments.dmcs.dmcs_environment import DMCSEnvironment

            env = DMCSEnvironment(config)
            eval_env = DMCSEnvironment(config)
        elif isinstance(config, cfg.OpenAIConfig):
            from environments.openai.openai_environment import OpenAIEnvironment

            env = OpenAIEnvironment(config)
            eval_env = OpenAIEnvironment(config)
        elif isinstance(config, cfg.PyBoyConfig):
            from environments.pyboy.pyboy_environment import PyboyEnvironment

            env = PyboyEnvironment(config)
            eval_env = PyboyEnvironment(config)
        elif isinstance(config, cfg.ShowdownConfig):
            from environments.showdown.showdown_environment import (
                ShowdownEnvironment,
            )

            env = ShowdownEnvironment(config, evaluation=False)
            eval_env = ShowdownEnvironment(config, evaluation=True)
        elif isinstance(config, cfg.GripperConfig):
            from environments.gripper.gripper_environment import GripperEnvironment

            env = GripperEnvironment(config, evaluation=False)
            eval_env = env
        else:
            raise ValueError(f"Unkown environment: {type(config)}")

        env = ImageWrapper(config, env) if bool(image_observation) else env
        eval_env = (
            ImageWrapper(config, eval_env) if bool(image_observation) else eval_env
        )
        return env, eval_env
