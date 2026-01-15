import util.configurations as cfg
from environments.base_environment import BaseEnvironment
from util.configurations import GymEnvironmentConfig

# Disable these as this is a deliberate use of dynamic imports
# pylint: disable=import-outside-toplevel


class EnvironmentFactory:
    def __init__(self) -> None:
        pass

    def create_environment(
        self,
        config: GymEnvironmentConfig,
        train_seed: int,
        eval_seed: int,
        image_observation: bool,
    ) -> tuple[BaseEnvironment, BaseEnvironment]:

        env: BaseEnvironment
        eval_env: BaseEnvironment
        if isinstance(config, cfg.DMCSConfig):
            from environments.sarl.dmcs.dmcs_environment import DMCSEnvironment

            env = DMCSEnvironment(config, train_seed, image_observation)
            eval_env = DMCSEnvironment(config, eval_seed, image_observation)
        elif isinstance(config, cfg.OpenAIConfig):
            from environments.sarl.openai.openai_environment import OpenAIEnvironment

            env = OpenAIEnvironment(config, train_seed, image_observation)
            eval_env = OpenAIEnvironment(config, eval_seed, image_observation)
        elif isinstance(config, cfg.PyBoyConfig):
            from environments.sarl.pyboy.pyboy_environment import PyboyEnvironment

            env = PyboyEnvironment(config, train_seed, image_observation)
            eval_env = PyboyEnvironment(config, eval_seed, image_observation)
        elif isinstance(config, cfg.ShowdownConfig):
            from environments.sarl.showdown.showdown_environment import (
                ShowdownEnvironment,
            )

            env = ShowdownEnvironment(config, train_seed, image_observation)
            eval_env = ShowdownEnvironment(config, eval_seed, image_observation)

        elif isinstance(config, cfg.DroneConfig):
            from environments.sarl.drone.drone_environment import DroneEnvironment

            env = DroneEnvironment(config, train_seed, image_observation)
            eval_env = env
        elif isinstance(config, cfg.GripperConfig):
            from environments.sarl.gripper.gripper_environment import GripperEnvironment

            env = GripperEnvironment(config, train_seed, image_observation)
            eval_env = env
        elif isinstance(config, cfg.MPEConfig):
            from environments.marl.mpe.mpe import MPE2Environment

            env = MPE2Environment(config, train_seed)
            eval_env = MPE2Environment(config, eval_seed)

        elif isinstance(config, cfg.SMACConfig):
            from environments.marl.smac.smac import SMACEnvironment

            env = SMACEnvironment(config, train_seed)
            eval_env = SMACEnvironment(config, eval_seed)
        elif isinstance(config, cfg.SMAC2Config):
            from environments.marl.smac2.smac2 import SMAC2Environment

            env = SMAC2Environment(config, train_seed)
            eval_env = SMAC2Environment(config, eval_seed)
        else:
            raise ValueError(f"Unkown environment: {type(config)}")

        return env, eval_env
