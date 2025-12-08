import util.configurations as cfg
from environments.base_environment import BaseEnvironment
from environments.gym_environment import GymEnvironment
from environments.marl_environment import MARLEnvironment
from environments.multimodal_wrapper import MultiModalWrapper
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
    ) -> tuple[
        BaseEnvironment | MultiModalWrapper,
        BaseEnvironment | MultiModalWrapper,
    ]:

        env: BaseEnvironment | MultiModalWrapper
        eval_env: BaseEnvironment | MultiModalWrapper
        if isinstance(config, cfg.DMCSConfig):
            from environments.dmcs.dmcs_environment import DMCSEnvironment

            env = DMCSEnvironment(config, train_seed)
            eval_env = DMCSEnvironment(config, eval_seed)
        elif isinstance(config, cfg.OpenAIConfig):
            from environments.openai.openai_environment import OpenAIEnvironment

            env = OpenAIEnvironment(config, train_seed)
            eval_env = OpenAIEnvironment(config, eval_seed)
        elif isinstance(config, cfg.PyBoyConfig):
            from environments.pyboy.pyboy_environment import PyboyEnvironment

            env = PyboyEnvironment(config, train_seed)
            eval_env = PyboyEnvironment(config, eval_seed)
        elif isinstance(config, cfg.ShowdownConfig):
            from environments.showdown.showdown_environment import ShowdownEnvironment

            env = ShowdownEnvironment(config, train_seed, evaluation=False)
            eval_env = ShowdownEnvironment(config, eval_seed, evaluation=True)

        elif isinstance(config, cfg.DroneConfig):
            from environments.drone.drone_environment import DroneEnvironment

            env = DroneEnvironment(config, train_seed)
            eval_env = env
        elif isinstance(config, cfg.GripperConfig):
            from environments.gripper.gripper_environment import GripperEnvironment

            env = GripperEnvironment(config, train_seed)
            eval_env = env
        elif isinstance(config, cfg.MPEConfig):
            from environments.mpe.mpe import MPE2Environment

            env = MPE2Environment(config, train_seed)
            eval_env = MPE2Environment(config, eval_seed)

        elif isinstance(config, cfg.SMACConfig):
            from environments.smac.smac import SMACEnvironment

            env = SMACEnvironment(config, train_seed)
            eval_env = SMACEnvironment(config, eval_seed)
        elif isinstance(config, cfg.SMAC2Config):
            from environments.smac2.smac2 import SMAC2Environment

            env = SMAC2Environment(config, train_seed)
            eval_env = SMAC2Environment(config, eval_seed)
        else:
            raise ValueError(f"Unkown environment: {type(config)}")

        if isinstance(env, GymEnvironment) and isinstance(eval_env, GymEnvironment):
            env = MultiModalWrapper(config, env) if bool(image_observation) else env
            eval_env = (
                MultiModalWrapper(config, eval_env)
                if bool(image_observation)
                else eval_env
            )
        return env, eval_env
