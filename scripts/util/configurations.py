"""
Configuration class for Gym Environments.
"""

from pathlib import Path
from typing import ClassVar

from cares_reinforcement_learning.util.configurations import SubscriptableClass

file_path = Path(__file__).parent.resolve()


class RunConfig(SubscriptableClass):
    command: str
    data_path: str | None

    seeds: list[int] | None = None
    episodes: int | None = None


class GymEnvironmentConfig(SubscriptableClass):
    """
    Configuration class for Gym Environment.

    Attributes:
        frames_to_stack (int): Number of frames to stack for image observation (default: 3)
        frame_width (int): Width of the image frames (default: 84)
        frame_height (int): Height of the image frames (default: 84)
        grey_scale (bool): Whether to convert frames to grayscale (default: False)
        display (int): Display mode for the environment (default: 0)
        save_train_checkpoints (int): Whether to save training checkpoints (default: 0)
    """

    gym: ClassVar[str]
    domain: str = ""
    task: str

    display: int = 0
    save_train_checkpoints: int = 0

    # stochastic noise configuration
    state_std: float = 0.0
    action_std: float = 0.0

    # image observation configurations
    frames_to_stack: int = 3
    frame_width: int = 84
    frame_height: int = 84
    grey_scale: int = 0

    def dict(self, *args, **kwargs):
        """Inject the class-level name into serialized dict."""
        data = super().dict(*args, **kwargs)
        data["gym"] = self.__class__.gym
        return data


class OpenAIConfig(GymEnvironmentConfig):
    gym: ClassVar[str] = "openai"


class DMCSConfig(GymEnvironmentConfig):
    gym: ClassVar[str] = "dmcs"


class PyBoyConfig(GymEnvironmentConfig):
    gym: ClassVar[str] = "pyboy"

    rom_path: str = f"{Path.home()}/cares_rl_configs"
    act_freq: int = 24
    emulation_speed: int = 0
    headless: int = 1


class ShowdownConfig(GymEnvironmentConfig):
    gym: ClassVar[str] = "showdown"


class GripperConfig(GymEnvironmentConfig):
    gym: ClassVar[str] = "gripper"

    gripper_id: int

    # camera_id: int = 0
    # blindable: bool = False
    # observation_type: int = 1
    # episode_horizon: int = 50

    # goal_selection_method: int = 0
    # reference_marker_id: int = 7
    # cube_ids: list[int] = [1, 2, 3, 4, 5, 6]

    # marker_size: int = 40
    # noise_tolerance: int = 15

    # elevator_device_name: str = "/dev/ttyUSB0"
    # elevator_baudrate: int = 1000000
    # elevator_servo_id: int = 13
    # elevator_limits: list[int] = [6000, 1500]

    # is_inverted: bool = False
    # camera_matrix: str = f"{Path.home()}/cares_rl_configs/12DOF_ID2/env_config.json"
    # camera_distortion: str = f"{Path.home()}/cares_rl_configs/12DOF_ID2/env_config.json"

    # gripper_config: str = (
    #     f"{Path.home()}/cares_rl_configs/12DOF_ID2/gripper_config.json"  # Path to the gripper configuration file
    # )
