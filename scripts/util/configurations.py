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
        image_observation (bool): Whether to use image observation (default: False)
        frames_to_stack (int): Number of frames to stack for image observation (default: 3)
        frame_width (int): Width of the image frames (default: 84)
        frame_height (int): Height of the image frames (default: 84)
        grey_scale (bool): Whether to convert frames to grayscale (default: False)
        display (int): Display mode for the environment (default: 0)
    """

    gym: ClassVar[str]
    domain: str = ""
    task: str

    display: int = 0

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
