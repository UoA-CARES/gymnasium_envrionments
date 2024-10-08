"""
Configuration class for Gym Environments.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field

from cares_reinforcement_learning.util.configurations import EnvironmentConfig

file_path = Path(__file__).parent.resolve()


class GymEnvironmentConfig(EnvironmentConfig):
    """
    Configuration class for Gym Environment.

    Attributes:
        gym (str): Gym Environment <openai, dmcs, pyboy>
        task (str): Task description
        domain (Optional[str]): Domain description (default: "")
        image_observation (Optional[bool]): Whether to use image observation (default: False)
        rom_path (Optional[str]): Path to ROM files (default: f"{Path.home()}/cares_rl_configs")
        act_freq (Optional[int]): Action frequency (default: 24)
        emulation_speed (Optional[int]): Emulation speed (default: 0)
        headless (Optional[bool]): Whether to run in headless mode (default: False)
    """

    gym: str = Field(description="Gym Environment <openai, dmcs, pyboy>")
    task: str
    domain: Optional[str] = ""
    display: Optional[int] = 0

    # image observation configurations
    frames_to_stack: Optional[int] = 3
    frame_width: Optional[int] = 84
    frame_height: Optional[int] = 84
    grey_scale: Optional[int] = 0

    # pyboy configurations TODO move...
    rom_path: Optional[str] = f"{Path.home()}/cares_rl_configs"
    act_freq: Optional[int] = 24
    emulation_speed: Optional[int] = 0
    headless: Optional[int] = 1
