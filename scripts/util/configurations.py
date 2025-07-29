"""
Configuration class for Gym Environments.
"""

from pathlib import Path

from cares_reinforcement_learning.util.configurations import EnvironmentConfig
from pydantic import Field

file_path = Path(__file__).parent.resolve()


class GymEnvironmentConfig(EnvironmentConfig):
    """
    Configuration class for Gym Environment.

    Attributes:
        gym (str): Gym Environment <openai, dmcs, pyboy>
        task (str): Task description
        domain (str): Domain description (default: "")
        image_observation (bool): Whether to use image observation (default: False)
        rom_path (str): Path to ROM files (default: f"{Path.home()}/cares_rl_configs")
        act_freq (int): Action frequency (default: 24)
        emulation_speed (int): Emulation speed (default: 0)
        headless (bool): Whether to run in headless mode (default: False)
    """

    gym: str = Field(description="Gym Environment <openai, dmcs, pyboy, drone>")
    task: str
    domain: str = ""
    display: int = 0

    # image observation configurations
    frames_to_stack: int = 3
    frame_width: int = 84
    frame_height: int = 84
    grey_scale: int = 0

    # pyboy configurations TODO move...
    # Default values are being set in multiple locations which may cause confusion
    rom_path: str = f"{Path.home()}/cares_rl_configs"
    act_freq: int = 24
    emulation_speed: int = 0
    headless: int = 1
