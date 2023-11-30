<<<<<<< HEAD
"""
Configuration class for Gym Environments.
"""

from pathlib import Path
from typing import Optional
=======
import logging
import json

from pathlib import Path

file_path = Path(__file__).parent.resolve()
>>>>>>> d58b901 (type hinting + linting across the board)

from pydantic import Field

from cares_reinforcement_learning.util.configurations import EnvironmentConfig

<<<<<<< HEAD
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

=======

class GymEnvironmentConfig(EnvironmentConfig):
>>>>>>> d58b901 (type hinting + linting across the board)
    gym: str = Field(description="Gym Environment <openai, dmcs, pyboy>")
    task: str
    domain: Optional[str] = ""
    image_observation: Optional[int] = 0

    rom_path: Optional[str] = f"{Path.home()}/cares_rl_configs"
    act_freq: Optional[int] = 24
    emulation_speed: Optional[int] = 0
    headless: Optional[int] = 0

<<<<<<< HEAD
=======
    rom_path: Optional[str] = f"{Path.home()}/cares_rl_configs"
    act_freq: Optional[int] = 24
    emulation_speed: Optional[int] = 0
    headless: Optional[bool] = False

>>>>>>> d58b901 (type hinting + linting across the board)

# class OpenAIEnvironmentConfig(GymEnvironmentConfig):
#     gym: str = Field("openai", Literal=True)

# class DMCSEnvironmentConfig(GymEnvironmentConfig):
#     gym: str = Field("dmcs", Literal=True)
#     domain: str

# class PyboyEnvironmentConfig(GymEnvironmentConfig):
#     gym: str = Field("pyboy", Literal=True)

# class PokemonEnvironmentConfig(PyboyEnvironmentConfig):
#     task: str = Field("Pokemon", Literal=True)
