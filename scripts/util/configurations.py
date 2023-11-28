import logging
import json 

from pathlib import Path
file_path = Path(__file__).parent.resolve()

from pydantic import BaseModel, Field
from typing import List, Optional, Literal

from cares_reinforcement_learning.util.configurations import EnvironmentConfig

class GymEnvironmentConfig(EnvironmentConfig):
    gym: str = Field(description='Gym Environment <openai, dmcs, pyboy>')
    task: str
    domain: Optional[str] = ""
    image_observation: Optional[bool] = False

    rom_path : Optional[str] = f'{Path.home()}/cares_rl_configs/'
    act_freq : Optional[int] = 24
    emulation_speed : Optional[int] = 0
    headless : Optional[bool] = False

# TODO future
# class OpenAIEnvironmentConfig(GymEnvironmentConfig):
#     gym: str = Field("openai", Literal=True)

# class DMCSEnvironmentConfig(GymEnvironmentConfig):
#     gym: str = Field("dmcs", Literal=True)
#     domain: str

# class PyboyEnvironmentConfig(GymEnvironmentConfig):
#     gym: str = Field("pyboy", Literal=True)

# class PokemonEnvironmentConfig(PyboyEnvironmentConfig):
#     task: str = Field("Pokemon", Literal=True)
