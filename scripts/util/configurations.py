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
    image_observation: Optional[bool] = True

# class OpenAIEnvironmentConfig(GymEnvironmentConfig):
#     gym: str = Field("openai", Literal=True)

# class DMCSEnvironmentConfig(GymEnvironmentConfig):
#     gym: str = Field("dmcs", Literal=True)
#     domain: str

# class PokemonEnvironmentConfig(EnvironmentConfig):
#     task: str = Field("Pokemon", Literal=True)
