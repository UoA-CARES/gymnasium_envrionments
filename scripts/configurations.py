import logging
import json 

from pathlib import Path
file_path = Path(__file__).parent.resolve()

from pydantic import BaseModel, Field
from typing import List, Optional, Literal

from cares_reinforcement_learning.util.configurations import EnvironmentConfig

class GymEnvironmentConfig(EnvironmentConfig):
    gym: str = Field(description='Gym Environment <openai, dmcs>')
    task: str
    domain: Optional[str] = None
    image_observation: Optional[bool] = False