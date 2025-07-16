from dataclasses import dataclass 
from typing import Optional


@dataclass
class ControllerConfig:
    """Configuration for the controller"""
    
    resume:bool = True 
    