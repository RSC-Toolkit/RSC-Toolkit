from typing import ClassVar
from abc import ABC, abstractmethod
from pydantic import BaseModel
import logging


logger = logging.getLogger(__name__)

    
class BaseRSCModel(ABC, BaseModel):
    """
    Interface models
    """


