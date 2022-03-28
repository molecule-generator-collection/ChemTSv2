from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Policy(ABC):
    @staticmethod
    @abstractmethod
    def evaluate(child_state, conf):
        raise NotImplementedError('Please check your reward file')
    