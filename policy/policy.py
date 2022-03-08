from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Policy(ABC):
    @staticmethod
    @abstractmethod
    def evaluate(total_reward: np.float64,
                 constant: float,
                 parent_visits: int,
                 child_visits: int) -> np.float64:
        raise NotImplementedError('Please check your reward file')
    