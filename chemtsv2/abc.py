from abc import ABC, abstractmethod
from typing import Dict, List, Callable

from rdkit.Chem import Mol


class Filter(ABC):
    @staticmethod
    @abstractmethod
    def check(mol: Mol, config: Dict) -> bool:
        raise NotImplementedError("Please check your filter file")


class Policy(ABC):
    @staticmethod
    @abstractmethod
    def evaluate(child_state, conf):
        raise NotImplementedError("Please check your policy file")


class Reward(ABC):
    @staticmethod
    @abstractmethod
    def get_objective_functions(conf: dict) -> List[Callable[[Mol], float]]:
        raise NotImplementedError("Please check your reward file")

    @staticmethod
    @abstractmethod
    def calc_reward_from_objective_values(values: List[float], conf: dict) -> float:
        raise NotImplementedError("Please check your reward file")


class BatchReward(ABC):
    @staticmethod
    @abstractmethod
    def get_batch_objective_functions() -> List[
        Callable[[List[Mol], List[dict]], float]
    ]:
        raise NotImplementedError("Please check your reward file")

    @staticmethod
    @abstractmethod
    def calc_reward_from_objective_values(values: List[float], conf: dict) -> float:
        raise NotImplementedError("Please check your reward file")
