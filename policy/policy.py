from abc import ABC, abstractmethod


class Policy(ABC):
    @staticmethod
    @abstractmethod
    def evaluate(child_state, conf):
        raise NotImplementedError('Please check your reward file')
    