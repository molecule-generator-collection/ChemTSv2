from abc import ABC, abstractmethod
from typing import Dict

from rdkit.Chem import Mol


class Filter(ABC):
    @staticmethod
    @abstractmethod
    def check(mol: Mol, config: Dict) -> bool:
        raise NotImplementedError('Please check your filter file')
