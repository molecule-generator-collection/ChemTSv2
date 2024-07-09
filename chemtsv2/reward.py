from abc import ABC, abstractmethod
from functools import wraps
from typing import List, Callable
import re

from rdkit import Chem
from rdkit.Chem import Mol


class Reward(ABC):
    @staticmethod
    @abstractmethod
    def get_objective_functions(conf: dict) -> List[Callable[[Mol], float]]:
        raise NotImplementedError('Please check your reward file')
    
    @staticmethod
    @abstractmethod
    def calc_reward_from_objective_values(values: List[float], conf: dict) -> float:
        raise NotImplementedError('Please check your reward file')


class BatchReward(ABC):
    @staticmethod
    @abstractmethod
    def get_batch_objective_functions() -> List[Callable[[List[Mol], List[dict]], float]]:
        raise NotImplementedError('Please check your reward file')
    
    @staticmethod
    @abstractmethod
    def calc_reward_from_objective_values(values: List[float], conf: dict) -> float:
        raise NotImplementedError('Please check your reward file')


def convert_to_linker_reward(conf: dict):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not isinstance(args[0], Mol):
                raise TypeError("Check this decorator is placed in the correct position.")
            if 'cores' not in conf:
                raise KeyError("Must specify SMILES strings corresponding to the key `cores` in the config file.")
            smi = Chem.MolToSmiles(args[0])
            if smi.count("*") != len(conf['cores']):
                return -1
            mol_ = Chem.MolFromSmiles(add_atom_index_in_wildcard(smi))
            rwmol = Chem.RWMol(mol_)
            cores_mol = [Chem.MolFromSmiles(s) for s in conf['cores']]
            for m in cores_mol:
                rwmol.InsertMol(m)
            try:
                prod = Chem.molzip(rwmol)
            except:
                return -1
            return func(prod) 
        return wrapper

    def add_atom_index_in_wildcard(smiles: str):
        c = iter(range(1, smiles.count('*')+1))
        labeled_smiles = re.sub(r'\*', lambda _: f'[*:{next(c)}]', smiles)
        return labeled_smiles
    
    return decorator