from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np

from chemtsv2.abc import Reward
from chemtsv2.utils import transform_linker_to_mol


class Linker_LogP_reward(Reward):
    def get_objective_functions(conf):
        @transform_linker_to_mol(conf)
        def LogP(mol):
            return Descriptors.MolLogP(mol)
        return [LogP]
    
    
    def calc_reward_from_objective_values(values, conf):
        return np.tanh(values[0]/10)
