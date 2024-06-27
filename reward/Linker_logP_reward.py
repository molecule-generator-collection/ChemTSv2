import re

from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np

from chemtsv2.reward import Reward, convert_linker_reward


class Linker_LogP_reward(Reward):
    def get_objective_functions(conf):
        @convert_linker_reward(conf)
        def LogP(mol):
            return Descriptors.MolLogP(mol)
        return [LogP]
    
    
    def calc_reward_from_objective_values(values, conf):
        return np.tanh(values[0]/10)
