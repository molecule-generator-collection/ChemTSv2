from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np


def get_objective_functions(conf):
    def LogP(mol):
        return Descriptors.MolLogP(mol)
    return [LogP]


def calc_reward_from_objective_values(values, conf):
    return np.tanh(values[0]/10)