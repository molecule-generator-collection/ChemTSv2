from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np


def calc_objective_values(smiles, conf):
    mol = Chem.MolFromSmiles(smiles)
    score = Descriptors.MolLogP(mol)
    return [score]


def calc_reward_from_objective_values(values, conf):
    return np.tanh(values[0]/10)