from rdkit.Chem import Descriptors
from rdkit import Chem
import numpy as np

from chemtsv2.abc import Reward
from chemtsv2.utils import attach_fragment_to_all_sites


class Scaffold_Decoration_LogP_reward(Reward):
    def get_objective_functions(conf):
        @attach_fragment_to_all_sites(conf)
        def LogP(mol):
            print(Chem.MolToSmiles(mol))
            return Descriptors.MolLogP(mol)

        return [LogP]

    def calc_reward_from_objective_values(values, conf):
        return np.tanh(values[0] / 10)
