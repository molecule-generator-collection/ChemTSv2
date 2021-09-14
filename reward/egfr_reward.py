import os
import pickle

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'model', 'lgb_egfr.pickle'), mode='rb') as f:
        lgb_egfr = pickle.load(f)

def calc_objective_values(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
        score = lgb_egfr.predict(fp)[0]
    else:
        score = -1
    return [score]


def calc_reward_from_objective_values(values, conf):
    return np.tanh(values[0]/10)