import pickle

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


EGFR_MODEL_PATH = 'data/model/lgb_egfr.pickle'
with open(EGFR_MODEL_PATH, mode='rb') as f:
    lgb_egfr = pickle.load(f)
    print(f"[INFO] loaded model from {EGFR_MODEL_PATH}")

def get_objective_functions(conf):
    def EGFR(mol):
        if mol is None:
            return None
        fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
        return lgb_egfr.predict(fp)[0]
    return [EGFR]


def calc_reward_from_objective_values(values, conf):
    return np.tanh(values[0]/10) if None not in values else -1
