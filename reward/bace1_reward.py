import pickle

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


BACE1_MODEL_PATH = 'data/model/lgb_bace1.pickle'
with open(BACE1_MODEL_PATH, mode='rb') as f:
    lgb_bace1 = pickle.load(f)
    print(f"[INFO] loaded model from {BACE1_MODEL_PATH}")

def get_objective_functions(conf):
    def BACE1(mol):
        if mol is None:
            return None
        fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
        return lgb_bace1.predict(fp)[0]
    return [BACE1]


def calc_reward_from_objective_values(values, conf):
    return np.tanh(values[0]/10) if None not in values else -1
