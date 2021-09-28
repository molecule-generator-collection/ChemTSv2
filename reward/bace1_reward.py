import pickle

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


BACE1_MODEL_PATH = 'data/model/lgb_bace1.pickle'
with open(BACE1_MODEL_PATH, mode='rb') as f:
    lgb_bace1 = pickle.load(f)
    print(f"[INFO] loaded model from {BACE1_MODEL_PATH}")


def calc_objective_values(smiles, conf):
    mol = Chem.MolFromSmiles(smiles)
    fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
    score = lgb_bace1.predict(fp)[0]
    return [score]


def calc_reward_from_objective_values(values, conf):
    return np.tanh(values[0]/10)
