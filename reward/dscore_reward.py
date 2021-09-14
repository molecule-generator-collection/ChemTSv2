import os
import pickle
import sys

from rdkit import Chem
from rdkit.Chem import AllChem, RDConfig
import numpy as np
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'model', 'lgb_egfr.pickle'), mode='rb') as f:
        lgb_egfr = pickle.load(f)
with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'model', 'lgb_bace1.pickle'), mode='rb') as f:
        lgb_bace1 = pickle.load(f)

def minmax(x, min, max):
    return (x - min)/(max - min)

def max_gauss(x, a=1, mu=8, sigma=2):
    if x > mu:
        y = 1
    else :
        y = a * np.exp(-(x - mu)**2 / (2*sigma**2))
    return y

def min_gauss(x, a=1, mu=2, sigma=2):
    if x < mu:
        y = 1
    else :
        y = a * np.exp(-(x - mu)**2 / (2*sigma**2))
    return y

def calc_objective_values(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
        egfr = lgb_egfr.predict(fp)[0]
        bace1 = lgb_bace1.predict(fp)[0]
        sascore = sascorer.calculateScore(mol)
        qed = Chem.QED.qed(mol)
    else:
        egfr = -1
        bace1 = -1
        sascore = -1
        qed = -1
    return [egfr, bace1, sascore, qed]


def calc_reward_from_objective_values(values, conf):
    egfr = max_gauss(values[0])
    bace = min_gauss(values[1])
    sascore = minmax(-1 * values[2], -10, -1)
    return ((egfr ** 5) * (bace ** 3) * sascore * values[3]) ** (1/10)