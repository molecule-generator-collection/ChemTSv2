import os
import pickle
import sys
import traceback

from rdkit import Chem
from rdkit.Chem import AllChem, RDConfig
import numpy as np
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


EGFR_MODEL_PATH = 'data/model/lgb_egfr.pickle'
BACE1_MODEL_PATH = 'data/model/lgb_bace1.pickle'

with open(EGFR_MODEL_PATH, mode='rb') as f1, \
     open(BACE1_MODEL_PATH, mode='rb') as f2:
        lgb_egfr = pickle.load(f1)
        print(f"[INFO] loaded model from {EGFR_MODEL_PATH}")
        lgb_bace1 = pickle.load(f2)
        print(f"[INFO] loaded model from {BACE1_MODEL_PATH}")


def minmax(x, min, max):
    return (x - min)/(max - min)


def max_gauss(x, a=1, mu=8, sigma=2):
    if x > mu:
        return 1
    else :
        return a * np.exp(-(x-mu)**2 / (2*sigma**2))


def min_gauss(x, a=1, mu=2, sigma=2):
    if x < mu:
        return 1
    else :
        return a * np.exp(-(x-mu)**2 / (2*sigma**2))


def calc_objective_values(smiles):
    egfr = None
    bace1 = None
    sascore = None
    qed = None
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
        egfr = lgb_egfr.predict(fp)[0]
        bace1 = lgb_bace1.predict(fp)[0]
        sascore = sascorer.calculateScore(mol)
        try:
            qed = Chem.QED.qed(mol)
        except Chem.rdchem.AtomValenceException:
            traceback.print_exc()
    return [egfr, bace1, sascore, qed]


def calc_reward_from_objective_values(values, conf):
    weight = conf["weight"]
    scaling = conf["scaling_function"]
    if None in values:
        return -1
    egfr, bace1, sascore, qed = values
    if scaling["egfr"] == "max_gauss":
        scaled_egfr = max_gauss(egfr)
    elif scaling["egfr"] == "min_gauss":
        scaled_egfr = min_gauss(egfr)
    else:
        scaled_egfr = None
    if scaling["bace1"] == "max_gauss":
        scaled_bace1 = max_gauss(bace1)
    elif scaling["bace1"] == "min_gauss":
        scaled_bace1 = min_gauss(bace1)
    else:
        scaled_bace1 = None
    # SA score is made negative when scaling because a smaller value is more desirable.
    scaled_sascore = minmax(-1 * sascore, -10, -1)
    # Since QED is a value between 0 and 1, there is no need to scale it.
    scaled_values = [scaled_egfr, scaled_bace1, scaled_sascore, qed]
    multiplication_value = 1
    for v, w in zip(scaled_values, weight.values()):
        multiplication_value *= v**w
    dscore = multiplication_value ** (1/sum(weight.values()))
    return dscore
