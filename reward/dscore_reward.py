import os
import pickle
import sys
import traceback

from rdkit import Chem
from rdkit.Chem import AllChem, RDConfig
import numpy as np
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


egfr_model_path = '../data/model/lgb_egfr.pickle'
bace1_model_path = '../data/model/lgb_bace1.pickle'

with open(egfr_model_path, mode='rb') as f1, open(bace1_model_path , mode='rb') as f2:
        lgb_egfr = pickle.load(f1)
        print(f"[INFO] {egfr_model_path} has loaded.")
        lgb_bace1 = pickle.load(f2)
        print(f"[INFO] {bace1_model_path} has loaded.")


def minmax(x, min, max):
    return (x - min)/(max - min)


def max_gauss(x, a=1, mu=8, sigma=2):
    if x > mu:
        return 1
    else :
        return a * np.exp(-(x - mu)**2 / (2*sigma**2))


def min_gauss(x, a=1, mu=2, sigma=2):
    if x < mu:
        return 1
    else :
        return a * np.exp(-(x - mu)**2 / (2*sigma**2))


def calc_objective_values(smiles):
    egfr = bace1 = sascore = qed = None
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
    activity = conf["activity"]
    # If QED could not be calculated, 'values' contains None. In that case, -1 is returned.
    if all(values):
        if activity["egfr"] == 0:
            egfr = max_gauss(values[0])
        elif activity["egfr"] == 1:
            egfr = min_gauss(values[0])
        else:
            egfr = None
        if activity["bace1"] == 0:
            bace1 = max_gauss(values[1])
        elif activity["bace1"] == 1:
            bace1 = min_gauss(values[1])
        else:
            bace1 = None
        sascore = minmax(-1 * values[2], -10, -1)
        return ((egfr ** weight["egfr"]) * (bace1 ** weight["bace1"]) * (sascore * weight["sascore"]) * (values[3] * weight["qed"])) ** (1/sum(weight.values()))
    else:
        return -1