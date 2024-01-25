import os
import pickle
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
sys.path.append("./data/")
import sascorer

from chemtsv2.misc.scaler import minmax, max_gauss, min_gauss, rectangular
from chemtsv2.reward import Reward

LGB_MODELS_PATH = 'data/model/lgb_models.pickle'
SURE_CHEMBL_ALERTS_PATH = 'data/sure_chembl_alerts.txt'
CHEMBL_FPS_PATH = 'data/chembl_fps.npy'
with open(LGB_MODELS_PATH, mode='rb') as models,\
    open(SURE_CHEMBL_ALERTS_PATH, mode='rb') as alerts, \
    open(CHEMBL_FPS_PATH, mode='rb') as fps:
    lgb_models = pickle.load(models)
    smarts = pd.read_csv(alerts, header=None, sep='\t')[1].tolist()
    alert_mols = [Chem.MolFromSmarts(smart) for smart in smarts if Chem.MolFromSmarts(smart) is not None]
    chebml_fps = np.load(fps, allow_pickle=True).item()


def scale_objective_value(params, value):
    scaling = params["type"]
    if scaling == "max_gauss":
        return max_gauss(value, params["alpha"], params["mu"], params["sigma"])
    elif scaling == "min_gauss":
        return min_gauss(value, params["alpha"], params["mu"], params["sigma"])
    elif scaling == "minmax":
        return minmax(value, params["min"], params["max"])
    elif scaling == "rectangular":
        return rectangular(value, params["min"], params["max"])
    elif scaling == "identity":
        return value
    else:
        raise ValueError("Set the scaling function from one of 'max_gauss', 'min_gauss', 'minimax', rectangular, or 'identity'")


class Dscore_reward(Reward):
    def get_objective_functions(conf):
        def EGFR(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["EGFR"].predict(fp)[0]

        def ERBB2(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["ERBB2"].predict(fp)[0]

        def ABL(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["ABL"].predict(fp)[0]

        def SRC(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["SRC"].predict(fp)[0]

        def LCK(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["LCK"].predict(fp)[0]

        def PDGFRbeta(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["PDGFRbeta"].predict(fp)[0]

        def VEGFR2(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["VEGFR2"].predict(fp)[0]

        def FGFR1(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["FGFR1"].predict(fp)[0]

        def EPHB4(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["EPHB4"].predict(fp)[0]

        def Solubility(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["Sol"].predict(fp)[0]

        def Permeability(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["Perm"].predict(fp)[0]

        def Metabolic_stability(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["Meta"].predict(fp)[0]

        def Toxicity(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["Tox"].predict(fp)[0]

        def SAscore(mol):
            return sascorer.calculateScore(mol)

        def QED(mol):
            try:
                return Chem.QED.qed(mol)
            except (Chem.rdchem.AtomValenceException, Chem.rdchem.KekulizeException):
                return None

        # The following three functions were taken from　https://github.com/jrwnter/mso.
        def molecular_weight(mol):
            """molecular weight"""
            mw = Chem.Descriptors.MolWt(mol)
            return mw

        def tox_alert(mol):
            """
            0 if a molecule matches a structural alert as defined by the included list from surechembl.
            """
            if np.any([mol.HasSubstructMatch(alert) for alert in alert_mols]):
                score = 0
            else:
                score = 1
            return score

        def has_chembl_substruct(mol):
            """0 for molecuels with substructures (ECFP2 that occur less often than 5 times in ChEMBL."""
            fp_query = AllChem.GetMorganFingerprint(mol, 1, useCounts=False)
            if np.any([bit not in chebml_fps for bit in fp_query.GetNonzeroElements().keys()]):
                return 0
            else:
                return 1

        return [EGFR, ERBB2, ABL, SRC, LCK, PDGFRbeta, VEGFR2, FGFR1, EPHB4, Solubility, Permeability, Metabolic_stability,
                Toxicity, SAscore, QED, molecular_weight, tox_alert, has_chembl_substruct]


    def calc_reward_from_objective_values(values, conf):

        if None in values:
            return -1

        dscore_params = conf["Dscore_parameters"]
        objectives = [f.__name__ for f in Dscore_reward.get_objective_functions(conf)]

        scaled_values = []
        weights = []
        for objective, value in zip(objectives, values):
            if objective == "SAscore":
                # SAscore is made negative when scaling because a smaller value is more desirable.
                scaled_values.append(scale_objective_value(dscore_params[objective], -1 * value))
            else:
                scaled_values.append(scale_objective_value(dscore_params[objective], value))
            weights.append(dscore_params[objective]["weight"])

        multiplication_value = 1
        for v, w in zip(scaled_values, weights):
            multiplication_value *= v**w
        dscore = multiplication_value ** (1/sum(weights))

        return dscore