import os
import pickle
import sys

from rdkit import Chem
from rdkit.Chem import AllChem, RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from misc.scaler import minmax, max_gauss, min_gauss

LGB_MODELS_PATH = 'data/model/lgb_models.pickle'
with open(LGB_MODELS_PATH, mode='rb') as f:
    lgb_models = pickle.load(f)


def scale_objective_value(params, value):
    scaling = params["type"]
    if scaling == "max_gauss":
        return max_gauss(value, params["alpha"], params["mu"], params["sigma"])
    elif scaling == "min_gauss":
        return min_gauss(value, params["alpha"], params["mu"], params["sigma"])
    elif scaling == "minmax":
        return minmax(value, params["min"], params["max"])
    elif scaling == "identity":
        return value
    else:
        raise ValueError("Set the scaling function from one of 'max_gauss', 'min_gauss', 'minimax', or 'identity'")


def get_objective_functions(conf):
    def EGFR(mol):
        if mol is None:
            return None
        fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
        return lgb_models["EGFR"].predict(fp, num_iteration=lgb_models["EGFR"].best_iteration)[0]

    def BACE1(mol):
        if mol is None:
            return None
        fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
        return lgb_models["BACE1"].predict(fp, num_iteration=lgb_models["BACE1"].best_iteration)[0]

    def ERBB2(mol):
        if mol is None:
            return None
        fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
        return lgb_models["ERBB2"].predict(fp, num_iteration=lgb_models["ERBB2"].best_iteration)[0]

    def ABL(mol):
        if mol is None:
            return None
        fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
        return lgb_models["ABL"].predict(fp, num_iteration=lgb_models["ABL"].best_iteration)[0]

    def SRC(mol):
        if mol is None:
            return None
        fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
        return lgb_models["SRC"].predict(fp, num_iteration=lgb_models["SRC"].best_iteration)[0]

    def LCK(mol):
        if mol is None:
            return None
        fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
        return lgb_models["LCK"].predict(fp, num_iteration=lgb_models["LCK"].best_iteration)[0]

    def PDGFRbeta(mol):
        if mol is None:
            return None
        fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
        return lgb_models["PDGFRbeta"].predict(fp, num_iteration=lgb_models["PDGFRbeta"].best_iteration)[0]

    def VEGFR2(mol):
        if mol is None:
            return None
        fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
        return lgb_models["VEGFR2"].predict(fp, num_iteration=lgb_models["VEGFR2"].best_iteration)[0]

    def FGFR1(mol):
        if mol is None:
            return None
        fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
        return lgb_models["FGFR1"].predict(fp, num_iteration=lgb_models["FGFR1"].best_iteration)[0]

    def EPHB4(mol):
        if mol is None:
            return None
        fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
        return lgb_models["EPHB4"].predict(fp, num_iteration=lgb_models["EPHB4"].best_iteration)[0]

    def Solubility(mol):
        if mol is None:
            return None
        fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
        return lgb_models["Sol"].predict(fp, num_iteration=lgb_models["Sol"].best_iteration)[0]

    def Permeability(mol):
        if mol is None:
            return None
        fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
        return lgb_models["Perm"].predict(fp, num_iteration=lgb_models["Perm"].best_iteration)[0]

    def Toxicity(mol):
        if mol is None:
            return None
        fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
        return lgb_models["Tox"].predict(fp, num_iteration=lgb_models["Tox"].best_iteration)[0]

    def SAScore(mol):
        return sascorer.calculateScore(mol)

    def QED(mol):
        try:
            return Chem.QED.qed(mol)
        except Chem.rdchem.AtomValenceException:
            return None

    return [EGFR, BACE1, ERBB2, ABL, SRC, LCK, PDGFRbeta, VEGFR2, FGFR1, EPHB4, Solubility, Permeability, Toxicity, SAScore, QED]


def calc_reward_from_objective_values(values, conf):
    if None in values:
        return -1
    egfr, bace1, erbb2, abl, src, lck, pdgfrbeta, vegfr2, fgfr1, ephb4, solubility, permeability, toxicity, sascore, qed = values
    dscore_params = conf["Dscore_parameters"]
    scaled_values = []
    scaled_values.append(scale_objective_value(dscore_params["EGFR"], egfr))
    scaled_values.append(scale_objective_value(dscore_params["BACE1"], bace1))
    scaled_values.append(scale_objective_value(dscore_params["ERBB2"], erbb2))
    scaled_values.append(scale_objective_value(dscore_params["ABL"], abl))
    scaled_values.append(scale_objective_value(dscore_params["SRC"], src))
    scaled_values.append(scale_objective_value(dscore_params["LCK"], lck))
    scaled_values.append(scale_objective_value(dscore_params["PDGFRbeta"], pdgfrbeta))
    scaled_values.append(scale_objective_value(dscore_params["VEGFR2"], vegfr2))
    scaled_values.append(scale_objective_value(dscore_params["FGFR1"], fgfr1))
    scaled_values.append(scale_objective_value(dscore_params["EPHB4"], ephb4))
    scaled_values.append(scale_objective_value(dscore_params["Solubility"], solubility))
    scaled_values.append(scale_objective_value(dscore_params["Permeability"], permeability))
    scaled_values.append(scale_objective_value(dscore_params["Toxicity"], toxicity))
    # SAscore is made negative when scaling because a smaller value is more desirable.
    scaled_values.append(scale_objective_value(dscore_params["SAscore"], -1 * sascore))
    scaled_values.append(scale_objective_value(dscore_params["QED"], qed))
    weight = [v["weight"] for v in dscore_params.values()]
    multiplication_value = 1
    for v, w in zip(scaled_values, weight):
        multiplication_value *= v**w
    dscore = multiplication_value ** (1/sum(weight))
    return dscore