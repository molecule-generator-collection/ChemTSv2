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


def scale_objective_values(target_name, value, conf):
    scaling = conf["scaling_function"]
    if scaling[target_name] == "max_gauss":
        return max_gauss(value)
    elif scaling[target_name] == "min_gauss":
        return min_gauss(value)
    else:
        return None


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
    scaled_values = []
    target_names = list(conf['scaling_function'].keys())
    affinity_values = values[:len(target_names)]
    for n, v in zip(target_names, affinity_values):
        scaled_values.append(scale_objective_values(n, v, conf))
    solubility, permeability, toxicity, sascore, qed = values[len(affinity_values):]
    scaled_values.append(max_gauss(solubility, a=1, mu=-2, sigma=0.6))
    scaled_values.append(max_gauss(permeability, a=1, mu=-4.5, sigma=0.5))
    scaled_values.append(min_gauss(toxicity, a=1, mu=5.5, sigma=0.5))
    # SA score is made negative when scaling because a smaller value is more desirable.
    scaled_values.append(minmax(-1 * sascore, -10, -1))
    # Since QED is a value between 0 and 1, there is no need to scale it.
    scaled_values.append(qed)
    weight = conf["weight"]
    multiplication_value = 1
    for v, w in zip(scaled_values, weight.values()):
        multiplication_value *= v**w
    dscore = multiplication_value ** (1/sum(weight.values()))
    return dscore