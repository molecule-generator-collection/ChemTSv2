import pickle

from rdkit.Chem import AllChem
import numpy as np

from chemtsv2.reward import Reward


LGB_MODELS_PATH = 'data/model/lgb_models.pickle'
with open(LGB_MODELS_PATH, mode='rb') as f:
    lgb_models = pickle.load(f)
    print(f"[INFO] loaded model from {LGB_MODELS_PATH}")


class EGFR_reward(Reward):
    def get_objective_functions(conf):
        def EGFR(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["EGFR"].predict(fp)[0]
        return [EGFR]

    def calc_reward_from_objective_values(values, conf):
        return np.tanh(values[0]/10) if None not in values else -1
