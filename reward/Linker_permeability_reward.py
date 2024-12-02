import re
import pickle
import numpy as np
import pandas as pd
import math
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from mordred import Calculator, descriptors

from chemtsv2.reward import Reward
from chemtsv2.misc.scaler import max_gauss
from chemtsv2.utils import transform_linker_to_mol


LGBM_PATH = "reward/protac_linker_gen/model_all.pkl"
FEAT_NAME_PATH = "reward/protac_linker_gen/feature_names.pkl"
with open(FEAT_NAME_PATH, mode='rb') as f:
    FEAT_NAMES = pickle.load(f)
with open(LGBM_PATH, mode='rb') as f:
    LGB_MODEL = pickle.load(f)
CALC_ZAGREB1 = Calculator(descriptors.ZagrebIndex.ZagrebIndex(version=1, variable=1))


class Linker_permeability_reward(Reward):
    def get_objective_functions(conf):
        @transform_linker_to_mol(conf)
        def Permeability(mol):
            calc = Calculator(descriptors, ignore_3D=False)
            mol_3D = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_3D, useRandomCoords=True, randomSeed=0)
            try: 
                AllChem.MMFFOptimizeMolecule(mol_3D)
            except:
                return -3
            mordred = calc.pandas([mol_3D])
            df_mordred = pd.DataFrame(mordred)
            X = df_mordred[FEAT_NAMES]
            X = np.array(X)
            y_pred = LGB_MODEL.predict(X)
            return y_pred[0]

        def First_Zagreb_Index(mol):
            smi = Chem.MolToSmiles(mol)
            if smi.count("*") != len(conf['cores']):
                return -1
            _mol = Chem.MolFromSmiles(smi.strip('*'))
            if _mol is None:
                return -1
            zi = CALC_ZAGREB1(_mol)[0]
            na = _mol.GetNumAtoms()
            return zi / na
        
        return [Permeability, First_Zagreb_Index]
    
    
    def calc_reward_from_objective_values(values, conf):
        perm, zagreb_norm = values
        lt_thr = zagreb_norm <= 4.2
        return max_gauss(perm, mu=1.1, sigma=1.3) if lt_thr else -1
