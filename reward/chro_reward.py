import os
import pickle
import shutil
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
sys.path.append("./data/")
import sascorer

from reward.reward import Reward
from qcforever.gaussian_run import GaussianRunPack


class UV_reward(Reward):
    def get_objective_functions(conf):
        def SAScore(mol):
            return sascorer.calculateScore(mol)
        
        def UV(mol):
            sdf_input = f"InputMol{conf['gid']}.sdf"
            mol_wH = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_wH, AllChem.ETKDG())
            Chem.MolToMolFile(mol_wH, sdf_input)
            try:
                AllChem.UFFOptimizeMolecule(mol_wH, maxIters=200)
                sdf_input_opt = f"InputMolopt_ok_{conf['gid']}.sdf"
            except:
                sdf_input_opt = f"InputMolopt_fail_{conf['gid']}.sdf"
            Chem.MolToMolFile(mol_wH, sdf_input_opt)
            
            #Gaussian calculation
            current_dir = os.getcwd()
            calc_dir = sdf_input_opt.split('.')[0]
            try:
                run_pack = GaussianRunPack.GaussianDFTRun(
                    conf['gau_functional'],
                    conf['gau_basis'], 
                    conf['gau_core_num'],
                    conf['gau_option'],
                    sdf_input_opt,
                    solvent=str(conf['gau_solvent']),
                    error=str(conf['gau_error']))
                run_pack.mem = conf['gau_memory']  # 5GB minimum
                result_dict = run_pack.run_gaussian()
                run_pack = None  # 
            except Exception as e:
                print('Gaussian DFT calculation failed:', e)
                result_dict = {}
            os.chdir(current_dir)

            #Post-processing
            #If Gaussian failed and did not make calc_dir, calc_dir for gaussian is required for the later process.
            if not os.path.exists(calc_dir):
                os.mkdir(calc_dir)
            result_base_dir = os.path.join(conf['output_dir'], 'gaussian_result')
            result_dir = os.path.join(result_base_dir, calc_dir)
            if os.path.isdir(result_dir):
                shutil.rmtree(result_dir)
            shutil.move(calc_dir, result_dir)
            shutil.move(sdf_input, result_dir)
            shutil.move(sdf_input_opt, result_dir)
            with open(os.path.join(result_dir, 'gaussian_result.pickle'), mode='wb') as f:
                pickle.dump(result_dict, f)

            if 'uv' in result_dict and len(result_dict['uv']) > 0:
                uv_abs_wl = result_dict['uv'][0][0]
            else:
                uv_abs_wl = 0
            return uv_abs_wl

        return [SAScore, UV]


    def calc_reward_from_objective_values(values, conf):
        # https://www.tandfonline.com/doi/pdf/10.1080/14686996.2022.2075240
        sascore, uv_abs_wl = values
        f_score = np.tanh(0.003*(uv_abs_wl-400)) / 2
        g_score = (-np.tanh(sascore-4)+1) / 2
        return f_score * g_score
