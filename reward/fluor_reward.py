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


class Fluor_UV_reward(Reward):
    def get_objective_functions(conf):
        def Fluor_UV(mol):
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
                uv_abs_it = result_dict['uv'][1][0]
            else:
                uv_abs_wl = 0
                uv_abs_it = 0
            if 'fluor' in result_dict and len(result_dict['fluor']) > 0:
                fl_abs_wl = result_dict['fluor'][0][0]
                fl_abs_it = result_dict['fluor'][1][0]
            else:
                fl_abs_wl = 0
                fl_abs_it = 0
            return [uv_abs_wl, uv_abs_it, fl_abs_wl, fl_abs_it]

        return [Fluor_UV]


    def calc_reward_from_objective_values(values, conf):
        # https://www.science.org/doi/full/10.1126/sciadv.abj3906
        def gauss_scaler(x, a=1, mu=0, sigma=1):
            return a * np.exp(-(x-mu)**2/(2*sigma**2))

        uv_abs_wl, uv_abs_it, fl_abs_wl, fl_abs_it = values[0]

        uv_target = 700
        fl_target = 1200
        uv_it_target = 0.01
        fl_it_target = 0.01

        R_aw = gauss_scaler(uv_abs_wl, mu=uv_target, sigma=150)
        R_ai = np.tanh(np.log10(uv_abs_it+10**(-8)) - np.log10(uv_it_target)) / 2
        R_fw = gauss_scaler(fl_abs_wl, mu=fl_target, sigma=150)
        R_fs = np.tanh(np.log10(fl_abs_it+10**(-8)) - np.log10(fl_it_target)) / 2
        
        return 0.4*R_aw + 0.1*R_ai + 0.4*R_fw + 0.1*R_fs
