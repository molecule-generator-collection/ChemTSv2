import os
import sys

from rdkit.Chem import Descriptors, RDConfig
sys.path.append("./data/")
import sascorer

from reward.reward import Reward

import GaussianRunPack
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import gc, shutil
import numpy as np
import datetime
import pickle

class IPE_reward(Reward):
    def get_objective_functions(conf):
        def LogP(mol):
            return Descriptors.MolLogP(mol)

        def SAScore(mol):
            return sascorer.calculateScore(mol)
        
        def IPE(mol):

            mol_index =  str(conf['gid'])
            print('gid', mol_index, Chem.MolToSmiles(mol))

            #Geometrical Optimization and save mol as SDF file
            SDFinput = 'InputMol'+mol_index+'.sdf'
            SDFinput_opt = 'InputMolopt'+mol_index+'.sdf'

            mol_wH = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_wH, AllChem.ETKDG())

            Chem.MolToMolFile(mol_wH, SDFinput)
            try:
                opt = AllChem.UFFOptimizeMolecule(mol_wH, maxIters=200)
            except:
                opt=None

            Chem.MolToMolFile(mol_wH, SDFinput_opt)
            
            #(Optional) ChemLAQA

            #Gaussian calculation
            functional = conf['gau_functional']#'B3LYP'
            basis = conf['gau_basis']
            core_num = conf['gau_core_num']
            option = conf['gau_option'] #"opt deen homolumo stable2o2"
            solvent = str(conf['gau_solvent'])
            error = str(conf['gau_error'])
            current_dir = os.getcwd()
            try:
                run_pack = GaussianRunPack.GaussianDFTRun(functional, basis, core_num, option, SDFinput_opt, solvent = solvent, error = error)
                run_pack.mem = '12GB'
                outdic = run_pack.run_gaussian()

                del run_pack
                gc.collect()

            except Exception as e:
                print('DFT failed:', e)
                outdic = {}
            
            print('outdic', outdic)
            os.chdir(current_dir)

            #Post-processing
            #If Gaussian failed and did not make calc_dir, calc_dir for gaussian is required for the later process.
            calc_dir = SDFinput_opt.split('.')[0]
            if not os.path.exists(calc_dir):
                os.mkdir(calc_dir)
                
            result_dir = os.path.join(conf['output_dir'], 'gaussian_result')
            #if not os.path.isdir(result_dir):
            #    os.mkdir(result_dir)
            
            if os.path.isdir(os.path.join(result_dir, calc_dir)):
                shutil.rmtree(os.path.join(result_dir, calc_dir))
            print('shutil.move(calc_dir, result_dir)', shutil.move(calc_dir, result_dir))
            shutil.move(SDFinput, os.path.join(result_dir, calc_dir))
            shutil.move(SDFinput_opt, os.path.join(result_dir, calc_dir))
            with open(os.path.join(result_dir, calc_dir, 'gaussian_result.pickle'), mode='wb') as f:
                pickle.dump(outdic, f)

            #Extract values
            if 'ipe' in outdic.keys():
                if len(outdic['ipe']) > 0:
                    ipe_value = outdic['ipe'][0] if outdic['ipe'][1] < 1.0 else 0
                    gaussian_result = ipe_value
                else:
                    gaussian_result = 0
            else:
                gaussian_result = 0
            return gaussian_result

        return [LogP, SAScore, IPE]



   

    def calc_reward_from_objective_values(values, conf):
        def intensity_scaler_tanh( data, epsilon = 2):
            return np.tanh((np.log10(10**-epsilon + data) + epsilon) )

        def gauss(x, a=1, mu=0, sigma=1):
            return a * np.exp(-(x - mu)**2 / (2*sigma**2))

        def sa_scaler(v, sa_threshold = 3.5, sigma = 1):
            if v < sa_threshold:
                return 1
            else:
                return gauss(v, mu = sa_threshold, sigma = sigma)


        #minizing IPE value. Note that ipe_val = 0 when calculation failed or mol had charge.
        logP, sascore, ipe_val = values
        ipe_val = 100 if ipe_val == 0 else ipe_val
        ipe_score = 1 - 0.1*ipe_val / (1 + 0.1*ipe_val)
        score = ipe_score * sa_scaler(sascore, sa_threshold = 3)
        #print(ipe_val, sascore, score, ipe_score)

        return score
