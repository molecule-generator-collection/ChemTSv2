import sys, os, subprocess, yaml
import pandas as pd
from pathlib import Path
import rdkit
from rdkit import Chem
from IPython.core.debugger import Pdb

from extend_driver import setup_custom_logger
from ChemTSv2.chemts_mothods import select_weight_model, set_rearrange_smiles, make_config_file, csv_to_mol2, plot_reward

import logging
logs_dir = 'logs'
out_log_file = os.path.join(logs_dir, 'ChemTS.log')
logger = setup_custom_logger('ChemTS', out_log_file)

cwd = os.path.dirname(os.path.abspath(__file__))
target_dirname = 'work/results'

def run(workflow_config, trajectory_dirs):
    for trajectory_dir in trajectory_dirs:
        logger.info(trajectory_dir)
        sincho_result_file = os.path.join(trajectory_dir, 'sincho_result.yaml')

        trajectory_name = trajectory_dir.split(os.sep)[-1]
        trajectory_num = trajectory_name.split('_')[-1]
        ChemTS_output_dir = os.path.join(workflow_config['GENERATE_WORKFLOW']['working_directory'], 'ChemTS')
        trajectory_output_dir = os.path.join(ChemTS_output_dir, trajectory_name)
        os.makedirs(trajectory_output_dir, exist_ok = True)
        
        with open(sincho_result_file, 'r')as f:
            sincho_results = yaml.safe_load(f)
        sincho_results = sincho_results['SINCHO_result']
        
        input_compound_file = os.path.join(trajectory_dir, 'lig_'+ trajectory_num + '.pdb')
        input_compound_smiles = Chem.MolToSmiles(Chem.MolFromPDBFile(input_compound_file))

        for rank, sincho_result in sincho_results.items():
            logger.info(f"rank , {rank}")

            estimate_add_mw = sincho_result['mw']
            weight_model_dir = select_weight_model(input_compound_smiles, estimate_add_mw)
            rearrange_smi = set_rearrange_smiles(input_compound_file, sincho_result['atom_num'])
            logger.info(f"smi , {input_compound_smiles}")
            logger.info(f"rearrange_smi , {rearrange_smi}")
            os.makedirs(os.path.join(cwd, 'work'), exist_ok=True)
            subprocess.run(['rm', os.path.join('work', '_setting.yaml')], cwd=cwd)
            make_config_file({**workflow_config, **sincho_result}, weight_model_dir)
            
            df_result_all = pd.DataFrame()
            for n in range(1, int(workflow_config['ChemTS']['num_chemts_loops'])+1):
                with open(out_log_file, 'a') as stdout_f:
                    subprocess.run(['python', 'run.py', '-c', os.path.join('work', '_setting.yaml'), '--input_smiles', rearrange_smi], cwd=cwd, stdout=stdout_f, stderr=stdout_f)
                    subprocess.run(' '.join(['mv', os.path.join(target_dirname, 'result_C*'), os.path.join(target_dirname, 'result.csv')]), shell=True, cwd=cwd, stdout=stdout_f, stderr=stdout_f)
                    df_result_one_cycle = pd.read_csv(os.path.join(cwd, target_dirname, 'result.csv'))
                    df_result_one_cycle.insert(0, 'trial', n) 
                    df_result_all = pd.concat([df_result_all, df_result_one_cycle])
                    subprocess.run(' '.join(['cat', os.path.join(target_dirname, 'run.log'), '>>', os.path.join(target_dirname, 'run.log.all')]), shell=True, cwd=cwd)
                    
            df_result_all.to_csv(os.path.join(cwd, target_dirname, 'results.csv'))
            subprocess.run(['rm', os.path.join(target_dirname, 'result.csv'), os.path.join(target_dirname, 'header'), os.path.join(target_dirname, 'record')], cwd=cwd)
            
            plot_reward(os.path.join(cwd, target_dirname, 'results.csv'))
            csv_to_mol2(csv = os.path.join(cwd, target_dirname, 'results.csv'), prefix = os.path.join(cwd, target_dirname, 'lead'), \
                        cutoff=float(workflow_config['Clustering']['cutoff']), nsamples = int(workflow_config['ChemTS']['num_chemts_pickups']), ligand_pdb = input_compound_file)

            rank_output_dir = os.path.join(trajectory_output_dir, rank)
            os.makedirs(rank_output_dir, exist_ok = True)

            subprocess.run(' '.join(['mv', os.path.join(cwd, target_dirname, '*'), rank_output_dir]), shell=True)

if __name__ == '__main__':
    args = sys.argv
    workflow_config_file = args[1]
    trajectory_dirs = args[2]
    run(workflow_config_file, trajectory_dirs)