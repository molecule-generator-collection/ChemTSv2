import numpy as np
from rdkit import Chem
import pickle
import subprocess
import time
import shutil
import os

def get_jobID(submitted_jobID_list):
    
    qstat = subprocess.run('qstat', encoding='utf-8', stdout=subprocess.PIPE)
    pooled_all_jobID_list = [l.split()[0] for l in (qstat.stdout).split('\n')[2:-1]]
    pooled_jobID_list = [jobID for jobID in pooled_all_jobID_list if jobID in submitted_jobID_list]
    
    return pooled_jobID_list

def check_values(valid_conf_list):
    conf = valid_conf_list[0]
    result_dir = os.path.join(conf['output_dir'], 'gaussian_result')
    
    for i, mol in enumerate(valid_conf_list):
        conf = valid_conf_list[i]
        gid = conf['gid']
        calc_dir = f'InputMolopt{gid}'
        if not os.path.exists(os.path.join(result_dir, calc_dir, 'values.pickle')):
            return True
    return False
    

def run_qsub_parallel(valid_mol_list, reward_calculator, valid_conf_list):
    conf = valid_conf_list[0]
    n_cpus_qsub_parallel = conf['gau_total_core_num']
    cpu_cluster = conf['cpu_cluster']
   
    result_dir = os.path.join(conf['output_dir'], 'gaussian_result')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    
    if cpu_cluster == 'pcc-skl':
        gaussian_cores = 40
    elif cpu_cluster == 'pcc-normal':
        gaussian_cores = 28
    qsub_max = int(n_cpus_qsub_parallel/gaussian_cores)

    submitted_jobID_list = []
    objective_values_list = []
    for i, mol in enumerate(valid_mol_list):
        conf = valid_conf_list[i]
        conf['gau_core_num'] = gaussian_cores
        gid = conf['gid']

        calc_obj = [mol, conf]
        with open(f'calc_obj_{gid}.pickle', mode='wb') as f:
            pickle.dump(calc_obj, f)

        if cpu_cluster == 'pcc-skl':
            cp = subprocess.run(f'qsub misc/qsub_parallel_job_pcc-skl.sh calc_obj_{gid}.pickle', shell=True, encoding='utf-8', stdout=subprocess.PIPE)
        elif cpu_cluster == 'pcc-normal':
            cp = subprocess.run(f'qsub misc/qsub_parallel_job_pcc-normal.sh calc_obj_{gid}.pickle' , shell=True, encoding='utf-8', stdout=subprocess.PIPE)
        jobID = cp.stdout.split(' ')[2]
        submitted_jobID_list.append(jobID)
        pooled_jobID_list = get_jobID(submitted_jobID_list)

        while(len(pooled_jobID_list) >= qsub_max):
            time.sleep(0.5)
            pooled_jobID_list = get_jobID(submitted_jobID_list)
    
    while(len(pooled_jobID_list) > 0 or check_values(valid_conf_list)):
        print(len(pooled_jobID_list) > 0, check_values(valid_conf_list))
        time.sleep(0.5)
        pooled_jobID_list = get_jobID(submitted_jobID_list)
    
    objective_values_list = []
    for i, mol in enumerate(valid_mol_list):
        conf = valid_conf_list[i]
        gid = conf['gid']
        calc_dir = f'InputMolopt{gid}'
        with open(os.path.join(result_dir, calc_dir, 'values.pickle'), mode='rb') as f:
            values = pickle.load(f)
        objective_values_list.append(values)
        
        if cpu_cluster == 'pcc-skl':
            shutil.move('qsub_parallel_job_pcc-skl.sh.o'+submitted_jobID_list[i], os.path.join(result_dir, calc_dir))
            shutil.move('qsub_parallel_job_pcc-skl.sh.e'+submitted_jobID_list[i], os.path.join(result_dir, calc_dir))
        elif cpu_cluster == 'pcc-normal':
            shutil.move('qsub_parallel_job_pcc-normal.sh.o'+submitted_jobID_list[i], os.path.join(result_dir, calc_dir))
            shutil.move('qsub_parallel_job_pcc-normal.sh.e'+submitted_jobID_list[i], os.path.join(result_dir, calc_dir))
    
    return objective_values_list
    
    


if __name__ == "__main__":
    pass
