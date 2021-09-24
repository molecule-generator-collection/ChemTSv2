from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess
from multiprocessing import Process
import os
import shutil
import time

def docking_calculation(cmd):
    proc = subprocess.call( cmd , shell=True)  
    #logger.info('end in thread')

def calc_objective_values(smiles, cmpd_id, dict_id, logger, score_type='SCORE.INTER', target_dir='', docking_num = 10):
    # check RBT_ROOT setting:
    RBT_ROOT=os.environ.get('RBT_ROOT')
    if os.getenv('RBT_ROOT') == None:
        logger.info("The RBT_ROOT has not defined, please set it before use it!")
        exit(10000)
    else:
        RBT_ROOT=os.getenv('RBT_ROOT')
        logger.info('RBT_ROOT is: ', RBT_ROOT)
    
    num_cpus = docking_num # total number of docking    

    sdf_file = 'tmp_comp.sdf' # The 3D structure of generaed molecular used in docking simulation.
    docking_result_file = 'rdock_out_'
    min_score = 10**10
    min_score_inter = 10**10
    min_score_id = 0
    min_score_inter_id = 0

    path=os.getcwd()
    os.chdir(path)
    
    #----Translation from SMILES to sdf
    fw = Chem.SDWriter(sdf_file)
    m1 = Chem.MolFromSmiles(smiles)
    try:
        if m1!= None:
            m = Chem.AddHs(m1)

            cid = AllChem.EmbedMolecule(m)
            #fw.write(m)

            opt = AllChem.UFFOptimizeMolecule(m,maxIters=200)
            logger.info('3D structure optimization by AllChem.UFFOptimizeMolecule(m,maxIters=200) :', opt)

            fw.write(m)
            fw.close()

            #----rdock calculation
            start_time = time.time()
            processes = []
            os.system(f"cp {target_dir}/cavity.* ./")
            os.system(f"cp {target_dir}/receptor.mol2 ./")
            for i in range(num_cpus):
                cmd = RBT_ROOT + '/bin/rbdock -allH -r cavity.prm -p '+ RBT_ROOT + '/data/scripts/dock.prm -i ' + sdf_file + ' -o ' + docking_result_file + str(i) + ' -T 0 -s '+str(i)+' -n 1' 
                logger.debug('cmd', cmd)
                t = Process(target=docking_calculation, args=(cmd,))
                processes.append(t)

            for p in processes:
                p.start()
		
            for p in processes:
                p.join()

            logger.info('docking simulation is end!')
            end_time = time.time()
            logger.info('docking time_used', end_time - start_time)

            for i in range(num_cpus):
                #----find the minimum score of rdock from multiple docking results
                f = open(docking_result_file+str(i)+'.sd')
                lines = f.readlines()
                f.close()

                line_count = 0
                score_line = -1
                score_inter_line = -1
                for line in lines:
                    v_list = line.split()
                    if line_count == score_line:
                        logger.debug(v_list[0])
                        if float(v_list[0]) < min_score:
                            min_score = float(v_list[0])
                            min_score_id = i
                    
                    if line_count == score_inter_line:
                        logger.debug(v_list[0])
                        if float(v_list[0]) < min_score_inter:
                            min_score_inter = float(v_list[0])
                            min_score_inter_id = i

                    if len(v_list) <= 1:
                        line_count += 1
                        continue
                        
                    if v_list[1] == '<SCORE>':
                        score_line = line_count + 1
                    if v_list[1] == '<SCORE.INTER>':
                        score_inter_line = line_count + 1
                    
                    line_count += 1
    except:
        logger.info('smiles error')
        min_score=10**10
        min_score_inter = 10**10

    os.chdir(path)
    
    best_docking_id = min_score_id if score_type == 'SCORE' else min_score_inter_id
    min_score = min_score if score_type == 'SCORE' else min_score_inter
    return [min_score, best_docking_id, f"3D_pose_{dict_id}_{cmpd_id}_{best_docking_id}.sd"]

def calc_reward_from_objective_values(values, conf):
    base_rdock_score = conf['base_docking_score']
    re = (- (values - base_rdock_score)*0.1) / (1+abs(values - base_rdock_score)*0.1)
    return re