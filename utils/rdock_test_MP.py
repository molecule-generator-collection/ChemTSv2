from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess
#import threading
from multiprocessing import Process
import os
import shutil
import time

def docking_calculation(cmd):
    proc = subprocess.call( cmd , shell=True)  
    print('end in thread')

#----parameter & preparation
def rdock_score(compound, score_type, target_dir, docking_num = 10):

    # check RBT_ROOT setting:
    RBT_ROOT=os.environ.get('RBT_ROOT')
    if os.getenv('RBT_ROOT') == None:
        print("The RBT_ROOT has not defined, please set it before use it!")
        exit(0)
    else:
        RBT_ROOT=os.getenv('RBT_ROOT')
        print('RBT_ROOT is: ', RBT_ROOT)

    input_smiles = str(compound)
    
    num_cpus = docking_num # total number of docking    
    score_name = score_type # SCORE or SCORE.INTER  ## edit by Biao Ma at 20190920

    target_dir = target_dir
    sdf_file = 'tmp_comp.sdf'
    docking_result_file = 'rdock_out_'
    min_score = 10**10
    min_score_inter = 10**10
    min_score_id = 0
    min_score_inter_id = 0

    path=os.getcwd()
    os.chdir(path)
    
    #----Translation from SMILES to sdf
    fw = Chem.SDWriter(sdf_file)
    m1 = Chem.MolFromSmiles(input_smiles)
    try:
        if m1!= None:
            m = Chem.AddHs(m1)

            cid = AllChem.EmbedMolecule(m)
            #fw.write(m)

            opt = AllChem.UFFOptimizeMolecule(m,maxIters=200)
            print('3D structure optimization by AllChem.UFFOptimizeMolecule(m,maxIters=200) :', opt)

            fw.write(m)
            fw.close()

            #----rdock calculation
            #cmd = '/home/terayama/rdock/fiexed_side_chain/build/exe/rbdock -r cavity.prm -p /home/terayama/rdock/fiexed_side_chain/data/scripts/dock.prm -i ' + sdf_file + ' -o ' + docking_result_file + ' -T 0 -n' + str(num_docking_unit)
            #proc = subprocess.call( cmd , shell=True)
            #proc = subprocess.popen(cmd, shell=True)
            #r1 = p1.wait()
            #print('r1', p1)
            
            start_time = time.time()
            processes = []
            for i in range(num_cpus):
                #cmd = '/home/terayama/rdock/fiexed_side_chain/build/exe/rbdock -r cavity.prm -p /home/terayama/rdock/fiexed_side_chain/data/scripts/dock.prm -i ' + sdf_file + ' -o ' + docking_result_file + str(i) + ' -T 0 -s '+str(i)+' -n 1'
                #cmd = 'rbdock -r cavity.prm -p /home/mki/SideEffectPrediction/08_rDock/prm/dock.prm -i ' + sdf_file + ' -o ' + docking_result_file + str(i) + ' -T 0 -s '+str(i)+' -n 1' 
                cmd = RBT_ROOT + '/bin/rbdock -allH -r '+ target_dir + '/cavity.prm -p '+ RBT_ROOT + '/data/scripts/dock.prm -i ' + sdf_file + ' -o ' + docking_result_file + str(i) + ' -T 0 -s '+str(i)+' -n 1' 
                print('cmd', cmd)
                #t = threading.Thread(target=docking_calculation, args=(cmd,))
                #threads.append(t)
                #t.start()
                t = Process(target=docking_calculation, args=(cmd,))
                processes.append(t)
            for p in processes:
                p.start()
		
            for p in processes:
                p.join()
            print('end')
            end_time = time.time()
            #print('time used:%s' % end_time - start_time)
            print('docking time_used', end_time - start_time)


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
                        print(v_list[0])
                        if float(v_list[0]) < min_score:
                            min_score = float(v_list[0])
                            min_score_id = i
                    
                    if line_count == score_inter_line:
                        print(v_list[0])
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


            print('minimum rdock score', min_score, 'score_inter', min_score_inter)
    except:
        print('error')
        min_score=10**10
        min_score_inter = 10**10

    os.chdir(path)
    #os.mkdir('out') # creat the directory for rdock best output for new compound.
    best_docking_id = min_score_id if score_type == 'SCORE' else min_score_inter_id
    return [min_score, min_score_inter, best_docking_id]


#rdock_score('Oc1ccc(NC(=O)C)cc1', 2, '<SCORE.INTER>', docking_num = 1, protein_target = 'kith')
