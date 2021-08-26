import sys
import os
# check SBMolGen_PATH setting
if os.getenv('SBMolGen_PATH') == None:
    print("THe SBMolGen_PATH has not defined, please set it before use it!")
    exit(0)
else:
    SBMolGen_PATH=os.getenv('SBMolGen_PATH')
    sys.path.append(SBMolGen_PATH+'/utils')
from math import *
import random as pr
import numpy as np
import time
from load_model import loaded_model
from make_smile import zinc_data_with_bracket_original, zinc_processed_with_bracket
from add_node_type_zinc import chem_kn_simulation, make_input_smile,predict_smile,check_node_type,node_to_add,expanded_node
import yaml

class chemical:
    def __init__(self):

        self.position=['&']
        self.num_atom=8
        self.vl = ['\n', '&', 'C', '1', 'N', '[C@@H]', '2', '[C@H]', '(', '=', 'O', ')', 'S', 'c', '[S@]', '[nH]', '[O-]', '[N+]', 'n', 'F', '#', '[C@]', '[C@@]', '[S@@]', 'P', '/', '\\', 'Cl', 's', 'Br', 'o', '[NH3+]', 'I', '[n+]', '[nH+]', '3', '[N-]', '[S-]', 'B', '4', '5', '[NH+]', '[Si]', '[P@]', '[NH2+]', '[P@@]', '[N@+]', '6', '[N@@+]', '[S@@+]', '7', '8', '[P@@H]', '[n-]', '[C-]', '[P+]', '[Cu]', '[Ni]', '[Zn]', '[Au-]', '[OH+]']
        
    def Clone(self):

        st = chemical()
        st.position= self.position[:]
        return st

    def SelectPosition(self,m):
        self.position.append(m)

    def Getatom(self):
        return [i for i in range(self.num_atom)]

class Node:
    def __init__(self, position = None,  parent = None, state = None):
        self.position = position
        self.parentNode = parent
        self.childNodes = []
        self.child=None
        self.wins = 0
        self.visits = 0
        self.nonvisited_atom=state.Getatom()
        self.type_node=[]
        self.depth=0


    def Selectnode(self):
        ucb=[]
        print('UCB:')
        for i in range(len(self.childNodes)):
            ucb_tmp = self.childNodes[i].wins/self.childNodes[i].visits+ c_val*sqrt(2*log(self.visits)/self.childNodes[i].\
visits)
            ucb.append(ucb_tmp)
            print(self.childNodes[i].position, ucb_tmp,) 
        m = np.amax(ucb)
        indices = np.nonzero(ucb == m)[0]
        ind=pr.choice(indices)
        s=self.childNodes[ind]
        print('\n', 'index', ind, self.position, m,) 
        return s

    def Addnode(self, m, s):
        n = Node(position = m, parent = self, state = s)
        self.childNodes.append(n)

    def simulation(self,state):
        predicted_smile=predict_smile(model,state)
        input_smile=make_input_smile(predicted_smile)
        logp,valid_smile,all_smile=logp_calculation(input_smile)

        return logp,valid_smile,all_smile

    def Update(self, result):
        self.visits += 1
        self.wins += result


def MCTS(root, verbose = False):
    """initialization of the chemical trees and grammar trees"""
    start_time = time.time()
    run_time = time.time() + 3600*hours # 3600*24
    rootnode = Node(state = root)
    state = root.Clone()
    """----------------------------------------------------------------------"""


    """global variables used for save valid compounds and simulated compounds"""
    valid_compound=[]
    depth=[]
    min_score=1000
    score_distribution=[]
    min_score_distribution=[]
    
    generated_dict = {} #dictionary of generated compounds
    dict_id = 1  ## this id used for save best docking pose.
    """----------------------------------------------------------------------"""
    out_f = open(output_dir, 'a')

    while time.time()<=run_time:
        node = rootnode # important !    this node is different with state / node is the tree node
        state = root.Clone() # but this state is the state of the initialization .  too important !!!
        """selection step"""
        node_pool=[]

        while node.childNodes!=[]:
            node = node.Selectnode()
            state.SelectPosition(node.position)
        print("state position:,",state.position)

        if len(state.position)>= 70:
            re= -1.0
            while node != None:
                node.Update(re)
                node = node.parentNode
            continue
        if node.position == '\n':
            re = -1.0
            while node != None:
                node.Update(re)
                node = node.parentNode
            continue
        
        """------------------------------------------------------------------"""

        """expansion step"""
        expanded=expanded_node(model,state.position,val,loop_num_nodeExpansion)
        
        new_compound = []
        nodeadded = []
        for n in range(simulation_num):
            nodeadded_tmp = node_to_add(expanded, val)
            all_posible=chem_kn_simulation(model,state.position,val,nodeadded_tmp)
            generate_smile=predict_smile(all_posible,val)
            new_compound_tmp = make_input_smile(generate_smile)
            nodeadded.extend(nodeadded_tmp)
            new_compound.extend(new_compound_tmp)
        print('nodeadded', nodeadded) 
        print('new compound', new_compound)
        print('generated_dict', generated_dict)
        print('dict_id', dict_id)
        for comp in new_compound:
            print('lastcomp', comp[-1], ' ... ',comp[-1] == '\n')
        node_index,rdock_score,valid_smile,generated_dict = check_node_type(new_compound, score_type, generated_dict, sa_threshold = sa_threshold, rule = rule5, radical = radical_check, docking_num = docking_num, target_dir = target_dir, hashimoto_filter = hashimoto_filter, dict_id = dict_id, trial = trial)
        valid_compound.extend(valid_smile)
        score_distribution.extend(rdock_score)
        
        print('node', node_index, 'rdock_score', rdock_score, 'valid', valid_smile)
        out_f.write(str(valid_smile) + ', '+ str(rdock_score)+', '+str(min_score)+', '+str(len(state.position))+', '+str(time.time()-start_time))
        out_f.write('\n')
        out_f.flush()
        dict_id += 1

        if len(node_index)==0:
            re=-1.0
            while node != None:
                node.Update(re)
                node = node.parentNode
        else:
            re_list = []
            atom_checked = []
            for i in range(len(node_index)):
                m=node_index[i]
                atom = nodeadded[m]
                
                if atom not in atom_checked: 
                    node.Addnode(atom, state)
                    node_pool.append(node.childNodes[len(atom_checked)])
                    depth.append(len(state.position))
                    atom_checked.append(atom)
                else:
                    node_pool.append(node.childNodes[atom_checked.index(atom)])
                
                for child in node.childNodes:
                    print(child.position)
                print('\n')
                
                score_index = 0 if score_type == 'SCORE' else 1

                print("current minmum score",min_score)
                if rdock_score[i][score_index]<=min_score:
                    min_score_distribution.append(rdock_score[i][score_index])
                    min_score=rdock_score[i][score_index]
                else:
                    min_score_distribution.append(min_score)
                """simulation"""
                
                if atom == '\n':
                    re = -1
                else:
                    re=(- (rdock_score[i][score_index] - base_rdock_score)*0.1)/(1+abs(rdock_score[i][score_index] -base_rdock_score)*0.1)
                re_list.append(re)
                print('atom', atom, 're_list', re_list)
                """backpropation step"""

            for i in range(len(node_pool)):

                node=node_pool[i]
                while node != None:
                    node.Update(re_list[i])
                    node = node.parentNode
            
            for child in node_pool:
                print(child.position, child.wins, child.visits)
            

    out_f.close()
                    
    """check if found the desired compound"""
    print("rdock_score",score_distribution)
    print("num valid_compound:",len(valid_compound))
    print("valid compounds",valid_compound)
    print("depth",depth)
    print("min_score",min_score_distribution)

    return valid_compound


def UCTchemical():
    state = chemical()
    best = MCTS(root = state,verbose = False)

    return best


if __name__ == "__main__":
    # set parameter
    argvs = sys.argv

    """read yaml file for configuration"""
    f = open(str(argvs[1]), "r+")
    conf = yaml.load(f, Loader=yaml.SafeLoader)
    f.close()

    trial = conf.get('trial', 1)
    c_val = conf.get('c_val', 1.0)
    loop_num_nodeExpansion = conf.get('loop_num_nodeExpansion', 1000) 
    target = conf.get('target', 'CDK2')
    target_dir = conf.get('target_path', './')
    hours = conf.get('hours', 1) 
    score_type = conf.get('score_type', 'SCORE.INTER') #<SCORE> or <SCORE.INTER>
    docking_num = conf.get('docking_num', 10)
    sa_threshold = conf.get('sa_threshold', 3.5) #if SA > sa_threshold, score = 0. Default sa_threshold = 10
    rule5 = conf.get('rule5', 1) #0:none, 1: rule of 5, 2: rule of 3  #RO5: if a compound does not satisfy rule of 5, score = 0.
    radical_check = conf.get('radical_check', True)
    simulation_num = conf.get('simulation_num', 3)
    hashimoto_filter = conf.get('hashimoto_filter', True)  # or False, use/not use hashimoto filter 
    base_rdock_score = conf.get('base_rdock_score', -20)
    model_name = conf.get('model_name', 'model')

    print('========== display configuration ==========')
    print('trial num is: ', trial)
    print('c_val: ', c_val)
    print('loop_num_nodeExpansion: ', loop_num_nodeExpansion)
    print('target: ', target)
    print('target_dir: ',target_dir)
    print('max run time: ',hours)
    print('score_type: ', score_type)
    print('docking_num: ',docking_num)
    print('sa_threshold: ',sa_threshold)
    print('model_name: ', model_name)
    print('base_rdock_score: ', base_rdock_score)
    print('simulation_num: ',simulation_num)
    print('hashimoto_filter: ', hashimoto_filter)
    """----------------------------------------------------------------------"""

    output_dir = 'result_'+target+'_C'+str(c_val)+'_trial'+str(trial)+'.txt'

    smile_old=zinc_data_with_bracket_original(SBMolGen_PATH + '/data/250k_rndm_zinc_drugs_clean.smi')
    val,smile=zinc_processed_with_bracket(smile_old)
    print('val is ', val)

    out_f = open(output_dir, 'w')
    out_f.write('#valid_smile, rdock_score, min_score, depth, used_time')
    out_f.write('\n')
    out_f.close()

    model=loaded_model(SBMolGen_PATH + '/RNN-model/'+ model_name)  #WM300 not tested  
    valid_compound=UCTchemical()
