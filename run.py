from math import sqrt, log
import random
import sys
import time
import yaml

import numpy as np

from utils.add_node_type_zinc import chem_kn_simulation, make_input_smile, predict_smile, check_node_type, node_to_add, expanded_node
from utils.load_model import loaded_model
from utils.make_smiles import zinc_data_with_bracket_original, zinc_processed_with_bracket


class chemical:
    def __init__(self):
        self.position = ['&']
        self.num_atom = 8
        
    def Clone(self):
        st = chemical()
        st.position = self.position[:]
        return st

    def SelectPosition(self, m):
        self.position.append(m)

    def Getatom(self):
        return [i for i in range(self.num_atom)]

class Node:
    def __init__(self, position=None, parent=None, state=None):
        self.position = position
        self.parentNode = parent
        self.childNodes = []
        self.child = None
        self.wins = 0
        self.visits = 0
        self.nonvisited_atom = state.Getatom()
        self.type_node = []
        self.depth = 0

    def Selectnode(self):
        ucb=[]
        print('UCB:')
        for i in range(len(self.childNodes)):
            ucb_tmp = (self.childNodes[i].wins / self.childNodes[i].visits
                + c_val * sqrt(2 * log(self.visits) / self.childNodes[i].visits)
                )
            ucb.append(ucb_tmp)
            print(f"{self.childNodes[i].position} {ucb_tmp}") 
        m = np.amax(ucb)
        indices = np.nonzero(ucb == m)[0]
        ind = random.choice(indices)
        s = self.childNodes[ind]
        print(f"\nindex {ind} {self.position} {m}") 
        return s

    def Addnode(self, m, s):
        n = Node(position=m, parent=self, state=s)
        self.childNodes.append(n)

    def simulation(self):
        raise SystemExit("[ERROR] Do NOT use this method")

    def Update(self, result):
        self.visits += 1
        self.wins += result


def MCTS(root, verbose = False):
    """initialization of the chemical trees and grammar trees"""
    start_time = time.time()
    #run_time = time.time() + 3600*hours # 3600*24
    run_time = time.time() + 60
    rootnode = Node(state=root)
    state = root.Clone()

    """global variables used for save valid compounds and simulated compounds"""
    valid_compound = []
    depth = []
    min_score = 1000
    score_distribution = []
    min_score_distribution = []
    generated_dict = {}  # dictionary of generated compounds

    out_f = open(output_file, 'a')

    while time.time()<=run_time:
        node = rootnode  # important! This node is different with state / node is the tree node
        state = root.Clone()  # but this state is the state of the initialization. Too important!

        """selection step"""
        node_pool = []
        while node.childNodes!=[]:
            node = node.Selectnode()
            state.SelectPosition(node.position)
        print(f"state position: {state.position}")

        if len(state.position) >= 70:
            re = -1.0
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

        """expansion step"""
        expanded = expanded_node(model, state.position, val, loop_num_nodeExpansion)
        
        new_compound = []
        nodeadded = []
        for _ in range(simulation_num):
            nodeadded_tmp = node_to_add(expanded, val)
            all_posible = chem_kn_simulation(model,state.position, val, nodeadded_tmp)
            generate_smile = predict_smile(all_posible, val)
            new_compound_tmp = make_input_smile(generate_smile)
            nodeadded.extend(nodeadded_tmp)
            new_compound.extend(new_compound_tmp)
        print(f"nodeadded {nodeadded}\n"
              f"new compound {new_compound}\n"
              f"generated_dict {generated_dict}\n") 
        for comp in new_compound:
            print(f"lastcomp {comp[-1]} ... ", comp[-1] == '\n')
        node_index, score, valid_smile, generated_dict = check_node_type(
            new_compound,
            generated_dict,
            sa_threshold=sa_threshold,
            rule=rule5,
            radical=radical_check,
            hashimoto_filter=hashimoto_filter,
            trial=trial,
            )
        valid_compound.extend(valid_smile)
        score_distribution.extend(score)
        
        print(f"node {node_index} score {score} valid {valid_smile}")
        out_f.write(f"{valid_smile}, {score}, {min_score}, {len(state.position)}, {time.time()-start_time}\n")
        out_f.flush()

        if len(node_index) == 0:
            re = -1.0
            while node != None:
                node.Update(re)
                node = node.parentNode
        else:
            re_list = []
            atom_checked = []
            for i in range(len(node_index)):
                m = node_index[i]
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

                print(f"current minmum score: {min_score}")
                if score[i][score_index] <= min_score:
                    min_score_distribution.append(score[i][score_index])
                    min_score = score[i][score_index]
                else:
                    min_score_distribution.append(min_score)

                """simulation"""
                if atom == '\n':
                    re = -1
                else:
                    re = ((-(score[i][score_index] - base_score) * 0.1)
                        / (1 + abs(score[i][score_index] - base_score) * 0.1))
                re_list.append(re)
                print(f"atom: {atom} re_list: {re_list}")

            """backpropation step"""
            for i in range(len(node_pool)):
                node = node_pool[i]
                while node != None:
                    node.Update(re_list[i])
                    node = node.parentNode
            
            for child in node_pool:
                print(child.position, child.wins, child.visits)
    out_f.close()
                    
    """check if found the desired compound"""
    print(f"score: {score_distribution}\n"
          f"num valid_compound: {len(valid_compound)}\n"
          f"valid compounds: {valid_compound}\n"
          f"depth: {depth}\n"
          f"min_score: {min_score_distribution}")
    return valid_compound


def UCTchemical():
    state = chemical()
    best = MCTS(root=state, verbose=False)
    return best


if __name__ == "__main__":
    argvs = sys.argv
    """read yaml file for configuration"""
    with open(str(argvs[1]), "r+") as f:
        conf = yaml.load(f, Loader=yaml.SafeLoader)
    trial = conf.get('trial', 1)
    c_val = conf.get('c_val', 1.0)
    loop_num_nodeExpansion = conf.get('loop_num_nodeExpansion', 1000) 
    hours = conf.get('hours', 1) 
    score_type = conf.get('score_type', 'SCORE.INTER') #<SCORE> or <SCORE.INTER>
    docking_num = conf.get('docking_num', 10)
    sa_threshold = conf.get('sa_threshold', 3.5) #if SA > sa_threshold, score = 0. Default sa_threshold = 10
    rule5 = conf.get('rule5', 1) #0:none, 1: rule of 5, 2: rule of 3  #RO5: if a compound does not satisfy rule of 5, score = 0.
    radical_check = conf.get('radical_check', True)
    simulation_num = conf.get('simulation_num', 3)
    hashimoto_filter = conf.get('hashimoto_filter', True)  # or False, use/not use hashimoto filter 
    base_score = conf.get('base_score', -20)
    model_name = conf.get('model_name', 'model')
    print(f"========== Configuration ==========\n"
          f"trial num is: {trial}\n"
          f"c_val: {c_val}\n"
          f"loop_num_nodeExpansion: {loop_num_nodeExpansion}\n"
          f"max run time: {hours}\n"
          f"score_type: {score_type}\n"
          f"docking_num: {docking_num}\n"
          f"sa_threshold: {sa_threshold}\n"
          f"model_name: {model_name}\n"
          f"base_score: {base_score}\n"
          f"simulation_num: {simulation_num}\n"
          f"hashimoto_filter: {hashimoto_filter}")

    output_file = f"result_C{c_val}_trial{trial}.txt"

    smile_old = zinc_data_with_bracket_original('data/250k_rndm_zinc_drugs_clean.smi')
    val, smile = zinc_processed_with_bracket(smile_old)
    print(f"val is {val}")

    with open(output_file, 'w') as f:
        f.write('#valid_smile, score, min_score, depth, used_time\n')

    model = loaded_model('model/' + model_name)  #WM300 not tested  
    valid_compound = UCTchemical()
