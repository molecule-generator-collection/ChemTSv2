import argparse
from math import sqrt, log
import random
import os
import time
import yaml

import numpy as np

from utils.utils import chem_kn_simulation, make_input_smiles, predict_smiles, evaluate_node, node_to_add, expanded_node
from utils.load_model import loaded_model
from utils.make_smiles import zinc_data_with_bracket_original, zinc_processed_with_bracket
from reward.logP_reward import scaling_score


def get_parser():
    parser = argparse.ArgumentParser(
        description="",
        usage=f"python {os.path.basename(__file__)} -c CONFIG_FILE"
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True,
        help="path to a config file"
    )
    return parser.parse_args()


class State:
    def __init__(self):
        self.position = ['&']
        self.num_atom = 8
        
    def Clone(self):
        st = State()
        st.position = self.position[:]
        return st

    def SelectPosition(self, m):
        self.position.append(m)

    def Getatom(self):
        return [i for i in range(self.num_atom)]

class Node:
    def __init__(self, position=None, parent=None, state=None, conf=None):
        self.position = position
        self.parentNode = parent
        self.childNodes = []
        self.child = None
        self.wins = 0
        self.visits = 0
        self.nonvisited_atom = state.Getatom()
        self.type_node = []
        self.depth = 0
        self.conf = conf

    def Selectnode(self):
        ucb=[]
        print('UCB:')
        for i in range(len(self.childNodes)):
            ucb_tmp = (self.childNodes[i].wins / self.childNodes[i].visits
                + self.conf['c_val'] * sqrt(2 * log(self.visits) / self.childNodes[i].visits)
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
        n = Node(position=m, parent=self, state=s, conf=self.conf)
        self.childNodes.append(n)

    def simulation(self):
        raise SystemExit("[ERROR] Do NOT use this method")

    def Update(self, result):
        self.visits += 1
        self.wins += result


def MCTS(root, conf, val, model, verbose=False):
    """initialization of the chemical trees and grammar trees"""
    start_time = time.time()
    run_time = time.time() + 3600 * conf['hours']
    rootnode = Node(state=root, conf=conf)
    state = root.Clone()

    """global variables used for save valid compounds and simulated compounds"""
    valid_compound = []
    depth = []
    min_score = 1000
    score_distribution = []
    min_score_distribution = []
    generated_dict = {}  # dictionary of generated compounds

    out_f = open(os.path.join(conf['output_dir'], f"result_C{conf['c_val']}_trial{conf['trial']}.txt"), 'a')

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
        expanded = expanded_node(model, state.position, val, conf['max_len'])
        
        new_compound = []
        nodeadded = []
        for _ in range(conf['simulation_num']):
            nodeadded_tmp = node_to_add(expanded, val)
            all_posible = chem_kn_simulation(model, state.position, val, nodeadded_tmp, conf['max_len'])
            generate_smiles = predict_smiles(all_posible, val)
            new_compound_tmp = make_input_smiles(generate_smiles)
            nodeadded.extend(nodeadded_tmp)
            new_compound.extend(new_compound_tmp)
        print(f"nodeadded {nodeadded}\n"
              f"new compound {new_compound}\n"
              f"generated_dict {generated_dict}\n") 
        for comp in new_compound:
            print(f"lastcomp {comp[-1]} ... ", comp[-1] == '\n')
        node_index, score, valid_smiles, generated_dict = evaluate_node(
            new_compound,
            generated_dict,
            sa_threshold=conf['sa_threshold'],
            rule=conf['rule5'],
            radical=conf['radical_check'],
            hashimoto_filter=conf['hashimoto_filter'],
            trial=conf['trial'],
            )
        valid_compound.extend(valid_smiles)
        score_distribution.extend(score)
        
        print(f"node {node_index} score {score} valid {valid_smiles}")
        out_f.write(f"{valid_smiles}\t{score}\t{min_score}\t{len(state.position)}\t{time.time()-start_time}\n")
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

                print(f"current minmum score: {min_score}")
                if score[i][0] <= min_score:
                    min_score_distribution.append(score[i][0])
                    min_score = score[i][0]
                else:
                    min_score_distribution.append(min_score)

                # Score scaling
                re = -1 if atom == '\n' else scaling_score(scores=score[i], conf=conf)
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


def update_config(conf):
    conf.setdefault('trial', 1)
    conf.setdefault('c_val', 1.0)
    conf.setdefault('hours', 1) 
    conf.setdefault('sa_threshold', 3.5)  #if SA > sa_threshold, score = 0. Default sa_threshold = 10
    conf.setdefault('rule5', 1)  #0:none, 1: rule of 5, 2: rule of 3  #RO5: if a compound does not satisfy rule of 5, score = 0.
    conf.setdefault('radical_check', True)
    conf.setdefault('simulation_num', 3)
    conf.setdefault('hashimoto_filter', True)  # or False, use/not use hashimoto filter 
    conf.setdefault('model_json', 'model/model.json')
    conf.setdefault('model_weight', 'model/model.h5')
    conf.setdefault('output_dir', 'result')


def main():
    args = get_parser()
    with open(args.config, "r") as f:
        conf = yaml.load(f, Loader=yaml.SafeLoader)
    update_config(conf)
    os.makedirs(conf['output_dir'], exist_ok=True)
    model = loaded_model(conf['model_json'], conf['model_weight'])  #WM300 not tested  
    conf["max_len"] = model.input_shape[1]
    print(f"========== Configuration ==========")
    for k, v in conf.items():
        print(f"{k}: {v}")
    print(f"===================================")

    smiles_old = zinc_data_with_bracket_original('data/250k_rndm_zinc_drugs_clean.smi')
    val, _ = zinc_processed_with_bracket(smiles_old)
    print(f"val is {val}")
    with open(os.path.join(conf['output_dir'], f"result_C{conf['c_val']}_trial{conf['trial']}.txt"), 'w') as f:
        f.write('#valid_smiles\tscore\tmin_score\tdepth\tused_time\n')

    state = State()
    _ = MCTS(root=state, conf=conf, val=val, model=model, verbose=False)


if __name__ == "__main__":
    main()
