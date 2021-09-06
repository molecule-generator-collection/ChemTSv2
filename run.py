import argparse
from math import sqrt, log
import random
import os
import time
import yaml

import numpy as np
import pandas as pd

from utils.utils import chem_kn_simulation, make_input_smiles, predict_smiles, evaluate_node, node_to_add, expanded_node, back_propagation
from utils.load_model import loaded_model
from utils.make_smiles import zinc_data_with_bracket_original, zinc_processed_with_bracket
from reward.logP_reward import calc_reward_score


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
    valid_smiles_list = []
    depth_list = []
    objective_value_list = []
    elapsed_time_list = []
    generated_dict = {}  # dictionary of generated compounds

    while time.time()<=run_time:
        node = rootnode  # important! This node is different with state / node is the tree node
        state = root.Clone()  # but this state is the state of the initialization. Too important!

        """selection step"""
        node_pool = []
        while node.childNodes!=[]:
            node = node.Selectnode()
            state.SelectPosition(node.position)
        print(f"state position: {state.position}")

        if len(state.position) >= 70 or node.position == '\n':
            back_propagation(node, reward=-1.0)
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
        node_index, score, valid_smiles, generated_dict = evaluate_node(new_compound, generated_dict, conf)

        valid_smiles_list.extend(valid_smiles)
        depth = len(state.position)
        depth_list.extend([depth for _ in range(len(valid_smiles))])
        elapsed_time = round(time.time()-start_time, 1)
        elapsed_time_list.extend([elapsed_time for _ in range(len(valid_smiles))])
        objective_value_list.extend(score)
        
        print(f"node {node_index} score {score} valid {valid_smiles} time {elapsed_time}")

        if len(node_index) == 0:
            back_propagation(node, reward=-1.0)
        else:
            re_list = []
            atom_checked = []
            for i in range(len(node_index)):
                m = node_index[i]
                atom = nodeadded[m]
                
                if atom not in atom_checked: 
                    node.Addnode(atom, state)
                    node_pool.append(node.childNodes[len(atom_checked)])
                    atom_checked.append(atom)
                else:
                    node_pool.append(node.childNodes[atom_checked.index(atom)])
                
                for child in node.childNodes:
                    print(child.position)
                print('\n')

                # calculate reward score
                re = -1 if atom == '\n' else calc_reward_score(scores=score[i], conf=conf)
                re_list.append(re)
                print(f"atom: {atom} re_list: {re_list}")

            """backpropation step"""
            for i in range(len(node_pool)):
                node = node_pool[i]
                back_propagation(node, reward=re_list[i])
            
            for child in node_pool:
                print(child.position, child.wins, child.visits)
                    
    """check if found the desired compound"""
    print(f"num valid_smiles: {len(valid_smiles_list)}\n"
          f"valid smiles: {valid_smiles_list}\n"
          f"depth: {depth_list}\n"
          f"objective value: {objective_value_list}\n"
          f"time: {elapsed_time_list}")
    df = pd.DataFrame({
        "smiles": valid_smiles_list,
        "objective_value": objective_value_list,
        "depth": depth_list,
        "elapsed_time": elapsed_time_list,
    })
    return df


def set_default_config(conf):
    conf.setdefault('trial', 1)
    conf.setdefault('c_val', 1.0)
    conf.setdefault('hours', 1) 
    conf.setdefault('sa_threshold', 3.5)
    conf.setdefault('use_lipinski_filter', 'rule_of_5')  # rule_of_5, rule_of_3
    conf.setdefault('radical_check', True)
    conf.setdefault('simulation_num', 3)
    conf.setdefault('use_hashimoto_filter', True) 
    conf.setdefault('model_json', 'model/model.json')
    conf.setdefault('model_weight', 'model/model.h5')
    conf.setdefault('output_dir', 'result')


def main():
    args = get_parser()
    with open(args.config, "r") as f:
        conf = yaml.load(f, Loader=yaml.SafeLoader)
    set_default_config(conf)
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

    state = State()
    df = MCTS(root=state, conf=conf, val=val, model=model, verbose=False)
    output_path = os.path.join(conf['output_dir'], f"result_C{conf['c_val']}.pkl")
    print(f"[INFO] save results at {output_path}")
    df.to_pickle(output_path)


if __name__ == "__main__":
    main()
