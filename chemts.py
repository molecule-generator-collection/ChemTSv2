from math import sqrt, log
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import pickle

from misc.utils import chem_kn_simulation, make_input_smiles, predict_smiles, \
    evaluate_node, node_to_add, expanded_node, back_propagation


class State:
    def __init__(self, position=['&']):
        self.position = position
        self.num_atom = 8  # no longer used?
        
    def Clone(self):
        st = State()
        st.position = self.position[:]
        return st

    def SelectPosition(self, m):
        self.position.append(m)

    def Getatom(self):
        return [i for i in range(self.num_atom)]


class Node:
    def __init__(self, policy_evaluator, position=None, parent=None, state=None, conf=None):
        self.position = position
        self.parentNode = parent
        self.childNodes = []
        self.child = None
        self.wins = 0
        self.visits = 0
        self.nonvisited_atom = state.Getatom()  # no longer used?
        self.type_node = []
        self.depth = 0
        self.policy_evaluator = policy_evaluator
        self.conf = conf

    def Selectnode(self, logger):
        score_list = []
        logger.debug('UCB:')
        for i in range(len(self.childNodes)):
            score = self.policy_evaluator.evaluate(
                self.childNodes[i].wins, self.conf['c_val'], self.visits, self.childNodes[i].visits)
            score_list.append(score)
            logger.debug(f"{self.childNodes[i].position} {score}") 
        m = np.amax(score_list)
        indices = np.nonzero(score_list == m)[0]
        ind = random.choice(indices)
        s = self.childNodes[ind]
        logger.debug(f"\nindex {ind} {self.position} {m}") 
        return s

    def Addnode(self, m, s, policy_evaluator):
        n = Node(policy_evaluator, position=m, parent=self, state=s, conf=self.conf)
        self.childNodes.append(n)

    def simulation(self):
        raise SystemExit("[ERROR] Do NOT use this method")

    def Update(self, result):
        self.visits += 1
        self.wins += result


class MCTS:
    def __init__(self, root_state, conf, val, model, reward_calculator, policy_evaluator, logger):
        self.start_time = time.time()
        self.rootnode = Node(policy_evaluator, state=root_state, conf=conf)
        self.root_state = root_state
        self.conf = conf
        self.val = val
        self.model = model
        self.reward_calculator = reward_calculator
        self.policy_evaluator = policy_evaluator
        self.logger = logger

        self.valid_smiles_list = []
        self.depth_list = []
        self.objective_values_list = []
        self.reward_values_list = []
        self.elapsed_time_list = []
        self.generated_dict = {}  # dictionary of generated compounds
        self.generated_id_list = []
        self.filter_check_list = []
        self.total_valid_num = 0

        self.obj_column_names = [f.__name__ for f in self.reward_calculator.get_objective_functions(self.conf)]
        self.output_path = os.path.join(conf['output_dir'], f"result_C{conf['c_val']}.csv")
        if os.path.exists(self.output_path):
            sys.exit(f"[ERROR] {self.output_path} already exists. Please specify a different file name.")

        if conf['threshold_type'] == "time":
            self.threshold = time.time() + 3600 * conf['hours']
        elif conf['threshold_type'] == "generation_num":
            self.threshold = conf['generation_num']
        else:
            sys.exit("[ERROR] Specify 'threshold_type': [time, generation_num]")

    def flush(self):
        df = pd.DataFrame({
            "generated_id": self.generated_id_list,
            "smiles": self.valid_smiles_list,
            "reward": self.reward_values_list,
            "depth": self.depth_list,
            "elapsed_time": self.elapsed_time_list,
            "is_through_filter": self.filter_check_list,
        })
        df_obj = pd.DataFrame(self.objective_values_list, columns=self.obj_column_names)
        df = pd.concat([df, df_obj], axis=1)
        if os.path.exists(self.output_path):
            df.to_csv(self.output_path, mode='a', index=False, header=False)
        else:
            df.to_csv(self.output_path, mode='w', index=False)
        self.logger.info(f"save results at {self.output_path}")

        self.generated_id_list.clear()
        self.valid_smiles_list.clear()
        self.reward_values_list.clear()
        self.depth_list.clear()
        self.elapsed_time_list.clear()
        self.filter_check_list.clear()
        self.objective_values_list.clear()

    def search(self):
        """initialization of search tree"""
        if self.conf['restart'] and os.path.exists(os.path.join(self.conf['output_dir'], self.conf['checkpoint_file'])):
            self.logger.info("Load restart data from checkpoint file")
            counter_list = self.load_checkpoint()
            gid, infinite_loop_counter_for_selection, infinite_loop_counter_for_expansion, expanded_before = counter_list
        else:
            gid = 0
            infinite_loop_counter_for_selection = 0
            infinite_loop_counter_for_expansion = 0
            expanded_before = {}
        
        while (time.time() if self.conf['threshold_type']=="time" else self.total_valid_num) <= self.threshold:
            node = self.rootnode  # important! This node is different with state / node is the tree node
            state = self.root_state.Clone()  # but this state is the state of the initialization. Too important!

            """selection step"""
            node_pool = []
            while node.childNodes!=[]:
                node = node.Selectnode(self.logger)
                state.SelectPosition(node.position)
            self.logger.info(f"state position: {state.position}")

            self.logger.debug(f"infinite loop counter (selection): {infinite_loop_counter_for_selection}")
            if len(state.position) >= 70 or node.position == '\n':
                back_propagation(node, reward=-1.0)
                infinite_loop_counter_for_selection += 1
                if infinite_loop_counter_for_selection > self.conf['infinite_loop_threshold_for_selection']:
                    self.flush()
                    sys.exit('[WARN] Infinite loop is detected in the selection step. Change hyperparameters or RNN model.')
                continue
            else:
                infinite_loop_counter_for_selection = 0

            """expansion step"""
            expanded = expanded_node(self.model, state.position, self.val, self.logger, threshold=self.conf['expansion_threshold'])
            self.logger.debug(f"infinite loop counter (expansion): {infinite_loop_counter_for_expansion}")
            if set(expanded) == expanded_before:
                infinite_loop_counter_for_expansion += 1
                if infinite_loop_counter_for_expansion > self.conf['infinite_loop_threshold_for_expansion']:
                    self.flush()
                    sys.exit('[WARN] Infinite loop is detected in the expansion step. Change hyperparameters or RNN model.')
            else:
                infinite_loop_counter_for_expansion = 0
            expanded_before = set(expanded)

            new_compound = []
            nodeadded = []
            for _ in range(self.conf['simulation_num']):
                nodeadded_tmp = node_to_add(expanded, self.val, self.logger)
                all_posible = chem_kn_simulation(self.model, state.position, self.val, nodeadded_tmp, self.conf['max_len'])
                generate_smiles = predict_smiles(all_posible, self.val)
                new_compound_tmp = make_input_smiles(generate_smiles)
                nodeadded.extend(nodeadded_tmp)
                new_compound.extend(new_compound_tmp)

            _gids = list(range(gid, gid+len(new_compound)))
            gid += len(new_compound)

            self.logger.debug(f"nodeadded {nodeadded}")
            self.logger.info(f"new compound {new_compound}")
            self.logger.debug(f"generated_dict {self.generated_dict}") 
            if self.conf["debug"]:
                self.logger.debug('\n' + '\n'.join([f"lastcomp {comp[-1]} ... " + str(comp[-1] == '\n') for comp in new_compound]))
            node_index, objective_values, valid_smiles, generated_id_list, filter_check_list = evaluate_node(new_compound, self.generated_dict, self.reward_calculator, self.conf, self.logger, _gids)

            valid_num = len(valid_smiles)
            self.total_valid_num += valid_num
            self.valid_smiles_list.extend(valid_smiles)
            depth = len(state.position)
            self.depth_list.extend([depth for _ in range(valid_num)])
            elapsed_time = round(time.time()-self.start_time, 1)
            self.elapsed_time_list.extend([elapsed_time for _ in range(valid_num)])
            self.objective_values_list.extend(objective_values)
            self.generated_id_list.extend(generated_id_list)
            self.filter_check_list.extend(filter_check_list)

            self.logger.info(f"Number of valid SMILES: {self.total_valid_num}")
            self.logger.debug(f"node {node_index} objective_values {objective_values} valid smiles {valid_smiles} time {elapsed_time}")

            if len(node_index) == 0:
                back_propagation(node, reward=-1.0)
                continue

            re_list = []
            atom_checked = []
            for i in range(len(node_index)):
                m = node_index[i]
                atom = nodeadded[m]

                if atom not in atom_checked: 
                    node.Addnode(atom, state, self.policy_evaluator)
                    node_pool.append(node.childNodes[len(atom_checked)])
                    atom_checked.append(atom)
                else:
                    node_pool.append(node.childNodes[atom_checked.index(atom)])

                if self.conf["debug"]:
                    self.logger.debug('\n' + '\n'.join([f"Child node position ... {c.position}" for c in node.childNodes]))

                re = -1 if atom == '\n' else self.reward_calculator.calc_reward_from_objective_values(values=objective_values[i], conf=self.conf)
                if self.conf['include_filter_result_in_reward']:
                    re *= filter_check_list[i]
                    
                re_list.append(re)
                self.logger.debug(f"atom: {atom} re_list: {re_list}")
            self.reward_values_list.extend(re_list)

            """backpropation step"""
            for i in range(len(node_pool)):
                node = node_pool[i]
                back_propagation(node, reward=re_list[i])

            if self.conf['debug']:
                self.logger.debug('\n' + '\n'.join([f"child position: {c.position}, wins: {c.wins}, visits: {c.visits}" for c in node_pool]))

            if len(self.valid_smiles_list) > self.conf['flush_threshold'] and self.conf['flush_threshold'] != -1:
                self.flush()
            
            """save checkpoint file"""
            if self.conf['save_checkpoint']:
                counter_list = [gid, infinite_loop_counter_for_selection, infinite_loop_counter_for_expansion, expanded_before]
                self.save_checkpoint(counter_list)

        if len(self.valid_smiles_list) > 0:
            self.flush()
            
    def load_checkpoint(self):
        with open(os.path.join(self.conf['output_dir'], self.conf['checkpoint_file']), mode='rb') as f:
            gid = pickle.load(f)
            self.start_time = pickle.load(f)
            self.rootnode = pickle.load(f)
            self.root_state = pickle.load(f)
            self.conf = pickle.load(f)
            self.val = pickle.load(f)
            #self.model = pickle.load(f)
            self.reward_calculator = pickle.load(f)
            self.policy_evaluator = pickle.load(f)
            self.logger = pickle.load(f)
            self.valid_smiles_list = pickle.load(f)
            self.depth_list = pickle.load(f)
            self.objective_values_list = pickle.load(f)
            self.elapsed_time_list = pickle.load(f)
            self.generated_dict = pickle.load(f)
            self.generated_id_list = pickle.load(f) 
            self.filter_check_list = pickle.load(f)
            self.total_valid_num = pickle.load(f)
            self.obj_column_names = pickle.load(f)
            self.output_path = pickle.load(f)

        return gid

    def save_checkpoint(self, counter_list):
        cp_filename = self.conf['checkpoint_file']
        if os.path.exists(os.path.join(self.conf['output_dir'], f'{cp_filename}.1')):
            os.rename(os.path.join(self.conf['output_dir'], f'{cp_filename}.1'), os.path.join(self.conf['output_dir'], f'{cp_filename}.2'))
        if os.path.exists(os.path.join(self.conf['output_dir'], cp_filename)):
            os.rename(os.path.join(self.conf['output_dir'], f'{cp_filename}'), os.path.join(self.conf['output_dir'], f'{cp_filename}.1'))
        
        with open(os.path.join(self.conf['output_dir'], cp_filename), mode='wb') as f:
            pickle.dump(counter_list, f)
            pickle.dump(self.start_time, f)
            pickle.dump(self.rootnode, f)
            pickle.dump(self.root_state, f)
            pickle.dump(self.conf, f)
            pickle.dump(self.val, f)
            #pickle.dump(self.model, f)
            pickle.dump(self.reward_calculator, f)
            pickle.dump(self.policy_evaluator, f)
            pickle.dump(self.logger, f)
            pickle.dump(self.valid_smiles_list, f)
            pickle.dump(self.depth_list, f)
            pickle.dump(self.objective_values_list, f)
            pickle.dump(self.elapsed_time_list, f)
            pickle.dump(self.generated_dict, f)
            pickle.dump(self.generated_id_list, f)
            pickle.dump(self.filter_check_list, f)
            pickle.dump(self.total_valid_num, f)
            pickle.dump(self.obj_column_names, f)
            pickle.dump(self.output_path, f)


