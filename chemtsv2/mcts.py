import os
import sys
import time

import numpy as np
import pandas as pd
import pickle

from chemtsv2.utils import chem_kn_simulation, build_smiles_from_tokens,\
    evaluate_node, node_to_add, expanded_node, back_propagation


class State:
    def __init__(self, position=['&'], parent=None):
        self.position = position
        self.visits = 0
        self.total_reward = 0
        self.parent_node = parent
        self.child_nodes = []
        
    def clone(self, include_visit=False, include_total_reward=False, include_parent_node=False, include_child_node=False):
        st = State()
        st.position = self.position[:]
        st.visits = self.visits if include_visit else 0
        st.total_reward = self.total_reward if include_total_reward else 0
        st.parent_node = self.parent_node if include_parent_node else None
        st.child_nodes = self.child_nodes if include_child_node else []
        return st

    def add_position(self, m):
        self.position.append(m)


class Node:
    def __init__(self, policy_evaluator, position=None, state=None, conf=None):
        self.position = position
        self.state = state
        self.policy_evaluator = policy_evaluator
        self.conf = conf

    def select_node(self, logger):
        score_list = []
        logger.debug('UCB:')
        for i in range(len(self.state.child_nodes)):
            score = self.policy_evaluator.evaluate(self.state.child_nodes[i].state, self.conf)
            score_list.append(score)
            logger.debug(f"{self.state.child_nodes[i].position} {score}") 
        m = np.amax(score_list)
        indices = np.nonzero(score_list == m)[0]
        ind = int(self.conf['random_generator'].choice(indices))
        s = self.state.child_nodes[ind]
        logger.debug(f"\nindex {ind} {self.position} {m}") 
        return s

    def add_node(self, m, state, policy_evaluator):
        state.parent_node = self
        node = Node(policy_evaluator, position=m, state=state, conf=self.conf)
        self.state.child_nodes.append(node)

    def simulation(self):
        raise SystemExit("[ERROR] Do NOT use this method")

    def update(self, reward):
        self.state.visits += 1
        self.state.total_reward += reward


class MCTS:
    def __init__(self, root_state, conf, tokens, model, reward_calculator, policy_evaluator, logger):
        self.start_time = time.time()
        self.rootnode = Node(policy_evaluator, state=root_state, conf=conf)
        self.conf = conf
        self.tokens = tokens
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
        
        if conf['batch_reward_calculation']:
            self.obj_column_names = [f.__name__ for f in self.reward_calculator.get_batch_objective_functions()]
        else:
            self.obj_column_names = [f.__name__ for f in self.reward_calculator.get_objective_functions(self.conf)]
        self.output_path = os.path.join(conf['output_dir'], f"result_C{conf['c_val']}.csv")
        if os.path.exists(self.output_path) and not conf['restart']:
            sys.exit(f"[ERROR] {self.output_path} already exists. Please specify a different file name.")

        self.gid = 0
        self.loop_counter_for_selection = 0
        self.loop_counter_for_expansion = 0
        self.expanded_before = {}

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
        ckpt_path = os.path.join(self.conf['output_dir'], self.conf['checkpoint_file'])
        if self.conf['restart'] and os.path.exists(ckpt_path):
            self.logger.info(f"Load the checkpoint file from {ckpt_path}")
            self.load_checkpoint()
        
        while (time.time() if self.conf['threshold_type']=="time" else self.total_valid_num) <= self.threshold:
            node = self.rootnode  # important! This node is different with state / node is the tree node
            state = node.state.clone()  # but this state is the state of the initialization. Too important!

            """selection step"""
            node_pool = []
            while node.state.child_nodes != []:
                node = node.select_node(self.logger)
                state.add_position(node.position)
            self.logger.info(f"state position: {state.position}")

            self.logger.debug(f"infinite loop counter (selection): {self.loop_counter_for_selection}")
            if node.position == '\n':
                back_propagation(node, reward=-1.0)
                self.loop_counter_for_selection += 1
                if self.loop_counter_for_selection > self.conf['infinite_loop_threshold_for_selection']:
                    self.flush()
                    sys.exit('[WARN] Infinite loop is detected in the selection step. Change hyperparameters or RNN model.')
                continue
            else:
                self.loop_counter_for_selection = 0

            """expansion step"""
            expanded = expanded_node(self.model, state.position, self.tokens, self.logger, threshold=self.conf['expansion_threshold'])
            self.logger.debug(f"infinite loop counter (expansion): {self.loop_counter_for_expansion}")
            if set(expanded) == self.expanded_before:
                self.loop_counter_for_expansion += 1
                if self.loop_counter_for_expansion > self.conf['infinite_loop_threshold_for_expansion']:
                    self.flush()
                    sys.exit('[WARN] Infinite loop is detected in the expansion step. Change hyperparameters or RNN model.')
            else:
                self.loop_counter_for_expansion = 0
            self.expanded_before = set(expanded)

            new_compound = []
            nodeadded = []
            for _ in range(self.conf['simulation_num']):
                nodeadded_tmp = node_to_add(expanded, self.tokens, self.logger)
                nodeadded.extend(nodeadded_tmp)
                for n in nodeadded_tmp:
                    position_tmp = state.position + [n]
                    all_posible = chem_kn_simulation(self.model, position_tmp, self.tokens, self.conf)
                    new_compound.append(build_smiles_from_tokens(all_posible, self.tokens, use_selfies=self.conf['use_selfies']))

            _gids = list(range(self.gid, self.gid+len(new_compound)))
            self.gid += len(new_compound)

            self.logger.debug(f"nodeadded {nodeadded}")
            self.logger.info(f"new compound {new_compound}")
            self.logger.debug(f"generated_dict {self.generated_dict}") 
            if self.conf["debug"]:
                self.logger.debug('\n' + '\n'.join([f"lastcomp {comp[-1]} ... " + str(comp[-1] == '\n') for comp in new_compound]))
            node_index, objective_values, valid_smiles, generated_id_list, filter_check_list = evaluate_node(new_compound, self.generated_dict, self.reward_calculator, self.conf, self.logger, _gids)

            if len(valid_smiles) == 0:
                back_propagation(node, reward=-1.0)
                continue

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

            re_list = []
            atom_checked = []
            for i in range(len(node_index)):
                m = node_index[i]
                atom = nodeadded[m]
                state_clone = state.clone(include_visit=True, include_total_reward=True)

                if atom not in atom_checked: 
                    node.add_node(atom, state_clone, self.policy_evaluator)
                    node_pool.append(node.state.child_nodes[len(atom_checked)])
                    atom_checked.append(atom)
                else:
                    node_pool.append(node.state.child_nodes[atom_checked.index(atom)])

                if self.conf["debug"]:
                    self.logger.debug('\n' + '\n'.join([f"Child node position ... {c.position}" for c in node.state.child_nodes]))

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
                self.logger.debug('\n' + '\n'.join([f"child position: {c.position}, total_reward: {c.state.total_reward}, visits: {c.state.visits}" for c in node_pool]))

            if len(self.valid_smiles_list) > self.conf['flush_threshold'] and self.conf['flush_threshold'] != -1:
                self.flush()
            
            """save checkpoint file"""
            if self.conf['save_checkpoint']:
                self.save_checkpoint()

        if len(self.valid_smiles_list) > 0:
            self.flush()
            
    def load_checkpoint(self):
        ckpt_path = os.path.join(self.conf['output_dir'], self.conf['checkpoint_file'])
        with open(ckpt_path, mode='rb') as f:
            cp_obj = pickle.load(f)
        self.gid = cp_obj['gid']
        self.loop_counter_for_selection = cp_obj['loop_counter_for_selection']
        self.loop_counter_for_expansion = cp_obj['loop_counter_for_expansion']
        self.expanded_before = cp_obj['expanded_before']        
        self.start_time = cp_obj['start_time']
        self.rootnode = cp_obj['rootnode']
        self.conf = cp_obj['conf']
        self.generated_dict = cp_obj['generated_dict']
        self.total_valid_num = cp_obj['total_valid_num']


    def save_checkpoint(self):
        ckpt_fname = self.conf['checkpoint_file']
        ckpt_path = os.path.join(self.conf['output_dir'], ckpt_fname)
        stem, ext = ckpt_fname.rsplit('.', 1)
        # To keep the three most recent checkpoint files.
        ckpt1_path = os.path.join(self.conf['output_dir'], f'{stem}2.{ext}')
        ckpt2_path = os.path.join(self.conf['output_dir'], f'{stem}3.{ext}')
        if os.path.exists(ckpt1_path):
            os.rename(ckpt1_path, ckpt2_path)
        if os.path.exists(ckpt_path):
            os.rename(ckpt_path, ckpt1_path)

        cp_obj = {
            'gid': self.gid,
            'loop_counter_for_selection': self.loop_counter_for_selection,
            'loop_counter_for_expansion': self.loop_counter_for_expansion,
            'expanded_before': self.expanded_before,
            'start_time': self.start_time, 
            'conf': self.conf, 
            'rootnode': self.rootnode,
            'generated_dict': self.generated_dict,
            'total_valid_num': self.total_valid_num,
        }
        
        with open(ckpt_path, mode='wb') as f:
            pickle.dump(cp_obj, f)
        self.flush()
            