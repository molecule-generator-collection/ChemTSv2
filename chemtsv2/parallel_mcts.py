from collections import deque
from copy import deepcopy
from enum import Enum
from math import log, sqrt
import os
import random  # only for Hash table initialization
import sys
import time

from mpi4py import MPI
import numpy as np
import pandas as pd
from rdkit import Chem

from chemtsv2.mp_utils import (
    backtrack_tdsdfuct, backtrack_mpmcts, compare_ucb_tdsdfuct, compare_ucb_mpmcts, update_selection_ucbtable_mpmcts, update_selection_ucbtable_tdsdfuct,
    Item, HashTable)
from chemtsv2.utils import chem_kn_simulation, build_smiles_from_tokens, expanded_node, has_passed_through_filters

"""
classes defined distributed parallel mcts
"""
class JobType(Enum):
    '''
    defines JobType tag values
    values higher than PRIORITY_BORDER (128) mean high prority tags
    FINISH is not used in this implementation. It will be needed for games.
    '''
    SEARCH = 0
    BACKPROPAGATION = 1
    PRIORITY_BORDER = 128
    GATHER_RESULTS = 253
    TIMEUP = 254
    FINISH = 255

    @classmethod
    def is_high_priority(self, tag):
        return tag >= self.PRIORITY_BORDER.value


class Tree_Node():
    def __init__(self, state, parentNode=None, reward_calculator=None, conf=None):
        # todo: these should be in a numpy array
        # MPI payload [node.state, node.reward, node.wins, node.visits, node.num_thread_visited, node.path_ucb]
        self.state = state
        self.childNodes = []
        self.parentNode = parentNode
        self.wins = 0
        self.visits = 0
        self.virtual_loss = 0
        self.num_thread_visited = 0
        self.reward = 0
        self.check_childnode = []
        self.expanded_nodes = []
        self.path_ucb = []
        self.childucb = []
        self.conf = conf
        self.reward_calculator = reward_calculator
        self.val = conf['token']
        self.max_len=conf['max_len']

    def selection(self):
        ucb = []
        for i in range(len(self.childNodes)):
            ucb.append((self.childNodes[i].wins +
                        self.childNodes[i].virtual_loss) /
                       (self.childNodes[i].visits +
                        self.childNodes[i].num_thread_visited) +
                       self.conf['c_val'] *
                       sqrt(2 *log(self.visits +self.num_thread_visited) /
                            (self.childNodes[i].visits +
                                self.childNodes[i].num_thread_visited)))
        self.childucb = ucb
        m = np.amax(ucb)
        indices = np.nonzero(ucb == m)[0]
        ind = self.conf['random_generator'].choice(indices)
        self.childNodes[ind].num_thread_visited += 1
        self.num_thread_visited += 1
        return ind, self.childNodes[ind]

    def expansion(self, model, logger):
        node_idxs = expanded_node(model, self.state, self.val, logger)
        self.check_childnode.extend(node_idxs)
        self.expanded_nodes.extend(node_idxs)

    def addnode(self, m):
        self.expanded_nodes.remove(m)
        added_nodes = []
        added_nodes.extend(self.state)
        added_nodes.append(self.val[m])
        self.num_thread_visited += 1
        n = Tree_Node(state=added_nodes, parentNode=self, conf=self.conf)
        n.num_thread_visited += 1
        self.childNodes.append(n)
        return n

    def update_local_node(self, score):
        self.visits += 1
        self.wins += score
        self.reward = score

    def simulation(self, chem_model, state, gen_id, generated_dict):
        filter_flag = 0

        self.conf['gid'] = gen_id
        all_posible = chem_kn_simulation(chem_model, state, self.val, self.conf)
        smi = build_smiles_from_tokens(all_posible, self.val)

        if smi in generated_dict:
            values_list = generated_dict[smi][0]
            score = generated_dict[smi][1]
            filter_flag = generated_dict[smi][2]
            valid_flag = 1  # because only valid SMILES strings are stored in generated_dict
            return values_list, score, smi, filter_flag, valid_flag

        if has_passed_through_filters(smi, self.conf):
            mol = Chem.MolFromSmiles(smi)
            values_list = [f(mol) for f in self.reward_calculator.get_objective_functions(self.conf)]
            score = self.reward_calculator.calc_reward_from_objective_values(values=values_list, conf=self.conf)
            filter_flag = 1
            valid_flag = 1
        else:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                values_list = [-999 for _ in self.reward_calculator.get_objective_functions(self.conf)]
                valid_flag = 0
            else:
                values_list = [f(mol) for f in self.reward_calculator.get_objective_functions(self.conf)]
                valid_flag = 1
            score = 0
            filter_flag = 0
        if valid_flag:
            generated_dict[smi] = [values_list, score, filter_flag]
        return values_list, score, smi, filter_flag, valid_flag

    def backpropagation(self, cnode):
        self.wins += cnode.reward
        self.visits += 1
        self.num_thread_visited -= 1
        self.reward = cnode.reward
        for i in range(len(self.childNodes)):
            if cnode.state[-1] == self.childNodes[i].state[-1]:
                self.childNodes[i].wins += cnode.reward
                self.childNodes[i].num_thread_visited -= 1
                self.childNodes[i].visits += 1


class p_mcts:
    """
    parallel mcts algorithms includes TDS-UCT, TDS-df-UCT and MP-MCTS
    """
    # todo: use generated_dict

    def __init__(self, communicator, chem_model, reward_calculator, conf, logger):
        self.comm = communicator
        self.rank = self.comm.Get_rank()
        self.nprocs = self.comm.Get_size()

        self.chem_model = chem_model
        self.reward_calculator = reward_calculator
        self.conf = conf
        self.logger = logger
        self.threshold = 3600 * conf['hours']
        # Initialize HashTable
        root_node = Tree_Node(state=['&'], reward_calculator=reward_calculator, conf=conf)
        random.seed(conf['zobrist_hash_seed'])
        self.hsm = HashTable(self.nprocs, root_node.val, root_node.max_len, len(root_node.val))

        self.output_path = os.path.join(conf['output_dir'], f"result_C{conf['c_val']}.csv")
        if self.rank == 0:
            if os.path.exists(self.output_path):
                print(f"[ERROR] {self.output_path} already exists. Please specify a different file name.", file=sys.stderr, flush=True)
                self.comm.Abort()

        self.start_time = time.time()

        self.id_suffix = "_" + str(self.rank).zfill(len(str(self.nprocs)))

        # for results
        self.total_valid_num = 0

        # these are gathred to rank 0 in the end
        self.generated_id_list = []
        self.valid_smiles_list = []
        self.depth_list = []
        self.objective_values_list = [] # raw reward (could be list)
        self.reward_values_list = [] # normalized reward for UCT
        self.elapsed_time_list = []

        self.generated_dict = {}  # dictionary of generated compounds
        self.filter_check_list = [] # only needed for output (at rank 0)
        self.obj_column_names = [f.__name__ for f in self.reward_calculator.get_objective_functions(self.conf)]

    def get_generated_id(self):
        _id = str(self.total_valid_num) + self.id_suffix
        self.total_valid_num += 1
        return _id

    def elapsed_time(self):
        return time.time() - self.start_time

    def send_message(self, node, dest, tag, data=None):
        # send node using MPI_Bsend
        # typical usage of data is path_ucb for newly created child nodes
        if data is None:
            self.comm.bsend(np.asarray(
                [node.state, node.reward, node.wins, node.visits, node.num_thread_visited, node.path_ucb]),
                dest=dest, tag=tag)
        else:
            self.comm.bsend(np.asarray(
                [node.state, node.reward, node.wins, node.visits, node.num_thread_visited, data]),
                dest=dest, tag=tag)

    def send_search_childnode(self, node, ucb_table, dest):
        self.comm.bsend(np.asarray(
            [node.state, node.reward, node.wins, node.visits, node.num_thread_visited, ucb_table]),
            dest=dest, tag=JobType.SEARCH.value)

    def send_backprop(self, node, dest):
        self.comm.bsend(np.asarray(
            [node.state, node.reward, node.wins, node.visits, node.num_thread_visited, node.path_ucb]),
            dest=dest, tag=JobType.BACKPROPAGATION.value)

    def record_result(self, smiles, depth, reward, gen_id, raw_reward_list, filter_flag):
        self.valid_smiles_list.append(smiles)
        self.depth_list.append(depth)
        self.reward_values_list.append(reward)
        self.elapsed_time_list.append(self.elapsed_time())
        self.generated_id_list.append(gen_id)
        self.objective_values_list.append(raw_reward_list)
        self.filter_check_list.append(filter_flag)

    def gather_results(self):
        status = MPI.Status()
        if self.rank == 0:
            self.logger.info(f"Gather each rank result...")
        for id in range(1, self.nprocs):
            if self.rank == 0:
                (valid_smiles_list, depth_list, reward_values_list, elapsed_time_list,
                 generated_id_list, objective_values_list, filter_check_list) = self.comm.recv(source=id, tag=JobType.GATHER_RESULTS.value, status=status)
                self.valid_smiles_list.extend(valid_smiles_list)
                self.depth_list.extend(depth_list)
                self.reward_values_list.extend(reward_values_list)
                self.elapsed_time_list.extend(elapsed_time_list)
                self.generated_id_list.extend(generated_id_list)
                self.objective_values_list.extend(objective_values_list)
                self.filter_check_list.extend(filter_check_list)
            elif self.rank == id:
                self.comm.send((self.valid_smiles_list, self.depth_list, self.reward_values_list, self.elapsed_time_list,
                                self.generated_id_list, self.objective_values_list, self.filter_check_list),
                                dest=0, tag=JobType.GATHER_RESULTS.value)

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
        if self.rank == 0:
            self.logger.info(f"Save a result at {self.output_path}")

        self.generated_id_list.clear()
        self.valid_smiles_list.clear()
        self.reward_values_list.clear()
        self.depth_list.clear()
        self.elapsed_time_list.clear()
        self.filter_check_list.clear()
        self.objective_values_list.clear()

    def TDS_UCT(self):
        # self.comm.barrier()
        status = MPI.Status()

        self.start_time = time.time()
        _, rootdest = self.hsm.hashing(['&'])
        jobq = deque()
        timeup = False
        if self.rank == rootdest:
            root_job_message = np.asarray([['&'], None, 0, 0, 0, []])
            for i in range(3 * self.nprocs):
                temp = deepcopy(root_job_message)
                root_job = (JobType.SEARCH.value, temp)
                jobq.appendleft(root_job)
        while not timeup:
            if self.rank == 0:
                if self.elapsed_time() > self.threshold:
                    timeup = True
                    for dest in range(1, self.nprocs):
                        dummy_data = tag = JobType.TIMEUP.value
                        self.comm.bsend(dummy_data, dest=dest,
                                        tag=JobType.TIMEUP.value)
            while True:
                ret = self.comm.Iprobe(source=MPI.ANY_SOURCE,
                                        tag=MPI.ANY_TAG, status=status)
                if ret == False:
                    break
                else:
                    message = self.comm.recv(
                        source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                    cur_status = status
                    tag = cur_status.Get_tag()
                    job = (tag, message)
                    if JobType.is_high_priority(tag):
                        # high priority messages (timeup and finish)
                        jobq.append(job)
                    else:
                        # normal messages (search and backpropagate)
                        jobq.appendleft(job)

            jobq_non_empty = bool(jobq)
            if jobq_non_empty:
                (tag, message) = jobq.pop()
                if tag == JobType.SEARCH.value:
                    # if node is not in the hash table
                    if self.hsm.search_table(message[0]) is None:
                        node = Tree_Node(state=message[0], reward_calculator=self.reward_calculator, conf=self.conf)
                        #node.state = message[0]
                        if node.state == ['&']:
                            node.expansion(self.chem_model, self.logger)
                            m = self.conf['random_generator'].choice(node.expanded_nodes)
                            n = node.addnode(m)
                            self.hsm.insert(Item(node.state, node))
                            _, dest = self.hsm.hashing(n.state)
                            self.send_message(n, dest, tag=JobType.SEARCH.value)
                        else:
                            # or max_len_wavelength :
                            if len(node.state) < node.max_len:
                                gen_id = self.get_generated_id()
                                values_list, score, smi, filter_flag, is_valid_smi = node.simulation(
                                    self.chem_model, node.state, gen_id, self.generated_dict)
                                if is_valid_smi:
                                    self.record_result(smiles=smi, depth=len(node.state), reward=score,
                                                       gen_id=gen_id, raw_reward_list=values_list, filter_flag=filter_flag)
                                # backpropagation on local memory
                                node.update_local_node(score)
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(node.state[0:-1])
                                self.send_backprop(node, dest)
                            else:
                                score = -1
                                # backpropagation on local memory
                                node.update_local_node(score)
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(node.state[0:-1])
                                self.send_backprop(node, dest)

                    else:  # if node already in the local hashtable
                        node = self.hsm.search_table(message[0])
                        #print("debug:", node.visits,
                        #      node.num_thread_visited, node.wins)
                        if node.state == ['&']:
                            if node.expanded_nodes != []:
                                m = self.conf['random_generator'].choice(node.expanded_nodes)
                                n = node.addnode(m)
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(n.state)
                                self.comm.bsend(np.asarray([n.state, n.reward, n.wins, n.visits,
                                                n.num_thread_visited]), dest=dest, tag=JobType.SEARCH.value)
                            else:
                                ind, childnode = node.selection()
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(childnode.state)
                                self.send_message(childnode, dest, tag=JobType.SEARCH.value)
                        else:
                            #node.num_thread_visited = message[4]
                            if len(node.state) < node.max_len:
                                if node.state[-1] != '\n':
                                    if node.expanded_nodes != []:
                                        m = self.conf['random_generator'].choice(node.expanded_nodes)
                                        n = node.addnode(m)
                                        self.hsm.insert(Item(node.state, node))
                                        _, dest = self.hsm.hashing(n.state)
                                        self.send_message(n, dest, tag=JobType.SEARCH.value)
                                    else:
                                        if node.check_childnode == []:
                                            node.expansion(self.chem_model, self.logger)
                                            m = self.conf['random_generator'].choice(
                                                node.expanded_nodes)
                                            n = node.addnode(m)
                                            self.hsm.insert(Item(node.state, node))
                                            _, dest = self.hsm.hashing(n.state)
                                            self.send_message(n, dest, tag=JobType.SEARCH.value)
                                        else:
                                            ind, childnode = node.selection()
                                            self.hsm.insert(Item(node.state, node))
                                            _, dest = self.hsm.hashing(
                                                childnode.state)
                                            self.send_message(childnode, dest, tag=JobType.SEARCH.value)

                                else:
                                    gen_id = self.get_generated_id()
                                    values_list, score, smi, filter_flag, is_valid_smi = node.simulation(
                                        self.chem_model, node.state, gen_id, self.generated_dict)
                                    score = -1
                                    if is_valid_smi:
                                        self.record_result(smiles=smi, depth=len(node.state), reward=score,
                                                           gen_id=gen_id, raw_reward_list=values_list, filter_flag=filter_flag)
                                    # backpropagation on local memory
                                    node.update_local_node(score)
                                    self.hsm.insert(Item(node.state, node))
                                    _, dest = self.hsm.hashing(node.state[0:-1])
                                    self.send_backprop(node, dest)

                            else:
                                score = -1
                                # backpropagation on local memory
                                node.update_local_node(score)
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(node.state[0:-1])
                                self.send_backprop(node, dest)

                elif tag == JobType.BACKPROPAGATION.value:
                    node = Tree_Node(state=message[0], reward_calculator=self.reward_calculator, conf=self.conf)
                    node.reward = message[1]
                    local_node = self.hsm.search_table(message[0][0:-1])
                    if local_node.state == ['&']:
                        local_node.backpropagation(node)
                        self.hsm.insert(Item(local_node.state, local_node))
                        _, dest = self.hsm.hashing(local_node.state)
                        self.send_message(local_node, dest, tag=JobType.SEARCH.value)
                    else:
                        local_node.backpropagation(node)
                        self.hsm.insert(Item(local_node.state, local_node))
                        _, dest = self.hsm.hashing(local_node.state[0:-1])
                        self.send_backprop(local_node, dest)
                elif tag == JobType.TIMEUP.value:
                    timeup = True

        return

    def TDS_df_UCT(self):
        # self.comm.barrier()
        status = MPI.Status()
        self.start_time = time.time()
        bpm = 0
        bp = []
        _, rootdest = self.hsm.hashing(['&'])
        jobq = deque()
        timeup = False
        if self.rank == rootdest:
            root_job_message = np.asarray([['&'], None, 0, 0, 0, []])
            for i in range(3 * self.nprocs):
                temp = deepcopy(root_job_message)
                root_job = (JobType.SEARCH.value, temp)
                jobq.appendleft(root_job)
        while not timeup:
            if self.rank == 0:
                if self.elapsed_time() > self.threshold:
                    timeup = True
                    for dest in range(1, self.nprocs):
                        dummy_data = tag = JobType.TIMEUP.value
                        self.comm.bsend(dummy_data, dest=dest,
                                        tag=JobType.TIMEUP.value)
            while True:
                ret = self.comm.Iprobe(source=MPI.ANY_SOURCE,
                                        tag=MPI.ANY_TAG, status=status)
                if ret == False:
                    break
                else:
                    message = self.comm.recv(
                        source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                    cur_status = status
                    tag = cur_status.Get_tag()
                    job = (tag, message)
                    if JobType.is_high_priority(tag):
                        jobq.append(job)
                    else:
                        jobq.appendleft(job)
            jobq_non_empty = bool(jobq)
            if jobq_non_empty:
                (tag, message) = jobq.pop()
                if tag == JobType.SEARCH.value:
                    if self.hsm.search_table(message[0]) == None:
                        node = Tree_Node(state=message[0], reward_calculator=self.reward_calculator, conf=self.conf)
                        info_table = message[5]
                        #print ("not in table info_table:",info_table)
                        if node.state == ['&']:
                            node.expansion(self.chem_model, self.logger)
                            m = self.conf['random_generator'].choice(node.expanded_nodes)
                            n = node.addnode(m)
                            self.hsm.insert(Item(node.state, node))
                            _, dest = self.hsm.hashing(n.state)
                            self.send_message(n, dest, tag=JobType.SEARCH.value)
                        else:
                            if len(node.state) < node.max_len:
                                gen_id = self.get_generated_id()
                                values_list, score, smi, filter_flag, is_valid_smi = node.simulation(
                                    self.chem_model, node.state, gen_id, self.generated_dict)
                                if is_valid_smi:
                                    self.record_result(smiles=smi, depth=len(node.state), reward=score,
                                                       gen_id=gen_id, raw_reward_list=values_list, filter_flag=filter_flag)
                                node.update_local_node(score)
                                # update infor table
                                info_table = backtrack_tdsdfuct(
                                    info_table, score)
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(node.state[0:-1])
                                self.send_backprop(node, dest)
                            else:
                                score = -1
                                node.update_local_node(score)
                                info_table = backtrack_tdsdfuct(
                                    info_table, score)
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(node.state[0:-1])
                                self.send_backprop(node, dest)

                    else:  # if node already in the local hashtable
                        node = self.hsm.search_table(message[0])
                        info_table = message[5]
                        #print ("in table info_table:",info_table)
                        if node.state == ['&']:
                            # print ("in table root:",node.state,node.path_ucb,len(node.state),len(node.path_ucb))
                            if node.expanded_nodes != []:
                                m = self.conf['random_generator'].choice(node.expanded_nodes)
                                n = node.addnode(m)
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(n.state)
                                self.send_message(n, dest, tag=JobType.SEARCH.value)
                            else:
                                ind, childnode = node.selection()
                                self.hsm.insert(Item(node.state, node))
                                info_table = update_selection_ucbtable_tdsdfuct(
                                    info_table, node, ind)
                                #print ("info_table after selection:",info_table)
                                _, dest = self.hsm.hashing(childnode.state)
                                self.send_message(childnode, dest, tag=JobType.SEARCH.value)
                        else:
                            #node.path_ucb = message[5]
                            # info_table=message[5]
                            #print("check ucb:", node.reward, node.visits, node.num_thread_visited,info_table)
                            if len(node.state) < node.max_len:
                                if node.state[-1] != '\n':
                                    if node.expanded_nodes != []:
                                        m = self.conf['random_generator'].choice(node.expanded_nodes)
                                        n = node.addnode(m)
                                        self.hsm.insert(Item(node.state, node))
                                        _, dest = self.hsm.hashing(n.state)
                                        self.send_message(n, dest, tag=JobType.SEARCH.value)
                                    else:
                                        if node.check_childnode == []:
                                            node.expansion(self.chem_model, self.logger)
                                            m = self.conf['random_generator'].choice(
                                                node.expanded_nodes)
                                            n = node.addnode(m)
                                            self.hsm.insert(Item(node.state, node))
                                            _, dest = self.hsm.hashing(n.state)
                                            self.send_message(n, dest, tag=JobType.SEARCH.value)
                                        else:
                                            ind, childnode = node.selection()
                                            self.hsm.insert(Item(node.state, node))
                                            info_table = update_selection_ucbtable_tdsdfuct(
                                                info_table, node, ind)
                                            _, dest = self.hsm.hashing(
                                                childnode.state)
                                            self.send_message(childnode, dest, tag=JobType.SEARCH.value)
                                else:
                                    gen_id = self.get_generated_id()
                                    value_list, score, smi, filter_flag, is_valid_smi = node.simulation(
                                        self.chem_model, node.state, gen_id, self.generated_dict)
                                    score = -1
                                    if is_valid_smi:
                                        self.record_result(smiles=smi, depth=len(node.state), reward=score,
                                                           gen_id=gen_id, raw_reward_list=value_list, filter_flag=filter_flag)
                                    node.update_local_node(score)
                                    info_table = backtrack_tdsdfuct(
                                        info_table, score)

                                    self.hsm.insert(Item(node.state, node))
                                    _, dest = self.hsm.hashing(node.state[0:-1])
                                    self.send_backprop(node, dest)
                            else:
                                score = -1
                                node.update_local_node(score)
                                info_table = backtrack_tdsdfuct(
                                    info_table, score)
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(node.state[0:-1])
                                self.send_backprop(node, dest)

                elif tag == JobType.BACKPROPAGATION.value:
                    bpm += 1
                    node = Tree_Node(state=message[0], reward_calculator=self.reward_calculator, conf=self.conf)
                    node.reward = message[1]
                    local_node = self.hsm.search_table(message[0][0:-1])
                    #print ("report check message[5]:",message[5])
                    #print ("check:",len(message[0]), len(message[5]))
                    #print ("check:",local_node.wins, local_node.visits, local_node.num_thread_visited)
                    info_table=message[5]
                    if local_node.state == ['&']:
                        local_node.backpropagation(node)
                        self.hsm.insert(Item(local_node.state, local_node))
                        _, dest = self.hsm.hashing(local_node.state)
                        self.send_message(local_node, dest, tag=JobType.SEARCH.value)
                    else:
                        local_node.backpropagation(node)
                        #local_node,info_table = backtrack_tdsdf(info_table,local_node, node)
                        back_flag = compare_ucb_tdsdfuct(info_table,local_node)
                        self.hsm.insert(Item(local_node.state, local_node))
                        if back_flag == 1:
                            _, dest = self.hsm.hashing(local_node.state[0:-1])
                            self.send_backprop(local_node, dest)
                        if back_flag == 0:
                            _, dest = self.hsm.hashing(local_node.state)
                            self.send_message(local_node, dest, tag=JobType.SEARCH.value)
                elif tag == JobType.TIMEUP.value:
                    timeup = True
        bp.append(bpm)

        return

    def MP_MCTS(self):
        #self.comm.barrier()
        status = MPI.Status()
        self.start_time = time.time()
        _, rootdest = self.hsm.hashing(['&'])
        jobq = deque()
        timeup = False
        if self.rank == rootdest:
            root_job_message = np.asarray([['&'], None, 0, 0, 0, []])
            for i in range(3 * self.nprocs):
                temp = deepcopy(root_job_message)
                root_job = (JobType.SEARCH.value, temp)
                jobq.appendleft(root_job)
        while not timeup:
            if self.rank == 0:
                if self.elapsed_time() > self.threshold:
                    timeup = True
                    for dest in range(1, self.nprocs):
                        dummy_data = tag = JobType.TIMEUP.value
                        self.comm.bsend(dummy_data, dest=dest, tag=JobType.TIMEUP.value)

            while True:
                ret = self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                if ret == False:
                    break
                else:
                    message = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                    cur_status = status
                    tag = cur_status.Get_tag()
                    job = (tag, message)
                    if JobType.is_high_priority(tag):
                        jobq.append(job)
                    else:
                        jobq.appendleft(job)
            jobq_non_empty = bool(jobq)
            if jobq_non_empty:
                (tag, message) = jobq.pop()
                if tag == JobType.SEARCH.value:
                    if self.hsm.search_table(message[0]) == None:
                        node = Tree_Node(state=message[0], reward_calculator=self.reward_calculator, conf=self.conf)
                        if node.state == ['&']:
                            node.expansion(self.chem_model, self.logger)
                            m = self.conf['random_generator'].choice(node.expanded_nodes)
                            n = node.addnode(m)
                            self.hsm.insert(Item(node.state, node))
                            _, dest = self.hsm.hashing(n.state)
                            self.send_message(n, dest, tag=JobType.SEARCH.value)
                        else:
                            if len(node.state) < node.max_len:
                                gen_id = self.get_generated_id()
                                values_list, score, smi, filter_flag, is_valid_smi = node.simulation(
                                    self.chem_model, node.state, gen_id, self.generated_dict)
                                if is_valid_smi:
                                    self.record_result(smiles=smi, depth=len(node.state), reward=score,
                                                       gen_id=gen_id ,raw_reward_list=values_list, filter_flag=filter_flag)
                                node.update_local_node(score)
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(node.state[0:-1])
                                self.send_backprop(node, dest)
                            else:
                                score = -1
                                node.update_local_node(score)
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(node.state[0:-1])
                                self.send_backprop(node, dest)

                    else:  # if node already in the local hashtable
                        node = self.hsm.search_table(message[0])
                        if node.state == ['&']:
                            # print ("in table root:",node.state,node.path_ucb,len(node.state),len(node.path_ucb))
                            if node.expanded_nodes != []:
                                m = self.conf['random_generator'].choice(node.expanded_nodes)
                                n = node.addnode(m)
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(n.state)
                                self.send_message(n, dest, tag=JobType.SEARCH.value)
                            else:
                                ind, childnode = node.selection()
                                self.hsm.insert(Item(node.state, node))
                                ucb_table = update_selection_ucbtable_mpmcts(node, ind)
                                _, dest = self.hsm.hashing(childnode.state)
                                self.send_message(childnode, dest, tag=JobType.SEARCH.value, data=ucb_table)
                        else:
                            node.path_ucb = message[5]
                            #print("check ucb:", node.wins, node.visits, node.num_thread_visited)
                            if len(node.state) < node.max_len:
                                if node.state[-1] != '\n':
                                    if node.expanded_nodes != []:
                                        m = self.conf['random_generator'].choice(node.expanded_nodes)
                                        n = node.addnode(m)
                                        self.hsm.insert(Item(node.state, node))
                                        _, dest = self.hsm.hashing(n.state)
                                        self.send_message(n, dest, tag=JobType.SEARCH.value)
                                    else:
                                        if node.check_childnode == []:
                                            node.expansion(self.chem_model, self.logger)
                                            m = self.conf['random_generator'].choice(node.expanded_nodes)
                                            n = node.addnode(m)
                                            self.hsm.insert(Item(node.state, node))
                                            _, dest = self.hsm.hashing(n.state)
                                            self.send_message(n, dest, tag=JobType.SEARCH.value)
                                        else:
                                            ind, childnode = node.selection()
                                            self.hsm.insert(Item(node.state, node))
                                            ucb_table = update_selection_ucbtable_mpmcts(node, ind)
                                            _, dest = self.hsm.hashing(childnode.state)
                                            self.send_message(childnode, dest, tag=JobType.SEARCH.value, data=ucb_table)
                                else:
                                    gen_id = self.get_generated_id()
                                    values_list, score, smi, filter_flag, is_valid_smi = node.simulation(
                                        self.chem_model, node.state, gen_id, self.generated_dict)
                                    score = -1
                                    if is_valid_smi:
                                        self.record_result(smiles=smi, depth=len(node.state), reward=score,
                                                           gen_id=gen_id,raw_reward_list=values_list, filter_flag=filter_flag)
                                    node.update_local_node(score)
                                    self.hsm.insert(Item(node.state, node))
                                    _, dest = self.hsm.hashing(node.state[0:-1])
                                    self.send_backprop(node, dest)
                            else:
                                score = -1
                                node.update_local_node(score)
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(node.state[0:-1])
                                self.send_backprop(node, dest)

                elif tag == JobType.BACKPROPAGATION.value:
                    node = Tree_Node(state=message[0], reward_calculator=self.reward_calculator, conf=self.conf)
                    node.reward = message[1]
                    local_node = self.hsm.search_table(message[0][0:-1])
                    if local_node.state == ['&']:
                        local_node.backpropagation(node)
                        self.hsm.insert(Item(local_node.state, local_node))
                        _, dest = self.hsm.hashing(local_node.state)
                        self.send_message(local_node, dest, tag=JobType.SEARCH.value)
                    else:
                        local_node.backpropagation(node)
                        local_node = backtrack_mpmcts(local_node, node)
                        back_flag = compare_ucb_mpmcts(local_node)
                        self.hsm.insert(Item(local_node.state, local_node))
                        if back_flag == 1:
                            _, dest = self.hsm.hashing(local_node.state[0:-1])
                            self.send_backprop(local_node, dest)
                        if back_flag == 0:
                            _, dest = self.hsm.hashing(local_node.state)
                            self.send_message(local_node, dest, tag=JobType.SEARCH.value)

                elif tag == JobType.TIMEUP.value:
                    timeup = True

        return
