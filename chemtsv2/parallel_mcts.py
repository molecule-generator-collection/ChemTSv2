from collections import deque
from copy import deepcopy
import datetime
from enum import Enum
from math import log, sqrt
import os
import pickle
import random  # only for Hash table initialization
import sys
import time

from mpi4py import MPI
import numpy as np
import pandas as pd
from rdkit import Chem

from chemtsv2.mp_utils import (
    backtrack_mpmcts,
    compare_ucb_mpmcts,
    update_selection_ucbtable_mpmcts,
    Item,
    HashTable,
)
from chemtsv2.utils import (
    generate_smiles_as_token_index,
    build_smiles_from_token_index,
    get_expanded_node_index,
    has_passed_through_filters,
)

"""
classes defined distributed parallel mcts
"""


class JobType(Enum):
    """
    defines JobType tag values
    values higher than PRIORITY_BORDER (128) mean high prority tags
    FINISH is not used in this implementation. It will be needed for games.
    """

    SEARCH = 0
    BACKPROPAGATION = 1
    PRIORITY_BORDER = 128
    CHECKPOINT_PREPARE = 150
    CHECKPOINT_READY = 151
    CHECKPOINT_SAVE = 152
    CHECKPOINT_LOAD = 153
    GATHER_RESULTS = 253
    TIMEUP = 254
    FINISH = 255

    @classmethod
    def is_high_priority(self, tag):
        return tag >= self.PRIORITY_BORDER.value


class MPNode:
    def __init__(self, position=["&"], parentNode=None, conf=None):
        # todo: the payload of MPI should be in a numpy array. consider using @property for implementation
        # MPI payload [node.state, node.reward, node.wins, node.visits, node.num_thread_visited, node.path_ucb]
        self.state = ["&"] if position is None else position
        self.child_nodes = []
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

    def selection(self):
        ucb = []
        for i in range(len(self.child_nodes)):
            ucb.append(
                (self.child_nodes[i].wins + self.child_nodes[i].virtual_loss)
                / (self.child_nodes[i].visits + self.child_nodes[i].num_thread_visited)
                + self.conf["c_val"]
                * sqrt(
                    2
                    * log(self.visits + self.num_thread_visited)
                    / (self.child_nodes[i].visits + self.child_nodes[i].num_thread_visited)
                )
            )
        self.childucb = ucb
        m = np.amax(ucb)
        indices = np.nonzero(ucb == m)[0]
        ind = self.conf["random_generator"].choice(indices)
        self.child_nodes[ind].num_thread_visited += 1
        self.num_thread_visited += 1
        return ind, self.child_nodes[ind]

    def expansion(self, model, tokens, logger):
        node_idxs = get_expanded_node_index(model, self.state, tokens, logger)
        self.check_childnode.extend(node_idxs)
        self.expanded_nodes.extend(node_idxs)

    def addnode(self, m, tokens):
        self.expanded_nodes.remove(m)
        added_nodes = []
        added_nodes.extend(self.state)
        added_nodes.append(tokens[m])
        self.num_thread_visited += 1
        n = MPNode(position=added_nodes, parentNode=self, conf=self.conf)
        n.num_thread_visited += 1
        self.child_nodes.append(n)
        return n

    def update_local_node(self, score):
        self.visits += 1
        self.wins += score
        self.reward = score

    def simulation(self, chem_model, state, gen_id, generated_dict, reward_calculator, tokens):
        filter_flag = 0

        self.conf["gid"] = gen_id
        generated_token_indexes = generate_smiles_as_token_index(
            chem_model, state, tokens, self.conf
        )
        smi = build_smiles_from_token_index(
            generated_token_indexes, tokens, use_selfies=self.conf["use_selfies"]
        )

        if smi in generated_dict:
            values_list = generated_dict[smi][0]
            score = generated_dict[smi][1]
            filter_flag = generated_dict[smi][2]
            # because only valid SMILES strings are stored in generated_dict
            valid_flag = 1
            return values_list, score, smi, filter_flag, valid_flag

        if has_passed_through_filters(smi, self.conf):
            mol = Chem.MolFromSmiles(smi)
            values_list = [f(mol) for f in reward_calculator.get_objective_functions(self.conf)]
            score = reward_calculator.calc_reward_from_objective_values(
                values=values_list, conf=self.conf
            )
            filter_flag = 1
            valid_flag = 1
        else:
            mol = Chem.MolFromSmiles(smi)
            valid_flag = 0 if mol is None else 1
            values_list = [-999 for _ in reward_calculator.get_objective_functions(self.conf)]
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
        for i in range(len(self.child_nodes)):
            if cnode.state[-1] == self.child_nodes[i].state[-1]:
                self.child_nodes[i].wins += cnode.reward
                self.child_nodes[i].num_thread_visited -= 1
                self.child_nodes[i].visits += 1


class p_mcts:
    """
    parallel mcts algorithms includes TDS-UCT, TDS-df-UCT and MP-MCTS
    """

    # todo: use generated_dict

    def __init__(
        self,
        communicator,
        root_position,
        chem_model,
        reward_calculator,
        tokens,
        conf,
        logger,
    ):
        self.comm = communicator
        self.rank = self.comm.Get_rank()
        self.nprocs = self.comm.Get_size()

        self.chem_model = chem_model
        self.reward_calculator = reward_calculator
        self.conf = conf
        self.logger = logger
        self.threshold = 3600 * conf["hours"]
        self.root_position = ["&"] if root_position is None else root_position
        random.seed(conf["zobrist_hash_seed"])
        self.tokens = tokens
        # Initialize HashTable
        self.hsm = HashTable(self.nprocs, self.tokens, conf["max_len"])

        if self.conf["checkpoint_load"]:
            dt = datetime.datetime.now()
            stime = dt.strftime("%Y%m%d_%H%M%S")
            self.output_path = os.path.join(
                conf["output_dir"], f"result_C{conf['c_val']}_{stime}.csv"
            )
        else:
            self.output_path = os.path.join(conf["output_dir"], f"result_C{conf['c_val']}.csv")
        if self.rank == 0:
            if os.path.exists(self.output_path):
                print(
                    f"[ERROR] {self.output_path} already exists. Please specify a different file name.",
                    file=sys.stderr,
                    flush=True,
                )
                self.comm.Abort()

        self.start_time = time.time()

        self.id_suffix = "_" + str(self.rank).zfill(len(str(self.nprocs)))

        # for results
        self.total_valid_num = 0

        # these are gathred to rank 0 in the end
        self.generated_id_list = []
        self.valid_smiles_list = []
        self.depth_list = []
        self.objective_values_list = []  # raw reward (could be list)
        self.reward_values_list = []  # normalized reward for UCT
        self.elapsed_time_list = []

        self.generated_dict = {}  # dictionary of generated compounds
        self.filter_check_list = []  # only needed for output (at rank 0)
        self.obj_column_names = [
            f.__name__ for f in self.reward_calculator.get_objective_functions(self.conf)
        ]

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
            self.comm.bsend(
                np.asarray(
                    [
                        node.state,
                        node.reward,
                        node.wins,
                        node.visits,
                        node.num_thread_visited,
                        node.path_ucb,
                    ],
                    dtype=object,
                ),
                dest=dest,
                tag=tag,
            )
        else:
            self.comm.bsend(
                np.asarray(
                    [
                        node.state,
                        node.reward,
                        node.wins,
                        node.visits,
                        node.num_thread_visited,
                        data,
                    ],
                    dtype=object,
                ),
                dest=dest,
                tag=tag,
            )

    def send_search_childnode(self, node, ucb_table, dest):
        self.comm.bsend(
            np.asarray(
                [
                    node.state,
                    node.reward,
                    node.wins,
                    node.visits,
                    node.num_thread_visited,
                    ucb_table,
                ],
                dtype=object,
            ),
            dest=dest,
            tag=JobType.SEARCH.value,
        )

    def send_backprop(self, node, dest):
        self.comm.bsend(
            np.asarray(
                [
                    node.state,
                    node.reward,
                    node.wins,
                    node.visits,
                    node.num_thread_visited,
                    node.path_ucb,
                ],
                dtype=object,
            ),
            dest=dest,
            tag=JobType.BACKPROPAGATION.value,
        )

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
            self.logger.info("Gather each rank result...")
        for rid in range(1, self.nprocs):
            self.comm.barrier()
            if self.rank == 0:
                (
                    valid_smiles_list,
                    depth_list,
                    reward_values_list,
                    elapsed_time_list,
                    generated_id_list,
                    objective_values_list,
                    filter_check_list,
                ) = self.comm.recv(source=rid, tag=JobType.GATHER_RESULTS.value, status=status)
                self.valid_smiles_list.extend(valid_smiles_list)
                self.depth_list.extend(depth_list)
                self.reward_values_list.extend(reward_values_list)
                self.elapsed_time_list.extend(elapsed_time_list)
                self.generated_id_list.extend(generated_id_list)
                self.objective_values_list.extend(objective_values_list)
                self.filter_check_list.extend(filter_check_list)
            elif self.rank == rid:
                self.comm.send(
                    (
                        self.valid_smiles_list,
                        self.depth_list,
                        self.reward_values_list,
                        self.elapsed_time_list,
                        self.generated_id_list,
                        self.objective_values_list,
                        self.filter_check_list,
                    ),
                    dest=0,
                    tag=JobType.GATHER_RESULTS.value,
                )

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
            df.to_csv(self.output_path, mode="a", index=False, header=False)
        else:
            df.to_csv(self.output_path, mode="w", index=False)
        if self.rank == 0:
            self.logger.info(f"Save a result at {self.output_path}")

        self.generated_id_list.clear()
        self.valid_smiles_list.clear()
        self.reward_values_list.clear()
        self.depth_list.clear()
        self.elapsed_time_list.clear()
        self.filter_check_list.clear()
        self.objective_values_list.clear()

    def MP_MCTS(self):
        # self.comm.barrier()
        status = MPI.Status()
        self.start_time = time.time()
        _, rootdest = self.hsm.hashing(self.root_position)
        jobq = deque()
        timeup = False
        checkpoint_prepare = False
        checkpoint_saved = False
        checkpoint_ready_count = 0
        if self.conf["checkpoint_load"]:
            for i in range(self.nprocs):
                if self.rank == i:
                    self.comm.barrier()
                    ckpt_path = os.path.join(
                        self.conf["output_dir"], f"mp_checkpoint_rank{i:04}.pickle"
                    )
                    with open(ckpt_path, mode="rb") as f:
                        cp_obj = pickle.load(f)
                    print("Load complete", self.rank)
            # self.generated_id_list = cp_obj['generated_id_list']
            self.total_valid_num = cp_obj["total_valid_num"]
            # self.conf = cp_obj['conf']
            self.generated_dict = cp_obj["generated_dict"]
            self.hsm = cp_obj["hsm"]
            # self.start_time = cp_obj['start_time']
            jobq = cp_obj["jobq"]
        elif self.rank == rootdest:
            root_job_message = np.asarray([self.root_position, None, 0, 0, 0, []], dtype=object)
            for i in range(3 * self.nprocs):
                temp = deepcopy(root_job_message)
                root_job = (JobType.SEARCH.value, temp)
                jobq.appendleft(root_job)

        while not timeup:
            if self.rank == 0:
                if (
                    self.elapsed_time() > self.threshold
                    and self.conf["save_checkpoint"]
                    and not checkpoint_prepare
                ):
                    checkpoint_prepare = True
                    checkpoint_ready_count = 0
                    # print('Checkpoint prepare')
                    for dest in range(1, self.nprocs):
                        dummy_data = tag = JobType.CHECKPOINT_PREPARE.value
                        self.comm.bsend(dummy_data, dest=dest, tag=JobType.CHECKPOINT_PREPARE.value)
                if self.elapsed_time() > self.threshold and (
                    not self.conf["save_checkpoint"] or checkpoint_saved
                ):
                    timeup = True
                    for dest in range(1, self.nprocs):
                        dummy_data = tag = JobType.TIMEUP.value
                        self.comm.bsend(dummy_data, dest=dest, tag=JobType.TIMEUP.value)

            while True:
                ret = self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                if not ret:
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
                if checkpoint_prepare:
                    (tag, message) = jobq[-1]
                    if JobType.is_high_priority(tag):
                        pass
                    else:
                        continue
                (tag, message) = jobq.pop()
                if tag == JobType.SEARCH.value:
                    if self.hsm.search_table(message[0]) is None:
                        node = MPNode(position=message[0], conf=self.conf)
                        if node.state == self.root_position:
                            node.expansion(self.chem_model, self.tokens, self.logger)
                            m = self.conf["random_generator"].choice(node.expanded_nodes)
                            n = node.addnode(m, self.tokens)
                            self.hsm.insert(Item(node.state, node))
                            _, dest = self.hsm.hashing(n.state)
                            self.send_message(n, dest, tag=JobType.SEARCH.value)
                        else:
                            if len(node.state) < self.conf["max_len"]:
                                gen_id = self.get_generated_id()
                                values_list, score, smi, filter_flag, is_valid_smi = (
                                    node.simulation(
                                        self.chem_model,
                                        node.state,
                                        gen_id,
                                        self.generated_dict,
                                        self.reward_calculator,
                                        self.tokens,
                                    )
                                )
                                if is_valid_smi:
                                    self.record_result(
                                        smiles=smi,
                                        depth=len(node.state),
                                        reward=score,
                                        gen_id=gen_id,
                                        raw_reward_list=values_list,
                                        filter_flag=filter_flag,
                                    )
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
                        if node.state == self.root_position:
                            # print ("in table root:",node.state,node.path_ucb,len(node.state),len(node.path_ucb))
                            if node.expanded_nodes != []:
                                m = self.conf["random_generator"].choice(node.expanded_nodes)
                                n = node.addnode(m, self.tokens)
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(n.state)
                                self.send_message(n, dest, tag=JobType.SEARCH.value)
                            else:
                                ind, childnode = node.selection()
                                self.hsm.insert(Item(node.state, node))
                                ucb_table = update_selection_ucbtable_mpmcts(
                                    node, ind, self.root_position
                                )
                                _, dest = self.hsm.hashing(childnode.state)
                                self.send_message(
                                    childnode,
                                    dest,
                                    tag=JobType.SEARCH.value,
                                    data=ucb_table,
                                )
                        else:
                            node.path_ucb = message[5]
                            # print("check ucb:", node.wins, node.visits, node.num_thread_visited)
                            if len(node.state) < self.conf["max_len"]:
                                if node.state[-1] != "\n":
                                    if node.expanded_nodes != []:
                                        m = self.conf["random_generator"].choice(
                                            node.expanded_nodes
                                        )
                                        n = node.addnode(m, self.tokens)
                                        self.hsm.insert(Item(node.state, node))
                                        _, dest = self.hsm.hashing(n.state)
                                        self.send_message(n, dest, tag=JobType.SEARCH.value)
                                    else:
                                        if node.check_childnode == []:
                                            node.expansion(
                                                self.chem_model,
                                                self.tokens,
                                                self.logger,
                                            )
                                            m = self.conf["random_generator"].choice(
                                                node.expanded_nodes
                                            )
                                            n = node.addnode(m, self.tokens)
                                            self.hsm.insert(Item(node.state, node))
                                            _, dest = self.hsm.hashing(n.state)
                                            self.send_message(n, dest, tag=JobType.SEARCH.value)
                                        else:
                                            ind, childnode = node.selection()
                                            self.hsm.insert(Item(node.state, node))
                                            ucb_table = update_selection_ucbtable_mpmcts(
                                                node, ind, self.root_position
                                            )
                                            _, dest = self.hsm.hashing(childnode.state)
                                            self.send_message(
                                                childnode,
                                                dest,
                                                tag=JobType.SEARCH.value,
                                                data=ucb_table,
                                            )
                                else:
                                    gen_id = self.get_generated_id()
                                    (
                                        values_list,
                                        score,
                                        smi,
                                        filter_flag,
                                        is_valid_smi,
                                    ) = node.simulation(
                                        self.chem_model,
                                        node.state,
                                        gen_id,
                                        self.generated_dict,
                                        self.reward_calculator,
                                        self.tokens,
                                    )
                                    score = -1
                                    if is_valid_smi:
                                        self.record_result(
                                            smiles=smi,
                                            depth=len(node.state),
                                            reward=score,
                                            gen_id=gen_id,
                                            raw_reward_list=values_list,
                                            filter_flag=filter_flag,
                                        )
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
                    node = MPNode(position=message[0], conf=self.conf)
                    node.reward = message[1]
                    local_node = self.hsm.search_table(message[0][0:-1])
                    if local_node.state == self.root_position:
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
                    # print('Timeup', self.rank)
                    timeup = True
                elif tag == JobType.CHECKPOINT_PREPARE.value:
                    checkpoint_prepare = True
                    # print('Checkpoint prepare', self.rank)
                    dummy_data = JobType.CHECKPOINT_READY.value
                    self.comm.bsend(dummy_data, dest=0, tag=JobType.CHECKPOINT_READY.value)
                elif tag == JobType.CHECKPOINT_READY.value:
                    assert self.rank == 0
                    checkpoint_ready_count += 1
                    # print('Checkpoint ready count', checkpoint_ready_count)
                    if checkpoint_ready_count >= self.nprocs - 1:
                        # print('Checkpoint ready count', checkpoint_ready_count)
                        for dest in range(0, self.nprocs):
                            dummy_data = tag = JobType.CHECKPOINT_SAVE.value
                            self.comm.bsend(
                                dummy_data, dest=dest, tag=JobType.CHECKPOINT_SAVE.value
                            )
                elif tag == JobType.CHECKPOINT_SAVE.value:
                    cp_obj = {
                        #'generated_id_list': self.generated_id_list,
                        "total_valid_num": self.total_valid_num,
                        "conf": self.conf,
                        "generated_dict": self.generated_dict,
                        "hsm": self.hsm,
                        "start_time": self.start_time,
                        "jobq": jobq,
                    }
                    # print('Checkpoint save start')
                    for i in range(self.nprocs):
                        if self.rank == i:
                            self.comm.barrier()
                            ckpt_path = os.path.join(
                                self.conf["output_dir"],
                                f"mp_checkpoint_rank{i:04}.pickle",
                            )
                            with open(ckpt_path, mode="wb") as f:
                                pickle.dump(cp_obj, f)
                    # print('Checkpoint save finished')
                    checkpoint_saved = True

        return
