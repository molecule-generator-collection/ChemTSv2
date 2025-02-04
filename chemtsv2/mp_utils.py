from math import log, sqrt
from random import randint

import numpy as np


### Check UCB path ###
def backtrack_mpmcts(pnode, cnode):
    for path_ucb in reversed(pnode.path_ucb):
        ind = path_ucb[0][3]
        path_ucb[0][0] += cnode.reward
        path_ucb[0][1] += 1
        path_ucb[0][2] -= 1
        path_ucb[ind + 1][0] += cnode.reward
        path_ucb[ind + 1][1] += 1
        path_ucb[ind + 1][2] -= 1
    return pnode


def compare_ucb_mpmcts(pnode):
    # print ("check info_table:",info_table)
    for path_ucb in pnode.path_ucb:
        ucb = []
        for i in range(len(path_ucb) - 1):
            ind = path_ucb[0][3]
            ucb.append(
                (path_ucb[i + 1][0] + 0) / (path_ucb[i + 1][1] + path_ucb[i + 1][2])
                + pnode.conf["c_val"]
                * sqrt(
                    2
                    * log(path_ucb[0][1] + path_ucb[0][2])
                    / (path_ucb[i + 1][1] + path_ucb[i + 1][2])
                )
            )
        m = np.amax(ucb)
        indices = np.nonzero(ucb == m)[0]
        if ind in indices:
            back_flag = 0
        else:
            back_flag = 1
            break
    return back_flag


def update_selection_ucbtable_mpmcts(node, ind, root_position=["&"]):
    table = []
    final_table = []
    node_info = store_info(node)
    node_info.append(ind)
    table.append(node_info)
    for i in range(len(node.child_nodes)):
        child_info = store_info(node.child_nodes[i])
        table.append(child_info)
    if node.state == root_position:
        final_table.append(table)
    else:
        final_table.extend(node.path_ucb)
        final_table.append(table)
    return final_table


def store_info(node):
    table = [node.wins, node.visits, node.num_thread_visited]
    return table


### END Check UCB Path ###


### Zobrist Hash ###
class Item:
    key = ""
    value = 0

    def __init__(self, key, value):
        self.key = key
        self.value = value


class HashTable:
    "Common base class for a hash table"

    tableSize = 0
    entriesCount = 0
    alphabetSize = 2 * 26
    hashTable = []

    def __init__(self, nprocs, tokens, max_len):
        self.hashTable = dict()  # [[] for i in range(size)]

        # should be enough for our case, but should be larger for longer search
        self.hash_index_bits = 32
        self.hash_table_max_size = 2**self.hash_index_bits

        self.S = max_len
        self.P = len(tokens)
        self.tokens = tokens
        self.nprocs = nprocs
        self.zobristnum = [[0] * self.P for _ in range(self.S)]
        for i in range(self.S):
            for j in range(self.P):
                self.zobristnum[i][j] = randint(0, 2**64 - 1)

    def hashing(self, board):
        hashing_value = 0
        for i in range(self.S):
            piece = None
            if i <= len(board) - 1:
                if board[i] in self.tokens:
                    piece = self.tokens.index(board[i])
            if piece is not None:
                hashing_value ^= self.zobristnum[i][piece]

        # tail = int(math.log2(self.nprocs))
        # #print (tail)
        # head = int(64-math.log2(self.nprocs))
        # #print (head)
        # hash_key = format(hashing_value, '064b')[0:head]
        # hash_key = int(hash_key, 2)
        # core_dest = format(hashing_value, '064b')[-tail:]
        # core_dest = int(core_dest, 2)

        hash_key = hashing_value
        core_dest = (hashing_value >> self.hash_index_bits) % self.nprocs

        return hash_key, core_dest

    def insert(self, item):
        hash, _ = self.hashing(item.key)
        if self.hashTable.get(hash) is None:
            self.hashTable.setdefault(hash, [])
            self.hashTable[hash].append(item)
        else:
            for i, it in enumerate(self.hashTable[hash]):
                if it.key == item.key:
                    del self.hashTable[hash][i]
            self.hashTable[hash].append(item)

    def search_table(self, key):
        hash, _ = self.hashing(key)
        if self.hashTable.get(hash) is None:
            return None
        else:
            for i, it in enumerate(self.hashTable[hash]):
                if it.key == key:
                    return it.value
        return None


### END Zoblist Hash ###
