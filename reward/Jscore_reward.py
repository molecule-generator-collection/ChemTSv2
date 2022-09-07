""" Refer to https://arxiv.org/abs/1710.00616 for Jscore """
import sys

import numpy as np
from rdkit.Chem import Descriptors
sys.path.append("./data/")
import sascorer

from reward.reward import Reward

LOGP_BASELINE = np.loadtxt('./data/logP_values.txt')
LOGP_MEAN = np.mean(LOGP_BASELINE)
LOGP_STD = np.std(LOGP_BASELINE)

SA_BASELINE = np.loadtxt('./data/SA_scores.txt')
SA_MEAN = np.mean(SA_BASELINE)
SA_STD = np.std(SA_BASELINE)

CS_BASELINE = np.loadtxt('./data/cycle_scores.txt')
CS_MEAN = np.mean(CS_BASELINE)
CS_STD = np.std(CS_BASELINE)

class Jscore_reward(Reward):
    def get_objective_functions(conf):
        def LogP(mol):
            return Descriptors.MolLogP(mol)

        def SAScore(mol):
            return sascorer.calculateScore(mol)

        def RingSizePenalty(mol):
            ri = mol.GetRingInfo()
            max_ring_size = max((len(r) for r in ri.AtomRings()), default=0)
            return max_ring_size - 6

        return [LogP, SAScore, RingSizePenalty]


    def calc_reward_from_objective_values(values, conf):
        """ref: https://github.com/tsudalab/ChemTS/blob/4174c3600ebb47ed136b433b22a29c879824a6ba/mcts_logp_improved_version/add_node_type.py#L172"""
        logP, sascore, ring_size_penalty = values
        logP_norm = (logP - LOGP_MEAN) / LOGP_STD
        sascore_norm = (-sascore - SA_MEAN) / SA_STD
        rs_penalty_norm = (-ring_size_penalty - CS_MEAN) / CS_STD
        #jscore = logP - sascore - ring_size_penalty
        jscore = logP_norm + sascore_norm + rs_penalty_norm
        return jscore / (1 + abs(jscore))
