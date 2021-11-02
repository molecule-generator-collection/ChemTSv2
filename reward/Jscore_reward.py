""" Refer to https://arxiv.org/abs/1710.00616 for Jscore """
import os
import sys

from rdkit import Chem
from rdkit.Chem import Descriptors, RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


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
    logP, sascore, ring_size_penalty = values
    jscore = logP - sascore - ring_size_penalty
    return jscore / (1 + jscore)