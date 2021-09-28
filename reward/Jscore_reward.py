""" Refer to https://arxiv.org/abs/1710.00616 for Jscore """
import os
import sys

from rdkit import Chem
from rdkit.Chem import Descriptors, RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


def calc_objective_values(smiles, conf):
    mol = Chem.MolFromSmiles(smiles)
    logP = Descriptors.MolLogP(mol)
    sascore = sascorer.calculateScore(mol)
    ri = mol.GetRingInfo()
    max_ring_size = max((len(r) for r in ri.AtomRings()), default=0)
    ring_size_penalty = max_ring_size - 6
    return [logP, sascore, ring_size_penalty]


def calc_reward_from_objective_values(values, conf):
    logP, sascore, ring_size_penalty = values
    jscore = logP - sascore - ring_size_penalty
    return jscore / (1 + jscore)