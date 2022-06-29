""" Refer to https://arxiv.org/abs/1710.00616 for Jscore """
import sys

from rdkit.Chem import Descriptors
sys.path.append("./data/")
import sascorer

from reward.reward import BatchReward

class Jscore_reward(BatchReward):
    def get_batch_objective_functions():
        def LogP(mols, confs):
            return [Descriptors.MolLogP(mol) for mol in mols]

        def SAScore(mols, confs):
            return [sascorer.calculateScore(mol) for mol in mols]

        def RingSizePenalty(mols, confs):
            scores = []
            for mol in mols:
                ri = mol.GetRingInfo()
                max_ring_size = max((len(r) for r in ri.AtomRings()), default=0)
                scores.append(max_ring_size-6)
            return scores

        return [LogP, SAScore, RingSizePenalty]


    def calc_reward_from_objective_values(values, conf):
        logP, sascore, ring_size_penalty = values
        jscore = logP - sascore - ring_size_penalty
        return jscore / (1 + jscore)