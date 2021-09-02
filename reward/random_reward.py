import random


def calc_reward_score(smiles):
    return [random.uniform(-100, 10)]

def scaling_score(scores, conf):  # scaling score of SBMolGen
    return ((-(scores[0] - conf['base_score']) * 0.1) / (1 + abs(scores[0] - conf['base_score']) * 0.1))
