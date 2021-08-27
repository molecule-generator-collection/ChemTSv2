import random


def calc_reward_score(compound):
    return [random.uniform(-100, 10)]

def calc_simulation_score(scores, base_score=-20):  # simulation score of SBMolGen
    return ((-(scores[0] - base_score) * 0.1) / (1 + abs(scores[0] - base_score) * 0.1))
