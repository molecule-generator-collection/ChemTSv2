from math import log, sqrt

from policy.policy import Policy

class Ucb1(Policy):
    def evaluate(total_reward, constant, parent_visits, child_visits):
        ucb1 = ((total_reward / child_visits)
                + constant * sqrt(2 * log(parent_visits) / child_visits))
        return ucb1