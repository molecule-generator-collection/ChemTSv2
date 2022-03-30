from math import log, sqrt

from policy.policy import Policy

class Ucb1(Policy):
    def evaluate(child_state, conf):
        ucb1 = ((child_state.total_reward / child_state.visits)
                + conf['c_val'] * sqrt(2 * log(child_state.parent_node.state.visits) / child_state.visits))
        return ucb1