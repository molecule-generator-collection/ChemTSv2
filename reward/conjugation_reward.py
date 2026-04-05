from collections import Counter

import numpy as np

from chemtsv2.abc import Reward


class Conjugation_reward(Reward):
    def get_objective_functions(conf):
        def MaxConjugatedSize(mol):
            n = mol.GetNumAtoms()
            parent = list(range(n))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a, b):
                a = find(a)
                b = find(b)
                if a != b:
                    parent[a] = b

            conj_atoms = set()
            for bond in mol.GetBonds():
                if bond.GetIsConjugated():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    union(i, j)
                    conj_atoms.update((i, j))

            if not conj_atoms:
                return 0
            return max(Counter(find(a) for a in conj_atoms).values())  

        return [MaxConjugatedSize]

    def calc_reward_from_objective_values(values, conf):
        x = values[0]
        sigmoid = 1.0 / (1.0 + np.exp(-conf["sigmoid_steepness"] * (x - conf["sigmoid_center"])))
        return sigmoid + conf["sigmoid_linear"] * x
