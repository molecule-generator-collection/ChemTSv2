from rdkit.Chem import Descriptors, rdMolDescriptors

from filter.filter import Filter


class AromaticRingFilter(Filter):
    def check(mol, conf):
        return Descriptors.NumAromaticRings(mol) > 0