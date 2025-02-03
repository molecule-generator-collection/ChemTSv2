from rdkit.Chem import Descriptors, rdMolDescriptors

from chemtsv2.abc import Filter
from chemtsv2.utils import transform_linker_to_mol


class AromaticRingFilter(Filter):
    def check(mol, conf):
        return Descriptors.NumAromaticRings(mol) > 0


class AromaticRingFilterForXMol(Filter):
    def check(mol, conf):
        @transform_linker_to_mol(conf)
        def _check(mol, conf):
            return AromaticRingFilter.check(mol, conf)
        return _check(mol, conf)
