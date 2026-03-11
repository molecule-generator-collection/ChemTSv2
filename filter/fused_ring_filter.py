from rdkit import Chem

from chemtsv2.abc import Filter
from chemtsv2.utils import transform_linker_to_mol, attach_fragment_to_all_sites


class FusedRingFilter(Filter):
    def check(mol, conf):
        pat = Chem.MolFromSmarts("[R&!R1][R&!R1]")
        return mol.HasSubstructMatch(pat) == conf["fused_ring_filter"]["has_fused_ring"]


class FusedRingFilterForXMol(Filter):
    def check(mol, conf):
        @transform_linker_to_mol(conf)
        def _check(mol, conf):
            return FusedRingFilter.check(mol, conf)

        return _check(mol, conf)


class FusedRingFilterForDecoration(Filter):
    def check(mol, conf):
        @attach_fragment_to_all_sites(conf)
        def _check(mol, conf):
            return FusedRingFilter.check(mol, conf)

        return _check(mol, conf)
