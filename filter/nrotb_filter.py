from rdkit.Chem import Descriptors

from chemtsv2.abc import Filter
from chemtsv2.utils import transform_linker_to_mol, attach_fragment_to_all_sites


class NRotBFilter(Filter):
    def check(mol, config):
        return Descriptors.NumRotatableBonds(mol) <= config["nrotb_filter"]["threshold"]


class NRotBFilterForXMol(Filter):
    def check(mol, conf):
        @transform_linker_to_mol(conf)
        def _check(mol, conf):
            return NRotBFilter.check(mol, conf)

        return _check(mol, conf)

class NRotBFilterForXMol(Filter):
    def check(mol, conf):
        @attach_fragment_to_all_sites(conf)
        def _check(mol, conf):
            return NRotBFilter.check(mol, conf)

        return _check(mol, conf)
