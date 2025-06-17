from rdkit.Chem import Descriptors

from chemtsv2.abc import Filter
from chemtsv2.utils import transform_linker_to_mol, attach_fragment_to_all_sites


class MWFilter(Filter):
    def check(mol, config):
        return Descriptors.MolWt(mol) <= config["mw_filter"]["threshold"]


class MWFilterForXMol(Filter):
    def check(mol, conf):
        @transform_linker_to_mol(conf)
        def _check(mol, conf):
            return MWFilter.check(mol, conf)

        return _check(mol, conf)


class MWFilterForDecoration(Filter):
    def check(mol, conf):
        @attach_fragment_to_all_sites(conf)
        def _check(mol, conf):
            return MWFilter.check(mol, conf)

        return _check(mol, conf)
