from rdkit.Chem import Descriptors

from chemtsv2.abc import Filter
from chemtsv2.utils import transform_linker_to_mol, attach_fragment_to_all_sites


class LogPFilter(Filter):
    def check(mol, config):
        return Descriptors.MolLogP(mol) <= config["logp_filter"]["threshold"]


class LogPFilterForXMol(Filter):
    def check(mol, conf):
        @transform_linker_to_mol(conf)
        def _check(mol, conf):
            return LogPFilter.check(mol, conf)

        return _check(mol, conf)


class LogPFilterForDecoration(Filter):
    def check(mol, conf):
        @attach_fragment_to_all_sites(conf)
        def _check(mol, conf):
            return LogPFilter.check(mol, conf)

        return _check(mol, conf)
