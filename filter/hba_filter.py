from rdkit.Chem import Descriptors

from chemtsv2.abc import Filter
from chemtsv2.utils import transform_linker_to_mol


class HBAFilter(Filter):
    def check(mol, config): 
        return Descriptors.NumHAcceptors(mol) <= config['hba_filter']['threshold']


class HBAFilterForXMol(Filter):
    def check(mol, conf):
        @transform_linker_to_mol(conf)
        def _check(mol, conf):
            return HBAFilter.check(mol, conf)
        return _check(mol, conf)
