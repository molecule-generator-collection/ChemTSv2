from rdkit.Chem import Descriptors

from chemtsv2.filter import Filter
from chemtsv2.utils import transform_linker_to_mol

class HBDFilter(Filter):
    def check(mol, config): 
        return Descriptors.NumHDonors(mol) <= config['hbd_filter']['threshold']

class HBDFilterForXMol(Filter):
    def check(mol, conf):
        @transform_linker_to_mol(conf)
        def _check(mol, conf):
            return HBDFilter.check(mol, conf)
        return _check(mol, conf)
