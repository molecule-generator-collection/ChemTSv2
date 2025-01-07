from rdkit.Chem import Descriptors

from chemtsv2.filter import Filter
from chemtsv2.utils import transform_linker_to_mol

class TPSAFilter(Filter):
    def check(mol, config): 
        return Descriptors.TPSA(mol) <= config['tpsa_filter']['threshold']
    
class TPSAFilterForXMol(Filter):
    def check(mol, conf):
        @transform_linker_to_mol(conf)
        def _check(mol, conf):
            return TPSAFilter.check(mol, conf)
        return _check(mol, conf)