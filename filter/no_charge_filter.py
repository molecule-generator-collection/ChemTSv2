from rdkit import Chem

from chemtsv2.abc import Filter
from chemtsv2.utils import transform_linker_to_mol


class NoChargeFilter(Filter):
    def check(mol, conf):
        return Chem.rdmolops.GetFormalCharge(mol) == 0


class NoChargeFilterForXMol(Filter):
    def check(mol, conf):
        @transform_linker_to_mol(conf)
        def _check(mol, conf):
            return NoChargeFilter.check(mol, conf)

        return _check(mol, conf)
