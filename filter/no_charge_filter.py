from rdkit import Chem

from chemtsv2.filter import Filter


class NoChargeFilter(Filter):
    def check(mol, conf):
        return Chem.rdmolops.GetFormalCharge(mol) == 0