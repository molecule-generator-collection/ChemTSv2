from rdkit.Chem import Descriptors

from chemtsv2.filter import Filter


class RadicalFilter(Filter):
    def check(mol, config):
        return Descriptors.NumRadicalElectrons(mol) == 0