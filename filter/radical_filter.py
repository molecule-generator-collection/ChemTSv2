from rdkit.Chem import Descriptors

from filter.filter import Filter


class RadicalFilter(Filter):
    def check(mol, config):
        return Descriptors.NumRadicalElectrons(mol) == 0