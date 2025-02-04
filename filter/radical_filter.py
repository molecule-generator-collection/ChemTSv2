from rdkit.Chem import Descriptors

from chemtsv2.abc import Filter
from chemtsv2.utils import transform_linker_to_mol


class RadicalFilter(Filter):
    def check(mol, conf):
        return Descriptors.NumRadicalElectrons(mol) == 0


class RadicalFilterForXMol(Filter):
    def check(mol, conf):
        @transform_linker_to_mol(conf)
        def _check(mol, conf):
            return RadicalFilter.check(mol, conf)

        return _check(mol, conf)
