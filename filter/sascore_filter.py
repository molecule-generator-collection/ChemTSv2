import sys
sys.path.append("./data/")
import sascorer # pyright: ignore[reportMissingImports]

from chemtsv2.abc import Filter
from chemtsv2.utils import transform_linker_to_mol


class SascoreFilter(Filter):
    def check(mol, conf):
        return conf["sascore_filter"]["threshold"] > sascorer.calculateScore(mol)


class SascoreFilterForXMol(Filter):
    def check(mol, conf):
        @transform_linker_to_mol(conf)
        def _check(mol, conf):
            return SascoreFilter.check(mol, conf)

        return _check(mol, conf)
