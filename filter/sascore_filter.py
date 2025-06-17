import sys
sys.path.append("./data/")
import sascorer # pyright: ignore[reportMissingImports]

from chemtsv2.abc import Filter
from chemtsv2.utils import transform_linker_to_mol, attach_fragment_to_all_sites


class SascoreFilter(Filter):
    def check(mol, conf):
        return conf["sascore_filter"]["threshold"] > sascorer.calculateScore(mol)


class SascoreFilterForXMol(Filter):
    def check(mol, conf):
        @transform_linker_to_mol(conf)
        def _check(mol, conf):
            return SascoreFilter.check(mol, conf)

        return _check(mol, conf)


class SascoreFilterForDecoration(Filter):
    def check(mol, conf):
        @attach_fragment_to_all_sites(conf)
        def _check(mol, conf):
            return SascoreFilter.check(mol, conf)

        return _check(mol, conf)
