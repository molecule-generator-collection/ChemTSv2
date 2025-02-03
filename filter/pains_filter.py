import sys

from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams

from chemtsv2.abc import Filter
from chemtsv2.utils import transform_linker_to_mol


class PainsFilter(Filter):
    def check(mol, conf):
        params = FilterCatalogParams()
        is_valid_key = False
        if 'pains_a' in conf['pains_filter']['type']:
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
            is_valid_key = True
        if 'pains_b' in conf['pains_filter']['type']:
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
            is_valid_key = True
        if 'pains_c' in conf['pains_filter']['type']:
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
            is_valid_key = True
        if not is_valid_key:
            print("`use_pains_filter` only accepts [pains_a, pains_b, pains_c]")
            sys.exit(1)
        filter_catalogs = FilterCatalog.FilterCatalog(params)
        return not filter_catalogs.HasMatch(mol)


class PainsFilterForXMol(Filter):
    def check(mol, conf):
        @transform_linker_to_mol(conf)
        def _check(mol, conf):
            return PainsFilter.check(mol, conf)
        return _check(mol, conf)
