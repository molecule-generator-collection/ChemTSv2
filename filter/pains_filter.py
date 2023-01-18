import sys

from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams

from filter.filter import Filter


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
