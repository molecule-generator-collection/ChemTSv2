from rdkit import Chem
from rdkit.Chem import rdfiltercatalog

from chemtsv2.filter import Filter
from chemtsv2.utils import transform_linker_to_mol


def get_catalog():
    covalent_warhead_dict = {
        'alpha-halomethyl_ketone': ['*[C;!R](=O)[CH2][F,Cl,Br,I]', 1],
        'nitrile': ['*C#N', 1],
        'epoxide': ['C1OC1', 1],
        'haloacetamide': ["*[NH]C(=O)[CH2][F,Cl,Br,I]", 1],
        'boronic_acid': ["*B([O;H1])[O;H1]", 1],
        'ketone': ['CC(=O)C', 1],
        'acrylamide': ['*[NH]C(=O)C=C', 1],
        'ab-unsaturated_carbonyl': ['*C(=O)C=C', 1],
        'aldehyde': ["*C(=O)[H]", 1],
    }
    catalog = rdfiltercatalog.FilterCatalog()
    for label, (smarts_patt, min_count) in covalent_warhead_dict.items():
        f = rdfiltercatalog.SmartsMatcher(label, smarts_patt, min_count)
        catalog.AddEntry(rdfiltercatalog.FilterCatalogEntry(label, f))
    return catalog

COVALENT_WARHEAD_FILTER_CATALOG = get_catalog()


class CovalentWarheadFilter(Filter):
    def check(mol, conf):
        if conf['debug']:
            matchs = [e.GetDescription() for e in COVALENT_WARHEAD_FILTER_CATALOG.GetMatches(mol)]
            if matchs != []:
                smi = Chem.MolToSmiles(mol)
                print(f"[DEBUG FILTER] SMILES -> {smi} ; Match -> {','.join(matchs)}")
        return not COVALENT_WARHEAD_FILTER_CATALOG.HasMatch(mol)


class CovalentWarheadFilterForXMol(Filter):
    def check(mol, conf):
        @transform_linker_to_mol(conf)
        def _check(mol, conf):
            return CovalentWarheadFilter.check(mol, conf)
        return _check(mol, conf)
