from rdkit import Chem
from canonicalize_psmiles.canonicalize import canonicalize, reduce_multiplication

from chemtsv2.abc import Filter


class MonomerFilter(Filter):
    def check(mol, conf):
        smiles = Chem.MolToSmiles(mol)
        if conf["monomer_filter"]["canonicalize"]:
            monomer = canonicalize(smiles)
        else:
            monomer = reduce_multiplication(smiles)
        return smiles == monomer
