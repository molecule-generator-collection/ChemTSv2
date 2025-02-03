import sys

from rdkit.Chem import Descriptors, rdMolDescriptors

from chemtsv2.abc import Filter
from chemtsv2.utils import transform_linker_to_mol


class LipinskiFilter(Filter):
    def check(mol, conf):
        weight = round(rdMolDescriptors._CalcMolWt(mol), 2)
        logp = Descriptors.MolLogP(mol)
        donor = rdMolDescriptors.CalcNumLipinskiHBD(mol)
        acceptor = rdMolDescriptors.CalcNumLipinskiHBA(mol)
        rotbonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        if conf['lipinski_filter']['type'] == 'rule_of_5':
            cond = weight <= 500 and logp <= 5 and donor <= 5 and acceptor <= 10
        elif conf['lipinski_filter']['type'] == 'rule_of_3':
            cond = weight <= 300 and logp <= 3 and donor <= 3 and acceptor <= 3 and rotbonds <= 3
        else:
            print("`use_lipinski_filter` only accepts [rule_of_5, rule_of_3]")
            sys.exit(1)
        return cond


class LipinskiFilterForXMol(Filter):
    def check(mol, conf):
        @transform_linker_to_mol(conf)
        def _check(mol, conf):
            return LipinskiFilter.check(mol, conf)
        return _check(mol, conf)
