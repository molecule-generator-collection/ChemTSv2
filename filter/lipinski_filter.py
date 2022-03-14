import sys

from rdkit.Chem import Descriptors, rdMolDescriptors

from filter.filter import Filter


class LipinskiFilter(Filter):
    def check(mol, conf):
        weight = round(rdMolDescriptors._CalcMolWt(mol), 2)
        logp = Descriptors.MolLogP(mol)
        donor = rdMolDescriptors.CalcNumLipinskiHBD(mol)
        acceptor = rdMolDescriptors.CalcNumLipinskiHBA(mol)
        rotbonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        if conf['lipinski_filter']['type'] == 'rule_of_5':
            cond = weight < 500 or logp < 5 or donor < 5 or acceptor < 10
        elif conf['lipinski_filter']['type'] == 'rule_of_3':
            cond = weight < 300 or logp < 3 or donor < 3 or acceptor < 3 or rotbonds < 3
        else:
            print("`use_lipinski_filter` only accepts [rule_of_5, rule_of_3]")
            sys.exit(1)
        return cond