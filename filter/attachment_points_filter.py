from rdkit import Chem

from chemtsv2.abc import Filter


class AttachmentPointsFilter(Filter):
    def check(mol, conf):
        smi = Chem.MolToSmiles(mol)
        return smi.count('*') == conf['attachment_points_filter']['threshold']
