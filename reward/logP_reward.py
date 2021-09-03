from rdkit import Chem
from rdkit.Chem import Descriptors


def calc_reward_score(smiles):
    mol = Chem.MolFromSmiles(smiles)
    score = Descriptors.MolLogP(mol) if mol is not None else -1.
    return [score]

def scaling_score(scores, conf):
    return (0.8 * scores[0]) / (1 + 0.8 * abs(scores[0]))