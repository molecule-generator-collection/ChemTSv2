from rdkit import Chem
from rdkit.Chem import Descriptors


def calc_objective_values(smiles):
    mol = Chem.MolFromSmiles(smiles)
    score = Descriptors.MolLogP(mol) if mol is not None else -1.
    return [score]


def calc_reward_from_objective_values(values, conf):
    return (0.8 * values[0]) / (1 + 0.8 * abs(values[0]))