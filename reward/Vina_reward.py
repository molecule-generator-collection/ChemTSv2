from vina import Vina
from oddt.toolkits.extras import rdkit as ordkit
from rdkit import Chem
from rdkit.Chem import AllChem


def calc_objective_values(smiles, conf):
    v = Vina(sf_name=conf['vina_sf_name'], cpu=conf['vina_cpus'], verbosity=0)
    v.set_receptor(rigid_pdbqt_filename=conf['receptor'])

    mol = Chem.MolFromSmiles(smiles)
    AllChem.EmbedMolecule(mol)
    mol_pdbqt = ordkit.MolToPDBQTBlock(mol, computeCharges=True)
    v.set_ligand_from_string(mol_pdbqt)

    v.compute_vina_maps(center=conf['center'], box_size=conf['box_size'])

    _ = v.optimize()

    v.dock(exhaustiveness=32, n_poses=10)
    min_inter_score = v.energies()[0][1]
    return [min_inter_score]


def calc_reward_from_objective_values(values, conf):
    min_inter_score = values[0]
    score_diff = min_inter_score - conf['base_score']
    return - score_diff * 0.1 / (1 + abs(score_diff) * 0.1)