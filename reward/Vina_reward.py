import pprint

from vina import Vina
from oddt.toolkits.extras import rdkit as ordkit
from rdkit import Chem
from rdkit.Chem import AllChem


def get_objective_functions(conf):
    def VinaScore(mol):
        verbosity = 1 if conf['debug'] else 0
        v = Vina(sf_name=conf['vina_sf_name'], cpu=conf['vina_cpus'], verbosity=verbosity)
        v.set_receptor(rigid_pdbqt_filename=conf['vina_receptor'])

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        mol_pdbqt = ordkit.MolToPDBQTBlock(mol, computeCharges=True)
        v.set_ligand_from_string(mol_pdbqt)

        v.compute_vina_maps(
            center=conf['vina_center'],
            box_size=conf['vina_box_size'],
            spacing=conf['vina_spacing'])

        _ = v.optimize()

        if conf['debug']:
            pprint.pprint(v.info())

        v.dock(
            exhaustiveness=conf['vina_exhaustiveness'],
            n_poses=conf['vina_n_poses'],
            min_rmsd=conf['vina_min_rmsd'],
            max_evals=conf['vina_max_evals'])
        if conf['debug']:
            print(f"Vina Docking energies: {v.energies()}")
        return v.energies()[0][1]
    return [VinaScore]


def calc_reward_from_objective_values(values, conf):
    min_inter_score = values[0]
    score_diff = min_inter_score - conf['vina_base_score']
    return - score_diff * 0.1 / (1 + abs(score_diff) * 0.1)
