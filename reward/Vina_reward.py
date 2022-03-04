import pprint

from vina import Vina
from oddt.toolkits.extras import rdkit as ordkit
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Geometry import Point3D


def get_objective_functions(conf):
    def VinaScore(mol):
        verbosity = 1 if conf['debug'] else 0
        v = Vina(sf_name=conf['vina_sf_name'], cpu=conf['vina_cpus'], verbosity=verbosity)
        v.set_receptor(rigid_pdbqt_filename=conf['vina_receptor'])

        mol = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol)
            mol_conf = mol.GetConformer(-1)
            centroid = list(rdMolTransforms.ComputeCentroid(mol_conf))
            tr = [conf['vina_center'][i] - centroid[i] for i in range(3)]
            for i, p in enumerate(mol_conf.GetPositions()):
                mol_conf.SetAtomPosition(i, Point3D(p[0]+tr[0], p[1]+tr[1], p[2]+tr[2]))
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
        except Exception as e:
            print(f"Error SMILES: {Chem.MolToSmiles(mol)}")
            print(e)
            return None
    return [VinaScore]


def calc_reward_from_objective_values(values, conf):
    min_inter_score = values[0]
    if min_inter_score is None:
        return -1
    score_diff = min_inter_score - conf['vina_base_score']
    return - score_diff * 0.1 / (1 + abs(score_diff) * 0.1)
