import os
import subprocess
import shutil
import tempfile

#from vina import Vina
from meeko import MoleculePreparation, PDBQTMolecule
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Geometry import Point3D

from reward.reward import Reward


class Vina_reward(Reward):
    def get_objective_functions(conf):
        def VinaScore(mol):
            verbosity = 1 if conf['debug'] else 0
            temp_dir = tempfile.mkdtemp()
            temp_ligand_fname = os.path.join(temp_dir, 'ligand_temp.pdbqt')
            pose_dir = os.path.join(conf['output_dir'], "3D_pose")
            os.makedirs(pose_dir, exist_ok=True)
            output_ligand_fname = os.path.join(pose_dir, f"mol_{conf['gid']}_out.pdbqt")

            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            try:
                mol_conf = mol.GetConformer(-1)
            except ValueError as e:
                print(f"Error SMILES: {Chem.MolToSmiles(mol)}")
                print(e) 
                if not conf['debug']:
                    shutil.rmtree(temp_dir)
                return None

            centroid = list(rdMolTransforms.ComputeCentroid(mol_conf))
            tr = [conf['vina_center'][i] - centroid[i] for i in range(3)]
            for i, p in enumerate(mol_conf.GetPositions()):
                mol_conf.SetAtomPosition(i, Point3D(p[0]+tr[0], p[1]+tr[1], p[2]+tr[2]))
            mol_prep = MoleculePreparation()
            mol_prep.prepare(mol)
            mol_prep.write_pdbqt_file(temp_ligand_fname)
            cmd = [
                conf['vina_bin_path'],
                '--receptor', conf['vina_receptor'],
                '--ligand', temp_ligand_fname,
                '--center_x', str(conf['vina_center'][0]),
                '--center_y', str(conf['vina_center'][1]),
                '--center_z', str(conf['vina_center'][2]),
                '--size_x', str(conf['vina_box_size'][0]),
                '--size_y', str(conf['vina_box_size'][1]),
                '--size_z', str(conf['vina_box_size'][2]),
                '--cpu', str(conf['vina_cpus']),
                '--exhaustiveness', str(conf['vina_exhaustiveness']),
                '--max_evals', str(conf['vina_max_evals']),
                '--num_modes', str(conf['vina_num_modes']),
                '--min_rmsd', str(conf['vina_min_rmsd']),
                '--energy_range', str(conf['vina_energy_range']),
                '--out', output_ligand_fname,
                '--spacing', str(conf['vina_spacing']),
                '--verbosity', str(verbosity)]
            if conf['debug']:
                print(cmd)
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error SMILES: {Chem.MolToSmiles(mol)}")
                print(e) 
                if not conf['debug']:
                    shutil.rmtree(temp_dir)
                return None
            try:
                pdbqt_mols = PDBQTMolecule.from_file(output_ligand_fname, skip_typing=True)
            except RuntimeError as e:
                print(f"Error SMILES: {Chem.MolToSmiles(mol)}")
                print(e) 
                if not conf['debug']:
                    shutil.rmtree(temp_dir)
                return None
                
            min_affinity_score = pdbqt_mols[0].score
            if conf['debug']:
                print(f"min_affinity_score: {min_affinity_score}")
            if not conf['debug']:
                shutil.rmtree(temp_dir)
            return min_affinity_score
        return [VinaScore]
    
    
    def calc_reward_from_objective_values(values, conf):
        min_inter_score = values[0]
        if min_inter_score is None:
            return -1
        score_diff = min_inter_score - conf['vina_base_score']
        return - score_diff * 0.1 / (1 + abs(score_diff) * 0.1)
