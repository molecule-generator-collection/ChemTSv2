import os
import shutil
import tempfile

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from spython.main import Client

from chemtsv2.reward import Reward

def check_structure_rmsd(dock_mol, conf):
    scaffold_smiles = conf['fixed_structure_smiles'] 
    scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
    reference_mol = Chem.MolFromMolFile(conf['Crystal_structure_path']) 

    # Extract coordinates (Crystal structure)
    ref_scaf_match_index = reference_mol.GetSubstructMatch(scaffold_mol)
    ref_match_coords = np.array([reference_mol.GetConformer(0).GetAtomPosition(idx) for idx in ref_scaf_match_index])
    
    # Extract coordinates (Docking pose)
    dock_scaf_indexs = dock_mol.GetSubstructMatch(scaffold_mol)
    dock_match_coords = np.array([dock_mol.GetConformer(0).GetAtomPosition(idx) for idx in dock_scaf_indexs])

    # Calculate RMSD
    squared_diff = np.square(ref_match_coords - dock_match_coords)
    rmsd = round(np.sqrt(np.sum(squared_diff) / len(squared_diff)),2)
    return rmsd  

class Gnina_reward(Reward):
    def get_objective_functions(conf):
        def GninaScore(mol):
            temp_dir = tempfile.mkdtemp()
            temp_ligand_fname = os.path.join(temp_dir, 'ligand_temp.sdf')
            pose_dir = os.path.join(conf['output_dir'], "3D_pose")
            os.makedirs(pose_dir, exist_ok=True)
            output_ligand_fname = os.path.join(pose_dir, f"mol_{conf['gid']}_out.sdf")

            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            with Chem.SDWriter(temp_ligand_fname) as f:
                f.write(mol)
            cmd = [
                'gnina',
                '--receptor', conf['gnina_receptor'],
                '--ligand', temp_ligand_fname,
                #'--center_x', str(conf['gnina_center'][0]),
                #'--center_y', str(conf['gnina_center'][1]),
                #'--center_z', str(conf['gnina_center'][2]),
                #'--size_x', str(conf['gnina_box_size'][0]),
                #'--size_y', str(conf['gnina_box_size'][1]),
                #'--size_z', str(conf['gnina_box_size'][2]),
                '--autobox_ligand', str(conf['gnina_autobox_ligand']),
                '--cpu', str(conf['gnina_cpus']),
                '--num_modes', str(conf['gnina_num_modes']),
                '--out', '/' + output_ligand_fname]
            if conf['debug']:
                print(cmd)
            Client.load(conf['gnina_bin_path'])
            message = Client.execute(
                cmd,
                options=['--nv', '--no-home'],
                bind=['data:/scr', 'result:/result']
            )
            print(message)
            mols = [m for m in Chem.SDMolSupplier(output_ligand_fname) if m is not None]
            top_CNNpose_mol = mols[0]
            structure_rmsd = check_structure_rmsd(top_CNNpose_mol)
            smina_affinity = float(top_CNNpose_mol.GetProp('minimizedAffinity'))
            cnn_score = float(top_CNNpose_mol.GetProp('CNNscore'))
            cnn_affinity = float(top_CNNpose_mol.GetProp('CNNaffinity'))
            if conf['debug']:
                print(f"smina_affinity: {smina_affinity}")
                print(f"cnn_score: {cnn_score}")
                print(f"cnn_affinity: {cnn_affinity}")
            if not conf['debug']:
                shutil.rmtree(temp_dir)
            return [smina_affinity, cnn_score, cnn_affinity, structure_rmsd]
        return [GninaScore]
    
    
    def calc_reward_from_objective_values(values, conf):
        smina_affinity, cnn_score, cnn_affinity, v161_rmsd = values[0]
        if v161_rmsd > 2.0 or v161_rmsd is None:
            return -1
        if smina_affinity is None:
            return -1

        smina_afy_diff = smina_affinity - conf['gnina_base_smina_affinity']
        smina_afy_diff_scaled = - smina_afy_diff*0.6 / (1 + abs(smina_afy_diff)*0.1)
        cnn_afy_diff = cnn_affinity - conf['gnina_base_cnn_affinity']
        cnn_afy_diff_scaled = cnn_afy_diff*0.6 / (1 + abs(cnn_afy_diff)*0.1)

        return (smina_afy_diff_scaled + cnn_score + cnn_afy_diff_scaled) / 3


