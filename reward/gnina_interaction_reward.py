import os
import shutil
import tempfile

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from spython.main import Client

from chemtsv2.scaler import min_gauss
from chemtsv2.abc import Reward
from reward.util import get_interaction_distances


class Gnina_interaction_reward(Reward):
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
                # '--center_x', str(conf['gnina_center'][0]),
                # '--center_y', str(conf['gnina_center'][1]),
                # '--center_z', str(conf['gnina_center'][2]),
                # '--size_x', str(conf['gnina_box_size'][0]),
                # '--size_y', str(conf['gnina_box_size'][1]),
                # '--size_z', str(conf['gnina_box_size'][2]),
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
            top_CNNpose_idx = 0
            top_CNNpose_mol = mols[top_CNNpose_idx]
            smina_affinity = float(top_CNNpose_mol.GetProp('minimizedAffinity'))
            cnn_score = float(top_CNNpose_mol.GetProp('CNNscore'))
            cnn_affinity = float(top_CNNpose_mol.GetProp('CNNaffinity'))
            min_distance_dict = get_interaction_distances(conf['gnina_receptor'], output_ligand_fname, top_CNNpose_idx, conf)
            if conf['debug']:
                print(f"smina_affinity: {smina_affinity}")
                print(f"cnn_score: {cnn_score}")
                print(f"cnn_affinity: {cnn_affinity}")
                print(f"min_distance_dict: {min_distance_dict}")
            if not conf['debug']:
                shutil.rmtree(temp_dir)
            return [smina_affinity, cnn_score, cnn_affinity, min_distance_dict]
        return [GninaScore]


    def calc_reward_from_objective_values(values, conf):
        smina_affinity, cnn_score, cnn_affinity, min_distance_dict = values[0]
        if smina_affinity is None:
            return -1
        if min_distance_dict is None:
            return -1

        smina_afy_diff = smina_affinity - conf['gnina_base_smina_affinity']
        smina_afy_diff_scaled = - smina_afy_diff*0.6 / (1 + abs(smina_afy_diff)*0.1)
        cnn_afy_diff = cnn_affinity - conf['gnina_base_cnn_affinity']
        cnn_afy_diff_scaled = cnn_afy_diff*0.6 / (1 + abs(cnn_afy_diff)*0.1)

        scaled_distance_list = []
        for c in conf['prolif_interactions']:
            residue = c['residue']
            if min_distance_dict[residue]['interaction_type'] is None:
                scaled_distance_list.append(0)
                continue
            distance = min_distance_dict[residue]['distance']
            detected_interaction = min_distance_dict[residue]['interaction_type']
            detected_interaction_idx = c['interaction_type'].index(detected_interaction)
            mu = c['cutoff'][detected_interaction_idx]
            tolerance = conf['prolif_tolerance']
            sigma = np.sqrt((-(tolerance-mu)**2) / (2*np.log(1e-3)))
            scaled_distance = min_gauss(distance, a=1, mu=mu, sigma=sigma)
            scaled_distance_list.append(scaled_distance)

        scaled_objective_values = []
        scaled_objective_values.append(smina_afy_diff_scaled)
        scaled_objective_values.append(cnn_score)
        scaled_objective_values.append(cnn_afy_diff_scaled)
        scaled_objective_values.extend(scaled_distance_list)

        return np.mean(scaled_objective_values)