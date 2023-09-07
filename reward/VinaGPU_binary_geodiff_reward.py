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

import sys

from chemtsv2.misc.embed_3d import Embed3D_Geodiff

class VinaGPU_reward(Reward):
    def get_objective_functions(conf):
        def VinaScore(mol):
            
            #### GeoDiff ####

            if conf['debug']:
                print("### Geodiff input structure")
                print(Chem.MolToMolBlock(mol))

            mol = Embed3D_Geodiff(mol=mol,ckpt_path=conf['geodiff_ckpt_path'], tag=conf['geodiff_tag'],
                                  device='cuda',
                                  clip=conf['geodiff_clip'],
                                  n_steps=conf['geodiff_n_steps'],
                                  global_start_sigma=conf['geodiff_global_start_sigma'],
                                  w_global=conf['geodiff_w_global'],
                                  sampling_type=conf['geodiff_sampling_type'],
                                  eta=conf['geodiff_eta'],
                                  smi=None,
                                  infile=None,
                                  edge_order=conf['geodiff_edge_order'],
                                  save_data=conf['geodiff_save_data'],
                                  log_dir=conf['geodiff_log_dir'],
                                  seed=conf['geodiff_seed'],
                                  gid=conf['gid'],
                                  debug=conf['debug'])

            if conf['debug']:
                print("### Geodiff output structure")
                print(Chem.MolToMolBlock(mol))

            ########

            mol = Chem.AddHs(mol, addCoords=True)
            #AllChem.EmbedMolecule(mol)


            verbosity = 1 if conf['debug'] else 0
            temp_dir = tempfile.mkdtemp()
            temp_ligand_fname = os.path.join(temp_dir, 'ligand_temp.pdbqt')
            pose_dir = os.path.join(conf['output_dir'], "3D_pose")
            os.makedirs(pose_dir, exist_ok=True)
            output_ligand_fname = os.path.join(pose_dir, f"mol_{conf['gid']}_out.pdbqt")


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
                "time", '-p',
                conf['vina_bin_path'],
                '--seed', '11111',
                '--receptor', str(conf['vina_receptor']),
                #'--flex', str(conf['vina_flex']),
                '--ligand', temp_ligand_fname,
                '--thread', str(conf['vina_thread']),
                #'--search_depth', str(conf['vina_search_depth']),

                '--center_x', str(conf['vina_center'][0]),
                '--center_y', str(conf['vina_center'][1]),
                '--center_z', str(conf['vina_center'][2]),
                '--size_x', str(conf['vina_box_size'][0]),
                '--size_y', str(conf['vina_box_size'][1]),
                '--size_z', str(conf['vina_box_size'][2]),

                '--out', output_ligand_fname,

                '--num_modes', str(conf['vina_num_modes']),
                '--energy_range', str(conf['vina_energy_range'])

                ]

            if conf['debug']:
                print(cmd)

            vina_env = os.environ.copy()
            if conf['boost_lib_path'] is not None:
                vina_env['LD_LIBRARY_PATH'] = str(conf['boost_lib_path'])+':'+vina_env['LD_LIBRARY_PATH']
            #vina_env['CUDA_VISIBLE_DEVICES'] = str(conf['vina_gpus'])

            try:
                subprocess.run(cmd, check=True, env=vina_env)

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
