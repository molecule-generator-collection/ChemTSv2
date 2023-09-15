import os
import subprocess
import shutil
import tempfile

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Geometry import Point3D

from reward.reward import Reward

import glob
import re

import pprint
from vina import Vina
from meeko import MoleculePreparation
from meeko import PDBQTMolecule

from pathlib import Path

class DiffDock_Vina_reward(Reward):
    def get_objective_functions(conf):
        def DiffDockScore(mol):

            print("### DiffDock ###") 

            workdir = conf['output_dir']
            diffdock_pose_dir = os.path.join(conf['output_dir'], "diffdock_pose")
            temp_ligand_fname = os.path.join(diffdock_pose_dir, f"mol_{conf['gid']}_temp.sdf")
            os.makedirs(diffdock_pose_dir, exist_ok=True)
            output_ligand_fname = os.path.join(diffdock_pose_dir, f"mol_{conf['gid']}_out.sdf")

            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)

            try:
                mol_conf = mol.GetConformer(-1)
            except ValueError as e:
                print(f"Error SMILES: {Chem.MolToSmiles(mol)}")
                print(e) 
                return None,None
            with Chem.SDWriter(temp_ligand_fname) as w:
                for m in [mol]:
                    w.write(m)

            print("DiffDock input ligand:", temp_ligand_fname)

            suppl = Chem.SDMolSupplier(temp_ligand_fname)
            print(suppl)

            cmd = [
                'eval "$('+str(conf['conda_cmd'])+' shell.bash hook)"',
                '&&',
                'conda activate '+str(conf['diffdock_conda_env']),
                '&&',
                'python -m inference',
                '--complex_name', str(conf['diffdock_complex_name'])+'-'+str(conf['gid']),
                '--protein_path', str(conf['diffdock_protein_path']),
                '--ligand_description', temp_ligand_fname,
                '--out_dir',  diffdock_pose_dir,
                '--protein_sequence', str(conf['diffdock_protein_sequence']),
                '--samples_per_complex', str(conf['diffdock_samples_per_complex']),
                '--model_dir', str(conf['diffdock_model_dir']),
                '--ckpt', str(conf['diffdock_ckpt']),
                '--confidence_model_dir', str(conf['diffdock_confidence_model_dir']),
                '--confidence_ckpt', str(conf['diffdock_confidence_ckpt']),
                '--batch_size', str(conf['diffdock_batch_size']),
                '--no_final_step_noise' if conf['diffdock_no_final_step_noise'] else '',
                '--inference_steps', str(conf['diffdock_inference_steps']),
                '--actual_steps', str(conf['diffdock_actual_steps'])

                ]

            if conf['debug']:
                print(' '.join(cmd))
            diffdock_env = os.environ.copy()
            diffdock_env['PYTHONPATH'] = str(conf['diffdock_pythonpath'])

            try:
                subprocess.run(' '.join(cmd), shell=True, check=True, env=diffdock_env)
            except subprocess.CalledProcessError as e:
                print(f"Error SMILES: {Chem.MolToSmiles(mol)}")
                print(e) 
                if not conf['debug']:
                    shutil.rmtree(os.path.join(conf['output_dir'], conf['diffdock_complex_name'])+'-'+str(conf['gid']))
                return None, None
            try:
                diffdock_best_sdf = glob.glob(diffdock_pose_dir+'/'+str(conf['diffdock_complex_name'])+'-'+str(conf['gid'])+'/rank1_confidence*.sdf')[0]
                diffdock_confidence_score = diffdock_best_sdf.replace(diffdock_pose_dir+'/'+str(conf['diffdock_complex_name'])+'-'+str(conf['gid'])+'/rank1_confidence', '').replace('.sdf', '')
            except RuntimeError as e:
                print(f"Error SMILES: {Chem.MolToSmiles(mol)}")
                print(e) 
                if not conf['debug']:
                    shutil.rmtree(os.path.join(conf['output_dir'], conf['diffdock_complex_name'])+'-'+str(conf['gid']))
                return None, None

            if not conf['debug']:
                os.remove(temp_ligand_fname)

            if conf['debug']:
                print(f"diffdock_confidence_score: {diffdock_confidence_score}")

            suppl = Chem.SDMolSupplier(diffdock_best_sdf)
            diffdock_outmol = [m for m in suppl if m is not None][0]
            diffdock_outmol = Chem.AddHs(diffdock_outmol, addCoords=True)

            mol_conf = diffdock_outmol.GetConformer(-1)

            if conf['debug']:
                print("DiffDock pose:")
                for a in diffdock_outmol.GetAtoms():
                    print(a.GetIdx(), a.GetSymbol(), mol_conf.GetPositions()[a.GetIdx()])

            print("### Autodock Vina ###") 

            verbosity = 1 if conf['debug'] else 0
            v = Vina(sf_name=conf['vina_sf_name'], cpu=conf['vina_cpus'], verbosity=verbosity)
            v.set_receptor(rigid_pdbqt_filename=conf['vina_receptor'])

            ignore_vina_center = True
    
            try:
                # AllChem.EmbedMolecule(mol)
                centroid = list(rdMolTransforms.ComputeCentroid(mol_conf))
                if not ignore_vina_center:
                    tr = [conf['vina_center'][i] - centroid[i] for i in range(3)]
                else:
                    tr = [.0, .0, .0]
                for i, p in enumerate(mol_conf.GetPositions()):
                    mol_conf.SetAtomPosition(i, Point3D(p[0]+tr[0], p[1]+tr[1], p[2]+tr[2]))
                if conf['debug']:
                    print('centroid:', centroid)
                    print("tr:", tr)
                    print("DiffDock pose (apply translation for Vina):")
                    for i, p in enumerate(mol_conf.GetPositions()):
                        print(i, diffdock_outmol.GetAtomWithIdx(i).GetSymbol(), p)

                mol_prep = MoleculePreparation()
                mol_prep.prepare(diffdock_outmol)
                mol_pdbqt = mol_prep.write_pdbqt_string()
                v.set_ligand_from_string(mol_pdbqt)
    
                if not ignore_vina_center:
                    v.compute_vina_maps(
                        center=conf['vina_center'],
                        box_size=conf['vina_box_size'],
                        spacing=conf['vina_spacing'])

                else:
                    v.compute_vina_maps(
                        center=centroid,
                        box_size=conf['vina_box_size'],
                        spacing=conf['vina_spacing'])
    
                if conf['debug']:
                    pprint.pprint(v.info())
    
                energy = v.score()
                if conf['debug']:
                    print(f"Vina Docking energies: {energy}")

                min_inter_score = 1000
                if energy[1] < min_inter_score:
                    min_inter_score = energy[1]

                # save pose
                vina_pose_dir = os.path.join(conf['output_dir'], "vina_pose")
                if not os.path.exists(vina_pose_dir):
                    os.mkdir(vina_pose_dir)
                pose_file_name = f"{vina_pose_dir}/mol_{conf['gid']}_3D_pose.pdbqt"
                file_path = Path(pose_file_name)
                text = mol_pdbqt
                text = 'REMARK DIFFDOCK CONFIDENCE: '+diffdock_confidence_score+'\n'+text
                print(text)
                file_path.write_text(text)

                if conf['debug']:
                    print(f"diffdock_confidence_score: {diffdock_confidence_score}, min_inter_score: {min_inter_score}.")
                return diffdock_confidence_score, min_inter_score
            except Exception as e:
                print(f"Vina Error SMILES: {Chem.MolToSmiles(mol)}")
                print(e)
                return None, None
        return [DiffDockScore]
    
    
    def calc_reward_from_objective_values(values, conf):
        min_inter_score = values[0][1]
        if min_inter_score is None:
            return -1
        score_diff = min_inter_score - conf['vina_base_score']
        return - score_diff * 0.1 / (1 + abs(score_diff) * 0.1)

