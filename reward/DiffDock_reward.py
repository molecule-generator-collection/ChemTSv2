import os
import subprocess
import shutil
import tempfile

#from vina import Vina
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
#from rdkit import Chem
#from rdkit.Chem import AllChem, rdMolTransforms
#from rdkit.Geometry import Point3D
#
#from reward.reward import Reward

from pathlib import Path

class DiffDock_Vina_reward(Reward):
    def get_objective_functions(conf):
        def DiffDockScore(mol):

            workdir = conf['output_dir']
            diffdock_pose_dir = os.path.join(conf['output_dir'], "diffdock_pose")
            temp_ligand_fname = os.path.join(diffdock_pose_dir, f"mol_{conf['gid']}_temp.sdf")
            os.makedirs(diffdock_pose_dir, exist_ok=True)
            output_ligand_fname = os.path.join(diffdock_pose_dir, f"mol_{conf['gid']}_out.sdf")

            #temp_dir = tempfile.mkdtemp()
            #os.mkdir(temp_dir+'/results')
            #temp_ligand_fname = os.path.join(temp_dir, 'ligand_temp.sdf')
            #pose_dir = os.path.join(conf['output_dir'], "3D_pose")
            #os.makedirs(pose_dir, exist_ok=True)
            #output_ligand_fname = os.path.join(pose_dir, f"mol_{conf['gid']}_difffdock_out.sdf")

            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)

            print(type(mol))

            try:
                mol_conf = mol.GetConformer(-1)
            except ValueError as e:
                print(f"Error SMILES: {Chem.MolToSmiles(mol)}")
                print(e) 
                if not conf['debug']:
                    shutil.rmtree(temp_dir)
                return None
            with Chem.SDWriter(temp_ligand_fname) as w:
                for m in [mol]:
                    w.write(m)

            print(temp_ligand_fname)

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
                    shutil.rmtree(workdir)
                    #shutil.rmtree(temp_dir)
                return None
            try:
                diffdock_best_sdf = glob.glob(diffdock_pose_dir+'/'+str(conf['diffdock_complex_name'])+'-'+str(conf['gid'])+'/rank1_confidence*.sdf')[0]
            except RuntimeError as e:
                print(f"Error SMILES: {Chem.MolToSmiles(mol)}")
                print(e) 
                if not conf['debug']:
                    shutil.rmtree(workdir)
                    #shutil.rmtree(temp_dir)
                return None
            #shutil.copyfile(best_sdf, output_ligand_fname)

            diffdock_confidence_score = diffdock_best_sdf.replace(diffdock_pose_dir+'/'+str(conf['diffdock_complex_name'])+'-'+str(conf['gid'])+'/rank1_confidence', '').replace('.sdf', '')

            if conf['debug']:
                print(f"diffdock_confidence_score: {diffdock_confidence_score}")
            if not conf['debug']:
                shutil.rmtree(workdir)
                #shutil.rmtree(temp_dir)

            print("### Autodock Vina ###") 
            verbosity = 1 if conf['debug'] else 0
            v = Vina(sf_name=conf['vina_sf_name'], cpu=conf['vina_cpus'], verbosity=verbosity)
            v.set_receptor(rigid_pdbqt_filename=conf['vina_receptor'])
    
            suppl = Chem.SDMolSupplier(diffdock_best_sdf)
            diffdock_outmol = [m for m in suppl if m is not None][0]
            diffdock_outmol = Chem.AddHs(diffdock_outmol, addCoords=True)

            try:
                #AllChem.EmbedMolecule(mol)
                mol_conf = diffdock_outmol.GetConformer(-1)
                centroid = list(rdMolTransforms.ComputeCentroid(mol_conf))
                tr = [conf['vina_center'][i] - centroid[i] for i in range(3)]
                #if conf['debug']:
                #    for a in diffdock_outmol.GetAtoms():
                #        print(a.GetIdx(), a.GetSymbol(), mol_conf.GetPositions()[a.GetIdx()])
                if conf['debug']:
                    print("diffdock_output_positions:")
                for i, p in enumerate(mol_conf.GetPositions()):
                    if conf['debug']:
                        print(i, diffdock_outmol.GetAtomWithIdx(i).GetSymbol(), p)
                    mol_conf.SetAtomPosition(i, Point3D(p[0]+tr[0], p[1]+tr[1], p[2]+tr[2]))
                mol_conf = diffdock_outmol.GetConformer(-1)
                if conf['debug']:
                    print("vina_centor:")
                    print(tr)
                    print("Centroided:")
                    for i, p in enumerate(mol_conf.GetPositions()):
                        print(i, diffdock_outmol.GetAtomWithIdx(i).GetSymbol(), p)

                mol_prep = MoleculePreparation()
                mol_prep.prepare(diffdock_outmol)
                mol_pdbqt = mol_prep.write_pdbqt_string()
                if conf['debug']:
                    print(mol_pdbqt)
                v.set_ligand_from_string(mol_pdbqt)
                #v.set_ligand_from_file(mol_pdbqt)
    
                v.compute_vina_maps(
                    center=conf['vina_center'],
                    box_size=conf['vina_box_size'],
                    spacing=conf['vina_spacing'])
    
                #_ = v.optimize()
    
                if conf['debug']:
                    pprint.pprint(v.info())
    
                v.dock(
                    exhaustiveness=conf['vina_exhaustiveness'],
                    n_poses=conf['vina_n_poses'],
                    min_rmsd=conf['vina_min_rmsd'],
                    max_evals=conf['vina_max_evals'])
                if conf['debug']:
                    print(f"Vina Docking energies: {v.energies()}")
                # get the best inter score, because v.energies()[0][1] is not the best inter_score in some case.
                scores=v.energies()

                min_inter_score = 1000
                best_model = 1
                for m, ene in enumerate(scores):
                    if ene[1] < min_inter_score:
                        min_inter_score = ene[1]
                        best_model = m + 1
                # save best pose
                vina_pose_dir = os.path.join(conf['output_dir'], "vina_pose")
                if not os.path.exists(vina_pose_dir):
                    os.mkdir(vina_pose_dir)
                pose_file_name = f"{vina_pose_dir}/mol_{conf['gid']}_3D_pose_{best_model}.pdbqt"
                v.write_poses(f"{vina_pose_dir}/vina_temp_out.pdbqt", n_poses=conf['vina_n_poses'], overwrite=True)
                pdbqt_mol = PDBQTMolecule.from_file(f"{vina_pose_dir}/vina_temp_out.pdbqt", skip_typing=True)
                for pose in pdbqt_mol:
                    if pose.pose_id == best_model - 1:
                        if conf['debug']:
                            print("Vina Best Pose:")
                            print(pose.write_pdbqt_string())
                        pose.write_pdbqt_file(pose_file_name)
                        file_path = Path(pose_file_name)
                        text = file_path.read_text()
                        text = 'REMARK DIFFDOCK CONFIDENCE: '+diffdock_confidence_score+'\n'+text
                        file_path.write_text(text)


                if conf['debug']:
                    print(f"min_inter_score: {min_inter_score}, best pose num is {best_model}")
                return diffdock_confidence_score, min_inter_score
            except Exception as e:
                print(f"Vina Error SMILES: {Chem.MolToSmiles(mol)}")
                print(e)
                return None
        return [DiffDockScore]
    
    
    def calc_reward_from_objective_values(values, conf):
        if conf['debug']:
            print("values")
            print(type(values))
            print(values)
        min_inter_score = values[0][1]
        if min_inter_score is None:
            return -1
        score_diff = min_inter_score - conf['vina_base_score']
        return - score_diff * 0.1 / (1 + abs(score_diff) * 0.1)

