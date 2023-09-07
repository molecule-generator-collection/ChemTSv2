import pprint
import os
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Geometry import Point3D

from reward.reward import Reward

# Tankbind env
import sys
#from Bio.PDB.PDBList import PDBList
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
import torch
torch.set_num_threads(1)

import tempfile

import logging
from torch_geometric.loader import DataLoader
from tqdm import tqdm    # pip install tqdm if fails.

import shutil

from vina import Vina
from meeko import MoleculePreparation
from meeko import PDBQTMolecule

from pathlib import Path

class Tankbind_reward(Reward):
    def get_objective_functions(conf):
        def TankbindScore(mol):
            ### add to PATH
            sys.path.insert(0, conf['tankbind_pythonpath'])
           
            from feature_utils import split_protein_and_ligand
            from feature_utils import get_protein_feature

            from feature_utils import extract_torchdrug_feature_from_mol
            from data import TankBind_prediction
            from model import get_model

            print("### TankBind ###") 

            workdir = conf['output_dir']
            pocket_dir = os.path.join(conf['output_dir'], "tankbind/p2rank_pocket")
            pose_dir = os.path.join(conf['output_dir'], "tankbind/input_pose")
            temp_ligand_fname = os.path.join(pose_dir, f"mol_{conf['gid']}_temp.sdf")
            os.makedirs(pose_dir, exist_ok=True)

            ### get protein.pdb, (ligand.sdf,) ligand_from_rdkit.sdf ready
            proteinFile = conf['tankbind_protein_path']

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
            with Chem.SDWriter(temp_ligand_fname) as w:
                for m in [mol]:
                    w.write(m)
            ligandFile = temp_ligand_fname 

            if conf['debug']:
                print('proteinFile:', proteinFile)
                print('ligandFile:', ligandFile)

            ### get protein feature
            parser = PDBParser(QUIET=True)
            s = parser.get_structure("x", proteinFile)
            res_list = list(s.get_residues())

            protein_dict = {}
            pdb = conf["tankbind_complex_name"]
            protein_dict[pdb] = get_protein_feature(res_list)

            ### get compound feature
            rdkitMolFile = ligandFile
            mol = Chem.MolFromMolFile(rdkitMolFile)
            compound_info = extract_torchdrug_feature_from_mol(mol, has_LAS_mask=True)

            #if conf['debug']:
            #        print('protein_info:', protein_dict)
            #        print('ligand_info:', compound_info)

            ### p2rank
            #print(protein_dict[conf["tankbind_complex_name"]][0])
            try:
                if os.path.isdir(pocket_dir):
                    print(pocket_dir, 'is exist. Skip p2rank.')
                else:
                    p2rank = str(conf['p2rank_path'])
                    cmd = f"bash {p2rank} predict -f {proteinFile} -o {pocket_dir} -threads 1"
                    print(cmd)
                    os.system(cmd)
    
                info = []
               
                ### use external center.
                com = ",".join([str(round(a,3)) for a in conf['vina_center']])
                info.append([conf['tankbind_complex_name'], 'ligand', "ext_center", com])


                ### use protein center as the block center.
                com = ",".join([str(a.round(3)) for a in protein_dict[conf["tankbind_complex_name"]][0].mean(axis=0).numpy()])
                proteinFileName = proteinFile.split('/')[-1]
                libandFilenName = ligandFile.split('/')[-1]
                info.append([conf['tankbind_complex_name'], 'ligand', "protein_center", com])
       
                p2rankFile = pocket_dir+'/'+proteinFileName+'_predictions.csv'
                pocket = pd.read_csv(p2rankFile)
                pocket.columns = pocket.columns.str.strip()
                pocket_coms = pocket[['center_x', 'center_y', 'center_z']].values
                for ith_pocket, com in enumerate(pocket_coms):
                    com = ",".join([str(a.round(3)) for a in com])
                    #info.append([proteinFileName, Chem.MolToSmiles(mol), f"pocket_{ith_pocket+1}", com])
                    info.append([conf['tankbind_complex_name'], 'ligand', f"pocket_{ith_pocket+1}", com])
                info = pd.DataFrame(info, columns=['protein_name', 'compound_name', 'pocket_name', 'pocket_com'])
                #if conf['debug']:
                #    print(info)

            except Exception as e:
                print(f"Error SMILES: {Chem.MolToSmiles(mol)}")
                print(e)
                return None, None
            
            ### construct dataset
            dataset_path = os.path.join(conf['output_dir'], "tankbind/dataset")
            #dataset_path = os.path.join(conf['output_dir'], "dataset")
            #shutil.rmtree(dataset_path)
            if os.path.isdir(dataset_path):
                os.system(f"rm -r {dataset_path}")
                os.system(f"mkdir -p {dataset_path}")
            dataset = TankBind_prediction(dataset_path, data=info, protein_dict=protein_dict, compound_dict=dict(ligand=compound_info))
            #print(dataset)   
            batch_size = 5
            device = 'cuda' if conf['tankbind_gpu'] and torch.cuda.is_available() else 'cpu'
            if conf['debug']:
                print("device:", device)
            logging.basicConfig(level=logging.INFO)
            model = get_model(0, logging, device)

            modelFile = conf['tankbind_modelfile']
            
            model.load_state_dict(torch.load(modelFile, map_location=device))
            _ = model.eval()
            
            data_loader = DataLoader(dataset, batch_size=batch_size, follow_batch=['x', 'y', 'compound_pair'], shuffle=False, num_workers=8)
            affinity_pred_list = []
            y_pred_list = []
            for data in tqdm(data_loader):
                data = data.to(device)
                y_pred, affinity_pred = model(data)
                affinity_pred_list.append(affinity_pred.detach().cpu())
                for i in range(data.y_batch.max() + 1):
                    y_pred_list.append((y_pred[data['y_batch'] == i]).detach().cpu())
            
            affinity_pred_list = torch.cat(affinity_pred_list)
          
          
            info = dataset.data
            info['affinity'] = affinity_pred_list


            print(info)

            info.to_csv(workdir+'/tankbind/info_with_predicted_affinity_'+str(conf["gid"])+'.csv')
            chosen = info.loc[info.groupby(['protein_name', 'compound_name'],sort=False)['affinity'].agg('idxmax')].reset_index()
            info_wo_bc = info.query( 'pocket_name != "protein_center"' )
            chosen_best_pocket = info_wo_bc.loc[info_wo_bc.groupby(['protein_name', 'compound_name'],sort=False)['affinity'].agg('idxmax')].reset_index()
            chosen_pocket_1 = info.query( '(pocket_name == "protein_center") or (pocket_name == "pocket_1")').reset_index()
            chosen_block_center = info.query( 'pocket_name == "protein_center"' ).reset_index()
            chosen_ext_center = info.query( 'pocket_name == "ext_center"' ).reset_index()
            
            chosen = chosen_ext_center
            print("Use The docking result of \"ext_center\" for Vina Docking.")
            print(chosen)
            tankbind_affinity = chosen['affinity'][0]
            print('TankBind Affinity:', tankbind_affinity)

            # from predicted interaction distance map to sdf
            from generation_utils import get_LAS_distance_constraint_mask, get_info_pred_distance, write_with_new_coords
            
            idx = chosen['index'][0]
            pocket_name = chosen['pocket_name']
            compound_name = chosen['compound_name'][0]
            ligandName = compound_name
            device = 'cpu'
            coords = dataset[idx].coords.to(device)
            coords = dataset[idx].coords.to(device)
            protein_nodes_xyz = dataset[idx].node_xyz.to(device)
            n_compound = coords.shape[0]
            n_protein = protein_nodes_xyz.shape[0]
            y_pred = y_pred_list[idx].reshape(n_protein, n_compound).to(device)
            y = dataset[idx].dis_map.reshape(n_protein, n_compound).to(device)
            compound_pair_dis_constraint = torch.cdist(coords, coords)
            rdkitMolFile = ligandFile
            mol = Chem.MolFromMolFile(rdkitMolFile)
            LAS_distance_constraint_mask = get_LAS_distance_constraint_mask(mol).bool()
            info = get_info_pred_distance(coords, y_pred, protein_nodes_xyz, compound_pair_dis_constraint, 
                                          LAS_distance_constraint_mask=LAS_distance_constraint_mask,
                                          n_repeat=1, show_progress=False)

            tankbind_result_folder = f'{workdir}/tankbind/docking_result/'
            os.system(f'mkdir -p {tankbind_result_folder}')
            #toFile = tankbind_result_folder+'/'+str(ligandName)+'_'+str(conf['gid'])+'_tankbind.sdf'
            toFile = tankbind_result_folder+ligandName+'_'+str(conf['gid'])+'_'+chosen['pocket_name'][0]+'.sdf'
            new_coords = info.sort_values("loss")['coords'].iloc[0].astype(np.double)
            write_with_new_coords(mol, new_coords, toFile)
    
            suppl = Chem.SDMolSupplier(toFile)
            print([m for m in suppl if m is not None])
            tankbind_outmol = [m for m in suppl if m is not None][0]
            tankbind_outmol = Chem.AddHs(tankbind_outmol, addCoords=True)

            mol_conf = tankbind_outmol.GetConformer(-1)

            if conf['debug']:
                print("TankBind pose:")
                for a in tankbind_outmol.GetAtoms():
                    print(a.GetIdx(), a.GetSymbol(), mol_conf.GetPositions()[a.GetIdx()])

            print("### Autodock Vina ###") 

            verbosity = 1 if conf['debug'] else 0
            v = Vina(sf_name=conf['vina_sf_name'], cpu=conf['vina_cpus'], verbosity=verbosity)
            v.set_receptor(rigid_pdbqt_filename=conf['vina_receptor'])
    
            ignore_vina_center = False

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
                    print("centroid:", centroid)
                    print("tr:", tr)
                    print("TankBind pose (apply translation for Vina):")
                    for i, p in enumerate(mol_conf.GetPositions()):
                        print(i, tankbind_outmol.GetAtomWithIdx(i).GetSymbol(), p)
                mol_prep = MoleculePreparation()
                mol_prep.prepare(tankbind_outmol)
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
                #text = 'REMARK DIFFDOCK CONFIDENCE: '+diffdock_confidence_score+'\n'+text
                file_path.write_text(text)

                if conf['debug']:
                    # print('min_inter_score: '+str(min_inter_score)+', tankbind_affinity: '+str(np.round(tankbind_affinity,decimals=3))+'. best pose num is '+str(best_model)+'.')
                    print('min_inter_score: '+str(min_inter_score)+', tankbind_affinity: '+str(np.round(tankbind_affinity,decimals=3))+'.')
                return min_inter_score, tankbind_affinity

            except Exception as e:
                print(f"Vina Error SMILES: {Chem.MolToSmiles(mol)}")
                print(e)
                return None, None

        return [TankbindScore]
    
    
    def calc_reward_from_objective_values(values, conf):
        min_inter_score = values[0][1]
        if min_inter_score is None:
            return -1
        score_diff = min_inter_score - conf['vina_base_score']
        return - score_diff * 0.1 / (1 + abs(score_diff) * 0.1)
