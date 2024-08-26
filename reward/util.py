import os
import sys
import random
import re
import string
import subprocess
import tempfile

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize, normalize
import pandas as pd
import numpy as np


def read_reactions(reaction_file):
    reaction_df = pd.read_csv(reaction_file,header=None)
    reaction_list = [smarts for smarts in reaction_df[0]]
    return reaction_list


def read_substruct_mol(path, ss_list_file):
    file_list_df = pd.read_csv(ss_list_file, header = None)
    mol_list=[]
    for file_name in file_list_df[0]:
        mol_list.append(Chem.MolFromMolFile(str(os.path.join(path, file_name))))
    return mol_list


def sort_sdf_confs(sdf_inp, sdf_out):
    suppl = AllChem.SDMolSupplier(sdf_inp, removeHs=False)
    energy_id_list = []
    for i in range(0, len(suppl) - 1):
        energy_id = [suppl[i].GetProp("Energy"), i]
        energy_id_list.append(energy_id)
    energy_id_list.sort(key = lambda x: float(x[0]))
    writer = Chem.SDWriter(sdf_out)
    for energy, mol_id in energy_id_list:
        writer.write(suppl[mol_id])
    writer.close()
    return 1


def get_appropriate_ligand3d(mol):
    smiles = Chem.MolToSmiles(mol)
    smiles_neutralized = neutralize_atoms(smiles)
    smiles_tautomer = calc_canon_tautomer(smiles_neutralized)
    smiles_modtautomer = mod_tautomer_smiles(smiles_tautomer, smiles_neutralized)
    smiles_protomer = calc_protomer(smiles_modtautomer)
    mol_3d = calc_3dstructure(smiles_protomer)
    return mol_3d


def calc_canon_tautomer(smiles):
    mol = Chem.MolFromSmiles(smiles) 
    enumerator = rdMolStandardize.TautomerEnumerator()
    try:  
        tmp_smiles = Chem.MolToSmiles(enumerator.Canonicalize(mol))
    except:
        tmp_smiles = smiles

    return tmp_smiles


def neutralize_atoms(smiles):
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    norm = normalize.Normalizer()
    smiles_tmp = smiles
    mol = Chem.MolFromSmiles(smiles_tmp)
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
        try:
            mol_norm = norm.normalize(mol)
        except:
            mol_norm = Chem.MolFromSmiles(smiles)
        smiles_tmp = Chem.MolToSmiles(mol_norm)
    else:
        smiles_tmp = Chem.MolToSmiles(mol)
    
    return smiles_tmp


def mod_tautomer_smiles(tautomer_smiles, canon_smiles):
    tmp_smiles = tautomer_smiles
    taut_mol = Chem.MolFromSmiles(tmp_smiles)
    data_dir = os.path.join(os.getcwd(), 'data')
    reaction_file = os.path.join(data_dir,'reactions_mod_tautomer.txt') 
    ss_list_file = os.path.join(data_dir,'ss_list_mod_tautomer.txt')
    ss_path = os.path.join(data_dir,'ss_mod_tautomer')
    
    reaction_list = read_reactions(reaction_file)
    ss_list = read_substruct_mol(ss_path, ss_list_file)
    
    for reaction, ss in zip(reaction_list,ss_list):
        if tmp_smiles != tautomer_smiles:
            taut_mol = Chem.MolFromSmiles(tmp_smiles)
        try: 
            if taut_mol.HasSubstructMatch(ss):
                rxn = AllChem.ReactionFromSmarts(reaction)
                tmp_smiles = canon_smiles
                prev_smiles = tmp_smiles
                match_count = 0
                while True:
                    canon_mol = Chem.MolFromSmiles(tmp_smiles)
                    ps = rxn.RunReactants((canon_mol,))
                    if len(ps) < 1:
                        if match_count == 2:
                            tmp_smiles = canon_smiles
                        break
                    tmp_mol = ps[0][0]
                    tmp_smiles2 = Chem.MolToSmiles(tmp_mol)
                    tmp_mol2 = Chem.MolFromSmiles(tmp_smiles2)
                    tmp_smiles = Chem.MolToSmiles(tmp_mol2)
                    if not(tmp_mol2.HasSubstructMatch(ss)):
                        break
                    else:
                        if tmp_smiles == prev_smiles:
                            match_count = match_count + 1
                        else:
                            match_count = 0
                            prev_smiles = tmp_smiles
                    if match_count > 2:
                        break

        except:
            print(tmp_smiles)
    return tmp_smiles


def mod_protomer(smiles, reaction_list, ss_list):
    tmp_smiles = smiles
    taut_mol = Chem.MolFromSmiles(tmp_smiles)
    for reaction, ss in zip(reaction_list,ss_list):
        if tmp_smiles != smiles:
            taut_mol = Chem.MolFromSmiles(tmp_smiles)
        try: 
            if taut_mol.HasSubstructMatch(ss):

                rxn = AllChem.ReactionFromSmarts(reaction)
                prev_smiles = tmp_smiles
                prev_smiles2 = tmp_smiles
                match_count = 0
                while True:
                    canon_mol = Chem.MolFromSmiles(tmp_smiles)
                    ps = rxn.RunReactants((canon_mol,))
                    if len(ps) < 1:
                        if match_count == 2:
                            tmp_smiles = prev_smiles
                        break
                    tmp_mol = ps[0][0]
                    tmp_smiles2 = Chem.MolToSmiles(tmp_mol)
                    tmp_mol2 = Chem.MolFromSmiles(tmp_smiles2)
                    tmp_smiles = Chem.MolToSmiles(tmp_mol2)
                    if not(tmp_mol2.HasSubstructMatch(ss)):
                        break
                    else:
                        if tmp_smiles == prev_smiles2:
                            match_count = match_count + 1
                        else:
                            match_count = 0
                            prev_smiles2 = tmp_smiles
                    if match_count > 2:
                        break
        except:
            print(tmp_smiles)

    return tmp_smiles


def mod_charge(smiles, num_plus):
    tmp_smiles = smiles
    rm_priority_list = ["[NH+]","[NH2+]", "[NH3+]"]
    while num_plus > 1:
        num_plus_old = num_plus
        non_plus_parts = re.split("\[N[^\+]*\+\]", tmp_smiles)
        plus_parts = re.findall("\[N[^\+]*\+\]", tmp_smiles)
        for rm_pattern in rm_priority_list:            
            for i, plus_part in enumerate(plus_parts):                
                if rm_pattern == plus_part:
                    plus_parts[i] = "N"
                    num_plus = num_plus - 1
                    break
            else:
                continue
            break        
        if num_plus == num_plus_old:
            break
        plus_parts.append("")
        tmp_smiles = ""
        for non_plus_part, plus_part in zip(non_plus_parts, plus_parts):
            tmp_smiles = tmp_smiles + non_plus_part + plus_part 
    return tmp_smiles


def calc_protomer(smiles):
    smi_part_list = ["[c,C][-,\+]]", "CH[-,\+]]", "[n,N]-]", "NH-]", "NH2-]", "[o,O]\+]", "OH2[-,\+]]", "OH[-,\+]]"]

    data_dir = os.path.join(os.getcwd(), 'data')
    reaction_file = os.path.join(data_dir,'reactions_protomer.txt') 
    ss_list_file = os.path.join(data_dir,'ss_list_protomer.txt')
    ss_path = os.path.join(data_dir,'ss_protomer')
    
    reaction_list = read_reactions(reaction_file)
    ss_list = read_substruct_mol(ss_path, ss_list_file)

    command = ["obabel", "-:"+smiles, "-ocan", "-p", "7.4" ]
    pybel_smiles = subprocess.run(command, capture_output=True, text=True).stdout
    mod_flag = False
    rdk_mol = Chem.MolFromSmiles(pybel_smiles)
    if rdk_mol:
        tmp_smiles = Chem.MolToSmiles(rdk_mol)
        for smi_part in smi_part_list:
            if re.search(smi_part, pybel_smiles):
                tmp_smiles = smiles
                mod_flag = True
    else:
        tmp_smiles = smiles
        mod_flag = True
    if mod_flag:
        tmp_smiles = mod_protomer(tmp_smiles,reaction_list, ss_list)

    return tmp_smiles


def read_sdfs_make_confs_sdf(sdf_inp_list, sdf_out, num_conformers=40, num_opt_iters=1000):
    result_id = []
    mol_list = []
    for i, sdf_inp in enumerate(sdf_inp_list):
        suppl = AllChem.SDMolSupplier(sdf_inp)
        mol_H = Chem.AddHs(suppl[0])
        cids = AllChem.EmbedMultipleConfs(mol_H, numConfs=num_conformers)
        if len(cids) == 0:

            forceTol=0.01
            maxAttempts=1000
            while(len(cids) ==0):
                cids = AllChem.EmbedMultipleConfs(mol_H, numConfs=num_conformers, maxAttempts=maxAttempts, forceTol=forceTol, numZeroFail=38)
                forceTol= forceTol * 10
                if forceTol > 100:
                    break

        results = AllChem.MMFFOptimizeMoleculeConfs(mol_H, maxIters=num_opt_iters)
        mol_list.append(mol_H)
        for j, result in enumerate(results):
             result_id.append((result[1], j, i))
    result_id.sort()
    writer = Chem.SDWriter(sdf_out)
    for result in result_id:
        mol_H = mol_list[result[2]]
        mol_H.SetProp("Energy", "")
        mol_H.SetProp("_Name", "conformer_" + str(result[1]) + "_chiral" + str(result[2]))
        mol_H.SetProp("optimized_energy", str(result[0]))
        writer.write(mol_H, confId=result[1])
        break
    
    mol_H.SetProp("Energy", "")
    mol_H.SetProp("_Name", "conformer_" + str(result_id[0][1]))
    mol_H.SetProp("optimized_energy", str(result_id[0][1]))
    writer.close()
    return 1


def calc_stereo_centor(smiles, reaction_a_list, reaction_aa_list, ss_list, mol_type):
    tmp_smiles = smiles
    tmp_mol = Chem.MolFromSmiles(tmp_smiles)
    mod_mol_list = [tmp_mol]
    mod_flag = 1
    include_flag = False
    num_match_sum = 0
    for i, (reaction_a, reaction_aa, ss) in enumerate(zip(reaction_a_list, reaction_aa_list, ss_list)):
        try:
            num_match = len(tmp_mol.GetSubstructMatches(ss))
            num_match_sum = num_match_sum + num_match
            if num_match == 1:
                mod_flag = mod_flag * i
                if mod_flag < 4:
                    mod_mol_tmp_list = []
                    for mod_mol in mod_mol_list:
                        rxn_a = AllChem.ReactionFromSmarts(reaction_a)
                        rxn_aa = AllChem.ReactionFromSmarts(reaction_aa)
                        ps_a = rxn_a.RunReactants((mod_mol,))
                        ps_aa = rxn_aa.RunReactants((mod_mol,))
                        if len(ps_a) > 0:
                            mod_mol_tmp_list.append(ps_a[0][0])
                            include_flag = True
                            molb_a = Chem.MolToMolBlock(ps_a[0][0])
                        if len(ps_aa) > 0:  
                            mod_mol_tmp_list.append(ps_aa[0][0])
                            molb_aa = Chem.MolToMolBlock(ps_aa[0][0])
                            include_flag = True
                        if molb_a == molb_aa:
                            mod_mol_tmp_list = []
                    if mod_mol_tmp_list:
                        mod_mol_list = mod_mol_tmp_list
                else:
                    mod_mol_list = []
                    include_flag = False
            elif num_match > 1:
                mod_mol_list = []
                include_flag = False
        except:
            print(tmp_smiles)
    if not(mod_mol_list):
        mod_mol_list.append(Chem.MolFromSmiles(smiles))
    return_mol_list = []
    for mol in mod_mol_list:
        if mol_type == "smiles":
            return_mol_list.append(Chem.MolToSmiles(mol))
        elif mol_type == "molblock":
            return_mol_list.append(Chem.MolToMolBlock(mol))
        elif mol_type == "direct":
            return_mol_list.append(mol)
        elif mol_type == "skip":
            return_mol_list = [smiles]
    return return_mol_list, include_flag


def calc_3dstructure(smiles):
    mol_id = ''.join([random.choice(string.ascii_letters + string.digits) for i in range(8)])
    ss_cyclohexane = Chem.MolFromSmiles('CC1CCCCC1')
    file_list = []
    threshold_time = 180

    data_dir = os.path.join(os.getcwd(), 'data')
    dest_dir = os.path.join(os.getcwd(), 'tmp')
    reaction_a_file = os.path.join(data_dir,'reaction_list_stereo_a.txt') 
    reaction_aa_file = os.path.join(data_dir,'reaction_list_stereo_aa.txt') 
    ss_list_file = os.path.join(data_dir,'ss_list_3d.txt')
    ss_path = os.path.join(data_dir,'ss_3d')    
    reaction_a_list = read_reactions(reaction_a_file)
    reaction_aa_list = read_reactions(reaction_aa_file)
    ss_list = read_substruct_mol(ss_path, ss_list_file)
    if not(os.path.exists(dest_dir)):
        os.mkdir(dest_dir)

    tmp_mol = Chem.MolFromSmiles(smiles)
    if tmp_mol.HasSubstructMatch(ss_cyclohexane):
        mol_type = "molblock"
    else:
        mol_type = "smiles"
    Timeout_flag = False
    ss_flag = False
    mol_list, ss_flag = calc_stereo_centor(smiles, reaction_a_list, reaction_aa_list, ss_list, mol_type)
    
    if mol_type == "molblock":
        mol_list = list(set(mol_list))
        charge_error = False
        for i, molb in enumerate(mol_list):
            mol = Chem.MolFromMolBlock(molb)
            no3d_file_name = str(mol_id) + '_chiral' + str(i) + '_rdk2d.sdf'
            d3_file_name = str(mol_id) + '_chiral' + str(i) + '.sdf'
            no3d_file = os.path.join(dest_dir, no3d_file_name)
            d3_file = os.path.join(dest_dir, d3_file_name)
            writer = Chem.SDWriter(no3d_file)
            writer.write(mol)
            writer.close()
            file_list.append(no3d_file)
            command1 = ['obabel', '-isdf', no3d_file, '--gen3d', '-p', '7.4', '--ff', 'Ghemical', '-osdf', '-O', d3_file ]
            pybel_out = subprocess.run(command1, capture_output=True, text=True, timeout=threshold_time).stdout
            file_list.append(d3_file)
            try:
                suppl_ob = Chem.SDMolSupplier(d3_file)
                ob_mol = suppl_ob[0]
                ob_smiles = Chem.MolToSmiles(ob_mol)
                num_plus = ob_smiles.count("+")
                if num_plus > 1:
                    charge_error = True
                    mol_list, ss_flag = calc_stereo_centor(smiles, reaction_a_list, reaction_aa_list, ss_list, "smiles")
            except:
                charge_error = True
                mol_list, ss_flag = calc_stereo_centor(smiles, reaction_a_list, reaction_aa_list, ss_list, "smiles")

    if mol_type == "smiles" or mol_type == "skip" or charge_error:
        mol_list = list(set(mol_list))
        mol = Chem.MolFromSmiles(mol_list[0])
        for i, smi in enumerate(mol_list):
            d3_file_name = str(mol_id) + '_chiral' + str(i) + '.sdf'
            d3_file = os.path.join(dest_dir, d3_file_name)
            command1 = ['obabel', '-:'+smi, '--gen3d', '-p', '7.4', '--ff', 'Ghemical', '-osdf', '-O', d3_file ]                        

            try:
                pybel_out = subprocess.run(command1, capture_output=True, text=True, timeout=threshold_time).stdout
                file_list.append(d3_file)
            except subprocess.TimeoutExpired as e:
                no3d_file_name = str(mol_id) + '_chiral' + str(i) + '_no3d.sdf'
                no3d_file = os.path.join(dest_dir, no3d_file_name)
                command0 = ['obabel', '-:'+smi, '-p', '7.4', '-osdf', '-O', no3d_file ]
                command1 = ['obabel', '-isdf', no3d_file, '--gen3d', '-p', '7.4', '--ff', 'Ghemical', '-osdf', '-O', d3_file ]
                try:
                    pybel_out = subprocess.run(command0, capture_output=True, text=True, timeout=threshold_time).stdout
                    file_list.append(no3d_file)
                    pybel_out = subprocess.run(command1, capture_output=True, text=True, timeout=threshold_time).stdout
                    file_list.append(d3_file)
                except subprocess.TimeoutExpired as e:
                    print('Second attempt to gen 3d Failed!\n',e.output)
                    print('Skip smiles:', smi)
                    Timeout_flag = True
                    break
                
    if Timeout_flag:
        mol = Chem.AddHs(mol)
        mol = AllChem.EmbedMolecule(mol)
        return mol

    for i, smi in enumerate(mol_list):
        confs_sdf_inp_list = []
        opt_file_name = str(mol_id) + '_chiral' + str(i) + '_minimized.sdf'
        opt_file = os.path.join(dest_dir, opt_file_name)
        confs_sdf_inp_list.append(opt_file)
        command2 = ['obabel', '-isdf', d3_file, '--minimize', '--ff', 'Ghemical', '--cut',  '--rele', '0.1', '-osdf', '-O', opt_file ]    
        pybel_out = subprocess.run(command2, capture_output=True, text=True).stdout
        file_list.append(opt_file)
        

    confs_sdf_name = str(mol_id) + '_confs_sorted.sdf'
    confs_opt_sdf_name = str(mol_id) + '_confs_minimized_rdsort.sdf'
    confs_sdf = os.path.join(dest_dir, confs_sdf_name)
    confs_opt_sdf = os.path.join(dest_dir, confs_opt_sdf_name)

    read_sdfs_make_confs_sdf(confs_sdf_inp_list, confs_sdf)                
    file_list.append(confs_sdf)
    command2_2 = ['obabel', '-isdf', confs_sdf, '--minimize', '--ff', 'Ghemical', '--cut',  '--rele', '0.1', '-osdf', '-O', confs_opt_sdf ]
    pybel_out = subprocess.run(command2_2, capture_output=True, text=True).stdout
    file_list.append(confs_opt_sdf)
    
    sdf_use = confs_opt_sdf
   
    suppl_ob2 = Chem.SDMolSupplier(sdf_use, removeHs=False) 
    tmp_mol = suppl_ob2[0]
    for f in file_list:
        if os.path.isfile(f):
            os.remove(f)


    return tmp_mol


"""Calculate RMSD of scaffold region between reference structure and docking pose"""
def calc_scaffold_rmsd(dock_mol, conf):
    scaffold_mol = Chem.MolFromSmiles(conf['fixed_structure_smiles'])
    reference_mol = Chem.MolFromMolFile(conf['reference_structure_path']) 

    # Extract coordinates (Reference structure)
    ref_scaf_match_index = reference_mol.GetSubstructMatch(scaffold_mol)
    ref_match_coords = np.array([reference_mol.GetConformer(0).GetAtomPosition(idx) for idx in ref_scaf_match_index])
    
    # Extract coordinates (Docking pose)
    dock_scaf_indexs = dock_mol.GetSubstructMatch(scaffold_mol)
    dock_match_coords = np.array([dock_mol.GetConformer(0).GetAtomPosition(idx) for idx in dock_scaf_indexs])

    squared_diff = np.square(ref_match_coords - dock_match_coords)
    rmsd = np.sqrt(np.sum(squared_diff) / len(squared_diff))
    return rmsd  

def calc_strain_energy(docking_pose_file, conf):
    scname = ["total_strain", "dihedral_torsion_strain"]

    basename, ext = os.path.splitext(os.path.basename(docking_pose_file))
    ext = ext.lower()
    mol2_path = basename + ".mol2"
    cmd = [
        'obabel', '-i'+ext[1:], docking_pose_file, '-omol2', '-O', mol2_path, '-xu'
    ]
    results = subprocess.run(
        cmd, capture_output=True, check=True, text=True
    )
    cw_dir = os.getcwd()
    os.chdir(conf["script_path"])
    cmd = [
        'python', 'Torsion_Strain.py', cw_dir +'/' + mol2_path
    ]
    results = subprocess.run(
        cmd, capture_output=True, check=True, text=True
    )

    os.chdir(cw_dir)

    csv_path = basename + "_Torsion_Strain.csv"
    total_strain_energy = None
    max_single_strain_energy = None

    try:
        if not os.path.exists(csv_path):
            raise Exception(f"CSV file {csv_path} does not exist.")
        df = pd.read_csv(csv_path, header=None)   
        df = df[[1, 5]] #column1: total strain energy, column5: max single strain energy. For more Details, please check the README in the STRAIN_FILTER directory.
        df.index.name = "SID"
        df.columns = scname
        top_pose_strain_energy = df.to_numpy()[0]
        if len(top_pose_strain_energy)  >=2:
            total_strain_energy = top_pose_strain_energy[0]
            max_single_strain_energy = top_pose_strain_energy[1]
        else:
            raise Exception("The CSV file does not contain sufficient data.")
    except Exception as e:
        print(f"Warning: {e}")

    if not conf["savescr"]:
        for file_path in [mol2_path, csv_path]:
            if os.path.exists(file_path): os.remove(file_path)

    return total_strain_energy, max_single_strain_energy


def get_interaction_distances(receptor_fname, output_ligand_fname, pose_idx, conf):

    # Import necessary modules
    plf = sys.modules.get('prolif') or __import__('prolif')
    mda = sys.modules.get('MDAnalysis') or __import__('MDAnalysis')

    # Load receptor
    if receptor_fname.startswith('/scr'):
        receptor_fname = receptor_fname.replace('/scr', 'data')
    u = mda.Universe(receptor_fname)
    protein_mol = plf.Molecule.from_mda(u)
    pose_iterable = plf.sdf_supplier(output_ligand_fname)

    # Load ligand
    output_ligand_ext = output_ligand_fname.split('.')[-1]
    if output_ligand_ext == 'pdb':
        pose_iterable = plf.sdf_supplier(output_ligand_fname)
    elif output_ligand_ext == 'pdbqt':
        meeko = sys.modules.get('meeko') or __import__('meeko')
        with open(output_ligand_fname, 'r') as f:
            string = f.read()
        pdbqt_mol = meeko.PDBQTMolecule(string, is_dlg=False, skip_typing=True)
        sdf_content = meeko.RDKitMolCreate.write_sd_string(pdbqt_mol, only_cluster_leads=False)[0]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as tf:
            tf.write(sdf_content)
            tf_path = tf.name
        pose_iterable = plf.sdf_supplier(tf.name)
        os.remove(tf_path)
    else:
        raise ValueError(f"Unsupported file format: {output_ligand_ext}")
    ligand_mol = pose_iterable[pose_idx]

    # Define the types of interactions to be detected
    interaction_types = []
    for c in conf['prolif_interactions']:
        interaction_types.extend(c['interaction_type'])
    interaction_types = list(set(interaction_types))

    # Check whether the selected interaction types are valid
    available_interaction_types = plf.Fingerprint.list_available()
    invalid_interaction_types = [
        i for i in interaction_types if i not in available_interaction_types
    ]
    if invalid_interaction_types:
        raise ValueError(
            f"Invalid interaction types: {invalid_interaction_types}."
            f"Available interaction types are {available_interaction_types}."
        )

    # Set the maximum distance of interaction to be detected
    tolerance = conf['prolif_tolerance']
    parameters = {}
    for interaction_type in interaction_types:
        param_key = 'tolerance' if interaction_type == 'VdWContact' else 'distance'
        if interaction_type == 'PiStacking':
            parameters[interaction_type] = {
                'ftf_kwargs': {param_key: tolerance},
                'etf_kwargs': {param_key: tolerance}
            }
        else:
            parameters[interaction_type] = {param_key: tolerance}

    # Run detection of interaction
    fp = plf.Fingerprint(
        interactions=interaction_types,
        parameters=parameters,
        count=True,
        )
    residues = [c['residue'] for c in conf['prolif_interactions']]
    try:
        fp.run_from_iterable(
            lig_iterable=[ligand_mol], prot_mol=protein_mol, residues=residues
        )
    except:
        return None

    # Get the minimum distance of detected interactions for each residue
    min_distance_dict = {}
    for c in conf['prolif_interactions']:
        res = c['residue']
        min_distance_dict[res] = {'interaction_type': None, 'distance': None}
        metadata = fp.metadata(ligand_mol, protein_mol[res])
        for interaction_type in metadata.keys():
            if interaction_type not in c['interaction_type']:
                continue
            current_min_distance = min_distance_dict[res]['distance']
            distance_list = [d['distance'] for d in metadata[interaction_type]]
            latest_min_distance = min(distance_list)
            if current_min_distance is None or latest_min_distance < current_min_distance:
                min_distance_dict[res]['interaction_type'] = interaction_type
                min_distance_dict[res]['distance'] = latest_min_distance
            else:
                continue

    return min_distance_dict