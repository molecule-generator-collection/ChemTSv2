import re
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize, normalize
import pandas as pd
import os
import random, string
import subprocess

from chemtsv2.utils import read_reactions, read_substruct_mol


def tokenize_smiles(smiles_list):
    tokenized_smiles_list = []
    unique_token_set = set()
    for smi in smiles_list:
        tokenized_smiles = smi_tokenizer(smi)
        tokenized_smiles.append('\n')
        unique_token_set |= set(tokenized_smiles)
        tokenized_smiles_list.append(tokenized_smiles)
    return sorted(list(unique_token_set)), tokenized_smiles_list


def read_smiles_dataset(filepath):
    with open(filepath, 'r') as f:
        smiles_list = [l.strip('\n') for l in f.readlines()]
    return smiles_list


def smi_tokenizer(smi):
    """
    This function is based on https://github.com/pschwllr/MolecularTransformer#pre-processing
    Modified by Shoichi Ishida
    """
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    tokens.insert(0, '&')
    return tokens

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
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'data')
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

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'data')
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

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'data')
    dest_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'tmp')
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

