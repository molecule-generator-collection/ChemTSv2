import os
import sys
from math import *
import numpy as np
from keras.preprocessing import sequence
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import rdMolDescriptors
from reward.random_reward import calc_reward_score
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
import networkx as nx
from rdkit.Chem import rdmolops
import SDF2xyzV2
from filter import HashimotoFilter


smiles_max_len = 82 #MW250:60, MW300:70 

def expanded_node(model,state,val,loop_num):
    all_nodes=[]

    position=[]
    position.extend(state)
    get_int_old=[]
    for j in range(len(position)):
        get_int_old.append(val.index(position[j]))

    get_int=get_int_old

    x=np.reshape(get_int,(1,len(get_int)))
    x_pad= sequence.pad_sequences(x, maxlen=smiles_max_len, dtype='int32',                               
        padding='post', truncating='pre', value=0.)

    for i in range(loop_num):
        predictions=model.predict(x_pad)
        preds=np.asarray(predictions[0][len(get_int)-1]).astype('float64')
        preds = np.log(preds) / 1.0
        preds = np.exp(preds) / np.sum(np.exp(preds))
        next_probas = np.random.multinomial(1, preds, 1)
        next_int=np.argmax(next_probas)
        all_nodes.append(next_int)
    print(all_nodes)
    all_nodes=list(set(all_nodes))
    print(all_nodes)
    return all_nodes


def node_to_add(all_nodes,val):
    added_nodes=[]
    for i in range(len(all_nodes)):
        added_nodes.append(val[all_nodes[i]])
    print(added_nodes)
    return added_nodes


def chem_kn_simulation(model,state,val,added_nodes):
    all_posible=[]

    end="\n"
    for i in range(len(added_nodes)):
        position=[]
        position.extend(state)
        position.append(added_nodes[i])
        total_generated=[]
        get_int_old=[]
        for j in range(len(position)):
            get_int_old.append(val.index(position[j]))

        get_int=get_int_old

        x=np.reshape(get_int,(1,len(get_int)))
        x_pad= sequence.pad_sequences(x, maxlen=smiles_max_len, dtype='int32',
            padding='post', truncating='pre', value=0.)
        while not get_int[-1] == val.index(end):
            predictions=model.predict(x_pad)
            preds=np.asarray(predictions[0][len(get_int)-1]).astype('float64')
            preds = np.log(preds) / 1.0
            preds = np.exp(preds) / np.sum(np.exp(preds))
            next_probas = np.random.multinomial(1, preds, 1)
            next_int=np.argmax(next_probas)
            a=predictions[0][len(get_int)-1]
            get_int.append(next_int)
            x=np.reshape(get_int,(1,len(get_int)))
            x_pad = sequence.pad_sequences(x, maxlen=smiles_max_len, dtype='int32',
                padding='post', truncating='pre', value=0.)
            if len(get_int)>smiles_max_len:
                break
        total_generated.append(get_int)
        all_posible.extend(total_generated)
    return all_posible


def predict_smile(all_posible,val):
    new_compound=[]
    for i in range(len(all_posible)):
        total_generated=all_posible[i]
        generate_smile=[]
        for j in range(len(total_generated)-1):
            generate_smile.append(val[total_generated[j]])
        generate_smile.remove("&")
        new_compound.append(generate_smile)
    return new_compound


def make_input_smile(generate_smile):
    new_compound=[]
    for i in range(len(generate_smile)):
        middle=[]
        for j in range(len(generate_smile[i])):
            middle.append(generate_smile[i][j])
        com=''.join(middle)
        new_compound.append(com)
    return new_compound


def check_node_type(new_compound, score_type, generated_dict, sa_threshold = 10, rule = 0, radical = False, docking_num = 10, target_dir = 'cdk2', hashimoto_filter=False, dict_id=1, trial = 1):
    node_index=[]
    valid_compound=[]
    all_smile=[]
    distance=[]

    score=[]
    f_list = open('list_docking_pose_%s.txt' % trial, 'a')
    for i in range(len(new_compound)):
        #check dictionary
        print('check dictionary', 'comp:', new_compound[i], 'check:', new_compound[i] in generated_dict)
        if new_compound[i] in generated_dict:
            node_index.append(i)
            valid_compound.append(new_compound[i])
            score.append(generated_dict[new_compound[i]])
            print('duplication!!')
            continue

        ko = Chem.MolFromSmiles(new_compound[i])
        if ko!=None:
            # check hashimoto_filter
            if hashimoto_filter:
                hashifilter = HashimotoFilter()
                hf,_ = hashifilter.filter([new_compound[i]])
                print('hashimoto filter check is', hf)
                if hf[0] == 0:
                    continue

            #check SA_score
            SA_score = -sascorer.calculateScore(MolFromSmiles(new_compound[i]))
            print('SA_score', SA_score)
            if sa_threshold < -SA_score:
                continue

            #check radical
            if radical:
                #koh = Chem.AddHs(ko)  ## get ValueError: Sanitization error: Explicit valence for atom # 3 C, 6, is
                try:
                    koh = Chem.AddHs(ko)
                except ValueError:
                    continue

                fw = Chem.SDWriter('radical_check.sdf')
                try:
                    fw.write(koh)
                    fw.close()
                except ValueError:
                    continue
                Mol_atom, Mol_CartX, Mol_CartY, Mol_CartZ,TotalCharge, SpinMulti = SDF2xyzV2.Read_sdf('radical_check.sdf')
                print('radical check', SpinMulti)
                if SpinMulti == 2: #2:open
                    continue

            #check Rule of Five
            weight = round(rdMolDescriptors._CalcMolWt(ko), 2)
            logp = Descriptors.MolLogP(ko)
            donor = rdMolDescriptors.CalcNumLipinskiHBD(ko)
            acceptor = rdMolDescriptors.CalcNumLipinskiHBA(ko)
            rotbonds = rdMolDescriptors.CalcNumRotatableBonds(ko)
            if rule == 1:
                if weight > 500 or logp > 5 or donor > 5 or acceptor > 10:
                    continue
            if rule == 2:
                if weight > 300 or logp > 3 or donor > 3 or acceptor > 3 or rotbonds > 3:
                    continue

            cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(new_compound[i]))))
            if len(cycle_list) == 0:
                cycle_length =0
            else:
                cycle_length = max([ len(j) for j in cycle_list ])
            if cycle_length <= 6:
                cycle_length = 0
            if cycle_length==0:
                m = calc_reward_score(new_compound[i])
                if m[0]<10**10:
                    node_index.append(i)
                    valid_compound.append(new_compound[i])
                    score.append(m)
                    generated_dict[new_compound[i]] = m

    f_list.close()				
    return node_index,score,valid_compound, generated_dict
