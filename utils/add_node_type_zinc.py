from subprocess import Popen, PIPE
from math import *
import random
import numpy as np
from copy import deepcopy
#from types import IntType, ListType, TupleType, StringTypes
import itertools
import time
import math
import argparse
import subprocess
from load_model import loaded_model
from keras.preprocessing import sequence
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
import sys
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import rdMolDescriptors
from make_smile import zinc_data_with_bracket_original, zinc_processed_with_bracket
#from rdock_test import rdock_score
from rdock_test_MP import rdock_score
import sascorer
import pickle
import gzip
import networkx as nx
from rdkit.Chem import rdmolops
import SDF2xyzV2
from filter import HashimotoFilter, Neutralizer
import shutil,os

#import tensorflow as tf
#os.environ["CUDA_VISIBLE_DEVICES"]="0" #"0,1,2,3,"                                                                                                                  
#from keras import backend as K
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config = config)
#K.set_session(sess)


smiles_max_len = 82 #MW250:60, MW300:70 

def expanded_node(model,state,val,loop_num):

    all_nodes=[]

    end="\n"

    #position=[]
    position=[]
    position.extend(state)
    total_generated=[]
    new_compound=[]
    get_int_old=[]
    for j in range(len(position)):
        get_int_old.append(val.index(position[j]))

    get_int=get_int_old

    x=np.reshape(get_int,(1,len(get_int)))
    #x_pad= sequence.pad_sequences(x, maxlen=82, dtype='int32',
    #    padding='post', truncating='pre', value=0.)
    x_pad= sequence.pad_sequences(x, maxlen=smiles_max_len, dtype='int32',                               
        padding='post', truncating='pre', value=0.)

    for i in range(loop_num):
        predictions=model.predict(x_pad)
        #print "shape of RNN",predictions.shape
        preds=np.asarray(predictions[0][len(get_int)-1]).astype('float64')
        preds = np.log(preds) / 1.0
        preds = np.exp(preds) / np.sum(np.exp(preds))
        next_probas = np.random.multinomial(1, preds, 1)
        next_int=np.argmax(next_probas)
        #get_int.append(next_int)
        all_nodes.append(next_int)

    print(all_nodes)
    all_nodes=list(set(all_nodes))

    print(all_nodes)





#total_generated.append(get_int)
#all_posible.extend(total_generated)






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
    #val2=['C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[C@@H]', 'n', '-', '#', 'S', 'Cl', '[O-]', '[C@H]', '[NH+]', '[C@]', 's', 'Br', '/', '[nH]', '[NH3+]', '4', '[NH2+]', '[C@@]', '[N+]', '[nH+]', '\\', '[S@]', '5', '[N-]', '[n+]', '[S@@]', '[S-]', '6', '7', 'I', '[n-]', 'P', '[OH+]', '[NH-]', '[P@@H]', '[P@@]', '[PH2]', '[P@]', '[P+]', '[S+]', '[o+]', '[CH2-]', '[CH-]', '[SH+]', '[O+]', '[s+]', '[PH+]', '[PH]', '8', '[S@@+]']
    for i in range(len(added_nodes)):
        #position=[]
        position=[]
        position.extend(state)
        position.append(added_nodes[i])
        #print state
        #print position
        #print len(val2)
        total_generated=[]
        new_compound=[]
        get_int_old=[]
        for j in range(len(position)):
            get_int_old.append(val.index(position[j]))

        get_int=get_int_old

        x=np.reshape(get_int,(1,len(get_int)))
        x_pad= sequence.pad_sequences(x, maxlen=smiles_max_len, dtype='int32',
            padding='post', truncating='pre', value=0.)
        while not get_int[-1] == val.index(end):
            predictions=model.predict(x_pad)
            #print "shape of RNN",predictions.shape
            preds=np.asarray(predictions[0][len(get_int)-1]).astype('float64')
            preds = np.log(preds) / 1.0
            preds = np.exp(preds) / np.sum(np.exp(preds))
            next_probas = np.random.multinomial(1, preds, 1)
            #print predictions[0][len(get_int)-1]
            #print "next probas",next_probas
            #next_int=np.argmax(predictions[0][len(get_int)-1])
            next_int=np.argmax(next_probas)
            a=predictions[0][len(get_int)-1]
            next_int_test=sorted(range(len(a)), key=lambda i: a[i])[-10:]
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
    #print new_compound
    #print len(new_compound)

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


            #if new_compound[i][-1] == '\n':
            #    continue

            print('SA_score', SA_score)
            if sa_threshold < -SA_score:
                #node_index.append(i)
                #valid_compound.append(new_compound[i])
                #score.append([10**10, 10**10])
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
                    #node_index.append(i)
                    #valid_compound.append(new_compound[i])
                    #score.append([10**10, 10**10])
                    continue
            if rule == 2:
                if weight > 300 or logp > 3 or donor > 3 or acceptor > 3 or rotbonds > 3:
                    #node_index.append(i)                                                                                             
                    #valid_compound.append(new_compound[i])                                                                           
                    #score.append([10**10, 10**10])                                                                                   
                    continue

            cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(new_compound[i]))))
            if len(cycle_list) == 0:
                cycle_length =0
            else:
                cycle_length = max([ len(j) for j in cycle_list ])
            if cycle_length <= 6:
                cycle_length = 0
            if cycle_length==0:
                m=rdock_score(new_compound[i], score_type, target_dir, docking_num=docking_num)
                if m[0]<10**10:
                    node_index.append(i)
                    valid_compound.append(new_compound[i])
                    score.append(m[:2])
                    #add dictionary                                                                                                                          
                    generated_dict[new_compound[i]] = m[:2]

                    #copy best docking result
                    best_docking_id = m[2]
                    docking_result_file = 'rdock_out_'
                    compound_id = i
                    # creat the directory for best docking pose.
                    out_dir = 'mol_3D_pose_trial'+ str(trial)
                    if not os.path.isdir(out_dir):
                        os.mkdir(out_dir)  
                    f_list.write('pose_'+str(dict_id)+'_'+str(compound_id)+'_'+str(best_docking_id)+','+new_compound[i])
                    f_list.write('\n')
                    shutil.copyfile(docking_result_file+str(best_docking_id)+'.sd', out_dir + '/pose_'+ str(dict_id)+'_'+str(compound_id)+'_'+str(best_docking_id)+'.sd')
    f_list.close()				
    return node_index,score,valid_compound, generated_dict
