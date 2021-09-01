import os
import sys

from keras.preprocessing import sequence
import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MolFromSmiles, rdMolDescriptors, RDConfig, rdmolops
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from reward.random_reward import calc_reward_score
from utils import SDF2xyzV2
from utils.filter import HashimotoFilter


def expanded_node(model, state, val, loop_num, smiles_max_len, top_k=10):
    get_int = [val.index(state[j]) for j in range(len(state))]
    x = np.reshape(get_int, (1, len(get_int)))
    x_pad = sequence.pad_sequences(
        x,
        maxlen=smiles_max_len,
        dtype='int32',
        padding='post',
        truncating='pre',
        value=0.)
    preds = model.predict(x_pad)  # the sum of predictions is equal to the `conf['max_len']`
    state_pred = np.squeeze(preds)[len(get_int)-1]  # the sum of preds is equal to 1
    top_k_idxs = np.argpartition(state_pred, -top_k)[-top_k:]
    print(f"top_k_indices: {top_k_idxs}")
    return top_k_idxs


def node_to_add(all_nodes, val):
    added_nodes = [val[all_nodes[i]] for i in range(len(all_nodes))]
    print(added_nodes)
    return added_nodes


def chem_kn_simulation(model, state, val, added_nodes, smiles_max_len):
    all_posible = []
    end = "\n"
    for i in range(len(added_nodes)):
        position = []
        position.extend(state)
        position.append(added_nodes[i])
        total_generated = []
        get_int_old = [val.index(position[j]) for j in range(len(position))]
        get_int = get_int_old
        x = np.reshape(get_int, (1, len(get_int)))
        x_pad = sequence.pad_sequences(
            x,
            maxlen=smiles_max_len,
            dtype='int32',
            padding='post',
            truncating='pre',
            value=0.)

        while not get_int[-1] == val.index(end):
            predictions = model.predict(x_pad)
            preds = np.asarray(predictions[0][len(get_int) - 1]).astype('float64')
            preds = np.log(preds) / 1.0
            preds = np.exp(preds) / np.sum(np.exp(preds))
            next_probas = np.random.multinomial(1, preds, 1)
            next_int = np.argmax(next_probas)
            a = predictions[0][len(get_int) - 1]
            get_int.append(next_int)
            x = np.reshape(get_int, (1, len(get_int)))
            x_pad = sequence.pad_sequences(
                x,
                maxlen=smiles_max_len,
                dtype='int32',
                padding='post',
                truncating='pre',
                value=0.)
            if len(get_int) > smiles_max_len:
                break
        total_generated.append(get_int)
        all_posible.extend(total_generated)
    return all_posible


def predict_smiles(all_posible, val):
    new_compound = []
    for i in range(len(all_posible)):
        total_generated = all_posible[i]
        generate_smiles = [val[total_generated[j]] for j in range(len(total_generated) - 1)]
        generate_smiles.remove("&")
        new_compound.append(generate_smiles)
    return new_compound


def make_input_smiles(generate_smiles):
    new_compound = []
    for i in range(len(generate_smiles)):
        middle = [generate_smiles[i][j] for j in range(len(generate_smiles[i]))]
        com = ''.join(middle)
        new_compound.append(com)
    return new_compound


def check_node_type(new_compound, generated_dict, sa_threshold=10, rule=0, radical=False, hashimoto_filter=False, trial=1):
    node_index = []
    valid_compound = []
    score = []
    for i in range(len(new_compound)):
        print(f"check dictionary comp: {new_compound[i]} check: {new_compound[i] in generated_dict}")
        if new_compound[i] in generated_dict:
            node_index.append(i)
            valid_compound.append(new_compound[i])
            score.append(generated_dict[new_compound[i]])
            print('duplication!!')
            continue

        mol = Chem.MolFromSmiles(new_compound[i])
        if mol != None:
            # check hashimoto_filter
            if hashimoto_filter:
                hashifilter = HashimotoFilter()
                hf, _ = hashifilter.filter([new_compound[i]])
                print('hashimoto filter check is', hf)
                if hf[0] == 0:
                    continue

            #check SA_score
            SA_score = - sascorer.calculateScore(MolFromSmiles(new_compound[i]))
            print(f"SA_score: {SA_score}")
            if sa_threshold < -SA_score:
                continue

            #check radical
            if radical:
                try:
                    mol_addH = Chem.AddHs(mol)
                except ValueError:
                    continue

                fw = Chem.SDWriter('radical_check.sdf')
                try:
                    fw.write(mol_addH)
                    fw.close()
                except ValueError:
                    continue
                _, _, _, _, _, SpinMulti = SDF2xyzV2.Read_sdf('radical_check.sdf')
                print(f"radical check: {SpinMulti}")
                if SpinMulti == 2: #2:open
                    continue

            #check Rule of Five
            weight = round(rdMolDescriptors._CalcMolWt(mol), 2)
            logp = Descriptors.MolLogP(mol)
            donor = rdMolDescriptors.CalcNumLipinskiHBD(mol)
            acceptor = rdMolDescriptors.CalcNumLipinskiHBA(mol)
            rotbonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            if rule == 1:
                if weight > 500 or logp > 5 or donor > 5 or acceptor > 10:
                    continue
            if rule == 2:
                if weight > 300 or logp > 3 or donor > 3 or acceptor > 3 or rotbonds > 3:
                    continue

            cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(new_compound[i]))))
            cycle_length = 0 if len(cycle_list) == 0 else max([len(j) for j in cycle_list])
            if cycle_length <= 6:
                cycle_length = 0
            if cycle_length == 0:
                scores = calc_reward_score(new_compound[i])
                if scores[0] < 10 ** 10:
                    node_index.append(i)
                    valid_compound.append(new_compound[i])
                    score.append(scores)
                    generated_dict[new_compound[i]] = scores

    return node_index,score,valid_compound, generated_dict
