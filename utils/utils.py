import itertools
import os
import sys

from keras.preprocessing import sequence
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MolFromSmiles, rdMolDescriptors, RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from utils.filter import HashimotoFilter


def expanded_node(model, state, val, smiles_max_len, logger, threshold=0.995):
    get_int = [val.index(state[j]) for j in range(len(state))]
    x = np.reshape(get_int, (1, len(get_int)))
    x_pad = sequence.pad_sequences(
        x,
        maxlen=smiles_max_len,
        dtype='int32',
        padding='post',
        truncating='pre',
        value=0.)
    preds = model.predict(x_pad)  # the sum of preds is equal to the `conf['max_len']`
    state_preds = np.squeeze(preds)[len(get_int)-1]  # the sum of state_pred is equal to 1
    sorted_idxs = np.argsort(state_preds)[::-1]
    sorted_preds = state_preds[sorted_idxs]
    for i, v in enumerate(itertools.accumulate(sorted_preds)):
        if v > threshold:
            i = i if i != 0 else 1  # return one index if the first prediction value exceeds the threshold.
            break 
    logger.debug(f"indices for expansion: {sorted_idxs[:i]}")
    return sorted_idxs[:i]


def node_to_add(all_nodes, val, logger):
    added_nodes = [val[all_nodes[i]] for i in range(len(all_nodes))]
    logger.debug(added_nodes)
    return added_nodes


def back_propagation(node, reward):
    while node != None:
        node.Update(reward)
        node = node.parentNode


def chem_kn_simulation(model, state, val, added_nodes, smiles_max_len):
    all_posible = []
    end = "\n"
    for i in range(len(added_nodes)):
        position = []
        position.extend(state)
        position.append(added_nodes[i])
        total_generated = []
        get_int = [val.index(position[j]) for j in range(len(position))]
        x = np.reshape(get_int, (1, len(get_int)))
        x_pad = sequence.pad_sequences(
            x,
            maxlen=smiles_max_len,
            dtype='int32',
            padding='post',
            truncating='pre',
            value=0.)

        while not get_int[-1] == val.index(end):
            preds = model.predict(x_pad)  # the sum of preds is equal to the `conf['max_len']` 
            state_pred = np.squeeze(preds)[len(get_int)-1]  # the sum of state_pred is equal to 1
            next_int = np.random.choice(range(len(state_pred)), p=state_pred)
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


def evaluate_node(new_compound, generated_dict, reward_calculator, conf, logger):
    node_index = []
    valid_compound = []
    objective_values_list = []
    for i in range(len(new_compound)):
        if new_compound[i] in generated_dict:
            node_index.append(i)
            valid_compound.append(new_compound[i])
            objective_values_list.append(generated_dict[new_compound[i]])
            continue

        mol = Chem.MolFromSmiles(new_compound[i])
        if mol is None:
            continue

        if conf['use_hashimoto_filter']:
            hashifilter = HashimotoFilter()
            hf, _ = hashifilter.filter([new_compound[i]])
            if hf[0] == 0:
                continue

        SA_score = - sascorer.calculateScore(MolFromSmiles(new_compound[i]))
        if conf['sa_threshold'] < -SA_score:
            continue

        if conf['radical_check']:
            if Descriptors.NumRadicalElectrons(mol) != 0:
                continue

        if conf['use_lipinski_filter']:
            weight = round(rdMolDescriptors._CalcMolWt(mol), 2)
            logp = Descriptors.MolLogP(mol)
            donor = rdMolDescriptors.CalcNumLipinskiHBD(mol)
            acceptor = rdMolDescriptors.CalcNumLipinskiHBA(mol)
            rotbonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            if conf['use_lipinski_filter'] == 'rule_of_5':
                if weight > 500 or logp > 5 or donor > 5 or acceptor > 10:
                    continue
            elif conf['use_lipinski_filter'] == 'rule_of_3':
                if weight > 300 or logp > 3 or donor > 3 or acceptor > 3 or rotbonds > 3:
                    continue
            else:
                logger.error("`use_lipinski_filter` only accepts [rule_of_5, rule_of_3]")
                sys.exit(1)
            
        # check ring size
        ri = mol.GetRingInfo()
        max_ring_size = max((len(r) for r in ri.AtomRings()), default=0)
        if max_ring_size > 6:
            continue
        values = reward_calculator.calc_objective_values(new_compound[i])
        node_index.append(i)
        valid_compound.append(new_compound[i])
        objective_values_list.append(values)
        generated_dict[new_compound[i]] = values
    logger.info(f"Valid SMILES ratio: {len(valid_compound)/len(new_compound)}")

    return node_index, objective_values_list, valid_compound, generated_dict
