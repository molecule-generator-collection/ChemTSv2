import copy
from functools import wraps
import itertools
import time

from tensorflow.keras.preprocessing import sequence
import numpy as np
from rdkit import Chem
from rdkit.Chem import MolFromSmiles


def calc_execution_time(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        elapsed_time = time.time() - start
        print(f"Execution time of {f.__name__}: {elapsed_time} sec")
        return result
    return wrapper
        
def expanded_node(model, state, val, logger, threshold=0.995):
    get_int = [val.index(state[j]) for j in range(len(state))]
    x = np.reshape(get_int, (1, len(get_int)))
    model.reset_states()
    preds = model.predict_on_batch(x)
    state_preds = np.squeeze(preds)  # the sum of state_pred is equal to 1
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
        get_int = [val.index(position[j]) for j in range(len(position))]
        x = np.reshape(get_int, (1, len(get_int)))
        model.reset_states()

        while not get_int[-1] == val.index(end):
            preds = model.predict_on_batch(x)
            state_pred = np.squeeze(preds)
            next_int = np.random.choice(range(len(state_pred)), p=state_pred)
            get_int.append(next_int)
            x = np.reshape([next_int], (1, 1))
            if len(get_int) > smiles_max_len:
                break
        all_posible.append(get_int)
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


def has_passed_through_filters(smiles, conf, logger):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # default check
        return False
    checks = [f.check(mol, conf) for f in conf['filter_list']]
    return all(checks)


def evaluate_node(new_compound, generated_dict, reward_calculator, conf, logger, gids):
    node_index = []
    valid_compound = []
    objective_values_list = []
    generated_ids = []
    filter_check_list = []
    for i in range(len(new_compound)):
        mol = Chem.MolFromSmiles(new_compound[i])
        if mol is None:
            continue
        _mol = copy.deepcopy(mol)  # Chem.SanitizeMol() modifies `mol` in place
        if Chem.SanitizeMol(_mol, catchErrors=True).name != 'SANITIZE_NONE':
            continue

        if new_compound[i] in generated_dict:
            node_index.append(i)
            valid_compound.append(new_compound[i])
            objective_values_list.append(generated_dict[new_compound[i]][0])
            generated_ids.append(gids[i])
            filter_check_list.append(generated_dict[new_compound[i]][1])
            continue

        if has_passed_through_filters(new_compound[i], conf, logger):
            filter_check_value = 1
            filter_check_list.append(filter_check_value)
        else:
            if conf['include_filter_result_in_reward']:
                filter_check_value = 0
                filter_check_list.append(filter_check_value)
            else:
                continue

        conf['gid'] = gids[i]
        values = [f(mol) for f in reward_calculator.get_objective_functions(conf)]
        node_index.append(i)
        valid_compound.append(new_compound[i])
        objective_values_list.append(values)
        generated_dict[new_compound[i]] = [values, filter_check_value]
        generated_ids.append(gids[i])
    logger.info(f"Valid SMILES ratio: {len(valid_compound)/len(new_compound)}")

    return node_index, objective_values_list, valid_compound, generated_ids, filter_check_list
