import copy
from functools import wraps
import itertools
import sys
import time

import joblib
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Embedding, GRU
import numpy as np
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from chemtsv2.misc.manage_qsub_parallel import run_qsub_parallel


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
        node.update(reward)
        node = node.state.parent_node


def chem_kn_simulation(model, state, val, conf):
    end = "\n"
    position = []
    position.extend(state)
    get_int = [val.index(position[j]) for j in range(len(position))]
    x = np.reshape(get_int, (1, len(get_int)))
    model.reset_states()

    while not get_int[-1] == val.index(end):
        preds = model.predict_on_batch(x)
        state_pred = np.squeeze(preds)
        next_int = conf['random_generator'].choice(range(len(state_pred)), p=state_pred)
        get_int.append(next_int)
        x = np.reshape([next_int], (1, 1))
        if len(get_int) > conf['max_len']:
            break
    return get_int


def build_smiles_from_tokens(all_posible, val):
    total_generated = all_posible
    generate_tokens = [val[total_generated[j]] for j in range(len(total_generated) - 1)]
    generate_tokens.remove("&")
    return ''.join(generate_tokens)


def has_passed_through_filters(smiles, conf):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # default check
        return False
    checks = [f.check(mol, conf) for f in conf['filter_list']]
    return all(checks)


def neutralize_atoms(mol):
    #https://baoilleach.blogspot.com/2019/12/no-charge-simple-approach-to.html
    #https://www.rdkit.org/docs/Cookbook.html#neutralizing-molecules
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
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
    return mol


def get_model_structure_info(model_json, logger):
    with open(model_json, 'r') as f:
        loaded_model_json = f.read()
    loaded_model = model_from_json(loaded_model_json)
    logger.info(f"Loaded model_json from {model_json}")
    input_shape = None
    vocab_size = None
    output_size = None
    for layer in loaded_model.get_config()['layers']:
        config = layer.get('config')
        if layer.get('class_name') == 'InputLayer':
            input_shape = config['batch_input_shape'][1]
        if layer.get('class_name') == 'Embedding':
            vocab_size = config['input_dim']
        if layer.get('class_name') == 'TimeDistributed':
            output_size = config['layer']['config']['units']
    if input_shape is None or vocab_size is None or output_size is None:
        logger.error('Confirm if the version of Tensorflow is 2.5. If so, please consult with ChemTSv2 developers on the GitHub repository. At that time, please attach the file specified as `model_json`')
        sys.exit()
            
    return input_shape, vocab_size, output_size

    
def loaded_model(model_weight, logger, conf):
    model = Sequential()
    model.add(Embedding(input_dim=conf['rnn_vocab_size'], output_dim=conf['rnn_vocab_size'],
                        mask_zero=False, batch_size=1))
    model.add(GRU(256, batch_input_shape=(1, None, conf['rnn_vocab_size']), activation='tanh',
                  return_sequences=True, stateful=True))
    model.add(GRU(256, activation='tanh', return_sequences=False, stateful=True))
    model.add(Dense(conf['rnn_output_size'], activation='softmax'))
    model.load_weights(model_weight)
    logger.info(f"Loaded model_weight from {model_weight}")

    return model


def evaluate_node(new_compound, generated_dict, reward_calculator, conf, logger, gids):
    node_index = []
    valid_compound = []
    generated_ids = []
    filter_check_list = []

    valid_conf_list = []
    valid_mol_list = []
    valid_filter_check_value_list = []
    dup_compound_info = {}

    #check valid smiles
    for i in range(len(new_compound)):
        mol = Chem.MolFromSmiles(new_compound[i])
        if mol is None:
            continue
        _mol = copy.deepcopy(mol)  # Chem.SanitizeMol() modifies `mol` in place
        
        if Chem.SanitizeMol(_mol, catchErrors=True).name != 'SANITIZE_NONE':
            continue

        #Neutralize
        if conf['neutralization']:
            if conf['neutralization_strategy'] == 'Uncharger':
                un = rdMolStandardize.Uncharger()
                un.uncharge(mol)
            elif conf['neutralization_strategy'] == 'nocharge':
                neutralize_atoms(mol)
            new_compound[i] = Chem.MolToSmiles(mol)

        if new_compound[i] in generated_dict:
            dup_compound_info[i] = {
                'valid_compound': new_compound[i],
                'objective_values': generated_dict[new_compound[i]][0], 
                'generated_id': gids[i],
                'filter_check': generated_dict[new_compound[i]][1]}
            continue

        if has_passed_through_filters(new_compound[i], conf):
            filter_check_value = 1
            filter_check_list.append(filter_check_value)
        else:
            if conf['include_filter_result_in_reward']:
                filter_check_value = 0
                filter_check_list.append(filter_check_value)
            else:
                continue

        _conf = copy.deepcopy(conf)
        _conf['gid'] = gids[i]
        node_index.append(i)
        valid_compound.append(new_compound[i])
        generated_ids.append(gids[i])

        valid_conf_list.append(_conf)
        valid_mol_list.append(mol)
        valid_filter_check_value_list.append(filter_check_value)

    if len(valid_mol_list) == 0: 
        return [], [], [], [], []
    
    #calculation rewards of valid molecules
    def _get_objective_values(mol, conf):
        return [f(mol) for f in reward_calculator.get_objective_functions(conf)]

    if conf['leaf_parallel']:
        if conf['qsub_parallel']:
            if len(valid_mol_list) > 0:
                values_list = run_qsub_parallel(valid_mol_list, reward_calculator, valid_conf_list)
        else:
            # standard parallelization
            values_list = joblib.Parallel(n_jobs=conf['leaf_parallel_num'])(
                joblib.delayed(_get_objective_values)(m, c) for m, c in zip(valid_mol_list, valid_conf_list))
    elif conf['batch_reward_calculation']:
        values_list = [f(valid_mol_list, valid_conf_list) for f in reward_calculator.get_batch_objective_functions()]
        values_list = np.array(values_list).T.tolist()
    else:
        values_list = [_get_objective_values(m, c) for m, c in zip(valid_mol_list, valid_conf_list)]

    #record values and other data
    for i in range(len(valid_mol_list)):
        values = values_list[i]
        filter_check_value = valid_filter_check_value_list[i]
        generated_dict[valid_compound[i]] = [values, filter_check_value]
    # add duplicate compounds' data if duplicated compounds are generated
    for k, v in sorted(dup_compound_info.items()):
        node_index.append(k)
        valid_compound.append(v['valid_compound'])
        generated_ids.append(v['generated_id'])
        values_list.append(v['objective_values'])
        filter_check_list.append(v['filter_check'])

    logger.info(f"Valid SMILES ratio: {len(valid_compound)/len(new_compound)}")

    return node_index, values_list, valid_compound, generated_ids, filter_check_list
