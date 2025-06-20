import copy
from functools import wraps
import itertools
import re
import sys
import time

import joblib
from tensorflow.keras.models import Sequential, model_from_json  # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Dense, Embedding, GRU  # pyright: ignore[reportMissingImports]
import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.MolStandardize import rdMolStandardize
import selfies as sf


def calc_execution_time(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        elapsed_time = time.time() - start
        print(f"Execution time of {f.__name__}: {elapsed_time} sec")
        return result

    return wrapper


def get_expanded_node_index(model, state, tokens, logger, threshold=0.995):
    get_int = [tokens.index(state[j]) for j in range(len(state))]
    x = np.reshape(get_int, (1, len(get_int)))
    model.reset_states()
    preds = model.predict_on_batch(x)
    state_preds = np.squeeze(preds)  # the sum of state_pred is equal to 1
    sorted_idxs = np.argsort(state_preds)[::-1]
    sorted_preds = state_preds[sorted_idxs]
    for i, v in enumerate(itertools.accumulate(sorted_preds)):
        if v > threshold:
            # return one index if the first prediction value exceeds the threshold.
            i = i if i != 0 else 1
            break
    logger.debug(f"Indices for expansion: {sorted_idxs[:i]}")
    return sorted_idxs[:i]


def get_token_to_add(all_nodes, tokens, logger):
    added_nodes = [tokens[all_nodes[i]] for i in range(len(all_nodes))]
    logger.debug(f"Added nodes: {added_nodes}")
    return added_nodes


def back_propagation(node, reward):
    while node is not None:
        node.update(reward)
        node = node.state.parent_node


def generate_smiles_as_token_index(model, state, tokens, conf):
    end = "\n"
    position = []
    position.extend(state)
    get_int = [tokens.index(position[j]) for j in range(len(position))]
    x = np.reshape(get_int, (1, len(get_int)))
    model.reset_states()

    while not get_int[-1] == tokens.index(end):
        preds = model.predict_on_batch(x)
        state_pred = np.squeeze(preds)
        next_int = conf["random_generator"].choice(range(len(state_pred)), p=state_pred)
        get_int.append(next_int)
        x = np.reshape([next_int], (1, 1))
        if len(get_int) > conf["max_len"]:
            break
    return get_int


def build_smiles_from_token_index(generated_token_indexes, tokens, use_selfies=False):
    generate_tokens = [
        tokens[generated_token_indexes[j]] for j in range(len(generated_token_indexes) - 1)
    ]
    generate_tokens.remove("&")
    concat_tokens = "".join(generate_tokens)
    if use_selfies:
        # "[*]" is replaced with [Lr] because SELFIES (v2.1.0) currently does not support a wildcard representation.
        if "[Lr]" in concat_tokens:
            concat_tokens = sf.decoder(concat_tokens).replace("[Lr]", "[*]")
        else:
            concat_tokens = sf.decoder(concat_tokens)
    return concat_tokens


def has_passed_through_filters(smiles, conf):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # default check
        return False
    checks = [f.check(mol, conf) for f in conf["filter_list"]]
    return all(checks)


def neutralize_monovalent_charges(mol):
    """ 
    Neutralizes monovalent charges in a molecule while ignoring charge-delocalized functional 
    groups (e.g., nitro).
    This function is inspired by the work by Noel O’Boyle (https://www.rdkit.org/docs/Cookbook.html#neutralizing-molecules).
    """
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        num_total_hs = atom.GetTotalNumHs()
        if (charge != 1 and charge != -1) or (charge == 1 and num_total_hs == 0):
            continue
        if any(natom.GetFormalCharge() == -charge for natom in atom.GetNeighbors()):
            continue
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(num_total_hs - charge)
        atom.UpdatePropertyCache()
    return mol


def get_model_structure_info(model_json, logger):
    with open(model_json, "r") as f:
        loaded_model_json = f.read()
    loaded_model = model_from_json(loaded_model_json)
    logger.info(f"Loaded model_json from {model_json}")
    input_shape = None
    vocab_size = None
    output_size = None
    num_gru_units = None
    for layer in loaded_model.get_config()["layers"]:
        config = layer.get("config")
        if layer.get("class_name") == "InputLayer":
            input_shape = config["batch_input_shape"][1]
        if layer.get("class_name") == "Embedding":
            vocab_size = config["input_dim"]
        # Two GRU layers are included in the default RNN model. The two layers are assumed to have the same number of units.
        if layer.get("class_name") == "GRU":
            num_gru_units = config["units"]
        if layer.get("class_name") == "TimeDistributed":
            output_size = config["layer"]["config"]["units"]
    if input_shape is None or vocab_size is None or output_size is None or num_gru_units is None:
        logger.error(
            "Confirm if the version of Tensorflow is 2.14. If so, please consult with ChemTSv2 developers on the GitHub repository. At that time, please attach the file specified as `model_json`"
        )
        sys.exit()

    return input_shape, vocab_size, output_size, num_gru_units


def load_tensorflow_model(model_weight, logger, conf):
    model = Sequential()
    model.add(
        Embedding(
            input_dim=conf["rnn_vocab_size"],
            output_dim=conf["rnn_vocab_size"],
            mask_zero=False,
            batch_size=1,
        )
    )
    model.add(
        GRU(
            conf["num_gru_units"],
            batch_input_shape=(1, None, conf["rnn_vocab_size"]),
            activation="tanh",
            return_sequences=True,
            stateful=True,
        )
    )
    model.add(
        GRU(
            conf["num_gru_units"],
            activation="tanh",
            return_sequences=False,
            stateful=True,
        )
    )
    model.add(
        Dense(
            conf["rnn_output_size"],
            activation="softmax",
        )
    )
    model.load_weights(model_weight)
    logger.info(f"Model weights loaded from {model_weight}")

    return model


def evaluate_node(new_compound, generated_dict, reward_calculator, conf, logger, gids):
    node_index = []
    valid_compound = []
    generated_ids = []
    filter_check_list = []
    valid_conf_list = []
    valid_mol_list = []
    dup_compound_info = {}

    for i in range(len(new_compound)):
        mol = Chem.MolFromSmiles(new_compound[i])
        if mol is None:
            continue
        _mol = copy.deepcopy(mol)  # Chem.SanitizeMol() modifies `mol` in place

        if Chem.SanitizeMol(_mol, catchErrors=True).name != "SANITIZE_NONE":
            continue

        if conf["neutralization"]:
            if conf["neutralization_strategy"] == "Uncharger":
                un = rdMolStandardize.Uncharger()
                un.uncharge(mol)
            elif conf["neutralization_strategy"] == "nocharge":
                neutralize_monovalent_charges(mol)
            new_compound[i] = Chem.MolToSmiles(mol)

        if new_compound[i] in generated_dict:
            dup_compound_info[i] = {
                "valid_compound": new_compound[i],
                "objective_values": generated_dict[new_compound[i]][0],
                "generated_id": gids[i],
                "filter_check": generated_dict[new_compound[i]][1],
            }
            continue

        if has_passed_through_filters(new_compound[i], conf):
            filter_check_value = 1
            filter_check_list.append(filter_check_value)
        else:
            if conf["include_filter_result_in_reward"]:
                filter_check_value = 0
                filter_check_list.append(filter_check_value)
            else:
                continue

        _conf = copy.deepcopy(conf)
        _conf["gid"] = gids[i]
        node_index.append(i)
        valid_compound.append(new_compound[i])
        generated_ids.append(gids[i])

        valid_conf_list.append(_conf)
        valid_mol_list.append(mol)

    if len(valid_mol_list) == 0:
        return [], [], [], [], []

    def _get_objective_values(mol, conf):
        return [f(mol) for f in reward_calculator.get_objective_functions(conf)]

    if conf["leaf_parallel"]:
        values_list = joblib.Parallel(n_jobs=conf["leaf_parallel_num"])(
            joblib.delayed(_get_objective_values)(m, c)
            for m, c in zip(valid_mol_list, valid_conf_list)
        )
    elif conf["batch_reward_calculation"]:
        values_list = [
            f(valid_mol_list, valid_conf_list)
            for f in reward_calculator.get_batch_objective_functions()
        ]
        values_list = np.array(values_list).T.tolist()
    else:
        values_list = [_get_objective_values(m, c) for m, c in zip(valid_mol_list, valid_conf_list)]

    assert len(valid_compound) == len(values_list) == len(filter_check_list)
    for c, vs, fc in zip(valid_compound, values_list, filter_check_list):
        generated_dict[c] = [vs, fc]
    # add duplicate compounds' data if duplicates are generated
    for k, v in sorted(dup_compound_info.items()):
        node_index.append(k)
        valid_compound.append(v["valid_compound"])
        generated_ids.append(v["generated_id"])
        values_list.append(v["objective_values"])
        filter_check_list.append(v["filter_check"])

    logger.info(f"Valid SMILES ratio: {len(valid_compound) / len(new_compound)}")

    return node_index, values_list, valid_compound, generated_ids, filter_check_list


def add_atom_index_in_wildcard(smiles: str):
    c = iter(range(1, smiles.count("*") + 1))
    labeled_smiles = re.sub(r"\*", lambda _: f"[*:{next(c)}]", smiles)
    return labeled_smiles


def transform_linker_to_mol(conf: dict):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not isinstance(args[0], Mol):
                raise TypeError("Check this decorator is placed in the correct position.")
            if "cores" not in conf:
                raise KeyError(
                    "Must specify SMILES strings corresponding to the key `cores` in the config file."
                )
            smi = Chem.MolToSmiles(args[0])
            if smi.count("*") != len(conf["cores"]):
                if func.__code__.co_argcount == 1:  # for reward function
                    raise ValueError(
                        "The number of '*' in smi does not match the number of 'cores' in configuration. "
                        "Please set 'use_attachment_points_filter' to True in the configuration when performing linker generation."
                    )
                elif func.__code__.co_argcount == 2:  # for filter function
                    return False
                else:
                    raise TypeError("Check that this decorator is placed in the correct position.")
            mol_ = Chem.MolFromSmiles(add_atom_index_in_wildcard(smi))
            rwmol = Chem.RWMol(mol_)
            cores_mol = [Chem.MolFromSmiles(s) for s in conf["cores"]]
            for m in cores_mol:
                rwmol.InsertMol(m)
            try:
                prod = Chem.molzip(rwmol)
                Chem.SanitizeMol(prod)
                prod = Chem.MolFromSmiles(Chem.MolToSmiles(prod))  # Clear props
            except Exception:
                if func.__code__.co_argcount == 1:  # for reward function
                    return -1
                elif func.__code__.co_argcount == 2:  # for filter function
                    return False
                else:
                    raise TypeError("Check that this decorator is placed in the correct position.")
            if func.__code__.co_argcount == 1:  # for reward function
                return func(prod)
            elif func.__code__.co_argcount == 2:  # for filter function
                return func(prod, conf)
            else:
                raise TypeError("Check that this decorator is placed in the correct position.")

        return wrapper

    return decorator


def set_wildcard_index(smiles: str, idx: int = 1):
    return re.sub(r"\*", lambda _: f"[*:{idx}]", smiles)


def attach_fragment_to_all_sites(conf: dict):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not isinstance(args[0], Mol):
                raise TypeError("Check this decorator is placed in the correct position.")
            if "scaffold" not in conf:
                raise KeyError(
                    "Must specify SMILES strings corresponding to the key `scaffold` in the config file."
                )
            smi = Chem.MolToSmiles(args[0])
            if smi.count("*") != 1:
                if func.__code__.co_argcount == 1:  # for reward function
                    raise ValueError(
                        "The number of '*' in a smiles string must be 1. "
                        "Please set 'use_attachment_points_filter' to True and threshold to 1 in the configuration."
                    )
                elif func.__code__.co_argcount == 2:  # for filter function
                    return False
                else:
                    raise TypeError("Check that this decorator is placed in the correct position.")
            scaffold_mol = Chem.MolFromSmiles(conf["scaffold"])
            rwmol = Chem.RWMol(scaffold_mol)
            fragment_mols = [
                Chem.MolFromSmiles(set_wildcard_index(smi, i))
                for i in range(1, conf["scaffold"].count("*")+1)
            ]
            for m in fragment_mols:
                rwmol.InsertMol(m)
            try:
                prod = Chem.molzip(rwmol)
                Chem.SanitizeMol(prod)
                prod = Chem.MolFromSmiles(Chem.MolToSmiles(prod))  # Clear props
            except Exception:
                if func.__code__.co_argcount == 1:  # for reward function
                    return -1
                elif func.__code__.co_argcount == 2:  # for filter function
                    return False
                else:
                    raise TypeError("Check that this decorator is placed in the correct position.")
            if func.__code__.co_argcount == 1:  # for reward function
                return func(prod)
            elif func.__code__.co_argcount == 2:  # for filter function
                return func(prod, conf)
            else:
                raise TypeError("Check that this decorator is placed in the correct position.")

        return wrapper

    return decorator
