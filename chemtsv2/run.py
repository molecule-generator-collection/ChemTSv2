import argparse
from logging import getLogger, StreamHandler, FileHandler, Formatter, INFO, DEBUG
from importlib import import_module
import os
import sys
sys.path.append(os.getcwd())
import pickle
import re
import requests
import yaml

from numpy.random import default_rng
from rdkit import RDLogger
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from chemtsv2.mcts import MCTS, State
from chemtsv2.utils import loaded_model, get_model_structure_info
from chemtsv2.preprocessing import smi_tokenizer, selfies_tokenizer_from_smiles


def get_parser():
    parser = argparse.ArgumentParser(
        description="",
        usage=f"python {os.path.basename(__file__)} -c CONFIG_FILE"
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True,
        help="path to a config file"
    )
    parser.add_argument(
        "-d", "--debug", action='store_true',
        help="debug mode"
    )
    parser.add_argument(
        "-g", "--gpu", type=str,
        help="constrain gpu. (e.g. 0,1)"
    )
    parser.add_argument(
        "--use_gpu_only_reward", action='store_true',
        help="use GPUs exclusively for reward calculations"
    )
    parser.add_argument(
        "--input_smiles", type=str,
        help="SMILES string (Need to put the atom you want to extend at the end of the string)"
    )
    return parser.parse_args()


def get_logger(level, save_dir):
    logger = getLogger(__name__)
    logger.setLevel(level)
    logger.propagate = False

    formatter = Formatter("%(asctime)s : %(levelname)s : %(message)s ")

    fh = FileHandler(filename=os.path.join(save_dir, "run.log"), mode='w')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    sh = StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def set_default_config(conf):
    conf.setdefault('c_val', 1.0)
    conf.setdefault('threshold_type', 'time')
    conf.setdefault('hours', 1) 
    conf.setdefault('generation_num', 1000)
    conf.setdefault('simulation_num', 3)
    conf.setdefault('expansion_threshold', 0.995)
    conf.setdefault('flush_threshold', -1)
    conf.setdefault('infinite_loop_threshold_for_selection', 1000)
    conf.setdefault('infinite_loop_threshold_for_expansion', 20)
    conf.setdefault('fix_random_seed', False)
    conf.setdefault('random_seed', -1)

    conf.setdefault('use_lipinski_filter', False)
    conf.setdefault('lipinski_filter', {
        'module': 'filter.lipinski_filter',
        'class': 'LipinskiFilter',
        'type': 'rule_of_5'})
    conf.setdefault('use_radical_filter', False)
    conf.setdefault('radical_filter', {
        'module': 'filter.radical_filter',
        'class': 'RadicalFilter'})
    conf.setdefault('use_pubchem_filter', False) 
    conf.setdefault('pubchem_filter', {
        'module': 'filter.pubchem_filter',
        'class': 'PubchemFilter'}) 
    conf.setdefault('use_sascore_filter', False)
    conf.setdefault('sascore_filter', {
        'module': 'filter.sascore_filter',
        'class': 'SascoreFilter',
        'threshold': 3.5})
    conf.setdefault('use_ring_size_filter', False)
    conf.setdefault('ring_size_filter', {
        'module': 'filter.ring_size_filter',
        'class': 'RingSizeFilter',
        'threshold': 6})
    conf.setdefault('use_pains_filter', False)
    conf.setdefault('pains_filter', {
        'module': 'filter.pains_filter',
        'class': 'PainsFilter',
        'type': ['pains_a']})
    conf.setdefault('include_filter_result_in_reward', False)

    conf.setdefault('model_setting', {
        'model_json': 'model/model.tf25.json',
        'model_weight': 'model/model.tf25.best.ckpt.h5'})
    conf.setdefault('output_dir', 'result')
    conf.setdefault('reward_setting', {
        'reward_module': 'reward.logP_reward',
        'reward_class': 'LogP_reward'})
    conf.setdefault('batch_reward_calculation', False)
    conf.setdefault('policy_setting', {
        'policy_module': 'policy.ucb1',
        'policy_class': 'Ucb1'})
    conf.setdefault('token', 'model/tokens.pkl')

    conf.setdefault('leaf_parallel', False)
    conf.setdefault('leaf_parallel_num', 4)
    conf.setdefault('qsub_parallel', False)
    
    conf.setdefault('save_checkpoint', False)
    conf.setdefault('restart', False)
    conf.setdefault('checkpoint_file', "chemtsv2.ckpt.pkl")

    conf.setdefault('neutralization', False)
    conf.setdefault('use_selfies', False)
    
    

def get_filter_modules(conf):
    pat = re.compile(r'^use.*filter$')
    module_list = []
    for k, frag in conf.items():
        if not pat.search(k) or frag != True:
            continue
        _k = k.replace('use_', '')
        module_list.append(getattr(import_module(conf[_k]['module']), conf[_k]['class']))
    return module_list


def main():
    args = get_parser()
    with open(args.config, "r") as f:
        conf = yaml.load(f, Loader=yaml.SafeLoader)
    set_default_config(conf)
    os.makedirs(conf['output_dir'], exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1" if args.gpu is None else args.gpu

    # set log level
    conf["debug"] = args.debug
    log_level = DEBUG if args.debug else INFO
    logger = get_logger(log_level, conf["output_dir"])
    if not args.debug:
        RDLogger.DisableLog("rdApp.*")

    if args.use_gpu_only_reward:
        logger.info("Use GPUs exclusively for reward caluculations")
        tf.config.set_visible_devices([], 'GPU')

    if args.debug:
        conf['fix_random_seed'] = True
        conf['random_seed'] = 1234

    if conf['random_seed'] != -1:
        conf['fix_random_seed'] = True

    # download additional data if files don't exist
    if not os.path.exists('data/sascorer.py'):
        url = 'https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/sascorer.py'
        with open('data/sascorer.py', 'w') as f:
            f.write(requests.get(url).text)
    if not os.path.exists('data/fpscores.pkl.gz'):
        url = 'https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz'
        with open('data/fpscores.pkl.gz', 'wb') as f:
            f.write(requests.get(url).content)
    
    rs = conf['reward_setting']
    reward_calculator = getattr(import_module(rs["reward_module"]), rs["reward_class"])
    ps = conf['policy_setting']
    policy_evaluator = getattr(import_module(ps['policy_module']), ps['policy_class'])
    conf['max_len'], conf['rnn_vocab_size'], conf['rnn_output_size'], conf['num_gru_units'] = get_model_structure_info(conf['model_setting']['model_json'], logger)
    model = loaded_model(conf['model_setting']['model_weight'], logger, conf)  #WM300 not tested  
    if args.input_smiles is not None:
        logger.info(f"Extend mode: input SMILES = {args.input_smiles}")
        conf["input_smiles"] = args.input_smiles
        conf["tokenized_smiles"] = selfies_tokenizer_from_smiles(conf["input_smiles"]) if conf['use_selfies'] else smi_tokenizer(conf["input_smiles"])

    if conf['threshold_type'] == 'time':  # To avoid user confusion
        conf.pop('generation_num')
    elif conf['threshold_type'] == 'generation_num':
        conf.pop('hours')

    logger.info(f"========== Configuration ==========")
    for k, v in conf.items():
        logger.info(f"{k}: {v}")
    logger.info(f"GPU devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    logger.info(f"===================================")
            
    conf['filter_list'] = get_filter_modules(conf)

    conf['random_generator'] = default_rng(conf['random_seed']) if conf['fix_random_seed'] else default_rng()

    with open(conf['token'], 'rb') as f:
        tokens = pickle.load(f)
    logger.debug(f"Loaded tokens are {tokens}")

    state = State() if args.input_smiles is None else State(position=conf["tokenized_smiles"])
    mcts = MCTS(root_state=state, conf=conf, tokens=tokens, model=model, reward_calculator=reward_calculator, policy_evaluator=policy_evaluator, logger=logger)
    mcts.search()
    logger.info("Finished!")


if __name__ == "__main__":
    main()
