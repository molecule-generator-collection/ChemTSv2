import argparse
from logging import getLogger, StreamHandler, FileHandler, Formatter, INFO, DEBUG
import importlib
import os
import pickle
import yaml

from rdkit import RDLogger

from chemts import MCTS, State
from misc.load_model import loaded_model, loaded_model_struct
from misc.preprocessing import smi_tokenizer
from misc.filter import HashimotoFilter


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
    conf.setdefault('trial', 1)
    conf.setdefault('c_val', 1.0)
    conf.setdefault('threshold_type', 'time')
    conf.setdefault('hours', 1) 
    conf.setdefault('generation_num', 1000)
    conf.setdefault('simulation_num', 3)
    conf.setdefault('expansion_threshold', 0.995)

    conf.setdefault('use_lipinski_filter', True)
    conf.setdefault('lipinski_filter_type', 'rule_of_5')
    conf.setdefault('use_radical_filter', True)
    conf.setdefault('use_hashimoto_filter', True) 
    conf.setdefault('use_sascore_filter', True)
    conf.setdefault('sa_threshold', 3.5)
    conf.setdefault('use_ring_size_filter', True)
    conf.setdefault('ring_size_threshold', 6)
    conf.setdefault('include_filter_result_in_reward', False)

    conf.setdefault('model_json', 'model/model.json')
    conf.setdefault('model_weight', 'model/model.h5')
    conf.setdefault('output_dir', 'result')
    conf.setdefault('reward_calculator', 'reward.logP_reward')
    conf.setdefault('token', 'model/tokens.pkl')


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

    model = loaded_model(conf['model_weight'], logger)  #WM300 not tested  
    reward_calculator = importlib.import_module(conf["reward_calculator"])
    model_struct = loaded_model_struct(conf['model_json'], logger)
    conf["max_len"] = model_struct.input_shape[1]
    if args.input_smiles is not None:
        logger.info(f"Extend mode: input SMILES = {args.input_smiles}")
        conf["input_smiles"] = args.input_smiles
        conf["tokenized_smiles"] = smi_tokenizer(conf["input_smiles"])

    if conf['threshold_type'] == 'time':  # To avoid user confusion
        conf.pop('generation_num')
    elif conf['threshold_type'] == 'generation_num':
        conf.pop('hours')

    logger.info(f"========== Configuration ==========")
    for k, v in conf.items():
        logger.info(f"{k}: {v}")
    logger.info(f"GPU devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    logger.info(f"===================================")
            
    conf["hashimoto_filter"] = HashimotoFilter()
    with open(conf['token'], 'rb') as f:
        val = pickle.load(f)
    logger.debug(f"val is {val}")

    state = State() if args.input_smiles is None else State(position=conf["tokenized_smiles"])
    mcts = MCTS(root_state=state, conf=conf, val=val, model=model, reward_calculator=reward_calculator, logger=logger)
    df = mcts.search()
    output_path = os.path.join(conf['output_dir'], f"result_C{conf['c_val']}.pkl")
    logger.info(f"save results at {output_path}")
    df.to_pickle(output_path)


if __name__ == "__main__":
    main()
