import argparse
from logging import getLogger, StreamHandler, FileHandler, Formatter, INFO, DEBUG
import importlib
import os
import yaml

from rdkit import RDLogger

from chemts import MCTS, State
from utils.load_model import loaded_model
from utils.make_smiles import zinc_data_with_bracket_original, zinc_processed_with_bracket, smi_tokenizer
from utils.filter import HashimotoFilter


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
    conf.setdefault('hours', 1) 
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

    conf.setdefault('model_json', 'model/model.json')
    conf.setdefault('model_weight', 'model/model.h5')
    conf.setdefault('output_dir', 'result')
    conf.setdefault('reward_calculator', 'reward.logP_reward')


def main():
    args = get_parser()
    with open(args.config, "r") as f:
        conf = yaml.load(f, Loader=yaml.SafeLoader)
    set_default_config(conf)
    os.makedirs(conf['output_dir'], exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1" if args.gpu is None else args.gpu

    # set log level
    log_level = DEBUG if args.debug else INFO
    logger = get_logger(log_level, conf["output_dir"])
    if not args.debug:
        RDLogger.DisableLog("rdApp.*")

    model = loaded_model(conf['model_json'], conf['model_weight'], logger)  #WM300 not tested  
    reward_calculator = importlib.import_module(conf["reward_calculator"])
    conf["max_len"] = model.input_shape[1]
    if args.input_smiles is not None:
        logger.info(f"Extend mode: input SMILES = {args.input_smiles}")
        conf["input_smiles"] = args.input_smiles
        conf["tokenized_smiles"] = smi_tokenizer(conf["input_smiles"])

    logger.info(f"========== Configuration ==========")
    for k, v in conf.items():
        logger.info(f"{k}: {v}")
    logger.info(f"GPU devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    logger.info(f"===================================")
            
    conf["hashimoto_filter"] = HashimotoFilter()
    smiles_old = zinc_data_with_bracket_original('data/250k_rndm_zinc_drugs_clean.smi')
    val, _ = zinc_processed_with_bracket(smiles_old)
    logger.debug(f"val is {val}")

    state = State() if args.input_smiles is None else State(position=conf["tokenized_smiles"])
    df = MCTS(root=state, conf=conf, val=val, model=model, reward_calculator=reward_calculator, logger=logger)
    output_path = os.path.join(conf['output_dir'], f"result_C{conf['c_val']}.pkl")
    logger.info(f"save results at {output_path}")
    df.to_pickle(output_path)


if __name__ == "__main__":
    main()
