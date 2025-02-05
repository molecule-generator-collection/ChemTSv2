import re

import selfies as sf


def tokenize_smiles(smiles_list, use_selfies=False):
    tokenized_smiles_list = []
    unique_token_set = set()
    for smi in smiles_list:
        tokenized_smiles = selfies_tokenizer_from_smiles(smi) if use_selfies else smi_tokenizer(smi)
        tokenized_smiles.append("\n")
        unique_token_set |= set(tokenized_smiles)
        tokenized_smiles_list.append(tokenized_smiles)
    return sorted(list(unique_token_set)), tokenized_smiles_list


def read_smiles_dataset(filepath):
    with open(filepath, "r") as f:
        smiles_list = [line.strip("\n") for line in f.readlines()]
    return smiles_list


def smi_tokenizer(smi):
    """
    This function is based on https://github.com/pschwllr/MolecularTransformer#pre-processing
    Modified by Shoichi Ishida
    """
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == "".join(tokens)
    tokens.insert(0, "&")
    return tokens


def selfies_tokenizer_from_smiles(smi):
    if "[*]" in smi:
        # Because SELFIES (v2.1.0) currently does not support a wildcard (*) representation.
        smi = smi.replace("[*]", "[Lr]")
    slfs = sf.encoder(smi)
    tokens = list(sf.split_selfies(slfs))
    assert slfs == "".join(tokens)
    tokens.insert(0, "&")
    return tokens
