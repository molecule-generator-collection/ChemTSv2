import re

from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np

from chemtsv2.reward import Reward


def add_atom_labels(smiles):
    c = iter(range(1, smiles.count('*')+1))
    labeled_smiles = re.sub(r'\*', lambda _: f'[*:{next(c)}]', smiles)
    return labeled_smiles


class Linker_LogP_reward(Reward):
    def get_objective_functions(conf):
        def LogP(mol):
            smi = Chem.MolToSmiles(mol)
            if smi.count("*") != len(conf['warheads']):
                return -1
            mol_ = Chem.MolFromSmiles(add_atom_labels(smi))
            rwmol = Chem.RWMol(mol_)
            warheads_mol = [Chem.MolFromSmiles(s) for s in conf['warheads']]
            for m in warheads_mol:
                rwmol.InsertMol(m)
            prod = Chem.molzip(rwmol)
            print(Chem.MolToSmiles(prod))
            return Descriptors.MolLogP(prod)
        return [LogP]
    
    
    def calc_reward_from_objective_values(values, conf):
        return np.tanh(values[0]/10)


if __name__ == "__main__":
    import os
    import sys
    import pandas as pd
    import yaml

    def __build_smiles_from_warheads_and_linker(linker, warheads):
        if linker.count("*") != len(warheads):
            return None
        linker_mol = Chem.MolFromSmiles(add_atom_labels(linker))
        rwmol = Chem.RWMol(linker_mol)
        for m in warheads:
            rwmol.InsertMol(m)
        prod = Chem.molzip(rwmol)
        return Chem.MolToSmiles(prod)

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        conf = yaml.load(f, Loader=yaml.SafeLoader)
    result_path = os.path.join(conf['output_dir'], f"result_C{conf['c_val']}.csv")
    df = pd.read_csv(result_path)
    warheads_mol = [Chem.MolFromSmiles(s) for s in conf['warheads']]
    df['smiles_w_warheads'] = df['smiles'].apply(lambda x: __build_smiles_from_warheads_and_linker(x, warheads_mol))

    stem, ext = os.path.splitext(result_path)
    output_fname = f"{stem}_add_warheads{ext}"
    df.to_csv(output_fname, mode='w', index=False)
    print(f"[INFO] Save to {output_fname}\n"
          f"[INFO] Done!")
