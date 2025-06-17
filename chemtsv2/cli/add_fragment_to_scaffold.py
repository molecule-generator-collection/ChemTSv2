import argparse
import os
import re

import pandas as pd
from rdkit import Chem
import yaml


def get_parser():
    parser = argparse.ArgumentParser(
        description="", usage="chemtsv2-add_fragment_to_scaffold -c PATH_TO_CONFIG_FILE"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file used with the molecule generation",
    )
    return parser.parse_args()


def main():
    def __set_wildcard_index(smiles: str, idx: int = 1):
        return re.sub(r"\*", lambda _: f"[*:{idx}]", smiles)

    def __build_smiles_from_scaffold_and_fragment(scaffold: str, fragment: str):
        if fragment.count("*") != 1:
            return None
        scaffold_mol = Chem.MolFromSmiles(scaffold)
        rwmol = Chem.RWMol(scaffold_mol)
        fragment_mols = [
            Chem.MolFromSmiles(__set_wildcard_index(fragment, i))
            for i in range(1, scaffold.count("*")+1)
        ]
        for m in fragment_mols:
            rwmol.InsertMol(m)
        try:
            prod = Chem.molzip(rwmol)
            Chem.SanitizeMol(prod)
            prod = Chem.MolFromSmiles(Chem.MolToSmiles(prod))  # Clear props
        except Exception:
            return None
        return Chem.MolToSmiles(prod)

    args = get_parser()
    config_file = args.config
    print(f"[INFO] Load config file: {args.config}")
    with open(config_file, "r") as f:
        conf = yaml.load(f, Loader=yaml.SafeLoader)
    result_path = os.path.join(conf["output_dir"], f"result_C{conf['c_val']}.csv")
    print(f"[INFO] Load result CSV file: {result_path}")
    df = pd.read_csv(result_path)
    df["smiles_w_cores"] = df["smiles"].apply(
        lambda x: __build_smiles_from_scaffold_and_fragment(conf['scaffold'], x)
    )
    stem, ext = os.path.splitext(result_path)
    output_fname = f"{stem}_add_scaffold{ext}"
    df.to_csv(output_fname, mode="w", index=False)
    print(f"[INFO] Save to {output_fname}\n[INFO] Done!")


if __name__ == "__main__":
    main()
