import argparse
import os
import re

import pandas as pd
from rdkit import Chem
import yaml


def get_parser():
    parser = argparse.ArgumentParser(
        description="", usage="chemtsv2-add_cores_to_linker -c PATH_TO_CONFIG_FILE"
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
    def __add_atom_index_in_wildcard(smiles: str):
        c = iter(range(1, smiles.count("*") + 1))
        labeled_smiles = re.sub(r"\*", lambda _: f"[*:{next(c)}]", smiles)
        return labeled_smiles

    def __build_smiles_from_cores_and_linker(linker, cores):
        if linker.count("*") != len(cores):
            return None
        linker_mol = Chem.MolFromSmiles(__add_atom_index_in_wildcard(linker))
        rwmol = Chem.RWMol(linker_mol)
        for m in cores:
            rwmol.InsertMol(m)
        try:
            prod = Chem.molzip(rwmol)
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
    cores_mol = [Chem.MolFromSmiles(s) for s in conf["cores"]]
    df["smiles_w_cores"] = df["smiles"].apply(
        lambda x: __build_smiles_from_cores_and_linker(x, cores_mol)
    )

    stem, ext = os.path.splitext(result_path)
    output_fname = f"{stem}_add_cores{ext}"
    df.to_csv(output_fname, mode="w", index=False)
    print(f"[INFO] Save to {output_fname}\n[INFO] Done!")


if __name__ == "__main__":
    main()
