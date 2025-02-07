import argparse
import os
import random
from rdkit import Chem


def get_parser():
    parser = argparse.ArgumentParser(
        description="", usage="chemtsv2-augment-dataset -d PATH_TO_DATASET"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to a dataset file",
    )
    parser.add_argument(
        "-n",
        "--num_augment",
        type=int,
        default=4,
        required=False,
        help="number of randomized SMILES strings. note that a canonical SMILES is included in augmented dataset",
    )
    return parser.parse_args()


def randomize_smiles(mol):
    # This code is a part of the followin function:
    # https://github.com/undeadpixel/reinvent-randomized/blob/df63cab67df2a331afaedb4d0cea93428ef8a9f7/utils/chem.py#L90
    # MIT License
    # Ref. Randomized SMILES strings improve the quality of molecular generative models.
    # J Cheminform 11, 71 (2019). https://doi.org/10.1186/s13321-019-0393-0
    if not mol:
        return None
    new_atom_order = list(range(mol.GetNumAtoms()))
    random.shuffle(new_atom_order)
    random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
    return Chem.MolToSmiles(random_mol, canonical=False)


def main():
    args = get_parser()
    print(
        f"[INFO] Original dataset: {args.dataset}\n"
        f"[INFO] Number of generated randomized SMILES: {args.num_augment}"
    )

    print("[INFO] Process Start...")
    suppl = Chem.SmilesMolSupplier(args.dataset, titleLine=False, nameColumn=False)
    mols = [m for m in suppl if m is not None]
    smi_out = []
    for m in mols:
        smi_out.append(Chem.MolToSmiles(m))
        for _ in range(args.num_augment):
            smi_out.append(randomize_smiles(m))
    base, ext = os.path.splitext(args.dataset)
    ofname = f"{base}_randomized{ext}"
    with open(ofname, "w") as f:
        f.write("\n".join(smi_out))
    print(f"[INFO] Augmented dataset: {ofname}")
    print("[INFO] DONE!")


if __name__ == "__main__":
    main()
