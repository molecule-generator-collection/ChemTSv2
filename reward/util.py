import os
import sys
import subprocess
import tempfile

from rdkit import Chem
import pandas as pd
import numpy as np


def calc_scaffold_rmsd(dock_mol, conf):
    scaffold_mol = Chem.MolFromSmiles(conf["fixed_structure_smiles"])
    reference_mol = Chem.MolFromMolFile(conf["reference_structure_path"])

    # Extract coordinates (Reference structure)
    ref_scaf_match_index = reference_mol.GetSubstructMatch(scaffold_mol)
    ref_match_coords = np.array([
        reference_mol.GetConformer(0).GetAtomPosition(idx) for idx in ref_scaf_match_index
    ])

    # Extract coordinates (Docking pose)
    dock_scaf_indexs = dock_mol.GetSubstructMatch(scaffold_mol)
    dock_match_coords = np.array([
        dock_mol.GetConformer(0).GetAtomPosition(idx) for idx in dock_scaf_indexs
    ])

    squared_diff = np.square(ref_match_coords - dock_match_coords)
    rmsd = np.sqrt(np.sum(squared_diff) / len(squared_diff))
    return rmsd


def calc_strain_energy(docking_pose_file, conf):
    scname = ["total_strain", "dihedral_torsion_strain"]

    basename, ext = os.path.splitext(os.path.basename(docking_pose_file))
    ext = ext.lower()
    mol2_path = basename + ".mol2"
    cmd = [
        "obabel",
        "-i" + ext[1:],
        docking_pose_file,
        "-omol2",
        "-O",
        mol2_path,
        "-xu",
    ]
    results = subprocess.run(cmd, capture_output=True, check=True, text=True)
    cw_dir = os.getcwd()
    os.chdir(conf["script_path"])
    cmd = ["python", "Torsion_Strain.py", cw_dir + "/" + mol2_path]
    results = subprocess.run(cmd, capture_output=True, check=True, text=True)

    os.chdir(cw_dir)

    csv_path = basename + "_Torsion_Strain.csv"
    total_strain_energy = None
    max_single_strain_energy = None

    try:
        if not os.path.exists(csv_path):
            raise Exception(f"CSV file {csv_path} does not exist.")
        df = pd.read_csv(csv_path, header=None)
        df = df[
            [1, 5]
        ]  # column1: total strain energy, column5: max single strain energy. For more Details, please check the README in the STRAIN_FILTER directory.
        df.index.name = "SID"
        df.columns = scname
        top_pose_strain_energy = df.to_numpy()[0]
        if len(top_pose_strain_energy) >= 2:
            total_strain_energy = top_pose_strain_energy[0]
            max_single_strain_energy = top_pose_strain_energy[1]
        else:
            raise Exception("The CSV file does not contain sufficient data.")
    except Exception as e:
        print(f"Warning: {e}")

    if not conf["savescr"]:
        for file_path in [mol2_path, csv_path]:
            if os.path.exists(file_path):
                os.remove(file_path)

    return total_strain_energy, max_single_strain_energy


def get_interaction_distances(receptor_fname, output_ligand_fname, pose_idx, conf):
    # Import necessary modules
    plf = sys.modules.get("prolif") or __import__("prolif")
    mda = sys.modules.get("MDAnalysis") or __import__("MDAnalysis")

    # Load receptor
    # This process is for cases where GNINA reward is used
    if receptor_fname.startswith("/scr"):
        receptor_fname = receptor_fname.replace("/scr", "data")
    u = mda.Universe(receptor_fname)
    protein_mol = plf.Molecule.from_mda(u)

    # Load ligand
    output_ligand_ext = output_ligand_fname.split(".")[-1]
    if output_ligand_ext == "sdf":
        pose_iterable = plf.sdf_supplier(output_ligand_fname)
    elif output_ligand_ext == "pdbqt":
        meeko = sys.modules.get("meeko") or __import__("meeko")
        with open(output_ligand_fname, "r") as f:
            string = f.read()
        pdbqt_mol = meeko.PDBQTMolecule(string, is_dlg=False, skip_typing=True)
        sdf_content = meeko.RDKitMolCreate.write_sd_string(pdbqt_mol, only_cluster_leads=False)[0]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sdf", delete=False) as tf:
            tf.write(sdf_content)
            tf_path = tf.name
        pose_iterable = plf.sdf_supplier(tf.name)
        os.remove(tf_path)
    else:
        raise ValueError(f"Unsupported file format: {output_ligand_ext}")
    ligand_mol = pose_iterable[pose_idx]

    # Define the types of interactions to be detected
    interaction_types = []
    for c in conf["prolif_interactions"]:
        interaction_types.extend(c["interaction_type"])
    interaction_types = list(set(interaction_types))

    # Check whether the selected interaction types are valid
    available_interaction_types = plf.Fingerprint.list_available()
    invalid_interaction_types = [
        i for i in interaction_types if i not in available_interaction_types
    ]
    if invalid_interaction_types:
        raise ValueError(
            f"Invalid interaction types: {invalid_interaction_types}."
            f"Available interaction types are {available_interaction_types}."
        )

    # Set the maximum distance of interaction to be detected
    tolerance = conf["prolif_tolerance"]
    parameters = {}
    for interaction_type in interaction_types:
        param_key = "tolerance" if interaction_type == "VdWContact" else "distance"
        if interaction_type == "PiStacking":
            parameters[interaction_type] = {
                "ftf_kwargs": {param_key: tolerance},
                "etf_kwargs": {param_key: tolerance},
            }
        else:
            parameters[interaction_type] = {param_key: tolerance}

    # Run detection of interaction
    fp = plf.Fingerprint(
        interactions=interaction_types,
        parameters=parameters,
        count=True,
    )
    residues = [c["residue"] for c in conf["prolif_interactions"]]
    try:
        fp.run_from_iterable(lig_iterable=[ligand_mol], prot_mol=protein_mol, residues=residues)
    except Exception:
        return None

    # Get the minimum distance of detected interactions for each residue
    min_distance_dict = {}
    for c in conf["prolif_interactions"]:
        res = c["residue"]
        min_distance_dict[res] = {"interaction_type": None, "distance": None}
        metadata = fp.metadata(ligand_mol, protein_mol[res])
        for interaction_type in metadata.keys():
            if interaction_type not in c["interaction_type"]:
                continue
            current_min_distance = min_distance_dict[res]["distance"]
            distance_list = [d["distance"] for d in metadata[interaction_type]]
            latest_min_distance = min(distance_list)
            if current_min_distance is None or latest_min_distance < current_min_distance:
                min_distance_dict[res]["interaction_type"] = interaction_type
                min_distance_dict[res]["distance"] = latest_min_distance
            else:
                continue

    return min_distance_dict
