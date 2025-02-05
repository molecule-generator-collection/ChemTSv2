import os
import subprocess

import pandas as pd
from rdkit import Chem

from chemtsv2.abc import Reward


class ROCS_reward(Reward):
    def get_objective_functions(conf):
        def RocsScore(mol):
            # verbosity = 'true' if conf['debug'] else 'false'
            openeye_out_dir = os.path.join(
                conf["output_dir"], "omega_rocs_result", f"gid_{conf['gid']}"
            )
            os.makedirs(openeye_out_dir, exist_ok=True)
            gen_mol_fname = "gen_mol.smi"
            omega_db_fname = "db.oeb"

            # OMEGA
            with Chem.SmilesWriter(os.path.join(openeye_out_dir, gen_mol_fname)) as f:
                f.write(mol)
            cmd_omega = [
                "oeomega",
                "rocs",
                "-in",
                gen_mol_fname,
                "-out",
                omega_db_fname,
                "-maxconfs",
                str(conf["omega"]["maxconfs"]),
                "-strictstereo",
                "false",
            ]
            try:
                subprocess.run(cmd_omega, check=True, cwd=openeye_out_dir, env=os.environ)
            except subprocess.CalledProcessError as e:
                print(f"Error SMILES: {Chem.MolToSmiles(mol)}, in OMEGA calculation")
                print(e)
                return [None, None]

            if os.path.exists(os.path.join(openeye_out_dir, "oeomega_rocs.fail")):
                print(f"Error SMILES: {Chem.MolToSmiles(mol)}, in OMEGA calculation.")
                return [None, None]

            # ROCS
            cmd_rocs = [
                "rocs",
                "-query",
                conf["rocs"]["query"],
                "-mcquery",
                "-dbase",
                omega_db_fname,
            ]
            try:
                subprocess.run(cmd_rocs, check=True, cwd=openeye_out_dir, env=os.environ)
            except subprocess.CalledProcessError as e:
                print(f"Error SMILES: {Chem.MolToSmiles(mol)}, in ROCS calculation.")
                print(e)
                return [None, None]

            #
            df = pd.read_csv(os.path.join(openeye_out_dir, "rocs_1.rpt"), sep="\t")
            return [df["ShapeTanimoto"][0], df["ColorTanimoto"][0]]

        return [RocsScore]

    def calc_reward_from_objective_values(values, conf):
        shape_tanimoto, color_tanimoto = values[0]
        if shape_tanimoto is None:
            return -1
        return (shape_tanimoto + color_tanimoto) / 2
