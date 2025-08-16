import re

import pytest
from rdkit import Chem

from chemtsv2.utils import transform_linker_to_mol, attach_fragment_to_all_sites


class TestTransformLinkerToMol:
    def setup_method(self):
        self.valid_conf = {"cores": ["CCCC[*:1]"]}
        self.invalid_conf = {}
        self.valid_linker_mol = Chem.MolFromSmiles("On1ccc(*)c1")
        self.invalid_linker_mol = Chem.MolFromSmiles("CC1*CCC(C)1")
        self.two_attachment_points_linker_mol = Chem.MolFromSmiles("On1ccc(*)c1*")

    def test_transform_linker_to_mol_invalid_mol(self):
        @transform_linker_to_mol(self.valid_conf)
        def func(mol):
            pass

        with pytest.raises(
            TypeError, match="Check this decorator is placed in the correct position."
        ):
            func("not_a_mol")

    def test_transform_linker_to_mol_missing_cores_key(self):
        @transform_linker_to_mol(self.invalid_conf)
        def func(mol):
            pass

        with pytest.raises(
            KeyError,
            match="Must specify SMILES strings corresponding to the key `cores` in the config file.",
        ):
            func(self.valid_linker_mol)

    def test_transform_linker_to_mol_wildcard_count_mismatch(self):
        @transform_linker_to_mol(self.valid_conf)
        def func(mol):
            pass

        error_message = re.escape(
            "The number of '*' in smi does not match the number of 'cores' in configuration. "
            "Please set 'use_attachment_points_filter' to True in the configuration when performing linker generation."
        )
        with pytest.raises(
            ValueError,
            match=error_message,
        ):
            func(self.two_attachment_points_linker_mol)

    def test_transform_linker_to_mol_wildcard_count_mismatch_for_filter(self):
        @transform_linker_to_mol(self.valid_conf)
        def func(mol, conf):
            pass

        result = func(self.two_attachment_points_linker_mol, self.valid_conf)
        assert not result, f"Expected output is `False` but got {result}"

    def test_transform_linker_to_mol_reward_molzip_failure(self):
        @transform_linker_to_mol(self.valid_conf)
        def func(mol):
            pass

        result = func(self.invalid_linker_mol)
        assert result == -1, f"Expected output is `-1` but got {result}"

    def test_transform_linker_to_mol_filter_molzip_failure(self):
        @transform_linker_to_mol(self.valid_conf)
        def func(mol, conf):
            pass

        result = func(self.invalid_linker_mol, self.valid_conf)
        assert not result, f"Expected output is `False` but got {result}"

    def test_transform_linker_to_mol_with_one_argument(self):  # for reward function
        @transform_linker_to_mol(self.valid_conf)
        def func(mol):
            return Chem.MolToSmiles(mol)

        result = func(self.valid_linker_mol)
        assert result == "CCCCc1ccn(O)c1", f"Expected 'CCCCc1ccn(O)c1' but got {result}"

    def test_transform_linker_to_mol_with_two_arguments(self):  # for filter function
        @transform_linker_to_mol(self.valid_conf)
        def func(mol, conf):
            return f"{Chem.MolToSmiles(mol)} {conf['cores'][0]}"

        result = func(self.valid_linker_mol, self.valid_conf)
        assert result == "CCCCc1ccn(O)c1 CCCC[*:1]", (
            f"Expected 'CCCCc1ccn(O)c1 CCCC[*:1]' but got {result}"
        )

    def test_transform_linker_to_mol_invalid_decorator_placement(self):
        @transform_linker_to_mol(self.valid_conf)
        def func(mol, conf, extra_arg):
            pass

        with pytest.raises(
            TypeError, match="Check that this decorator is placed in the correct position."
        ):
            func(self.valid_linker_mol, self.valid_conf, "extra_arg")


class TestAttachFragmentToAllSites:
    def setup_method(self):
        self.valid_conf = {"scaffold": "C1C([*:1])=CC([*:2])=CC=1[*:3]"}
        self.invalid_conf = {}
        self.valid_fragment_mol = Chem.MolFromSmiles("On1ccc(*)c1")
        self.invalid_fragment_mol = Chem.MolFromSmiles("CC1*CCC(C)1")
        self.two_attachment_points_mol = Chem.MolFromSmiles("On1ccc(*)c1*")

    def test_attach_fragment_to_all_sites_invalid_mol(self):
        @attach_fragment_to_all_sites(self.valid_conf)
        def func(mol):
            pass

        with pytest.raises(
            TypeError, match="Check this decorator is placed in the correct position."
        ):
            func("not_a_mol")

    def test_attach_fragment_to_all_sites_missing_scaffold_key(self):
        @attach_fragment_to_all_sites(self.invalid_conf)
        def func(mol):
            pass

        with pytest.raises(
            KeyError,
            match="Must specify SMILES strings corresponding to the key `scaffold` in the config file.",
        ):
            func(self.valid_fragment_mol)

    def test_attach_fragment_to_all_sites_wildcard_count_mismatch(self):
        @attach_fragment_to_all_sites(self.valid_conf)
        def func(mol):
            pass

        error_message = re.escape(
            "The number of '*' in a smiles string must be 1. "
            "Please set 'use_attachment_points_filter' to True and threshold to 1 in the configuration."
        )
        with pytest.raises(
            ValueError,
            match=error_message,
        ):
            func(self.two_attachment_points_mol)

    def test_attach_fragment_to_all_sites_wildcard_count_mismatch_for_filter(self):
        @attach_fragment_to_all_sites(self.valid_conf)
        def func(mol, conf):
            pass

        result = func(self.two_attachment_points_mol, self.valid_conf)
        assert not result, f"Expected output is `False` but got {result}"

    def test_attach_fragment_to_all_sites_reward_molzip_failure(self):
        @attach_fragment_to_all_sites(self.valid_conf)
        def func(mol):
            pass

        result = func(self.invalid_fragment_mol)
        assert result == -1, f"Expected output is `-1` but got {result}"

    def test_attach_fragment_to_all_sites_filter_molzip_failure(self):
        @attach_fragment_to_all_sites(self.valid_conf)
        def func(mol, conf):
            pass

        result = func(self.invalid_fragment_mol, self.valid_conf)
        assert not result, f"Expected output is `False` but got {result}"

    def test_attach_fragment_to_all_sites_with_one_argument(self):  # for reward function
        @attach_fragment_to_all_sites(self.valid_conf)
        def func(mol):
            return Chem.MolToSmiles(mol)

        result = func(self.valid_fragment_mol)
        assert result == "On1ccc(-c2cc(-c3ccn(O)c3)cc(-c3ccn(O)c3)c2)c1", f"Expected 'On1ccc(-c2cc(-c3ccn(O)c3)cc(-c3ccn(O)c3)c2)c1' but got {result}"

    def test_attach_fragment_to_all_sites_with_two_argument(self):  # for filter function
        @attach_fragment_to_all_sites(self.valid_conf)
        def func(mol, conf):
            return f"{Chem.MolToSmiles(mol)} {conf['scaffold']}"

        result = func(self.valid_fragment_mol, self.valid_conf)
        assert result == "On1ccc(-c2cc(-c3ccn(O)c3)cc(-c3ccn(O)c3)c2)c1 C1C([*:1])=CC([*:2])=CC=1[*:3]", (
            f"Expected 'On1ccc(-c2cc(-c3ccn(O)c3)cc(-c3ccn(O)c3)c2)c1 C1C([*:1])=CC([*:2])=CC=1[*:3]' but got {result}"
        )

    def test_attach_fragment_to_all_sites_invalid_decorator_placement(self):
        @attach_fragment_to_all_sites(self.valid_conf)
        def func(mol, conf, extra_arg):
            pass

        with pytest.raises(
            TypeError, match="Check that this decorator is placed in the correct position."
        ):
            func(self.valid_fragment_mol, self.valid_conf, "extra_arg")
