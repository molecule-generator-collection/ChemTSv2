import os
import sys
sys.path.append(os.getcwd())

from rdkit import Chem

from filter import linker_validation_filter


class TestFilter():
    def setup_method(self):
        self.valid_conf = {"cores": ["CCCC[*:1]"]}
        self.invalid_conf = {}
        self.valid_linker_mol = Chem.MolFromSmiles("On1ccc(*)c1")
        self.invalid_linker_mol = Chem.MolFromSmiles("CC1*CCC(C)1")

    def test_linker_validation_filter_is_success(self):
        filter = linker_validation_filter.LinkerValidationFilter
        result = filter.check(self.valid_linker_mol, self.valid_conf)
        assert result, f"Expected output is `True` but got {result}"

    def test_linker_validation_filter_is_failed(self):
        filter = linker_validation_filter.LinkerValidationFilter
        result = filter.check(self.invalid_linker_mol, self.valid_conf)
        assert not result, f"Expected output is `False` but got {result}"
