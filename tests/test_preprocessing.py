import pytest

from chemtsv2.preprocessing import (
    tokenize_smiles,
    read_smiles_dataset,
    smi_tokenizer,
    selfies_tokenizer_from_smiles,
)


class TestPreprocessing():
    def setup_method(self):
        self.smiles_dataset_path = "data/2019PubChemQC_can_nocharge.smi"
        self.test_smiles_list = [
            "O=C(O)CCCc1cccnc1",
            "CC[C@H](C)C[C@@H](C)NC(=O)N1CCN(CC(=O)NC2CC2)CC1",
            "O=[N+]([O-])c1c(Nc2cccc3ncccc23)ncnc1N1CCN(c2cccc(Cl)c2)CC1",
            "Cc1occc1C(=O)/C(C#N)=C\c1cccc(C(F)(F)F)c1",
            "C/C(=C1/SC(=O)N(c2ccc(Cl)cc2)C1=O)c1ccc(Br)cc1",
            "O=C([O-])c1ccc(-c2ccncc2)cn1",
        ] 
        self.tokenized_smiles_list = [
            ['&', 'O', '=', 'C', '(', 'O', ')', 'C', 'C', 'C', 'c', '1', 'c', 'c', 'c', 'n', 'c', '1'],
            ['&', 'C', 'C', '[C@H]', '(', 'C', ')', 'C', '[C@@H]', '(', 'C', ')', 'N', 'C', '(', '=', 'O', ')', 'N', '1', 'C', 'C', 'N', '(', 'C', 'C', '(', '=', 'O', ')', 'N', 'C', '2', 'C', 'C', '2', ')', 'C', 'C', '1'],
            ['&', 'O', '=', '[N+]', '(', '[O-]', ')', 'c', '1', 'c', '(', 'N', 'c', '2', 'c', 'c', 'c', 'c', '3', 'n', 'c', 'c', 'c', 'c', '2', '3', ')', 'n', 'c', 'n', 'c', '1', 'N', '1', 'C', 'C', 'N', '(', 'c', '2', 'c', 'c', 'c', 'c', '(', 'Cl', ')', 'c', '2', ')', 'C', 'C', '1'],
            ['&', 'C', 'c', '1', 'o', 'c', 'c', 'c', '1', 'C', '(', '=', 'O', ')', '/', 'C', '(', 'C', '#', 'N', ')', '=', 'C', '\\', 'c', '1', 'c', 'c', 'c', 'c', '(', 'C', '(', 'F', ')', '(', 'F', ')', 'F', ')', 'c', '1'],
            ['&', 'C', '/', 'C', '(', '=', 'C', '1', '/', 'S', 'C', '(', '=', 'O', ')', 'N', '(', 'c', '2', 'c', 'c', 'c', '(', 'Cl', ')', 'c', 'c', '2', ')', 'C', '1', '=', 'O', ')', 'c', '1', 'c', 'c', 'c', '(', 'Br', ')', 'c', 'c', '1'],
            ['&', 'O', '=', 'C', '(', '[O-]', ')', 'c', '1', 'c', 'c', 'c', '(', '-', 'c', '2', 'c', 'c', 'n', 'c', 'c', '2', ')', 'c', 'n', '1'],
        ]
        self.tokenized_selfies_list = [
            ['&', '[O]', '[=C]', '[Branch1]', '[C]', '[O]', '[C]', '[C]', '[C]', '[C]', '[=C]', '[C]', '[=C]', '[N]', '[=C]', '[Ring1]', '[=Branch1]'],
            ['&', '[C]', '[C]', '[C@H1]', '[Branch1]', '[C]', '[C]', '[C]', '[C@@H1]', '[Branch1]', '[C]', '[C]', '[N]', '[C]', '[=Branch1]', '[C]', '[=O]', '[N]', '[C]', '[C]', '[N]', '[Branch1]', '[N]', '[C]', '[C]', '[=Branch1]', '[C]', '[=O]', '[N]', '[C]', '[C]', '[C]', '[Ring1]', '[Ring1]', '[C]', '[C]', '[Ring1]', '[=N]'],
            ['&', '[O]', '[=N+1]', '[Branch1]', '[C]', '[O-1]', '[C]', '[=C]', '[Branch1]', '[S]', '[N]', '[C]', '[=C]', '[C]', '[=C]', '[C]', '[=N]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[#Branch2]', '[Ring1]', '[=Branch1]', '[N]', '[=C]', '[N]', '[=C]', '[Ring1]', '[P]', '[N]', '[C]', '[C]', '[N]', '[Branch1]', '[N]', '[C]', '[=C]', '[C]', '[=C]', '[C]', '[Branch1]', '[C]', '[Cl]', '[=C]', '[Ring1]', '[#Branch1]', '[C]', '[C]', '[Ring1]', '[=N]'],
            ['&', '[C]', '[C]', '[O]', '[C]', '[=C]', '[C]', '[=Ring1]', '[Branch1]', '[C]', '[=Branch1]', '[C]', '[=O]', '[/C]', '[Branch1]', '[Ring1]', '[C]', '[#N]', '[=C]', '[\\C]', '[=C]', '[C]', '[=C]', '[C]', '[Branch1]', '[=Branch2]', '[C]', '[Branch1]', '[C]', '[F]', '[Branch1]', '[C]', '[F]', '[F]', '[=C]', '[Ring1]', '[#Branch2]'],
            ['&', '[C]', '[/C]', '[=Branch2]', '[Ring1]', '[=Branch2]', '[=C]', '[/S]', '[C]', '[=Branch1]', '[C]', '[=O]', '[N]', '[Branch1]', '[N]', '[C]', '[=C]', '[C]', '[=C]', '[Branch1]', '[C]', '[Cl]', '[C]', '[=C]', '[Ring1]', '[#Branch1]', '[C]', '[Ring1]', '[=N]', '[=O]', '[C]', '[=C]', '[C]', '[=C]', '[Branch1]', '[C]', '[Br]', '[C]', '[=C]', '[Ring1]', '[#Branch1]'],
            ['&', '[O]', '[=C]', '[Branch1]', '[C]', '[O-1]', '[C]', '[=C]', '[C]', '[=C]', '[Branch1]', '[=Branch2]', '[C]', '[=C]', '[C]', '[=N]', '[C]', '[=C]', '[Ring1]', '[=Branch1]', '[C]', '[=N]', '[Ring1]', '[N]'],
        ]
        self.expected_smiles_token_list = ['\n', '#', '&', '(', ')', '-', '/', '1', '2', '3', '=', 'Br', 'C', 'Cl', 'F', 'N', 'O', 'S', '[C@@H]', '[C@H]', '[N+]', '[O-]', '\\', 'c', 'n', 'o']
        self.expected_selfies_token_list = ['\n', '&', '[#Branch1]', '[#Branch2]', '[#N]', '[/C]', '[/S]', '[=Branch1]', '[=Branch2]', '[=C]', '[=N+1]', '[=N]', '[=O]', '[=Ring1]', '[Br]', '[Branch1]', '[C@@H1]', '[C@H1]', '[C]', '[Cl]', '[F]', '[N]', '[O-1]', '[O]', '[P]', '[Ring1]', '[S]', '[\\C]']
        
    def test_tokenize_smiles_for_smiles(self):
        token_list, tokenized_smiles_list = tokenize_smiles(self.test_smiles_list, use_selfies=False)
        assert token_list == self.expected_smiles_token_list
        for i, (r, t) in enumerate(zip(tokenized_smiles_list, self.tokenized_smiles_list)):
            r = r[:-1]  # remove '\n'
            assert r == t, f"Mismatch at index {i}: Result = {r}, Expected = {t}"
    
    def test_tokenize_smiles_for_selfies(self):
        token_list, tokenized_selfies_list = tokenize_smiles(self.test_smiles_list, use_selfies=True)
        assert token_list == self.expected_selfies_token_list
        for i, (r, t) in enumerate(zip(tokenized_selfies_list, self.tokenized_selfies_list)):
            r = r[:-1]  # remove '\n'
            assert r == t, f"Mismatch at index {i}: Result = {r}, Expected = {t}"
    
    def test_read_smiles_dataset(self):
        dataset = read_smiles_dataset(self.smiles_dataset_path)
        assert dataset[0] == "Oc1cc(O)c(O)c(O)c1"
        assert dataset[-1] == "CCCCCC#CCCCCC"
        assert dataset[3000] == "CCN(CC)CCOc1ccc(C(C)=O)cc1"
        assert dataset[6000] == "CCOC(=O)CCCCCC=O"
    
    def test_smi_tokenizer(self):
        result = [smi_tokenizer(s) for s in self.test_smiles_list]
        for i, (r, t) in enumerate(zip(result, self.tokenized_smiles_list)):
            assert r == t, f"Mismatch at index {i}: Result = {r}, Expected = {t}"
    
    def test_selfies_tokenizer_from_smiles(self):
        result = [selfies_tokenizer_from_smiles(s) for s in self.test_smiles_list]
        for i, (r, t) in enumerate(zip(result, self.tokenized_selfies_list)):
            assert r == t, f"Mismatch at index {i}: Result = {r}, Expected = {t}"
 