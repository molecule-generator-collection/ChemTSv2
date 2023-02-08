import os
import sys

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from data import metadata
from filter.filter import Filter


class Neutralizer:
    reactions = None

    def __init__(self):
        patts = metadata.reaction_patterns
        self.reactions = [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in patts]

    def NeutraliseCharges(self, mol, reactions=None):
        replaced = False
        for i, (reactant, product) in enumerate(self.reactions):
            while mol.HasSubstructMatch(reactant):
                replaced = True
                rms = AllChem.ReplaceSubstructs(mol, reactant, product)
                mol = rms[0]
        mol.SetProp('neutralized', str(replaced))
        return mol


class Evaluater:
    a_dict = {}
    b_dict = {}
    esPatterns = None

    _rawD = metadata.rawD

    dict_aEstate = metadata.aEstate
    dict_atEstate = metadata.atEstate

    def __init__(self):
        data_dir = os.path.join(os.getcwd(), 'data')
        dfb = pd.read_csv(os.path.join(data_dir, 'bonds_dict.txt'), delimiter='\t')
        for i, f in dfb.iterrows():
            if f['BondIs'] == 1:
                self.b_dict[f['ES_Index_Bond']] = f['BondIs']

        dfa = pd.read_csv(os.path.join(data_dir, 'atoms_dict.txt'), delimiter='\t')
        for i, f in dfa.iterrows():
            if f['AtomIs'] == 1:
                self.a_dict[f['ES_Index_AtomBond']] = f['AtomIs']

        rawV = self._rawD

        esPatterns = [None] * len(rawV)
        for i, (name, sma) in enumerate(rawV):
            patt = Chem.MolFromSmarts(sma)
            if patt is None:
                sys.stderr.write(f"WARNING: problems with pattern {sma} (name: {name}), skipped.\n")
            else:
                esPatterns[i] = name, patt
        self.esPatterns = esPatterns

    def Evaluate(self, mol):
        self.Det_UnknownAtoms(mol)
        self.Det_InvalidBonds(mol)
        self.Det_InvalidAtoms(mol)
        self.Det_FailMol(mol)
        return mol

    def TypeAtoms(self, mol):
        """  assigns each atom in a molecule to an EState type
        **Returns:**
        list of tuples (atoms can possibly match multiple patterns) with atom types
        """
        nAtoms = mol.GetNumAtoms()
        res = [None] * nAtoms
        for name, patt in self.esPatterns:
            matches = mol.GetSubstructMatches(patt, uniquify=0)
            for match in matches:
                idx = match[0]
                if res[idx] is None:
                    res[idx] = [name]
                elif name not in res[idx]:
                    res[idx].append(name)
        for i, v in enumerate(res):
            if v is not None:
                res[i] = tuple(v)
            else:
                res[i] = ()
        return res

    def aEstateMol(self, mol):
        aE_atoms = self.TypeAtoms(mol)
        aE_key = []
        for aE_atom in aE_atoms:
            if aE_atom != ():
                a = list(aE_atom)
                if a[0] in self.dict_aEstate:
                    aE_key.append(self.dict_aEstate[a[0]])
                else:
                    aE_key.append(-1)
            else:
                aE_key.append(-1)
        return aE_key
 
    def atEstateMol(self, mol):
        aE_atoms = self.TypeAtoms(mol)
        atE_key = []
        for aE_atom in aE_atoms:
            if aE_atom != ():
                c=list(aE_atom)
                if c[0] in self.dict_atEstate:
                    atE_key.append(self.dict_atEstate[c[0]])
                else:
                    atE_key.append(-2)
            else:
                atE_key.append(-2)
        return atE_key

    def Det_UnknownAtoms(self, mol):
        ctrue = 0
        cfalse = 0
        a_list = []
        aE_list = self.aEstateMol(mol)
        a_string = ''
        for atom in mol.GetAtoms():
            idx1 = atom.GetIdx()
            key1 = aE_list[idx1]
            if key1 == -1:
                cfalse += 1
                a_list.append(idx1)
            else:
                ctrue += 1
        if len(a_list) > 0:
            aa = map(str, a_list)
            a_string = ';'.join(aa)
        mol.SetProp('UnknownAtoms', a_string)
        if cfalse > 0:
            mol.SetProp('UnknownAtomIs', '1')
        else:
            mol.SetProp('UnknownAtomIs', '0')

    def Det_InvalidBonds(self, mol):
        aE_list = self.aEstateMol(mol)
        bonds = mol.GetBonds()
        a_string = ''
        invalid_atoms = []
        ctrue = 0
        cfalse = 0
        for bond in bonds:
            query_bE = None
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            key1 = aE_list[idx1]
            key2 = aE_list[idx2]
            query_bE = str(key1) + '_' + str(key2)
            if not query_bE in self.b_dict:
                cfalse += 1
                invalid_atoms.append(idx1)
                invalid_atoms.append(idx2)
            else:
                ctrue += 1

        if len(invalid_atoms) > 0:
            a_list = list(set(invalid_atoms))
            a_list.sort()
            aa = map(str,a_list)
            a_string = ';'.join(aa)

        mol.SetProp('InvalidBonds', a_string)
        if cfalse > 0:
            mol.SetProp('InvalidBondIs', '1')
        else:
            mol.SetProp('InvalidBondIs', '0')

    def Det_InvalidAtoms(self, mol):
        aE_list = self.aEstateMol(mol)
        atE_list = self.atEstateMol(mol)
        a_string = ''
        invalid_atoms = []
        ctrue = 0
        cfalse = 0
        for atom in mol.GetAtoms():
            query_aE = None
            idx1 = atom.GetIdx()
            key1 = aE_list[idx1]
            b = []
            for nbr in atom.GetNeighbors():
                idx2 = nbr.GetIdx()
                key2 = atE_list[idx2]
                b.append(key2)
            b.sort()
            b = list(map(str, b))
            b = '_'.join(b)
            query_aE = str(key1) + ':' + str(b)
            if not query_aE in self.a_dict:
                cfalse += 1
                invalid_atoms.append(idx1)
            else:
                ctrue += 1

        if len(invalid_atoms) > 0:
            a_list = list(set(invalid_atoms))
            a_list.sort()
            aa = map(str, a_list)
            a_string = ';'.join(aa)

        mol.SetProp('InvalidAtoms', a_string)
        if cfalse > 0:
            mol.SetProp('InvalidAtomIs', '1')
        else:
            mol.SetProp('InvalidAtomIs', '0')

    def Det_FailMol(self, mol):
        c = 0
        atoms = []
        atoms_string = ''
        if mol.HasProp('UnknownAtomIs'):
            if int(mol.GetProp('UnknownAtomIs')) != 0 :
                c += 1
                if mol.HasProp('UnknownAtomIs'):
                    if mol.GetProp('UnknownAtoms') is not '':
                        a1 = mol.GetProp('UnknownAtoms')
                        for idx in a1.split(';'):
                            atoms.append(idx)
        if mol.HasProp('InvalidBondIs'):
            if int(mol.GetProp('InvalidBondIs')) != 0 :
                c += 1
                if mol.HasProp('InvalidBonds'):
                    if mol.GetProp('InvalidBonds') is not '':
                        a2 = mol.GetProp('InvalidBonds')
                        for idx in a2.split(';'):
                            atoms.append(idx)
        if mol.HasProp('InvalidAtomIs'):
            if int(mol.GetProp('InvalidAtomIs')) != 0 :
                c += 1
                if mol.HasProp('InvalidAtoms'):
                    if mol.GetProp('InvalidAtoms') is not '':
                        a3 = mol.GetProp('InvalidAtoms')
                        for idx in a3.split(';'):
                            atoms.append(idx)
        if c == 0:
            mol.SetProp('ErrorAtoms', '')
            mol.SetProp('ErrorIs', '0')
        else:
            atoms = set(atoms)
            atoms = list(map(int, atoms))
            atoms.sort()
            atoms_string = ';'.join(map(str, atoms))
            mol.SetProp('ErrorAtoms', atoms_string)
            mol.SetProp('ErrorIs', '1')

NEUTRALIZER = Neutralizer()
EVALUATER = Evaluater()

class PubchemFilter(Filter):
    def check(mol, conf):
        try:
            mol1 = NEUTRALIZER.NeutraliseCharges(mol)

            neutralized = mol1.GetProp('neutralized')
            mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(mol1))  # Got an error in Evaluate() for some reason if not re-generating Mol object from SMILES.
            mol1.SetProp('neutralized', neutralized)

            mol2 = EVALUATER.Evaluate(mol1)
            if mol2 and int(mol2.GetProp('ErrorIs')) == 0:
                return True
            else:
                return False
        except:
            return False


