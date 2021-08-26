# -*- coding: utf-8 -*-

from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from os import path


class HashimotoFilter:

	neutralizer = None
	evaluater = None

	def __init__(self):
		self.neutralizer = Neutralizer()
		self.evaluater = Evaluater()

	def filter(self, smiles):
		'''
		input   list of str     smiles
		return  list of int     1: OK, 0: NG
		'''
		results = []
		mols = []  # debug用
		for smi in smiles:
			result = 0
			try:
				mol = Chem.MolFromSmiles(smi) # SMILES正当性チェック
				mol1 = self.neutralizer.NeutraliseCharges(mol)

				# 一度SMILESに直さないと、Evaluateで何故かエラーになる
				neutralized = mol1.GetProp('neutralized')
				mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(mol1))
				mol1.SetProp('neutralized', neutralized)

				mol2 = self.evaluater.Evaluate(mol1)
				if mol2 and int(mol2.GetProp('ErrorIs')) == 0:
					result = 1
			except:
				mol2 = mol
			results.append(result)
			mols.append(mol2)
		return (results, mols)


class Neutralizer:

	reactions = None

	def __init__(self):
		patts= (
			# Imidazoles
			('[n+;H]','n'),
			# Amines
			('[N+;!H0]','N'),
			# Carboxylic acids and alcohols
			('[$([O-]);!$([O-][#7])]','O'),
			# Thiols
			('[S-;X1]','S'),
			# Sulfonamides
			('[$([N-;X2]S(=O)=O)]','N'),
			# Enamines
			('[$([N-;X2][C,N]=C)]','N'),
			# Tetrazoles
			('[n-]','[nH]'),
			# Sulfoxides
			('[$([S-]=O)]','S'),
			# Amides
			('[$([N-]C=O)]','N'),
			)
		self.reactions = [(Chem.MolFromSmarts(x),Chem.MolFromSmiles(y,False)) for x,y in patts]


	def NeutraliseCharges(self, mol, reactions=None):
		replaced = False
		for i,(reactant, product) in enumerate(self.reactions):
			while mol.HasSubstructMatch(reactant):
				replaced = True
				rms = AllChem.ReplaceSubstructs(mol, reactant, product)
				mol = rms[0]
		mol.SetProp('neutralized', str(replaced))
		return mol
		#if replaced:
		#    return True
		#else:
		#    return False

class Evaluater:

	a_dict = {}
	b_dict = {}
	esPatterns = None

	_rawD = [
	  ('sLi', '[LiD1]-*'),
	  ('ssBe', '[BeD2](-*)-*'),
	  ('ssssBe', '[BeD4](-*)(-*)(-*)-*'),
	  ('ssBH', '[BD2H](-*)-*'),
	  ('sssB', '[BD3](-*)(-*)-*'),
	  ('ssssB', '[BD4](-*)(-*)(-*)-*'),
	  ('sCH3', '[CD1H3]-*'),
	  ('dCH2', '[CD1H2]=*'),
	  ('ssCH2', '[CD2H2](-*)-*'),
	  ('tCH', '[CD1H]#*'),
	  ('dsCH', '[CD2H](=*)-*'),
	  ('aaCH', '[C,c;D2H](:*):*'),
	  ('sssCH', '[CD3H](-*)(-*)-*'),
	  ('ddC', '[CD2H0](=*)=*'),
	  ('tsC', '[CD2H0](#*)-*'),
	  #('dssC', '[CD3H0](=*)(-*)-*'),
	  ('dssC', '[C,c;D3H0](=*)(~*)~*'),
	  ('aasC', '[C,c;D3H0](:*)(:*)-*'),
	  ('aaaC', '[C,c;D3H0](:*)(:*):*'),
	  ('ssssC', '[CD4H0](-*)(-*)(-*)-*'),
	  ('sNH3', '[ND1H3]-*'),
	  ('sNH2', '[ND1H2]-*'),
	  ('ssNH2', '[ND2H2](-*)-*'),
	  ('dNH', '[ND1H]=*'),
	  ('ssNH', '[ND2H](-*)-*'),
	  ('aaNH', '[N,nD2H](:*):*'),
	  ('tN', '[ND1H0]#*'),
	  ('sssNH', '[ND3H](-*)(-*)-*'),
	  ('dsN', '[ND2H0](=*)-*'),
	  ('aaN', '[N,nD2H0](:*):*'),
	  ('sssN', '[ND3H0](-*)(-*)-*'),
	  ('ddsN', '[ND3H0](~[OD1H0])(~[OD1H0])-,:*'),  # mod
	  ('aasN', '[N,nD3H0](:*)(:*)-,:*'),  # mod
	  ('ssssN', '[ND4H0](-*)(-*)(-*)-*'),
	  ('sOH', '[OD1H]-*'),
	  #('dO', '[OD1H0]=*'),
	  ('dO', '[$([OD1H0]=*),$([OD1H0-]-[*+])]'),
	  ('ssO', '[OD2H0](-*)-*'),
	  ('aaO', '[O,oD2H0](:*):*'),
	  ('sF', '[FD1]-*'),
	  ('sSiH3', '[SiD1H3]-*'),
	  ('ssSiH2', '[SiD2H2](-*)-*'),
	  ('sssSiH', '[SiD3H1](-*)(-*)-*'),
	  ('ssssSi', '[SiD4H0](-*)(-*)(-*)-*'),
	  ('sPH2', '[PD1H2]-*'),
	  ('ssPH', '[PD2H1](-*)-*'),
	  ('sssP', '[PD3H0](-*)(-*)-*'),
	  ('dsssP', '[PD4H0](=*)(-*)(-*)-*'),
	  ('sssssP', '[PD5H0](-*)(-*)(-*)(-*)-*'),
	  ('sSH', '[SD1H1]-*'),
	  ('dS', '[SD1H0]=*'),
	  ('ssS', '[SD2H0](-*)-*'),
	  ('aaS', '[S,sD2H0](:*):*'),
	  ('dssS', '[SD3H0](=*)(-*)-*'),
	  ('ddssS', '[SD4H0](~[OD1H0])(~[OD1H0])(-*)-*'),  # mod
	  ('sCl', '[ClD1]-*'),
	  ('sGeH3', '[GeD1H3](-*)'),
	  ('ssGeH2', '[GeD2H2](-*)-*'),
	  ('sssGeH', '[GeD3H1](-*)(-*)-*'),
	  ('ssssGe', '[GeD4H0](-*)(-*)(-*)-*'),
	  ('sAsH2', '[AsD1H2]-*'),
	  ('ssAsH', '[AsD2H1](-*)-*'),
	  ('sssAs', '[AsD3H0](-*)(-*)-*'),
	  ('sssdAs', '[AsD4H0](=*)(-*)(-*)-*'),
	  ('sssssAs', '[AsD5H0](-*)(-*)(-*)(-*)-*'),
	  ('sSeH', '[SeD1H1]-*'),
	  ('dSe', '[SeD1H0]=*'),
	  ('ssSe', '[SeD2H0](-*)-*'),
	  ('aaSe', '[SeD2H0](:*):*'),
	  ('dssSe', '[SeD3H0](=*)(-*)-*'),
	  ('ddssSe', '[SeD4H0](=*)(=*)(-*)-*'),
	  ('sBr', '[BrD1]-*'),
	  ('sSnH3', '[SnD1H3]-*'),
	  ('ssSnH2', '[SnD2H2](-*)-*'),
	  ('sssSnH', '[SnD3H1](-*)(-*)-*'),
	  ('ssssSn', '[SnD4H0](-*)(-*)(-*)-*'),
	  ('sI', '[ID1]-*'),
	  ('sPbH3', '[PbD1H3]-*'),
	  ('ssPbH2', '[PbD2H2](-*)-*'),
	  ('sssPbH', '[PbD3H1](-*)(-*)-*'),
	  ('ssssPb', '[PbD4H0](-*)(-*)(-*)-*'),
	]

	dict_aEstate ={'sLi':0,'ssBe':1,'ssssBe':2,'ssBH':3,'sssB':4,'ssssB':5,'sCH3':6,'dCH2':7,'ssCH2':8,'tCH':9,'dsCH':10,'aaCH':11,'sssCH':12,'ddC':13,'tsC':14,'dssC':15,'aasC':16,'aaaC':17,'ssssC':18,'sNH3':19,'sNH2':20,'ssNH2':21,'dNH':22,'ssNH':23,'aaNH':24,'tN':25,'sssNH':26,'dsN':27,'aaN':28,'sssN':29,'ddsN':30,'aasN':31,'ssssN':32,'sOH':33,'dO':34,'ssO':35,'aaO':36,'sF':37,'sSiH3':38,'ssSiH2':39,'sssSiH':40,'ssssSi':41,'sPH2':42,'ssPH':43,'sssP':44,'dsssP':45,'sssssP':46,'sSH':47,'dS':48,'ssS':49,'aaS':50,'dssS':51,'ddssS':52,'sCl':53,'sGeH3':54,'ssGeH2':55,'sssGeH':56,'ssssGe':57,'sAsH2':58,'ssAsH':59,'sssAs':60,'sssdAs':61,'sssssAs':62,'sSeH':63,'dSe':64,'ssSe':65,'aaSe':66,'dssSe':67,'ddssSe':68,'sBr':69,'sSnH3':70,'ssSnH2':71,'sssSnH':72,'ssssSn':73,'sI':74,'sPbH3':75,'ssPbH2':76,'sssPbH':77,'ssssPb':78}
	dict_atEstate ={'sLi':-1,'ssBe':-1,'ssssBe':-1,'ssBH':-1,'sssB':-1,'ssssB':-1,'sCH3':1,'dCH2':2,'ssCH2':1,'tCH':3,'dsCH':2,'aaCH':2,'sssCH':1,'ddC':4,'tsC':3,'dssC':2,'aasC':2,'aaaC':2,'ssssC':1,'sNH3':5,'sNH2':5,'ssNH2':5,'dNH':7,'ssNH':5,'aaNH':8,'tN':10,'sssNH':5,'dsN':7,'aaN':8,'sssN':5,'ddsN':9,'aasN':8,'ssssN':6,'sOH':11,'dO':12,'ssO':11,'aaO':13,'sF':14,'sSiH3':-1,'ssSiH2':-1,'sssSiH':-1,'ssssSi':-1,'sPH2':15,'ssPH':15,'sssP':15,'dsssP':15,'sssssP':15,'sSH':16,'dS':19,'ssS':16,'aaS':17,'dssS':18,'ddssS':18,'sCl':14,'sGeH3':-1,'ssGeH2':-1,'sssGeH':-1,'ssssGe':-1,'sAsH2':-1,'ssAsH':-1,'sssAs':-1,'sssdAs':-1,'sssssAs':-1,'sSeH':-1,'dSe':-1,'ssSe':-1,'aaSe':-1,'dssSe':-1,'ddssSe':-1,'sBr':20,'sSnH3':-1,'ssSnH2':-1,'sssSnH':-1,'ssssSn':-1,'sI':21,'sPbH3':-1,'ssPbH2':-1,'sssPbH':-1,'ssssPb':-1}

	def __init__(self):

		current_dir = path.dirname(path.abspath(__file__))
		dfb = pd.read_csv(current_dir + '/bonds_dict.txt', delimiter='\t')
		for i, f in dfb.iterrows():
			#print(f['Frequency'])
			if f['BondIs'] == 1:
				self.b_dict[f['ES_Index_Bond']]=f['BondIs']

		dfa = pd.read_csv(current_dir + '/atoms_dict.txt', delimiter='\t')
		for i, f in dfa.iterrows():
			#print(f['Frequency'])
			if f['AtomIs'] == 1:
				self.a_dict[f['ES_Index_AtomBond']]=f['AtomIs']

		rawV = self._rawD

		esPatterns = [None] * len(rawV)
		for i, (name, sma) in enumerate(rawV):
			patt = Chem.MolFromSmarts(sma)
			if patt is None:
				sys.stderr.write('WARNING: problems with pattern %s (name: %s), skipped.\n' % (sma, name))
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
		aE_key=[]
		for aE_atom in aE_atoms:
			if aE_atom != ():
				a=list(aE_atom)
				if a[0] in self.dict_aEstate:
					aE_key.append(self.dict_aEstate[a[0]])
				else:
					aE_key.append(-1)
			else:
				aE_key.append(-1)
		return aE_key
 
	def atEstateMol(self, mol):
		aE_atoms = self.TypeAtoms(mol)
		atE_key=[]
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
		ctrue=0
		cfalse=0
		a_list=[]
		aE_list=self.aEstateMol(mol)
		a_string=''
		#print(aE_list)
		for atom in mol.GetAtoms():
			idx1=atom.GetIdx()
			key1=aE_list[idx1]
			if key1 == -1:
				cfalse += 1
				a_list.append(idx1)
			else:
				ctrue += 1
		#print(cfalse)
		#print(ctrue)
		#print(a_list)
		if len(a_list)>0:
			aa=map(str,a_list)
			#aa=list(map(str,a_list))
			a_string=';'.join(aa)
			#print(a_string)
		mol.SetProp('UnknownAtoms', a_string)
		if cfalse > 0:
			mol.SetProp('UnknownAtomIs', '1')
		else:
			mol.SetProp('UnknownAtomIs', '0')

	def Det_InvalidBonds(self, mol):
		aE_list=self.aEstateMol(mol)
		bonds=mol.GetBonds()
		a_string=''
		invalid_atoms=[]
		#print(len(invalid_atoms))
		ctrue=0
		cfalse=0
		for bond in bonds:
			query_bE=None
			idx1=bond.GetBeginAtomIdx()
			idx2=bond.GetEndAtomIdx()
			key1=aE_list[idx1]
			key2=aE_list[idx2]
			query_bE=str(key1)+'_'+str(key2)
			if not query_bE in self.b_dict:
				cfalse += 1
				invalid_atoms.append(idx1)
				invalid_atoms.append(idx2)
			else:
				ctrue += 1

		if len(invalid_atoms)>0:
			a_list=list(set(invalid_atoms))
			a_list.sort()
			#aa=list(map(str,a_list))
			aa=map(str,a_list)
			#print(aa)
			a_string=';'.join(aa)
			#print(a_string)

		mol.SetProp('InvalidBonds', a_string)
		if cfalse > 0:
			mol.SetProp('InvalidBondIs', '1')
		else:
			mol.SetProp('InvalidBondIs', '0')

	def Det_InvalidAtoms(self, mol):
		aE_list=self.aEstateMol(mol)
		atE_list=self.atEstateMol(mol)
		a_string=''
		invalid_atoms=[]
		#print(len(invalid_atoms))
		ctrue=0
		cfalse=0
		for atom in mol.GetAtoms():
			query_aE=None
			idx1=atom.GetIdx()
			key1=aE_list[idx1]
			b=[]
			for nbr in atom.GetNeighbors():
				idx2=nbr.GetIdx()
				key2=atE_list[idx2]
				b.append(key2)
			b.sort()
			b=list(map(str,b))
			#print(b)
			b='_'.join(b)
			query_aE=str(key1)+':'+str(b)
			#print(query_aE)
			if not query_aE in self.a_dict:
				cfalse += 1
				invalid_atoms.append(idx1)
			else:
				ctrue += 1

		if len(invalid_atoms)>0:
			a_list=list(set(invalid_atoms))
			a_list.sort()
			aa=map(str,a_list)
			#aa=list(map(str,a_list))
			#print(aa)
			a_string=';'.join(aa)
			#print(a_string)
		#print(cfalse)
		#print(ctrue)

		mol.SetProp('InvalidAtoms', a_string)
		if cfalse > 0:
			mol.SetProp('InvalidAtomIs', '1')
		else:
			mol.SetProp('InvalidAtomIs', '0')

	def Det_FailMol(self, mol):
		c=0
		atoms=[]
		atoms_string=''
		if mol.HasProp('UnknownAtomIs'):
			if int(mol.GetProp('UnknownAtomIs')) != 0 :
				c += 1
				if mol.HasProp('UnknownAtomIs'):
					if mol.GetProp('UnknownAtoms') is not '':
						a1=mol.GetProp('UnknownAtoms')
						for idx in a1.split(';'):
							atoms.append(idx)
		if mol.HasProp('InvalidBondIs'):
			if int(mol.GetProp('InvalidBondIs')) != 0 :
				c += 1
				if mol.HasProp('InvalidBonds'):
					if mol.GetProp('InvalidBonds') is not '':
						a2=mol.GetProp('InvalidBonds')
						for idx in a2.split(';'):
							atoms.append(idx)
		if mol.HasProp('InvalidAtomIs'):
			if int(mol.GetProp('InvalidAtomIs')) != 0 :
				c += 1
				if mol.HasProp('InvalidAtoms'):
					if mol.GetProp('InvalidAtoms') is not '':
						a3=mol.GetProp('InvalidAtoms')
						for idx in a3.split(';'):
							atoms.append(idx)
		if c == 0:
			mol.SetProp('ErrorAtoms', '')
			mol.SetProp('ErrorIs', '0')
		else:
			atoms=set(atoms)
			atoms=list(map(int,atoms))
			atoms.sort()
			atoms_string=';'.join(map(str,atoms))
			mol.SetProp('ErrorAtoms',atoms_string)
			mol.SetProp('ErrorIs', '1')

