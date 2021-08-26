#!/sw/bin/python3.6
import sys, math
from numpy import *
import AtomInfo

def Read_sdf(infilename):

	ifile = open(infilename, 'r') #open file for reading

	count = 0 
	
	X = []
	Y = []
	Z = []
	element_symbol = []


	Bond_pair1 = []
	Bond_pair2 = []
	Bond_type = []

	TotalCharge = 0
	CHG_atom = []
	CHG = []

	for line in ifile:
		if count == 0:
			Header1 = line

			count += 1
			continue

		if count == 1:
			Header2 = line

			count += 1
			continue	

		if count == 2:
			Header3 = line
			count += 1

			continue

		if count == 3:
			a = line.split()
			N = int(a[0])
			N_Bond = int(a[1])

			count += 1
			continue

		if 3 < count <= N+4:
			i_atom = line.split()
			if len(i_atom) != 0:
				X.append(float(i_atom[0]))
				Y.append(float(i_atom[1]))
				Z.append(float(i_atom[2]))
				element_symbol.append(i_atom[3])

			count += 1
			continue

		if N+4 < count <= N+N_Bond+3 :
			bond_info = line.split()
			print (bond_info)
			bond_info = line.split()
			Bond_pair1.append(int(bond_info[0]))
			Bond_pair2.append(int(bond_info[1]))
			Bond_type.append(int(bond_info[2]))

			count +=1
			continue

		if count > N+N_Bond+3:
			mol_info = line.split()
			print (mol_info)
			if (mol_info[0] == "M"):
				if (mol_info[1] == "END"):
					break
				if (mol_info[1] == "CHG"):
					Num_CHGInfo = int(mol_info[2])
					for k in range(Num_CHGInfo):
						CHG_atom.append(int(mol_info[3+2*k]))
						CHG.append(int(mol_info[4+2*k]))
						TotalCharge += int(mol_info[4+2*k])
			else:
				print("The sdf file is invalid!")
				sys.exit()

			count +=1

#Copy to array of numpy###########################

	Mol_atom = []
	Mol_CartX = zeros(N)
	Mol_CartY = zeros(N)
	Mol_CartZ = zeros(N)

	CHG_atom = array(CHG_atom)
	CHG = array(CHG)

	for j in range(N):
		Mol_CartX[j] = X[j] 
		Mol_CartY[j] = Y[j] 
		Mol_CartZ[j] = Z[j] 
		Mol_atom.append(element_symbol[j])

#del element_symbol[N:TotalStep]

	del element_symbol[:]
	del X[:]
	del Y[:]
	del Z[:]

##################################################

	print (Mol_atom)
	print (N)
	print (len(Mol_CartX))

	print('Reading a sdf file has finished')

###Calculating the total number of electrons#################
	TotalNum_electron = 0 

	for j in range(N):
		if (len(CHG_atom) != 0):
			Judge = CHG_atom-j
			if (any(Judge) == 0):
				TotalNum_electron += AtomInfo.AtomicNumElec(Mol_atom[j])-CHG[where(Judge == 0)]
			else:
				TotalNum_electron += AtomInfo.AtomicNumElec(Mol_atom[j])
		else:
			TotalNum_electron += AtomInfo.AtomicNumElec(Mol_atom[j])

	print('Total number of electron: %7d ' %  (TotalNum_electron))

	if (TotalNum_electron%2==0):
		print ("This system is a closed shell!")
		SpinMulti = 1
	else:
		print ("This system is a open shell!")
		SpinMulti = 2
		
#############################################################

	return Mol_atom, Mol_CartX, Mol_CartY, Mol_CartZ,TotalCharge, SpinMulti

"""
usage ='Usage; %s infile' % sys.argv[0]

try:
    infilename = sys.argv[1]
except:
    print (usage); sys.exit()

Mol_atom, X, Y, Z, TotalCharge, SpinMulti = Read_sdf(infilename)

ofile = open('Test_FromSDF.xyz','w')

ofile.write('%5d \n' % len(Mol_atom))
ofile.write('%5d %5d \n' % (TotalCharge, SpinMulti))

for j in range(len(Mol_atom)):
        ofile.write('%-4s % 10.5f  % 10.5f  % 10.5f \n'
         % (Mol_atom[j],X[j], Y[j], Z[j]))

"""

