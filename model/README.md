# RNN model description

## model.tf25.best.ckpt.h5
This RNN model was trained using [data/250k_rndm_zinc_drugs_clean.smi](../data/README.md).  
This dataset is extracted from ZINC database and contains approximately 249,456 molecules.  
The tokens included in the dataset are as follows:
```
 \n, #, &, (, ), -, /, 1, 2, 3, 4, 5, 6, 7, 8, =, Br, C, Cl, F, I, N, O, P, S, 
 [C@@H], [C@@], [C@H], [C@], [CH-], [CH2-], [N+], [N-], [NH+], [NH-], [NH2+], [NH3+], 
 [O+], [O-], [OH+], [P+], [P@@H], [P@@], [P@], [PH+], [PH2], [PH], [S+], [S-], [S@@+], 
 [S@@], [S@], [SH+], [n+], [n-], [nH+], [nH], [o+], [s+], \\, c, n, o, s
```

## model_zinc_chon.tf25.best.ckpt.h5
This RNN model was trained using [data/250k_rndm_zinc_drugs_clean_std_woSsP+-.smi](../data/README.md)  
This dataset is extracted from ZINC database and contains 153,253 molecules.  
The tokens included in the dataset are as follows:
```
\n, #, &, (, ), -, /, 1, 2, 3, 4, 5, 6, 7, =, Br, C, Cl, F, I, N, O, [C@@H], [C@@], 
[C@H], [C@], [nH], \\, c, n, o
```

## model_pubchemqc.tf25.best.ckpt.h5
This RNN model was trained using [data/2019PubChemQC_can_nocharge.smi](../data/README.md)  
This dataset is extracted from PubChemQC database and contains 9,880 molecules.  
The tokens included in the dataset are as follows:
```
\n, #, &, (, ), -, 1, 2, 3, 4, =, C, N, O, c, n, o
```

## model_chembl220k.tf25.best.ckpt.h5
This RNN model was trained using [data/ChEMBL_220K.smi](../data/README.md)  
This dataset is extracted from ChEMBL database and curated by an expert medicinal chemist.
It contains 224,153 molecules.  
The tokens included in the dataset are as follows:
```
\n, #, &, (, ), -, /, 1, 2, 3, 4, 5, 6, 7, 8, =, C, Cl, F, N, O, S, [C@@H], [C@@],
[C@H], [C@], [nH], \\, c, n, o, s
```
