# README for reward function

## How to define reward function

ChemTSv2 frexibly accept user-defined reward file written in Python3.
A user-defined class should inherit from a [Reward base class](./reward.py). 
The reward class contains two static methods: `get_objective_functions()` and `calc_reward_from_objective_values()`. 
The former method takes a configuration parameter object in a dictionary format, has at least one inner function that calculates an objective value from a `Mol` object of RDKit, and returns a list of inner functions. 
The latter method takes a list of calculated objective values and the parameter object and returns a float value. 

Below is a simple example using only Python packages:
```python
import sys
import numpy as np
from rdkit.Chem import Descriptors
import sascorer
from reward.reward import Reward

class Jscore_reward(Reward):
    def get_objective_functions(conf):
        def LogP(mol):
            return Descriptors.MolLogP(mol)
        def SAScore(mol):
            return sascorer.calculateScore(mol)
        def RingSizePenalty(mol):
            ri = mol.GetRingInfo()
            max_ring_size = max((len(r) for r in ri.AtomRings()), default=0)
            return max_ring_size - 6
        return [LogP, SAScore, RingSizePenalty]

    def calc_reward_from_objective_values(values, conf):
        logP, sascore, ring_size_penalty = values
        jscore = logP - sascore - ring_size_penalty
        return jscore / (1 + abs(jscore))
```

If you want to use external software packages, such as Gaussian 16 and AutoDock Vina, you can use them in the reward file using `subprocess` Python module (cf. [Vina_binary_reward.py](./Vina_binary_reward.py))

## Additional packages for prepared reward files
### dscore_reward.py
```bash
pip install lightgbm
```

### Vina_binary_reward.py
```bash
pip install meeko
```
Need to install AutoDock Vina (v1.2.3) on your computer. Please download and install it. https://github.com/ccsb-scripps/AutoDock-Vina/releases

### Vina_reward.py
AutoDock Vina also provides the Python package, but the above binary version is more stable in ChemTSv2.
```bash
pip install vina
pip install meeko 
```

### fluor_reward.py and chro_reward.py
To facilitate the execution of Gaussian 16 software, ChemTSv2 utilizes [QCforever](https://github.com/molecule-generator-collection/QCforever). 
First, you need to install Gaussian 16 software, and then, just run the following command to install QCforever. 
```bash
pip install --upgrade git+https://github.com/molecule-generator-collection/QCforever.git
```

