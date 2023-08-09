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

### Vina_use_appropriate_lingand3d_reward.py
This reward function need AutoDock Vina Python package and openbabel binary.
Below is the command to install AutoDock Vina Python package.
```bash
pip install vina, meeko, scipy
```
Below is the example of command to install openbabel binary. 
In detail, please refer to the official instruction (https://openbabel.org/docs/dev/Installation/install.html#basic-build-procedure). 
```bash
git clone https://github.com/openbabel/openbabel/
cd openbabel
mkdir build; cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=(/path/to/install/)
make
make install
```

### Vina-GPU_reward.py
```bash
pip install meeko
```

Vina-GPU also needs to be installed and run on subprocess. See installation and usage instruction at https://github.com/DeltaGroupNJUPT/Vina-GPU .

[Note]
* Boost and proper version of GCC is required when both of build and run Vina-GPU. (Confirmed GCC 8 is worked properly)
* Copy OpenCL directory in the same location as Vina-GPU binary file.
* Copy Kernel1_Opt.bin and Kernel2_Opt.bin in working directory.
* Recommend to configure TF_FORCE_GPU_ALLOW_GROWTH enviroment variable to avoid insufficient memory.
```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

#### Parameter description

Vina-GPU binary and Boost library should be set in the configuration file.
|Paramaeter|Description|
|:---|:---|
|vina_bin_path|Vina-GPU binary file|
|boost_lib_path|Path to boost library|

### DiffDock_reward.py

Apart from ChemTSv2 environment, create DiffDock environment with conda.

* DiffDock
https://github.com/gcorso/DiffDock

Set the name of environment as the value of 'diffdock_conda_env' parameter, then it will be used in subprocess.

After DiffDock process, docking score is calculated by AutoDock Vina for the ligand which got the best DiffDock confidence score.
Thus, vina and meeko libraries are also required for working AutoDock Vina.

```bash
pip install vina
pip install meeko
```

#### Parameter description

|Paramaeter|Description|
|:---|:---|
|conda_cmd|conda command path|
|diffdock_conda_env|Conda environment of Diffdock|
|diffdock_pythonpath|Path to Downloaded (or cloned by git) DiffDock root directory|
|diffdock_complex_name|target PDB ID|
|diffdock_protein_path|target protein path (.pdb)|

for other parameters, see diffdock help displayed by running python as follows:

```bash
cd Path_to_DiffDock 
```
or 

```bash
export PYTHONPATH= Path_to_DiffDock
```

then

```bash
python -m inference --help

```

### TankBind_reward.py

Additional Python package installation is required for using TankBind.
p2rank is also required for binding-site estimation.

For further information, please see https://github.com/luwei0917/TankBind .

Here is an example of TankBind environment installation, under ChemTSv2 environment.
It could be better to install pytorch by PyPI than Conda.

The following is an example of installation for the TankBind reward function.
It might be better to use PyPI instead of Conda. (Althogh Conda is used to installation in [the official TankBind installation](https://github.com/luwei0917/TankBind#installation) )

```
conda activate mpchem
conda install gcc=8
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torchdrug==0.1.2 pyg biopython
pip install torchmetrics tqdm mlcrate pyallow
pip install torch-scatter
```

#### Parameter description

|Paramaeter|Description|
|:---|:---|
|p2rank_path|Conda environment of Diffdock|
|tankbind_pythonpath| Specifiy 'tankbind' directory path in Downloaded (or cloned by git) Tankbind directory |
|tankbind_modelfile| path to self_dock.pt (or re_dock.pt) in TankBind/saved_models directory |
|tankbind_complex_name|target PDB ID|
|tankbind_protein_path|target protein path (.pdb)|
|tankbind_gpu|Use GPU or not. (Boolean)|
