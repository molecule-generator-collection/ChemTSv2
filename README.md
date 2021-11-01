# ChemTSv2

This repository is an extended and refined version of [ChemTS[1]](https://www.tandfonline.com/doi/full/10.1080/14686996.2017.1401424). The original implementation is available at https://github.com/tsudalab/ChemTS.

[1] X. Yang, J. Zhang, K. Yoshizoe, K. Terayama, and K. Tsuda, "ChemTS: An Efficient Python Library for de novo Molecular Generation", Science and Technology of Advanced Materials, Vol.18, No.1, pp.972-976, 2017.

## Requirements

1. python: 3.6
2. rdkit: 2021.03.5
3. tensorflow: 2.5.0
4. pyyaml: 5.4.1

### How to setup (example)

```bash
conda create -n chemts -c conda-forge rdkit python=3.6
# switch a python virtual environment to `chemts`
pip install --upgrade tensorflow==2.5
conda install -c conda-forge matplotlib pyyaml
```

## How to run ChemTS

1. Clone this repository and move into it.

```bash
git clone git@github.com:molecule-generator-collection/ChemTSv2.git
cd ChemTS_dev
```

2. (Optional) Train the RNN model.

```bash
cd train_RNN
python train_RNN.py -c model_setting.yaml
```

3. (Optional) Create a config file for chemts.

Please refer to the sample file `config/setting.yaml` and you will see the following content:

```yaml
trial: 1
c_val: 1.0
hours: 0.01
expansion_threshold: 0.995
simulation_num: 3

use_lipinski_filter: True
#lipinski_filter_type: rule_of_5
use_radical_filter: True
use_hashimoto_filter: True
use_sascore_filter: True
#sa_threshold: 3.5
use_ring_size_filter: True
#ring_size_threshold: 6

model_json: model/model.tf25.json
model_weight: model/model.tf25.best.ckpt.h5
output_dir: result/example01
reward_calculator: reward.logP_reward
```

If you want to pass any value to `calc_reward_from_objective_values` (e.g., weights for each value), add it in the config file.

4. Generate molecules.

```bash
python run_chemts.py -c config/setting.yaml
```

## Advanced usege

### Extend user-specified SMILES

You can extend the SMILES string you input.
In this case, you need to put the atom you want to extend at the end of the string and run ChemTS with `--input_smiles` argument as follows.

```bash
python run_chemts.py -c config/setting.yaml --input_smiles 'C1=C(C)N=CC(N)=C1C'
```

## For developer

### How to define your own reward function

This ChemTS frexibly accept user-defined reward function. 
You only need to define two functions: `calc_objective_values()` and `calc_reward_from_objective_values()`.
If you want to use your own reward function, follow the instructions below.

1. Create a python file in `reward` (e.g., `reward/logP_reward.py`).
2. Define `calc_objective_values(smiles: str, conf: Dict[Any]) -> List[float]` and `calc_reward_from_objective_values(values: List[float], conf: Dict[Any]) -> float`.  
   2.1. `calc_objective_values()` takes an SMILES string as an input and returns raw objective values in the format of `List`.  
   2.2. `calc_reward_from_objective_values()` takes the objective values as `List` type and return a `float` value.  
3. Set `reward_calculator` to a dot path of the created file (e.g., reward.logP_reward).
4. Run ChemTS (e.g, python run_chemts.py -c `PATH_TO_YOUR_CONFIG_FILE`).

## License

This package is distributed under the MIT License.

## Contact
Shoichi Ishida (ishida.sho.nm@yokohama-cu.ac.jp) and Kei Terayama (terayama@yokohama-cu.ac.jp).
