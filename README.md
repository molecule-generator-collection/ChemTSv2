# ChemTS_dev

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
git clone git@github.com:ycu-iil/ChemTS_dev.git
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
c_val: 1.0
hours: 1
sa_threshold: 3.5
expansion_threshold: 0.995
use_lipinski_filter: rule_of_5
radical_check: True
simulation_num: 3
use_hashimoto_filter: True
model_json: model/model.tf25.json
model_weight: model/model.tf25.best.ckpt.h5
output_dir: result/example01
reward_calculator: reward.logP_reward
```

4. Generate molecules.

```bash
python run_chemts.py -c config/setting.yaml
```

## For developer

### How to define your own reward function

This ChemTS frexibly accept user-defined reward function. 
You only need to define two functions: `calc_objective_values()` and `calc_reward_from_objective_values`. 
If you want to use your own reward function, follow the instructions below.

1. Create a python file in `reward` (e.g., `reward/logP_reward.py`). 
2. Define `calc_objective_values()` and `calc_reward_from_objective_values`.  
   2.1. `calc_objective_values()` takes an SMILES string as an input and returns raw objective values in the format of `list`.  
   2.2. `calc_reward_from_objective_values` takes the objective values as `list` type and return a `float` value.  
3. Set `reward_calculator` to a dot path of the created file (e.g., reward.logP_reward). 
4. Run ChemTS (e.g, python run_chemts.py -c `PATH_TO_YOUR_CONFIG_FILE`). 

## License

This package is distributed under the MIT License.
