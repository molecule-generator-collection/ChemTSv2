# ChemTS_dev

## Requirements

### New

1. python: 3.6
2. rdkit: 2021.03.5
3. tensorflow: 2.5.0
4. pyyaml: 5.4.1

```bash
conda create -n chemts -c conda-forge rdkit python=3.6
# switch a python virtual environment to `chemts`
pip install --upgrade tensorflow==2.5
conda install -c conda-forge matplotlib pyyaml
```

## How to use

1. Get ChemTS_dev.

```bash
git clone git@github.com:ycu-iil/ChemTS_dev.git
cd ChemTS_dev
```

2. Train the RNN model.

```bash
cd train_RNN
python train_RNN.py -c model_setting.yaml
```

3. Make a setting file for molecule generate.

A sample of setting file.

setting.yaml

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

4. Molecule generate.

```bash
python run_chemts.py -c config/setting.yaml
```

## License

This package is distributed under the MIT License.
