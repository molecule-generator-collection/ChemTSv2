# ChemTS_dev

## Requirements

### New

1. python: 3.6
2. rdkit: 2021.03.5
3. tensorflow: 2.5.0
4. networkx: 2.6.2 
5. pyyaml: 5.4.1

### Old

1. [Python](https://www.anaconda.com/download/)>=3.7
2. [Keras](https://github.com/fchollet/keras) (version 2.0.5) If you installed the newest version of keras, some errors will show up. Please change it back to keras 2.0.5 by pip install keras==2.0.5. 
3. tensoflow (version 1.15.2, ver>=2.0 occurred error.) 
4. [rdkit](https://anaconda.org/rdkit/rdkit)

## How to use

1. Get ChemTS_dev.

```bash
git clone git@github.com:ycu-iil/ChemTS_dev.git
cd ChemTS_dev
```

2. Train the RNN model.

```bash
cd train_RNN
python train_RNN.py model_setting.yaml
```

3. Make a setting file for molecule generate.

A sample of setting file.

setting.yaml

```yaml
c_val: 1.0
loop_num_nodeExpansion: 1000
hours: 1
base_score: -20
sa_threshold: 3.5
rule5: 1
radical_check: True
simulation_num: 3
hashimoto_filter: True
model_name: model.tf25
```

4. Molecule generate.

```bash
python run.py setting.yaml
```

## License

This package is distributed under the MIT License.
