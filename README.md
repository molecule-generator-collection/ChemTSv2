# ChemTS_dev

## Requirements

1. [Python](https://www.anaconda.com/download/)>=3.7
2. [Keras](https://github.com/fchollet/keras) (version 2.0.5) If you installed the newest version of keras, some errors will show up. Please change it back to keras 2.0.5 by pip install keras==2.0.5. 
3. tensoflow (version 1.15.2, ver>=2.0 occurred error.) 
4. [rdkit](https://anaconda.org/rdkit/rdkit)
5. [rDock](http://rdock.sourceforge.net/installation/)

## How to use

1. Get SBMolGen.

```bash
git clone git@github.com:ycu-iil/ChemTS_dev.git
cd ChemTS_dev
```

Set the system path, here is the example for bash.

```bash
export SBMolGen_PATH={PATH_TO_CHEMTS_DEV} 
```

2. Train the RNN model.

```bash
cd train_RNN
python train_RNN.py train_RNN.yaml
```

3. Make a setting file for molecule generate.

A sample of setting file.

setting.yaml

```yaml
c_val: 1.0
loop_num_nodeExpansion: 1000
target: 'CDK2'
target_path: /home/apps/SBMolGen/example_ligand_design
hours: 1
score_target: 'SCORE.INTER'
docking_num: 10
base_rdock_score: -20
sa_threshold: 3.5
rule5: 1
radical_check: True
simulation_num: 3
hashimoto_filter: True
model_name: model
```

4. Molecule generate.

```bash
cd example_ligand_design
python ${SBMolGen_PATH}/sbmolgen.py setting.yaml
```

## License

This package is distributed under the MIT License.
