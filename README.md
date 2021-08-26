# Structure-based de novo Molecular Generator (SBMolGen)
Supporting Information for the paper _"[Structure-based de novo molecular generator combined with artificial intelligence and docking simulations](https://doi.org/10.26434/chemrxiv.14371967.v1)"_.

In this study, we developed a new deep learning-based molecular generator, SBMolGen, that integrates a recurrent neural network, a Monte Carlo tree search, and docking simulations. The results of an evaluation using four target proteins (two kinases and two G protein-coupled receptors) showed that the generated molecules had a better binding affinity score (docking score) than the known active compounds, and they possessed a broader chemical space distribution. SBMolGen not only generates novel binding active molecules but also presents 3D docking poses with target proteins, which will be useful in subsequent drug design.

SBMolGen uses and modifies some [ChemTS](https://github.com/tsudalab/ChemTS) features. For more information on ChemTS, please see the [paper of ChemTS](https://doi.org/10.1080/14686996.2017.1401424).

## Requirements
1. [Python](https://www.anaconda.com/download/)>=3.7
2. [Keras](https://github.com/fchollet/keras) (version 2.0.5) If you installed the newest version of keras, some errors will show up. Please change it back to keras 2.0.5 by pip install keras==2.0.5. 
3. tensoflow (version 1.15.2, ver>=2.0 occurred error.) 
4. [rdkit](https://anaconda.org/rdkit/rdkit)
5. [rDock](http://rdock.sourceforge.net/installation/)

## How to use

1. Get SBMolGen.

```
git clone https://github.com/clinfo/SBMolGen.git
cd SBMolGen
```
Set the system path, here is the example for bash.
```
export SBMolGen_PATH=/Path to SBMolGen/SBMolGen
export PATH=${SBMolGen_PATH}:${PATH}
export RBT_ROOT=/Path to rDock 
export LD_LIBRARY_PATH=${RBT_ROOT}/lib:${LD_LIBRARY_PATH}
``` 

2. Train the RNN model.

```
cd train_RNN
python train_RNN.py train_RNN.yaml
```
3. Make a setting file for molecule generate.

A sample of setting file.

setting.yaml
```
c_val: 1.0
loop_num_nodeExpansion: 1000
target: 'CDK2'
target_path: /home/apps/SBMolGen/example_ligand_design
hours: 1
score_target: 'SCORE.INTER'
docking_num: 10
base_rdock_score: -20
sa_threshold: 3.5
# rule5 1: weigth < 500 logp <5 donor < 5 acceptor < 10, 2: weigth < 300 logp <3 donor < 3 acceptor < 3 rotbonds <3
rule5: 1
radical_check: True
simulation_num: 3
hashimoto_filter: True
model_name: model
```
4. Prepare the target file.

Refer to the [rDock Tutorials](http://rdock.sourceforge.net/docking-in-3-steps/) for instructions on preparing the required files for docking.

5. Molecule generate.

```
cd example_ligand_design
python ${SBMolGen_PATH}/sbmolgen.py setting.yaml
```


## License
This package is distributed under the MIT License.
