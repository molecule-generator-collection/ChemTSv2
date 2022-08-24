# ChemTSv2

<div align="center">
  <img src="https://github.com/molecule-generator-collection/ChemTSv2/blob/master/img/logo.png" width="95%">
</div>

This repository is a refined and extended version of [ChemTS[1]](https://www.tandfonline.com/doi/full/10.1080/14686996.2017.1401424). The original implementation is available at https://github.com/tsudalab/ChemTS.

[1] X. Yang, J. Zhang, K. Yoshizoe, K. Terayama, and K. Tsuda, "ChemTS: An Efficient Python Library for de novo Molecular Generation", Science and Technology of Advanced Materials, Vol.18, No.1, pp.972-976, 2017.

## Requirements

1. python: 3.7
2. rdkit: 2021.03.5
3. tensorflow: 2.5.0
4. pyyaml
5. pandas
6. joblib

### How to setup (example)

```bash
conda create -n chemts -c conda-forge python=3.7
# switch a python virtual environment to `chemts`
pip install --upgrade git+https://github.com/molecule-generator-collection/ChemTSv2.git
```

## How to run ChemTSv2

### 1. Clone this repository and move into it.

```bash
git clone git@github.com:molecule-generator-collection/ChemTSv2.git
cd ChemTSv2
```

### 2. (Optional) Train the RNN model.

```bash
cd train_RNN
python train_RNN.py -c model_setting.yaml
```

If you want to use your trained model, please update `chemtsv2/misc/load_model:loaded_model` based on your model architecture.

### 3. (Optional) Create a config file for chemts.

Please refer to the sample file ([config/setting.yaml](config/setting.yaml)).
If you want to pass any value to `calc_reward_from_objective_values` (e.g., weights for each value), add it in the config file.

### 4. Generate molecules.

```bash
chemtsv2 -c config/setting.yaml
```

If you want to use GPU, run ChemTS with `--gpu GPU_ID` argument as follows.

```bash
chemtsv2 -c config/setting.yaml --gpu 0
```

## Advanced usege

### Extend user-specified SMILES

You can extend the SMILES string you input.
In this case, you need to put the atom you want to extend at the end of the string and run ChemTS with `--input_smiles` argument as follows.

```bash
chemtsv2 -c config/setting.yaml --input_smiles 'C1=C(C)N=CC(N)=C1C'
```

## Usage examples

### 1. [Multiobjective optimization using Dscore](./doc/multiobjective_optimization_using_dscore.md)

### 2. [AutoDock Vina as reward function](./doc/autodock_vina.md)

## License

This package is distributed under the MIT License.

## Contact

- Shoichi Ishida (ishida.sho.nm@yokohama-cu.ac.jp)
- Kei Terayama (terayama@yokohama-cu.ac.jp).
