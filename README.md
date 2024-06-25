# ChemTSv2

<div align="center">
  <img src="https://github.com/molecule-generator-collection/ChemTSv2/blob/master/img/logo.png" width="95%">
</div>

ChemTSv2[^13] is a refined and extended version of ChemTS[^1] and MPChemTS[^2].
The original implementations are available at https://github.com/tsudalab/ChemTS and https://github.com/yoshizoe/mp-chemts, respectively.

ChemTSv2 provides:

- easy-to-run interface by using only a configuration file
- easy-to-define framework for users' any reward function, molecular filter, and tree policy
- various usage examples in the GitHub repository

[^13]: Ishida, S. and Aasawat, T. and Sumita, M. and Katouda, M. and Yoshizawa, T. and Yoshizoe, K. and Tsuda, K. and Terayama, K. (2023). ChemTSv2: Functional molecular design using de novo molecule generator. <i>WIREs Computational Molecular Science</i> https://wires.onlinelibrary.wiley.com/doi/10.1002/wcms.1680

[^1]: Yang, X., Zhang, J., Yoshizoe, K., Terayama, K., & Tsuda, K. (2017). ChemTS: an efficient python library for de novo molecular generation. <i>Science and Technology of Advanced Materials</i>, 18(1), 972–976. https://doi.org/10.1080/14686996.2017.1401424

[^2]: Yang, X., Aasawat, T., & Yoshizoe, K. (2021). Practical Massively Parallel Monte-Carlo Tree Search Applied to Molecular Design. <i>In International Conference on Learning Representations</i>. https://openreview.net/forum?id=6k7VdojAIK

## How to setup :pushpin:

### Requirements :memo:

>[!NOTE]
>ARM architecture, e.g., Apple Silicon, is not currently supported

<details>
  <summary>Click to show/hide requirements</summary>

1. python: 3.11
2. rdkit: 2023.9.1
3. tensorflow: 2.14.1
4. pyyaml
5. pandas: 2.1.3
6. joblib
7. mpi4py: 3.1.5 (for massive parallel mode)

</details>

### ChemTSv2 with single process mode :red_car:

<details>
  <summary>Click to show/hide the instruction</summary>

```bash
cd YOUR_WORKSPACE
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade chemtsv2
```

</details>

### ChemTSv2 with massive parallel mode :airplane:

<details>
  <summary>Click to show/hide the instruction</summary>
NOTE: You need to run ChemTSv2-MP on a server where OpenMPI or MPICH is installed.
If you can't find `mpiexec` command, please consult your server administrator to install such an MPI library.

If you can use/prepare a server with MPI environment, please follow the (a) instruction; otherwise, please follow the (b) instruction.

#### (a) Installation on a server WITH a MPI environment

```bash
cd YOUR_WORKSPACE
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade chemtsv2[mp]
```

#### (b) Installation on a server WITHOUT a MPI environment

```bash
conda create -n mpchem python=3.11 -c conda-forge
# swith to the `mpchem` environment
conda install -c conda-forge openmpi cxx-compiler mpi mpi4py=3.1.5
pip install --upgrade chemtsv2
```

</details>

## How to run ChemTSv2 :pushpin:

### 1. Clone this repository and move into it

```bash
git clone git@github.com:molecule-generator-collection/ChemTSv2.git
cd ChemTSv2
```

### 2. Prepare a reward file

Please refer to `reward/README.md`.
An example of reward definition for LogP maximization task is as follows.

```python
from rdkit.Chem import Descriptors
import numpy as np
from chemtsv2.reward import Reward

class LogP_reward(Reward):
    def get_objective_functions(conf):
        def LogP(mol):
            return Descriptors.MolLogP(mol)
        return [LogP]
    
    def calc_reward_from_objective_values(objective_values, conf):
        logp = objective_values[0]
        return np.tanh(logp/10)
```

### 3. Prepare a config file

The explanation of options are described in the [Support option/function](#support-optionfunction-pushpin) section.
The prepared reward file needs to be specified in `reward_setting`.
For details, please refer to a sample file ([config/setting.yaml](config/setting.yaml)).
If you want to pass any value to `calc_reward_from_objective_values` (e.g., weights for each value), add it in the config file.

### 4. Generate molecules

#### ChemTSv2 with single process mode :red_car:

```bash
chemtsv2 -c config/setting.yaml
```

#### ChemTSv2 with massive parallel mode :airplane:

```bash
mpiexec -n 4 chemtsv2-mp --config config/setting_mp.yaml
```

#### ChemTSv2 with Docker

```bash
docker build -t chemtsv2:1.0.0 .
docker run -u $(id -u):$(id -g) \
           --rm \
           --mount type=bind,source=./,target=/app/ \
           chemtsv2:1.0.0 \
           chemtsv2 -c config/setting.yaml
```

## Example usage :pushpin:

|Target|Reward|Config|Additional requirement|Ref.|
|---|---|---|---|---|
|LogP|[logP_reward.py](reward/logP_reward.py)|[setting.yaml](config/setting.yaml)|-|-|
|Jscore|[Jscore_reward.py](reward/Jscore_reward.py)|[setting_jscore.yaml](config/setting_jscore.yaml)|-|[^1]|
|Absorption wavelength|[chro_reward.py](reward/chro_reward.py)|[setting_chro.yaml](config/setting_chro.yaml)|Gaussian 16[^3]<br> via QCforever[^10]|[^4]|
|Absorption wavelength|[chro_gamess_reward.py](reward/chro_gamess_reward.py)|[setting_chro_gamess.yaml](config/setting_chro_gamess.yaml)|GAMESS 2022.2[^12] via QCforever[^10]||
|Upper-absorption & fluorescence<br> wavelength|[fluor_reward.py](reward/fluor_reward.py)|[setting_fluor.yaml](config/setting_fluor.yaml)|Gaussian 16[^3]<br> via QCforever[^10]|[^5]|
|Kinase inhibitory activities|[dscore_reward.py](reward/dscore_reward.py)|[setting_dscore.yaml](config/setting_dscore.yaml)|LightGBM[^6]|[^7]|
|Docking score|[Vina_binary_reward.py](reward/Vina_binary_reward.py)|[setting_vina_binary.yaml](config/setting_vina_binary.yaml)|AutoDock Vina[^8]|[^9]|
|Pharmacophore|[pharmacophore_reward.py](reward/pharmacophore_reward.py)|[setting_pharmacophore.yaml](config/setting_pharmacophore.yaml)|-|[^11]|
|gnina docking|[gnina_singularity_reward.py](reward/gnina_singularity_reward.py)|[setting_gnina_singularity.yaml](config/setting_gnina_singularity.yaml)|-|-|
|Linker generation|[Linker_logP_reward.py](reward/Linker_logP_reward.py)|[setting_linker.yaml](config/setting_linker.yaml)|-|-|

[^3]: Frisch, M. J. et al. Gaussian 16 Revision C.01. 2016; Gaussian Inc. Wallingford CT.
[^4]: Sumita, M., Yang, X., Ishihara, S., Tamura, R., & Tsuda, K. (2018). Hunting for Organic Molecules with Artificial Intelligence: Molecules Optimized for Desired Excitation Energies. <i>ACS Central Science</i>, 4(9), 1126–1133. https://doi.org/10.1021/acscentsci.8b00213
[^5]: Sumita, M., Terayama, K., Suzuki, N., Ishihara, S., Tamura, R., Chahal, M. K., Payne, D. T., Yoshizoe, K., & Tsuda, K. (2022). De novo creation of a naked eye–detectable fluorescent molecule based on quantum chemical computation and machine learning. <i>Science Advances</i>, 8(10). https://doi.org/10.1126/sciadv.abj3906
[^6]: Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., … Liu, T.-Y. (2017). Lightgbm: A highly efficient gradient boosting decision tree. <i>Advances in Neural Information Processing Systems</i>, 30, 3146–3154.
[^7]: Yoshizawa, T., Ishida, S., Sato, T., Ohta, M., Honma, T., & Terayama, K. (2022). Selective Inhibitor Design for Kinase Homologs Using Multiobjective Monte Carlo Tree Search. <i>Journal of Chemical Information and Modeling</i>, 62(22), 5351–5360. https://doi.org/10.1021/acs.jcim.2c00787
[^8]: Eberhardt, J., Santos-Martins, D., Tillack, A. F., & Forli, S. (2021). AutoDock Vina 1.2.0: New Docking Methods, Expanded Force Field, and Python Bindings. <i>Journal of Chemical Information and Modeling</i>, 61(8), 3891–3898. https://doi.org/10.1021/acs.jcim.1c00203
[^9]: Ma, B., Terayama, K., Matsumoto, S., Isaka, Y., Sasakura, Y., Iwata, H., Araki, M., & Okuno, Y. (2021). Structure-Based de Novo Molecular Generator Combined with Artificial Intelligence and Docking Simulations. <i>Journal of Chemical Information and Modeling</i>, 61(7), 3304–3313. https://doi.org/10.1021/acs.jcim.1c00679
[^10]: Sumita, M., Terayama, K., Tamura, R., & Tsuda, K. (2022). QCforever: A Quantum Chemistry Wrapper for Everyone to Use in Black-Box Optimization. <i>Journal of Chemical Information and Modeling</i>, 62(18), 4427–4434. https://doi.org/10.1021/acs.jcim.2c00812
[^11]: 石田祥一, 吉澤竜哉, 寺山慧 (2023). 深層学習と木探索に基づくde novo分子設計, <i>SAR News</i>, 44.
[^12]: Barca, Giuseppe M. J. et al. (2020). Recent developments in the general atomic and molecular electronic structure system. <i>The Journal of Chemical Physics</i>, 152(15), 154102. https://doi.org/10.1063/5.0005188

## Support option/function :pushpin:

|Option|Single process|Massive parallel|Description|
|---|---|---|---|
|`c_val`|:white_check_mark:|:white_check_mark:|Exploration parameter to balance the trade-off between exploration and exploitation. A larger value (e.g., 1.0) prioritizes exploration, and a smaller value (e.g., 0.1) prioritizes exploitation.|
|`threshold_type`|:white_check_mark:|:heavy_check_mark:|Threshold type to select how long (`hours`) or how many (`generation_num`) molecule generation to perform. Massive parallel mode currently supports only the how long (`hours`) option.|
|`hours`|:white_check_mark:|:white_check_mark:|Time for molecule generation in hours|
|`generation_num`|:white_check_mark:|:white_large_square:|Number of molecules to be generated. Please note that the specified number is usually exceeded.|
|`expansion_threshold`|:white_check_mark:|:white_large_square:|(Advanced) Expansion threshold of the cumulative probability. The default is set to 0.995.|
|`simulation_num`|:white_check_mark:|:white_large_square:|(Advanced) Number of rollout runs in one cycle of MCTS. The default is set to 3.|
|`flush_threshold`|:white_check_mark:|:white_large_square:|Threshold for saving the progress of a molecule generation. If the number of generated molecules exceeds the threshold value, the result is saved. The default is set to -1, and this represents no progress is to be saved.|
|Molecule filter|:white_check_mark:|:white_check_mark:|Molecule filter to skip reward calculation of unfavorable generated molecules. Please refer to filter/README.md for details.|
|RNN model replacement|:white_check_mark:|:white_check_mark:|Users can switch RNN models used in expansion and rollout steps of ChemTSv2. The model needs to be trained using Tensorflow. `model_json` specifies the JSON file that contains the architecture of the RNN model, and `model_weight` specifies the file in H5 format that contains a set of the values of the weights. `token` specifies the pickle file that contains the token list used when training an RNN model.|
|Reward replacement|:white_check_mark:|:white_check_mark:|Users can use any reward function as long as they follow the reward base class ([reward/reward.py](reward/reward.py)). Please refer to reward/README.md for details.|
|Policy replacement|:white_check_mark:|:white_large_square:|(Advanced) Users can use any policy function as long as they follow the policy base class ([policy/policy.py](policy/policy.py)). Please refer to policy/README.md for details.|
|Restart|:beginner:|:beginner:|Users can save the checkpoint file and restart from the file. If users want to save a checkpoint file, <b>(SP mode)</b> set `save_checkpoint` to True and specify the file name in `checkpoint_file`. If users want to restart from the checkpoint, set `restart` to True and specify the checkpoint file in `checkpoint_file`. <b>(MP mode)</b> under development.|

- :white_check_mark: indicates that the option/function is supported.
- :heavy_check_mark: indicates that the option/function is partially supported.
- :beginner: indicates that the option/function is beta version.
- :white_large_square: indicates that the option/function is NOT supported.

Filter functions are described in [filter/README.md](./filter).

## Advanced usage :pushpin:

### Extend user-specified SMILES

You can extend the SMILES string you input.
In this case, you need to put the atom you want to extend at the end of the string and run ChemTS with `--input_smiles` argument as follows.

```bash
chemtsv2 -c config/setting.yaml --input_smiles 'C1=C(C)N=CC(N)=C1C'
```

#### Specify the last atom of SMILES string using OpenBabel

[OpenBabel](https://github.com/openbabel/openbabel) can be used to rearrange a SMILES string so that the specified atom comes last.
For example, if you want to rearrange `Br` in `NC1=CC(Br)=CC=C1` to the last position, run the following command:

```bash
# obabel -:"<SMILES>" -osmi -xl <atom no.>
# Atom numbers correspond to the order of atoms in an input SMILES string.
# In this example, `Br` appears fifth, so we specify `5` as a <atom no.>.
obabel -:"NC1=CC(Br)=CC=C1" -osmi -xl 5
# output: Nc1cc(ccc1)Br
```

Please refer to the [official documentation](https://openbabel.org/docs/current/FileFormats/SMILES_format.html) for detailed usage.

### Train RNN models using your own dataset

If you want to use the RNN models trained on your own datasets, use [train_model/train_RNN.py](train_model/train_RNN.py) and [train_model/model_setting.yaml](train_model/model_setting.yaml) to train the models.
You need to prepare a dataset that only contains SMILES string and modify the path in `dataset` key in `model_setting.yaml`. And then, run the following command:

```bash
cd train_model/
python train_RNN.py -c model_setting.yaml
```

Please note that the current version of ChemTSv2 does not support the change for RNN model structures, and users can only change the parameters described in `model_setting.yaml`.

Once you train the RNN model, specify the path to the checkpoint and token files in `model_setting` and `token` keys in ChemTSv2 config files to run ChemTSv2 with your own RNN model.

### GPU acceleration

If you want to use GPU, run ChemTS with `--gpu GPU_ID` argument as follows.

```bash
chemtsv2 -c config/setting.yaml --gpu 0
```

If you want to use GPU only for reward calculations, run ChemTSv2 with `--gpu` and `--use_gpu_only_reward` arguments as follows.

```bash
chemtsv2 -c config/setting_gnina_singularity.yaml --gpu 0 --use_gpu_only_reward
```

## How to cite

```text
@article{Ishida2023,
  doi = {10.1002/wcms.1680},
  url = {https://doi.org/10.1002/wcms.1680},
  year = {2023},
  month = jul,
  publisher = {Wiley},
  author = {Shoichi Ishida and Tanuj Aasawat and Masato Sumita and Michio Katouda and Tatsuya Yoshizawa and Kazuki Yoshizoe and Koji Tsuda and Kei Terayama},
  title = {ChemTSv2: Functional molecular design using de novo molecule generator},
  journal = {{WIREs} Computational Molecular Science}
}
```

## License :pushpin:

This package is distributed under the MIT License.

## Contact :pushpin:

- Shoichi Ishida (ishida.sho.nm@yokohama-cu.ac.jp)
- Kei Terayama (terayama@yokohama-cu.ac.jp).
