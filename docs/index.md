# Home

ChemTSv2[^1] is a refined and extended version of ChemTS[^2] and MPChemTS[^3].
The original implementations are available at [tsudalab/ChemTS][u1] and [yoshizoe/mp-chemts][u2], respectively.

ChemTSv2 provides:

- easy-to-run interface by using only a configuration file
- easy-to-define framework for users' any reward function, molecular filter, and tree policy
- various usage examples in the GitHub repository

[^1]: Ishida, S. and Aasawat, T. and Sumita, M. and Katouda, M. and Yoshizawa, T. and Yoshizoe, K. and Tsuda, K. and Terayama, K. (2023). ChemTSv2: Functional molecular design using de novo molecule generator. <i>WIREs Computational Molecular Science</i> https://wires.onlinelibrary.wiley.com/doi/10.1002/wcms.1680

[^2]: Yang, X., Zhang, J., Yoshizoe, K., Terayama, K., & Tsuda, K. (2017). ChemTS: an efficient python library for de novo molecular generation. <i>Science and Technology of Advanced Materials</i>, 18(1), 972–976. https://doi.org/10.1080/14686996.2017.1401424

[^3]: Yang, X., Aasawat, T., & Yoshizoe, K. (2021). Practical Massively Parallel Monte-Carlo Tree Search Applied to Molecular Design. <i>In International Conference on Learning Representations</i>. https://openreview.net/forum?id=6k7VdojAIK

[u1]: https://github.com/tsudalab/ChemTS

[u2]: https://github.com/yoshizoe/mp-chemts

## Installation

!!! info
    - Please set up a `Python 3.11` environment to use ChemTSv2.  
    - `OpenMPI` or `MPICH` must be installed on your server to use ChemTSv2 with massive parallel mode.

##### Single process mode

```bash
pip install chemtsv2
```

##### Massive parallel mode

```bash
pip install chemtsv2[mp]
```

## How to run ChemTSv2

!!! example

    Clone this repository and move into it.

    ```bash
    git clone git@github.com:molecule-generator-collection/ChemTSv2.git
    cd ChemTSv2
    ```

    === "Single process mode"
        ```bash
        chemtsv2 -c config/setting.yaml
        ```

    === "Massive parallel mode"
        ```bash
        mpiexec -n 4 chemtsv2-mp --config config/setting_mp.yaml
        ```

## How to cite

```bibtex
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
