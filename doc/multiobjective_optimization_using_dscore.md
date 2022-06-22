# Multiobjective optimization using Dscore

This document describes how to perform multiobjective optimization using Dscore, as implemented in the following paper.

## Reference

```
@article{Yoshizawa2022,
  doi = {10.26434/chemrxiv-2022-hcxx4-v2},
  url = {https://doi.org/10.26434/chemrxiv-2022-hcxx4-v2},
  year = {2022},
  month = jun,
  publisher = {American Chemical Society ({ACS})},
  journal = {ChemRxiv}
  author = {Tatsuya Yoshizawa and Shoichi Ishida and Tomohiro Sato and Masateru Ohta and Teruki Honma and Kei Terayama},
  title = {Selective Inhibitor Design for Kinase Homologs Using Multiobjective Monte Carlo Tree Search}
}
```

## How to run ChemTS with Dscore

1. Setup virtual environment (example).

    ```bash
    conda create -n chemts -c conda-forge python=3.7
    # switch a python virtual environment to `chemts`
    pip install --upgrade git+https://github.com/molecule-generator-collection/ChemTSv2.git
    pip install lightgbm
    ```

2. Clone this repository and move into it.

    ```bash
    git clone git@github.com:molecule-generator-collection/ChemTSv2.git
    cd ChemTSv2
    ```

3. (Optional) Adjust Dscore parameters.

   Adjust Dscore parameters according to how you want to optimize the properties of generated structures. Detailed instruction is provided [below](#How-to-adjust-Dscore-paramaters).
​
4. Generate molecules.

    ```bash
    chemtsv2 -c config/setting_dscore.yaml
    ```

## How to adjust Dscore paramaters

You need to specify a scaling function and weight for each objective in `./config/setting_dscore.yaml`, following the below format.

  ```yaml
  Dscore_parameters:
    EGFR: {type: max_gauss, alpha: 1, mu: 9, sigma: 2, weight: 8}
  ```

### Setting for scaling functions

For each objective, it is necessary to define a function that scales the raw objective value from 0 to 1.
In this repository, `max_gauss`, `min_gauss`, `minmax`, `rectangular`, and `identity` functions are provided.

- `max_gauss`: return 1 if x >= mu and 0–1 depending on a left-half Gaussian bell shape
  - parameters: mu, sigma, a (fixed to 1)
- `min_gauss`: return 1 if x <= mu and 0–1 depending on a right-half Gaussian bell shape
  - parameters: mu, sigma, a (fixed to 1)
- `minmax`: return min-max normalized value
  - parameters: min, max
- `rectangular`: return 1 if min <= x <= max and 0 otherwise
  - parameters: min, max
- `identity`: return x as is

For more information on scaling functions, please refer to the following [link](../chemtsv2/misc/README.md).

### Settings of weights
​
You can prioritize objectives to set importance weights (default=1) in `weight` key.
​

## Examples of paramater settings

### EGFR selective inhibitor design
​
To maximize inhibitory activity to EGFR while minimizing inhibitory activity to off-target proteins, `max_gauss` and `min_gauss` are set to EGFR and off-target proteins, respectively.

To equalize the weight of EGFR and ​the sum of the weights of 8 off-targets, the weight to EGFR is set to 8.
​
```yaml
Dscore_parameters:
  EGFR: {type: max_gauss, alpha: 1, mu: 9, sigma: 2, weight: 8}
  ERBB2: {type: min_gauss, alpha: 1, mu: 2, sigma: 2, weight: 1}
  ABL: {type: min_gauss, alpha: 1, mu: 2, sigma: 2, weight: 1}
  SRC: {type: min_gauss, alpha: 1, mu: 2, sigma: 2, weight: 1}
  LCK: {type: min_gauss, alpha: 1, mu: 2, sigma: 2, weight: 1}
  PDGFRbeta: {type: min_gauss, alpha: 1, mu: 2, sigma: 2, weight: 1}
  VEGFR2: {type: min_gauss, alpha: 1, mu: 2, sigma: 2, weight: 1}
  FGFR1: {type: min_gauss, alpha: 1, mu: 2, sigma: 2, weight: 1}
  EPHB4: {type: min_gauss, alpha: 1, mu: 2, sigma: 2, weight: 1}
  Solubility: {type: max_gauss, alpha: 1, mu: -2, sigma: 0.6, weight: 1}
  Permeability: {type: max_gauss, alpha: 1, mu: 1, sigma: 1, weight: 1}
  Metabolic_stability: {type: max_gauss, alpha: 1, mu: 75, sigma: 20, weight: 1}
  Toxicity: {type: min_gauss, alpha: 1, mu: 5.5, sigma: 0.5, weight: 1}
  # SAscore is made negative when scaling because a smaller value is more desirable.
  SAscore: {type: max_gauss, alpha: 1, mu: -3, sigma: 2, weight: 1}
  QED: {type: max_gauss, alpha: 1, mu: 0.8, sigma: 0.25, weight: 1}
  molecular_weight: {type: rectangular, min: 200, max: 600, weight: 1}
  tox_alert: {type: identity, weight: 1}
  has_chembl_substruct: {type: identity, weight: 1}
```
​
### EGFR selective inhibitor design without considering off-target proteins
​
You can perform the inhibitor design without considering off-target proteins by setting the weights for off-target proteins to 0.
​
```yaml
Dscore_parameters:
  EGFR: {type: max_gauss, alpha: 1, mu: 9, sigma: 2, weight: 8}
  ERBB2: {type: min_gauss, alpha: 1, mu: 2, sigma: 2, weight: 0}
  ABL: {type: min_gauss, alpha: 1, mu: 2, sigma: 2, weight: 0}
  SRC: {type: min_gauss, alpha: 1, mu: 2, sigma: 2, weight: 0}
  LCK: {type: min_gauss, alpha: 1, mu: 2, sigma: 2, weight: 0}
  PDGFRbeta: {type: min_gauss, alpha: 1, mu: 2, sigma: 2, weight: 0}
  VEGFR2: {type: min_gauss, alpha: 1, mu: 2, sigma: 2, weight: 0}
  FGFR1: {type: min_gauss, alpha: 1, mu: 2, sigma: 2, weight: 0}
  EPHB4: {type: min_gauss, alpha: 1, mu: 2, sigma: 2, weight: 0}
  Solubility: {type: max_gauss, alpha: 1, mu: -2, sigma: 0.6, weight: 1}
  Permeability: {type: max_gauss, alpha: 1, mu: 1, sigma: 1, weight: 1}
  Metabolic_stability: {type: max_gauss, alpha: 1, mu: 75, sigma: 20, weight: 1}
  Toxicity: {type: min_gauss, alpha: 1, mu: 5.5, sigma: 0.5, weight: 1}
  # SAscore is made negative when scaling because a smaller value is more desirable.
  SAscore: {type: max_gauss, alpha: 1, mu: -3, sigma: 2, weight: 1}
  QED: {type: max_gauss, alpha: 1, mu: 0.8, sigma: 0.25, weight: 1}
  molecular_weight: {type: rectangular, min: 200, max: 600, weight: 1}
  tox_alert: {type: identity, weight: 1}
  has_chembl_substruct: {type: identity, weight: 1}
```
​
### ERBB2 selective inhibitor design
​
Even if the target protein changes, the same way as selective EGFR inhibitors design can be used.
​
```yaml
Dscore_parameters:
  EGFR: {type: min_gauss, alpha: 1, mu: 2, sigma: 2, weight: 1}
  ERBB2: {type: max_gauss, alpha: 1, mu: 9, sigma: 2, weight: 8}
  ABL: {type: min_gauss, alpha: 1, mu: 2, sigma: 2, weight: 1}
  SRC: {type: min_gauss, alpha: 1, mu: 2, sigma: 2, weight: 1}
  LCK: {type: min_gauss, alpha: 1, mu: 2, sigma: 2, weight: 1}
  PDGFRbeta: {type: min_gauss, alpha: 1, mu: 2, sigma: 2, weight: 1}
  VEGFR2: {type: min_gauss, alpha: 1, mu: 2, sigma: 2, weight: 1}
  FGFR1: {type: min_gauss, alpha: 1, mu: 2, sigma: 2, weight: 1}
  EPHB4: {type: min_gauss, alpha: 1, mu: 2, sigma: 2, weight: 1}
  Solubility: {type: max_gauss, alpha: 1, mu: -2, sigma: 0.6, weight: 1}
  Permeability: {type: max_gauss, alpha: 1, mu: 1, sigma: 1, weight: 1}
  Metabolic_stability: {type: max_gauss, alpha: 1, mu: 75, sigma: 20, weight: 1}
  Toxicity: {type: min_gauss, alpha: 1, mu: 5.5, sigma: 0.5, weight: 1}
  # SAscore is made negative when scaling because a smaller value is more desirable.
  SAscore: {type: max_gauss, alpha: 1, mu: -3, sigma: 2, weight: 1}
  QED: {type: max_gauss, alpha: 1, mu: 0.8, sigma: 0.25, weight: 1}
  molecular_weight: {type: rectangular, min: 200, max: 600, weight: 1}
  tox_alert: {type: identity, weight: 1}
  has_chembl_substruct: {type: identity, weight: 1}
```
