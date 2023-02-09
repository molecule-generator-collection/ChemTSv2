# README for filter function

## How to define filter function

ChemTSv2 frexibly accept user-defined filter file written in Python3. 
A user-defined class should inherit from a [Filter base class](./filter.py). 
The filter class contains one static method: `check()`. 
The method takes a `Mol` object of RDKit and a configuration parameter object in a dictionary format and returns boolean value. 
In the filter function, the boolean value `True` indicates that an input molecule satisfies the filtering criteria, and `False` indicates otherwise. 
The molecule satisfying filter criteria continues to a reward calculation step. 

Below is a simple example. 
```python
from filter.filter import Filter

class RingSizeFilter(Filter):
    def check(mol, conf):
        ri = mol.GetRingInfo()
        max_ring_size = max((len(r) for r in ri.AtomRings()), default=0)
        return max_ring_size <= conf['ring_size_filter']['threshold']
```
This class filter a molecule whose largest ring structure exceeds the specified threshold in a configuration file.

## List of predefined filters

### lipinski_filter

`type` option: [`rule_of_5`, `rule_of_3`]
Lipinski's rule of 5[^1] and 3[^2].
[^1]: Lipinski, C. A., Lombardo, F., Dominy, B. W., & Feeney, P. J. (1997). Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings. Advanced Drug Delivery Reviews, 23(1–3), 3–25. https://doi.org/10.1016/s0169-409x(96)00423-1
[^2]: Jhoti, H., Williams, G., Rees, D. C., & Murray, C. W. (2013). The “rule of three” for fragment-based drug discovery: where are we now? Nature Reviews Drug Discovery, 12(8), 644–644. https://doi.org/10.1038/nrd3926-c1

### pains_filter

`type` option: [`pains_a`, `pains_b`, `pains_c`] (multiple choice)
Pan Assay Interference Compounds (PAINS)[^3]
[^3]: Baell, J. B., & Holloway, G. A. (2010). New Substructure Filters for Removal of Pan Assay Interference Compounds (PAINS) from Screening Libraries and for Their Exclusion in Bioassays. Journal of Medicinal Chemistry, 53(7), 2719–2740. https://doi.org/10.1021/jm901137j

### pubchem_filter

A filter, reported in Ma et al.[^4], based on the frequency of occurrence of molecular patterns in the PubChem database.
[^4]: Ma, B., Terayama, K., Matsumoto, S., Isaka, Y., Sasakura, Y., Iwata, H., Araki, M., & Okuno, Y. (2021). Structure-Based de Novo Molecular Generator Combined with Artificial Intelligence and Docking Simulations. Journal of Chemical Information and Modeling, 61(7), 3304–3313. https://doi.org/10.1021/acs.jcim.1c00679

### radical_filter

Filter a molecule with radical electrons.

### ring_size_filter

`threshold` option: `6 (default)`

### sascore_filter

`threshold` option: `3.5 (default)`
Synthetic accessibility score (SAscore)[^5]
[^5]: Ertl, P., & Schuffenhauer, A. (2009). Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions. Journal of Cheminformatics, 1(1). https://doi.org/10.1186/1758-2946-1-8
