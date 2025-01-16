# Usage Examples

|Target|Reward|Config|Additional requirement|Ref.|
|---|---|---|---|---|
|LogP|[logP_reward.py][r1]|[setting.yaml][c1]|-|-|
|Jscore|[Jscore_reward.py][r2]|[setting_jscore.yaml][c2]|-|[^2]|
|Absorption wavelength|[chro_reward.py][r3]|[setting_chro.yaml][c3]|Gaussian 16[^3]<br> via QCforever[^10]|[^4]|
|Absorption wavelength|[chro_gamess_reward.py][r4]|[setting_chro_gamess.yaml][c4]|GAMESS 2022.2[^12] via QCforever[^10]||
|Upper-absorption & fluorescence<br> wavelength|[fluor_reward.py][r5]|[setting_fluor.yaml][c5]|Gaussian 16[^3]<br> via QCforever[^10]|[^5]|
|Kinase inhibitory activities|[dscore_reward.py][r6]|[setting_dscore.yaml][c6]|LightGBM[^6]|[^7]|
|Docking score|[Vina_binary_reward.py][r7]|[setting_vina_binary.yaml][c7]|AutoDock Vina[^8]|[^9]|
|Pharmacophore|[pharmacophore_reward.py][r8]|[setting_pharmacophore.yaml][c8]|-|[^11]|
|gnina docking|[gnina_singularity_reward.py][r9]|[setting_gnina_singularity.yaml][c9]|-|-|
|Linker generation|[Linker_logP_reward.py][r10]|[setting_linker.yaml][c10]|-|-|

[^2]: Yang, X., Zhang, J., Yoshizoe, K., Terayama, K., & Tsuda, K. (2017). ChemTS: an efficient python library for de novo molecular generation. <i>Science and Technology of Advanced Materials</i>, 18(1), 972–976. https://doi.org/10.1080/14686996.2017.1401424
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

[r1]: https://github.com/molecule-generator-collection/ChemTSv2/blob/master/reward/logP_reward.py
[r2]: https://github.com/molecule-generator-collection/ChemTSv2/blob/master/reward/Jscore_reward.py
[r3]: https://github.com/molecule-generator-collection/ChemTSv2/blob/master/reward/chro_reward.py
[r4]: https://github.com/molecule-generator-collection/ChemTSv2/blob/master/reward/chro_gamess_reward.py
[r5]: https://github.com/molecule-generator-collection/ChemTSv2/blob/master/reward/fluor_reward.py
[r6]: https://github.com/molecule-generator-collection/ChemTSv2/blob/master/reward/dscore_reward.py
[r7]: https://github.com/molecule-generator-collection/ChemTSv2/blob/master/reward/Vina_binary_reward.py
[r8]: https://github.com/molecule-generator-collection/ChemTSv2/blob/master/reward/pharmacophore_reward.py
[r9]: https://github.com/molecule-generator-collection/ChemTSv2/blob/master/reward/gnina_singularity_reward.py
[r10]: https://github.com/molecule-generator-collection/ChemTSv2/blob/master/reward/Linker_logP_reward.py

[c1]: https://github.com/molecule-generator-collection/ChemTSv2/blob/master/config/setting.yaml
[c2]: https://github.com/molecule-generator-collection/ChemTSv2/blob/master/config/setting_jscore.yaml
[c3]: https://github.com/molecule-generator-collection/ChemTSv2/blob/master/config/setting_chro.yaml
[c4]: https://github.com/molecule-generator-collection/ChemTSv2/blob/master/config/setting_chro_gamess.yaml
[c5]: https://github.com/molecule-generator-collection/ChemTSv2/blob/master/config/setting_fluor.yaml
[c6]: https://github.com/molecule-generator-collection/ChemTSv2/blob/master/config/setting_dscore.yaml
[c7]: https://github.com/molecule-generator-collection/ChemTSv2/blob/master/config/setting_vina_binary.yaml
[c8]: https://github.com/molecule-generator-collection/ChemTSv2/blob/master/config/setting_pharmacophore.yaml
[c9]: https://github.com/molecule-generator-collection/ChemTSv2/blob/master/config/setting_gnina_singularity.yaml
[c10]: https://github.com/molecule-generator-collection/ChemTSv2/blob/master/config/setting_linker.yaml
