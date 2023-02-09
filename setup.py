""" ChemTSv2 is a flexible and versatile molecule generator based on reinforcement learning with natural language processing.

ChemTSv2 is a refined and extended version of [ChemTS](https://www.tandfonline.com/doi/full/10.1080/14686996.2017.1401424).
The original implementation is available at https://github.com/tsudalab/ChemTS.

It provides:

- easy-to-run interface by using only a configuration file
- easy-to-define framework for users' any reward function, molecular filter, and tree policy
- various usage examples in the GitHub repository

"""
import os
import shutil

from setuptools import setup


path = os.path.dirname(os.path.abspath(__file__))
shutil.copyfile(f"{path}/run.py", f"{path}/chemtsv2/run.py")
shutil.copyfile(f"{path}/run_mp.py", f"{path}/chemtsv2/run_mp.py")

DOCLINES = (__doc__ or '').split('\n')
INSTALL_REQUIRES = [
    'tensorflow==2.5',
    'numpy~=1.19.2',
    'protobuf~=3.9.2',
    'rdkit-pypi==2021.03.5',
    'pyyaml',
    'pandas',
    'joblib']
PACKAGES = [
    'chemtsv2',
    'chemtsv2.misc']
CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3.7',
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent"]
CONSOLE_SCRIPTS = [
    "chemtsv2 = chemtsv2.run:main",
    "chemtsv2-mp = chemtsv2.run_mp:main"
    ]

setup(
    name="chemtsv2",
    author="Shoichi Ishida",
    author_email="ishida.sho.nm@yokohama-cu.ac.jp",
    maintainer="Shoichi Ishida",
    maintainer_email="ishida.sho.nm@yokohama-cu.ac.jp",
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    long_description_content_type="text/markdown",
    license="MIT LIcense",
    url="https://github.com/molecule-generator-collection/ChemTSv2",
    version="0.9.10",
    download_url="https://github.com/molecule-generator-collection/ChemTSv2",
    python_requires=">=3.7",
    install_requires=INSTALL_REQUIRES,
    packages=PACKAGES,
    entry_points={'console_scripts': CONSOLE_SCRIPTS},
    classifiers=CLASSIFIERS
)
