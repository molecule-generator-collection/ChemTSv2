import os
import shutil

from setuptools import setup


path = os.path.dirname(os.path.abspath(__file__))
shutil.copyfile(f"{path}/run.py", f"{path}/chemtsv2/run.py")

INSTALL_REQUIRES = [
    'tensorflow==2.5',
    'rdkit-pypi==2021.03.5',
    'matplotlib',
    'pyyaml',
    'pandas']
PACKAGES = [
    'chemtsv2',
    'chemtsv2.misc']
CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3.7',
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent"]
CONSOLE_SCRIPTS = [
    "chemtsv2 = chemtsv2.run:main"]

setup(
    name="ChemTSv2",
    author="Shoichi Ishida",
    author_email="ishida.sho.nm@yokohama-cu.ac.jp",
    maintainer="Shoichi Ishida",
    maintainer_email="ishida.sho.nm@yokohama-cu.ac.jp",
    description="ChemTSv2",
    license="MIT LIcense",
    url="https://github.com/molecule-generator-collection/ChemTSv2",
    version="0.1",
    download_url="https://github.com/molecule-generator-collection/ChemTSv2",
    python_requires=">=3.7",
    install_requires=INSTALL_REQUIRES,
    packages=PACKAGES,
    entry_points={'console_scripts': CONSOLE_SCRIPTS},
    classifiers=CLASSIFIERS
)