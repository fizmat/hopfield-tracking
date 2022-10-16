# Minimal Win7 setup

How to run on Win7-x64 with 2Gb of memory?

## Make sure Git is installed, clone this repo

1. Is git bash available in the start menu? Does the `git` command work in any shell?
2. If not, install git for windows (both links point to the same setup):
    - https://gitforwindows.org/
    - https://git-scm.com/download/win
    - Setup asks many questions, default answers are ususally OK.
3. `git clone {repo url}` in any shell (cmd, Powershell, git bash)

## Conda

### Install miniconda
- Visit https://repo.anaconda.com/miniconda/
- Download and install version Miniconda3-py37_4.8.2
  - Any newer throws an error when installing 
- Start menu -> Anaconda Powershell Prompt
  - Update base conda environment `conda update --all`
  - Install experimental mamba solver (it's faster and uses less memory) `conda install conda-libmamba-solver`

### Create conda environment
In Anaconda Powershell Prompt
- `cd {local project repo directory}`
- `conda env create --experimental-solver=libmamba -f environment-win7.yml`
- activate the new environment `conda activate hopfield-tracking`
- `pytest` to check what works
