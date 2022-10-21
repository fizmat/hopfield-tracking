# Minimal Win7 setup

How to run on Win7-x64 with 2Gb of memory?

## Make sure Git is installed, clone this repo

1. Is git bash available in the start menu? Does the `git` command work in any shell?
2. If not, install git for windows (both links point to the same setup):
    - https://gitforwindows.org/
    - https://git-scm.com/download/win
    - Setup asks many questions, default answers are ususally OK.
3. `git clone {repo url}` in any shell (cmd, Powershell, git bash)

## 1. Without conda: python + venv

The virtual environment will take up ~ 1 Gb

### Install python 3.8
- Python 3.9 is not compatible with Windows 7 
- Python 3.7 and 3.8 installers require Windows 7 update KB2533623
    - it is no longer available, but [was replaced](https://www.reddit.com/r/windows/comments/ik7sp7/does_anybody_have_the_kb2533623_update_for/) by update KB3063858
    - 32-bit: https://www.microsoft.com/en-us/download/details.aspx?id=47409
    - 64-bit: https://www.microsoft.com/en-us/download/details.aspx?id=47442
- The latest release of Python 3.7 providing a binary installer is 3.7.9
    - https://www.python.org/downloads/release/python-379/
    - 3.7.6 did not run at all for me
    - in 3.7.7-3.7.9 creating a new venv failed
- The latest release of Python 3.8 providing a binary installer is 3.8.10
    - https://www.python.org/downloads/release/python-3810/

### Create virtual environment
1. Find the full path to your python 3.8 executable
    - for example `C:\Users\{username}\AppData\Local\Programs\Python38\python.exe`
2. Create and activate virtual python environment
    - in CMD:
        - `cd {local project repo directory}`
        - `{python.exe path} -m venv venv`
        - `venv\Scripts\activate.bat`
        - `pip install -r requirements.txt`
    - in git bash:
        - `cd {local project repo directory}`
        - `{python.exe path} -m venv venv`
        - `. venv/Scripts/activate`
        - `pip install -r requirements.txt`
3. `pytest` to check what works

## 2. With conda

The conda environment will take up ~ 5 Gb

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
- Cleanup to minimize disk usage: `conda clean --all`
- activate the new environment `conda activate hopfield-tracking`
- `pytest` to check what works
