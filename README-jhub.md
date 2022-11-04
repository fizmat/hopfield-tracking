# Running on Hybrilit jupyter servers

## First setup
Run in a web terminal on jhub2.
(The setup should also work on jhub and jlabhpc, but it's better to ensure compatibility with the server that will run most of the jupyter notebooks because it has GPUs for visualization).

**Not** over ssh on hydra, because Conda is only installed on jupyter servers.


[jupyter_rfb](https://github.com/vispy/jupyter_rfb) is a jupyter plugin needed by vispy (the GPU visualization library). 
It is server-side, not kernel-side, so it needs to be installed on the "system" python.
As we are a simple user on jhub2 we can only install it for the user.
No environment should be activated.

```shell
pip install jupyter_rfb simplejpeg --user
```

cd to the repository root directory

```shell
conda env create -f env-jhub.yml
conda activate hopfield-tracking
```

If conda activate does not work because conda is not initialized for your shell:
1. `conda init`
2. quit and start your shell again
3. cd to the repository root directory

Add to the local git config (.git/config) a filter to remove outputs from jupyter notebooks 
```shell
nb-clean add-filter --remove-empty-cells
```

Add the current env as a kernel to the server jupyter, by adding it to ~/.local/share/jupyter/kernels/
```shell
python -m ipykernel install --user --name=hopfield-tracking
```

Use this kernel 'hopfield-tracking' to run all jupyter notebooks. 

## Every time before working in the shell

It's even needed before using git, to make nb-clean available for the filter.

```shell
conda activate hopfield-tracking
```

