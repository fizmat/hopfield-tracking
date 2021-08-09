# hopfield-tracking

Some experiments with Hopfield networks for tracking

## Local development

### Prepare

```shell
conda env create
conda activate hopfield-tracking
nb-clean add-filter --remove-empty-cells
```

### Experiment

- easy generated tracks: `jupyter lab main.ipynb`
- noisy real data: `jupyter lab real-events.ipynb`

### Test locally

`pytest`

## Jhub on hybrilit

### First setup

```shell
wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
## !! install to /tmp/$USER/miniconda3 !!
bash Miniconda3-latest-Linux-x86_64.sh
/tmp/$USER/miniconda3/bin/conda env create -f env-jhub.yml
/tmp/$USER/miniconda3/envs/hopfield-tracking/bin/nb-clean add-filter --remove-empty-cells
git config --local --add filter.nb-clean.clean "/tmp/$USER/miniconda3/envs/hopfield-tracking/bin/nb-clean clean --remove-empty-cells"
/tmp/$USER/miniconda3/envs/hopfield-tracking/bin/python -m ipykernel install --user --name=hopfield-tracking
tar -cf hopfield-tracking.env.tar.lz4 -I lz4 -C /tmp/$USER/miniconda3/envs hopfield-tracking
tar -cf miniconda3.tar.lz4 -I lz4 -C /tmp/$USER/ miniconda3
```

### Test on hydra client node

```shell
sh hydra-env.sh
/tmp/$USER/miniconda3/envs/hopfield-tracking/bin/pytest
```

### Experiment interactively

- <https://jhub.jinr.ru>
- main.ipynb
- real-events.ipynb

### Run in Slurm

- <https://jlabhpc.jinr.ru>
- sbatch.ipynb
