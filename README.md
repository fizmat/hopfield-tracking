# hopfield-tracking

Some experiments with Hopfield networks for tracking

## Local development

### Prepare

```shell
conda env create
conda activate hopfield-tracking
nb-clean add-filter --remove-empty-cells
```

### Jupyter notebooks

`jupyter lab`

- [dataset_stats](dataset_stats.ipynb): visualize distribution of tracks/event and hits/event
- [demo_datasets](demo_datasets.ipynb): visualize hits/tracks of one event per dataset
- [demo_seg](demo_seg.ipynb): compare different segment generation methods
- [demo_track_seg](demo_track_seg.ipynb): compare different track-segment generators
- [stat_seg_length](stat_seg_length.ipynb): visualize distribution of segment length to predict neighbor-filtering effectiveness
- [stat_seg_neighbors](stat_seg_neighbors.ipynb): visualize segment length in a more detailed but less efficient way
- [profile_seg_pools](profile_seg_pools.ipynb): compare performance of parallelization methods (for stat_seg_neighbors): processes are best  
- [demo_event](demo_event.ipynb): demonstrate hopfield tracking for one event
- [hydra-setup](hydra-setup.ipynb): prepare to run on jhub2 and jlab-hpc
- [sbatch](sbatch.ipynb): run distributed optimization on jlab-hpc

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
