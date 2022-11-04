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
- [sbatch](sbatch.ipynb): run distributed optimization on jlab-hpc

### Testing

`pytest` - unit tests

`tox` - run pytest in multiple environments

## Hydra
[Instructions](README-jhub.md) for setting up on the hybrilit jupyter server jlab2.