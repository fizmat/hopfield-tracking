#!/bin/bash

n_tracks=${n_tracks:-10}
min_budget=${min_budget:-1}
max_budget=${max_budget:-10}
hopfield_steps=${hopfield_steps:-10}
n_iterations=${n_itreations:-10}
n_workers=${n_workers:-1}

if [ $SLURM_PROCID -eq 0 ]
then
    date
    "/tmp/$USER/miniconda3/envs/hopfield-tracking/bin/python" optimize.py \
    --n_tracks $n_tracks --hopfield_steps $hopfield_steps \
    --max_budget $max_budget --min_budget $min_budget \
    --n_iterations $n_iterations \
    --run_id "$SLURM_JOB_ID" \
    --shared_directory ./workdir --n_workers $n_workers
    date
else
    "/tmp/$USER/miniconda3/envs/hopfield-tracking/bin/python" optimize.py \
    --n_tracks $n_tracks --hopfield_steps $hopfield_steps \
    --max_budget $max_budget --min_budget $min_budget \
    --n_iterations $n_iterations \
    --run_id "$SLURM_JOB_ID" \
    --shared_directory ./workdir --worker
fi
