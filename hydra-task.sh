#!/bin/bash

n_tracks=${n_tracks:-10}
max_budget=${max_budget:-100}
n_iterations=${n_itreations:-10}
n_workers=${n_workers:-1}

if [ $SLURM_PROCID -eq 0 ]
then
    date
    "/tmp/$USER/env-hop/bin/python" optimize.py --n_tracks $n_tracks \
    --max_budget $max_budget --n_iterations $n_iterations \
    --run_id "$SLURM_JOB_ID" --nic_name enp4s0f0 \
    --shared_directory ./workdir --n_workers $n_workers
    date
else
    "/tmp/$USER/env-hop/bin/python" optimize.py --n_tracks $n_tracks \
    --max_budget $max_budget --n_iterations $n_iterations \
    --run_id "$SLURM_JOB_ID" --nic_name enp4s0f0 \
    --shared_directory ./workdir --worker
fi

