#!/bin/sh
if [ $SLURM_PROCID -eq 0 ]
then
    date
    /tmp/ikadochn/env-hop/bin/python optimize.py --n_tracks 10 --max_budget 1000 --n_iterations=500 --run_id "$SLURM_JOB_ID" --nic_name eth0 --shared_directory ./workdir --n_workers 80
    date
else
    /tmp/ikadochn/env-hop/bin/python optimize.py --n_tracks 10 --max_budget 1000 --n_iterations=500 --run_id "$SLURM_JOB_ID" --nic_name eth0 --shared_directory ./workdir --worker
fi

