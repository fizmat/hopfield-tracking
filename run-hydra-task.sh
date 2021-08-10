#!/bin/sh
#SBATCH --partition=cpu
#SBATCH --job-name=optimize
#SBATCH --ntasks=80
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=512
#SBATCH --no-requeue
srun ./hydra-task.sh 
