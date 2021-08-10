#!/bin/sh
#SBATCH --partition=cpu
#SBATCH --job-name=prepare-env
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=1024
#SBATCH --no-requeue
srun ./hydra-env.sh 
