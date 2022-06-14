#!/bin/bash
#SBATCH -c 2
#SBATCH -o job.%j
#SBATCH -p 
#SBATCH --time=infinite
#SBATCH -o slurm-%j.out
source /usr/local/bin/qatenv
time python3 ../mains/main_ucc_active_space.py