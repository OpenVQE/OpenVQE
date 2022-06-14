#!/bin/bash
#SBATCH -c 4
#SBATCH -o job.%j
#SBATCH -p 
#SBATCH --time=infinite
#SBATCH -o slurm-%j.out
source /usr/local/bin/qatenv
time python3 ../mains/main_quccsd_active_space.py