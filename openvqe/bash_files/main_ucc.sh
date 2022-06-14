#!/bin/bash
#SBATCH -c 6
#SBATCH -o job.%j
#SBATCH -p QLM
#SBATCH --time=infinite
#SBATCH -o slurm-%j.out
source /usr/local/bin/qatenv
time python3 ../mains/main_ucc.py