#!/bin/bash
#SBATCH -c 1
#SBATCH -o job.%j
#SBATCH -p 
#SBATCH --time=infinite
#SBATCH -o slurm-%j.out
source /usr/local/bin/qatenv
time python3 ../mains/main_qubit_adapt.py