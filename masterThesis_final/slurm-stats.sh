#!/bin/bash -l
#
# Single-core example job script for MPCDF Cobra.
# In addition to the Python example shown here, the script
# is valid for any single-threaded program, including
# sequential Matlab, Mathematica, Julia, and similar cases.
#
#SBATCH -o CAM_lag.out
#SBATCH -e CAM_lag.err
#SBATCH --ntasks=1         # launch job on a single core
#SBATCH --cpus-per-task=8  #   on a shared node
#SBATCH --mem=32000MB       # memory limit for the job
#SBATCH --time=24:00:00

module purge
module load gcc/10 impi/2021.2
module load R/4.1.2

# Set number of OMP threads to fit the number of available cpus, if applicable.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} 

srun Rscript ./CAM_lagTime.R 

