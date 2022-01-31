#!/bin/bash -l
#
# Single-core example job script for MPCDF Cobra.
# In addition to the Python example shown here, the script
# is valid for any single-threaded program, including
# sequential Matlab, Mathematica, Julia, and similar cases.
#
#SBATCH -o HLR_1.out
#SBATCH -e HLR_1.err
#SBATCH --ntasks=1         # launch job on a single core
#SBATCH --cpus-per-task=1  #   on a shared node
#SBATCH --mem=32000MB       # memory limit for the job
#SBATCH --time=24:00:00

module purge
module load gcc/10 impi/2021.2
module load anaconda/3/2021.05

# Set number of OMP threads to fit the number of available cpus, if applicable.
export OMP_NUM_THREADS=1

srun /ptmp/muellerg/Fiji.app/ImageJ-linux64 --headless --console -macro BigStitcher_headless_console.ijm "/ptmp/muellerg/PR012_11-53-16/ PR012_11-53-16"
