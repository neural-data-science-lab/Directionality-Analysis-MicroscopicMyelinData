#!/bin/bash -l
#
# Single-core example job script for MPCDF Cobra.
# In addition to the Python example shown here, the script
# is valid for any single-threaded program, including
# sequential Matlab, Mathematica, Julia, and similar cases.
#
#SBATCH -o HLR_18.out
#SBATCH -e HLR_18.err
#SBATCH --ntasks=1         # launch job on a single core
#SBATCH --cpus-per-task=8  #   on a shared node
#SBATCH --mem=32000MB       # memory limit for the job
#SBATCH --time=12:00:00

module purge
module load gcc/10 impi/2021.2
module load anaconda/3/2021.05

# Set number of OMP threads to fit the number of available cpus, if applicable.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun python3 ./directionality_pipeline_headless.py l PR018_l_ACx /ptmp/muellerg/PR018_l_ACx/ 92 25 False True True -260 -360 0 0 30 2160 2200 310
srun python3 ./directionality_pipeline_headless.py l PR018_l_ACx /ptmp/muellerg/PR018_l_ACx/ 92 25 True False True -260 -360 0 0 30 2160 2200 310
srun python3 ./directionality_pipeline_headless.py l PR018_l_ACx /ptmp/muellerg/PR018_l_ACx/ 37 25 False True True -260 -360 0 0 30 2160 2200 310
srun python3 ./directionality_pipeline_headless.py l PR018_l_ACx /ptmp/muellerg/PR018_l_ACx/ 37 25 True False True -260 -360 0 0 30 2160 2200 310

srun python3 ./directionality_pipeline_headless.py r PR018_r_ACx /ptmp/muellerg/PR018_r_ACx/ 92 25 False True True 360 380 0 0 30 2160 2200 310
srun python3 ./directionality_pipeline_headless.py r PR018_r_ACx /ptmp/muellerg/PR018_r_ACx/ 92 25 True False True 360 380 0 0 30 2160 2200 310
srun python3 ./directionality_pipeline_headless.py r PR018_r_ACx /ptmp/muellerg/PR018_r_ACx/ 37 25 False True True 360 380 0 0 30 2160 2200 310
srun python3 ./directionality_pipeline_headless.py r PR018_r_ACx /ptmp/muellerg/PR018_r_ACx/ 37 25 True False True 360 380 0 0 30 2160 2200 310
srun python3 ./plots_Advanced_headless.py PR018 /ptmp/muellerg/ 92
srun python3 ./plots_Advanced_headless.py PR018 /ptmp/muellerg/ 37
