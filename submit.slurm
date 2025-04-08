#!/bin/bash

# Job settings
#SBATCH -A oekochem.oekochem
#SBATCH -p smp
#SBATCH -t 01:00:00
#SBATCH --qos=12h
#SBATCH --ntasks=1

#SBATCH --job-name=test_array
#SBATCH --output=logs/output_%A_%a.out
#SBATCH --error=logs/error_%A_%a.err
#SBATCH --array=0-99
#SBATCH --cpus-per-task=1

# Load your Python environment if needed
module load conda
conda activate ocean_clustering_and_validation

# Call the Python script with the SLURM_ARRAY_TASK_ID as the argument
python test.py ${SLURM_ARRAY_TASK_ID}

