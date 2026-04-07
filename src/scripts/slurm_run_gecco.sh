#!/bin/bash                                                                                                                                                                                                      

#SBATCH -t 05:00:00
#SBATCH -J gecco_cross_validation
#SBATCH --mail-user="benrosenthal03@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --array=0-4
#SBATCH --cpus-per-task 16
#SBATCH -o gecco_%A_%a.out
#SBATCH -e gecco_%A_%a.err

source ~/.bashrc
# Activate the genecat environment
mamba activate genecat

# set bash strict mode http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'


cd /exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction
python src/scripts/gecco.py -k ${SLURM_ARRAY_TASK_ID}
