#!/bin/bash                                                                                                                                                                                                                                                              
#SBATCH -t 48:00:00
#SBATCH -J orthoANI_screen
#SBATCH --mail-user="benrosenthal03@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 32
#SBATCH -o slurm_output/orthoANI_%j.out
#SBATCH -e slurm_output/orthoANI_%j.err

source ~/.bashrc
mamba activate genecat

# set bash strict mode http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

OUT="/exports/lucid-grpzeller-work/${USER}/gLM-PUL-Prediction/src/data/data_collection/orthoANI_output.txt"
GENOME_DIR="/exports/lucid-grpzeller-work/${USER}/gLM-PUL-Prediction/src/data/genomes/selected_genomes/"

# run orthoANI script to generate matrix
python /exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/src/scripts/orthoANI.py -i $GENOME_DIR -o $OUT

# run deduplicate script to generate deduplicated cluster table
python /exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/src/scripts/deduplicate.py