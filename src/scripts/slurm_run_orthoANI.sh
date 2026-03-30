#!/bin/bash                                                                                                                                                                                                                                                              
#SBATCH -t 48:00:00
#SBATCH -J orthoANI_screen
#SBATCH --mail-user="benrosenthal03@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 32
#SBATCH -o orthoANI_%j.out
#SBATCH -e orthoANI_%j.err

source ~/.bashrc
mamba activate genecat

# set bash strict mode http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

OUT="/exports/lucid-grpzeller-work/${USER}/gLM-PUL-Prediction/src/data/results/orthoANI_output_new.txt"
GENOME_DIR="/exports/lucid-grpzeller-work/${USER}/gLM-PUL-Prediction/src/data/genomes/selected_genomes/"

python /exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/src/scripts/orthoANI.py -i $GENOME_DIR -o $OUT
