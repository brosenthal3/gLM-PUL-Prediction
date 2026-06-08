#!/bin/bash

#SBATCH -t 25:00:00
#SBATCH -J umap_embeddings
#SBATCH --mail-user="benrosenthal03@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -o slurm_output/umap_%j.out
#SBATCH -e slurm_output/umap_%j.err

source ~/.bashrc
mamba activate viz

# set bash strict mode http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

cd /exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/
python src/scripts/visualization/umap_embeddings.py