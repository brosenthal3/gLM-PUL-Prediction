#!/bin/bash                                                                                                                                                                                                                                                              
#SBATCH -t 10:00:00
#SBATCH -J esm_embeddings
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --mail-user="benrosenthal03@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH -o slurm_output/esm_embeddings_%j.out
#SBATCH -e slurm_output/esm_embeddings_%j.err

source ~/.bashrc
mamba activate bacformer

set -euo pipefail
IFS=$'\n\t'

# module load system/python/3.12.6
# mkdir -p $TMPDIR/esm_env
# python -m venv $TMPDIR/esm_env --system-site-packages
# source $TMPDIR/esm_env/bin/activate
# pip install biopython polars numpy torch esm

cd /exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction
python src/scripts/esmc_bacformer_script.py
