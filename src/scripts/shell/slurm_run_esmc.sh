#!/bin/bash                                                                                                                                                                                                                                                              
#SBATCH -t 24:00:00
#SBATCH -J esm_embeddings
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH -o esm_embeddings_%j.out
#SBATCH -e esm_embeddings_%j.err
#SBATCH --mail-user="benrosenthal03@gmail.com"
#SBATCH --mail-type="ALL"

set -euo pipefail
IFS=$'\n\t'

module load system/python/3.12.6

mkdir -p $TMPDIR/esm_env
python -m venv $TMPDIR/esm_env --system-site-packages
source $TMPDIR/esm_env/bin/activate

pip install biopython polars numpy torch esm

cd /exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction
python src/scripts/esmc_script.py
