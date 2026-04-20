#!/bin/bash                                                                                                                                                                                                                                                              
#SBATCH -t 24:00:00
#SBATCH -J esm_embeddings
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --cpus-per-task 12
#SBATCH -o esm_embeddings_%j.out
#SBATCH -e esm_embeddings_%j.err

set -euo pipefail
IFS=$'\n\t'

module load system/python/3.12.6

mkdir -p $TMPDIR/genecat_env
python -m venv $TMPDIR/genecat_env --system-site-packages
source $TMPDIR/genecat_env/bin/activate

pip install polars numpy torch esm

cd /exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction
python src/scripts/esmc_script.py