#!/bin/bash                                                                                                                                                                                                                                                              
#SBATCH -t 12:00:00
#SBATCH -J esm_bacformer_embeddings
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
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH='/exports/archive/lucid-grpzeller-primary/hackett/glm_bench'

set -euo pipefail
IFS=$'\n\t'

# extract embeddings
cd /exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction
python src/scripts/esmc_bacformer_script.py

# save folds for esmc:
python src/scripts/process_embeddings_output.py -e src/data/embeddings/esmc_bacformer_embeddings -o src/data/results/esmc/fold_data --dir --embedding_col embedding_esmc

# save folds for bacformer:
python src/scripts/process_embeddings_output.py -e src/data/embeddings/esmc_bacformer_embeddings -o src/data/results/bacformer/fold_data --dir --embedding_col embedding_bacformer