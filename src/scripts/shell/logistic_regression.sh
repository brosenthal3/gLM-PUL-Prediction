#!/bin/bash

#SBATCH -t 05:00:00
#SBATCH -J logistic_regression
#SBATCH --mail-user="benrosenthal03@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 32
#SBATCH -o slurm_output/logistic_regression_%j.out
#SBATCH -e slurm_output/logistic_regression_%j.err

source ~/.bashrc
mamba activate genecat

# set bash strict mode http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'
cd /exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/

### MODELS TRAINED ON ALL PULs ###

# genecat pfam:
python src/scripts/logistic_regression.py --input-df-file-path src/data/results/genecat_zeroshot_pfam/fold_data --output-dir src/data/results/genecat_zeroshot_pfam --model-name pfam --norm-type l2 --normalize

# genecat cazy:
python src/scripts/logistic_regression.py --input-df-file-path src/data/results/genecat_zeroshot_cazy/fold_data --output-dir src/data/results/genecat_zeroshot_cazy --model-name cazy --norm-type l2 --normalize

# ESM-C:
python src/scripts/logistic_regression.py --input-df-file-path src/data/results/esmc/fold_data --output-dir src/data/results/esmc --model-name esmc --norm-type l2 --normalize --embeddings-col embedding

# Bacformer:
python src/scripts/logistic_regression.py --input-df-file-path src/data/results/bacformer/fold_data --output-dir src/data/results/bacformer --model-name bacformer --norm-type l2 --normalize --embeddings-col embedding

### MODELS TRAINED EXCLUDING CRYPTIC PULs ###

# genecat pfam
python src/scripts/logistic_regression.py --input-df-file-path src/data/results/genecat_zeroshot_pfam/fold_data --output-dir src/data/results/genecat_zeroshot_pfam_masked --model-name pfam_masked --norm-type l2 --normalize --mask-cryptic-puls

# genecat cazy
python src/scripts/logistic_regression.py --input-df-file-path src/data/results/genecat_zeroshot_cazy/fold_data --output-dir src/data/results/genecat_zeroshot_cazy_masked --model-name cazy_masked --norm-type l2 --normalize --mask-cryptic-puls

# ESM-C:
python src/scripts/logistic_regression.py --input-df-file-path src/data/results/esmc/fold_data --output-dir src/data/results/esmc_masked --model-name esmc_masked --norm-type l2 --normalize --embeddings-col embedding

# Bacformer:
python src/scripts/logistic_regression.py --input-df-file-path src/data/results/bacformer/fold_data --output-dir src/data/results/bacformer_masked --model-name bacformer_masked --norm-type l2 --normalize --embeddings-col embedding

### GENERATE VISUALIZATIONSSS ###
python src/scripts/visualization/evaluate_predictions.py --model all
python src/scripts/visualization/evaluate_predictions.py --model masked