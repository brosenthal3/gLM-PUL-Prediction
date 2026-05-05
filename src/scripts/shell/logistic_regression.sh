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

# ENSURE ALL EMBEDDINGS OUTPUT IS PROCESSED #
# for genecat pfam:
#python src/scripts/process_embeddings_output.py -e src/data/embeddings/genecat_embeddings/model_gene_multilabel_untied_april_sriqcx3c_v0_context_embedding.embeddings.parquet -o src/data/results/genecat_zeroshot_pfam/fold_data

# for genecat cazy:
#python src/scripts/process_embeddings_output.py -e src/data/embeddings/genecat_embeddings/model_gene_multilabel_pfam_cazy_april_goycr91w_v0_context_embedding.embeddings.parquet -o src/data/results/genecat_zeroshot_cazy/fold_data

# for esmc:
#python src/scripts/process_embeddings_output.py -e src/data/embeddings/esmc_bacformer_embeddings -o src/data/results/esmc/fold_data --dir --embedding_col embedding_esmc

# for bacformer
#python src/scripts/process_embeddings_output.py -e src/data/embeddings/esmc_bacformer_embeddings -o src/data/results/bacformer/fold_data --dir --embedding_col embedding_bacformer


# for genecat pfam:
python src/scripts/logistic_regression.py --input-df-file-path src/data/results/genecat_zeroshot_pfam/fold_data --output-dir src/data/results/genecat_zeroshot_pfam --model-name pfam --norm-type l2 --normalize --gridsearch

# for genecat cazy:
python src/scripts/logistic_regression.py --input-df-file-path src/data/results/genecat_zeroshot_cazy/fold_data --output-dir src/data/results/genecat_zeroshot_cazy --model-name cazy --norm-type l2 --normalize --gridsearch

# for ESM-C:
python src/scripts/logistic_regression.py --input-df-file-path src/data/results/esmc/fold_data --output-dir src/data/results/esmc --model-name esmc --norm-type l2 --normalize --embeddings-col embedding --gridsearch

# for Bacformer:
python src/scripts/logistic_regression.py --input-df-file-path src/data/results/bacformer/fold_data --output-dir src/data/results/bacformer --model-name bacformer --norm-type l2 --normalize --embeddings-col embedding --gridsearch