#!/bin/bash                                                                                                                                                                                                      

#SBATCH -t 05:00:00
#SBATCH -J genecat_extract_embs
#SBATCH --mail-user="benrosenthal03@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH -o genecat_%j.out
#SBATCH -e genecat_%j.err

source ~/.bashrc
# Activate the genecat environment
mamba activate genecat

# set bash strict mode http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

# EXTRACT EMBEDDINGS #
export PYTHONPATH='/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/src'
BASEPATH=/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/data_split_class_level
PULPATH=/exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction
VOCAB=${BASEPATH}/models_multilabel_models/jan_model/BERT_train.fold_0.all_domains.min50.vocab.txt
MODEL_NAME=model_gene_multilabel_untied_jan_za9lmkbs_v0
MODEL=${BASEPATH}/models_multilabel_models/jan_model/${MODEL_NAME}.pt
GENES=${PULPATH}/src/data/genecat_output/genome.genes.parquet
FEATURES=${PULPATH}/src/data/genecat_output/genome.features.parquet
OUT=${PULPATH}/src/data/results/genecat/PUL_embs

python -m genecat.cli extract-embeddings -g $GENES -f $FEATURES -m $MODEL --vocab $VOCAB --batch-size 16 -j 1 -o $OUT --outtypes df

# PROCESS EMBEDDINGS INTO SEPARATE TABLES FOR EACH FOLD #
EMBS=${OUT}/${MODEL_NAME}_context_embedding.embeddings.parquet
cd /exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/

python src/scripts/process_embeddings_output.py --genes $GENES --embeddings $EMBS -k 7