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

export PYTHONPATH='/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/src'
BASEPATH=/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/data_split_class_level
PULPATH=/exports/lucid-grpzeller-primary/brosenthal/gLM-PUL-Prediction
VOCAB=${BASEPATH}/BERT_train.fold_0.unique_domains.min50.Pfam37.1.vocab.txt
MODEL=${BASEPATH}/models_multilabel_models/model_gene_multilabel_untied_march_s4spvlec_v0.pt
GENES=${PULPATH}/src/data/genecat_output/genome.genes.parquet
FEATURES=${PULPATH}/src/data/genecat_output/genome.features.parquet
OUT=${PULPATH}/src/data/results/genecat/PUL_embs

python -m genecat.cli extract-embeddings -g $GENES -f $FEATURES -m $MODEL --vocab $VOCAB --batch-size 16 -j 1 -o $OUT --outtypes df db