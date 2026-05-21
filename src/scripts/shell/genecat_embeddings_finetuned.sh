#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH -J genecat_extract_embs
#SBATCH --mail-user="benrosenthal03@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH -o slurm_output/genecat_embs_%j.out
#SBATCH -e slurm_output/genecat_embs_%j.err

source ~/.bashrc
# Activate the genecat environment
mamba activate genecat

# set bash strict mode http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

#-----EXTRACT EMBEDDINGS-----#
export PYTHONPATH='/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/src'
BASEPATH=/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/data_split_class_level
PULPATH=/exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction

# to fix GLIBCXX issues???
#export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# VOCABS
VOCAB_CAZY_PFAM=${BASEPATH}/models_multilabel_models/april_models/BERT_train.fold_0.unique_domains.min50.Pfam37.1_cazy_cayman_v0.12.vocab.txt
# MODELS
MODEL_PATH=${PULPATH}/src/data/results/genecat_finetuned_cazy_masked/models_fold_1/
MODEL_CAZY_PFAM=${MODEL_PATH}/model_cazy_fold_1_5wi7kw10_v0.pt
# INPUT DATA
GENES=${PULPATH}/src/data/genecat_output/genome.genes.parquet
FEATURES_CAZY_PFAM=${PULPATH}/src/data/genecat_output/dbcan.pfam.features.parquet
OUT=${PULPATH}/src/data/embeddings/genecat_finetuned_embeddings


# RUN SCRIPTS 
cd /exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/
python src/scripts/analysis/save_pretrained_model.py $MODEL_CAZY_PFAM ${MODEL_PATH}/pretrained_model_part

python -m genecat.cli extract-embeddings -g $GENES -f $FEATURES_CAZY_PFAM -m ${MODEL_PATH}/pretrained_model_part.pt --vocab $VOCAB_CAZY_PFAM --batch-size 16 -j 1 -o $OUT --outtypes df

#-----PROCESS EMBEDDINGS-----#
EMBS_CAZY_PFAM=${OUT}/${MODEL_PATH}/pretrained_model_part_context_embedding.embeddings.parquet
OUT_FOLDS="src/data/results/genecat_finetuned_cazy_masked/embeddings"

python src/scripts/process_embeddings_output.py --genes $GENES --embeddings $EMBS_CAZY_PFAM -k 7 -o ${OUT_FOLDS}  --embedding_col embeddings