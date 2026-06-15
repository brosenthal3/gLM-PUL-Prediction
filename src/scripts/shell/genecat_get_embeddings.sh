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

# VOCABS
VOCAB_PFAM=${BASEPATH}/models_multilabel_models/april_models/BERT_train.fold_0.unique_domains.min50.Pfam37.1.vocab.txt
VOCAB_CAZY_PFAM=${BASEPATH}/models_multilabel_models/april_models/BERT_train.fold_0.unique_domains.min50.Pfam37.1_cazy_cayman_v0.12.vocab.txt
# MODELS
MODEL_NAME_PFAM=model_gene_multilabel_untied_april_sriqcx3c_v0
MODEL_PFAM=${BASEPATH}/models_multilabel_models/april_models/${MODEL_NAME_PFAM}.pt
MODEL_NAME_CAZY_PFAM=model_gene_multilabel_pfam_cazy_april_goycr91w_v0
MODEL_CAZY_PFAM=${BASEPATH}/models_multilabel_models/april_models/${MODEL_NAME_CAZY_PFAM}.pt
# INPUT DATA
GENES=${PULPATH}/src/data/genecat_output/genome.genes.parquet
FEATURES_PFAM=${PULPATH}/src/data/genecat_output/pfam.features.parquet
FEATURES_CAZY_PFAM=${PULPATH}/src/data/genecat_output/dbcan.pfam.features.parquet
OUT=${PULPATH}/src/data/embeddings/genecat_embeddings

# RUN SCRIPTS 
python -m genecat.cli extract-embeddings -g $GENES -f $FEATURES_PFAM -m $MODEL_PFAM --vocab $VOCAB_PFAM --batch-size 16 -j 1 -o $OUT --outtypes df
python -m genecat.cli extract-embeddings -g $GENES -f $FEATURES_CAZY_PFAM -m $MODEL_CAZY_PFAM --vocab $VOCAB_CAZY_PFAM --batch-size 16 -j 1 -o $OUT --outtypes df

# PROCESS EMBEDDINGS
EMBS_PFAM=${OUT}/${MODEL_NAME_PFAM}_context_embedding.embeddings.parquet
EMBS_CAZY_PFAM=${OUT}/${MODEL_NAME_CAZY_PFAM}_context_embedding.embeddings.parquet
OUT_FOLDS="src/data/results/genecat_zeroshot"

cd /exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/
python src/scripts/process_embeddings_output.py --genes $GENES --embeddings $EMBS_PFAM -k 7 -o ${OUT_FOLDS}_pfam/fold_data
python src/scripts/process_embeddings_output.py --genes $GENES --embeddings $EMBS_CAZY_PFAM -k 7 -o ${OUT_FOLDS}_cazy/fold_data


#-----GET EMBEDDINGS FROM UNTRAINED-FINETUNED MODEL-------#
MODEL_PATH=${PULPATH}/src/data/results/genecat_untrained/models_fold_0/
MODEL_UNTRAINED=${MODEL_PATH}/model_cazy_fold_0_7t9fb3eu_v0.pt
FEATURES_UNTRAINED=${PULPATH}/src/data/genecat_output/dbcan.pfam.features.parquet
OUT=${PULPATH}/src/data/embeddings/genecat_untrained_embeddings

cd /exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/
python src/scripts/analysis/save_pretrained_model.py $MODEL_UNTRAINED ${MODEL_PATH}/pretrained_model_part
python -m genecat.cli extract-embeddings -g $GENES -f $FEATURES_UNTRAINED -m ${MODEL_PATH}/pretrained_model_part.pt --vocab $VOCAB_CAZY_PFAM --batch-size 16 -j 1 -o $OUT --outtypes df

EMBS_UNTRAINED=${OUT}/${MODEL_PATH}/pretrained_model_part_context_embedding.embeddings.parquet
OUT_FOLDS="src/data/results/genecat_untrained/embeddings"

python src/scripts/process_embeddings_output.py --genes $GENES --embeddings $EMBS_UNTRAINED -k 7 -o ${OUT_FOLDS}  --embedding_col embeddings


#------GET EMBEDDINGS FROM PRE-TRAINED FINETUNED MODEL-------# 
# TODO
# VOCAB=${BASEPATH}/models_multilabel_models/april_models/BERT_train.fold_0.unique_domains.min50.Pfam37.1_cazy_cayman_v0.12.vocab.txt
# FINETUNED_MODEL=$(ls -t "${OUT_CAZY}/models_fold_${FOLD}"/*.pt 2>/dev/null | head -n 1)