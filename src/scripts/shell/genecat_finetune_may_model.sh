#!/bin/bash                                                                                                                                                                                                      
#SBATCH -t 25:00:00
#SBATCH -J genecat_finetune
#SBATCH --mail-user="benrosenthal03@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=gpu-medium
#SBATCH --cpus-per-task 8
#SBATCH --array=0-6%4
#SBATCH -o slurm_output/genecat_finetune_may_%A_%a.out
#SBATCH -e slurm_output/genecat_finetune_may_%A_%a.err

source ~/.bashrc
mamba activate genecat

# set bash strict mode http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

BASEPATH=/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/multilabel_models/may_full_PG3_model
PULPATH=/exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction

# GENES
GENES_TRAIN=${PULPATH}/src/data/genecat_output/fold_${SLURM_ARRAY_TASK_ID}/train.genes.parquet
GENES_TEST=${PULPATH}/src/data/genecat_output/fold_${SLURM_ARRAY_TASK_ID}/test.genes.parquet

# CLUSTERS
CLUSTERS_TRAIN=${PULPATH}/src/data/splits/train_fold_${SLURM_ARRAY_TASK_ID}_with_cryptic.tsv
CLUSTERS_TEST=${PULPATH}/src/data/splits/test_fold_${SLURM_ARRAY_TASK_ID}_with_cryptic.tsv

# OUTPUT
OUT_PFAM=${PULPATH}/src/data/results/genecat_full_pg3
mkdir -p $OUT_PFAM

export PYTHONPATH='/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/src/:/exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/src/'
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

#----TRAIN PFAM ONLY MODEL----#

# MODEL
VOCAB=${BASEPATH}/representatives.unique_domains.min50.Pfam37.1.vocab.txt
MODEL_NAME=model_full_gene_multilabel_untied_qoflkopy_v0.pt
MODEL=${BASEPATH}/${MODEL_NAME}

# DOMAINS
DOMAINS_TRAIN=${PULPATH}/src/data/genecat_output/fold_${SLURM_ARRAY_TASK_ID}/train.pfam.parquet
DOMAINS_TEST=${PULPATH}/src/data/genecat_output/fold_${SLURM_ARRAY_TASK_ID}/test.pfam.parquet

python -m genecat.cli pul-finetune\
 -g ${GENES_TRAIN} -d ${DOMAINS_TRAIN} -c ${CLUSTERS_TRAIN}\
 --vocab ${VOCAB} -m ${MODEL} -o ${OUT_PFAM}/fold_${SLURM_ARRAY_TASK_ID}\
 --batch-size 128 -j 1 --offline --name fold_${SLURM_ARRAY_TASK_ID}\
 --test-gene-table ${GENES_TEST} --test-domain-table ${DOMAINS_TEST} --test-cluster-table ${CLUSTERS_TEST}\
 --middle-focus --epochs 30
