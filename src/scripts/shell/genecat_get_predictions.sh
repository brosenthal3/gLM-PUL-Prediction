#!/bin/bash                                                                                                                                                                                                      
#SBATCH -t 10:00:00
#SBATCH -J genecat_predict
#SBATCH --mail-user="benrosenthal03@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=gpu-medium
#SBATCH --cpus-per-task 8
#SBATCH -o slurm_output/genecat_predict_%j.out
#SBATCH -e slurm_output/genecat_predict_%j.err

source ~/.bashrc
mamba activate genecat

# set bash strict mode http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

BASEPATH=/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/data_split_class_level
PULPATH=/exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction
# OUTPUT
OUT_PFAM=${PULPATH}/src/data/results/genecat_finetuned_pfam_masked
OUT_CAZY=${PULPATH}/src/data/results/genecat_finetuned_cazy_masked
mkdir -p $OUT_PFAM
mkdir -p $OUT_CAZY
# ADD PYTHON PATH
export PYTHONPATH='/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/src/:/exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/src/'
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

for FOLD in {0..6}
do
    # INPUT
    GENES_TEST=${PULPATH}/src/data/genecat_output/fold_${FOLD}/test.genes.parquet
    CLUSTERS_TEST=${PULPATH}/src/data/splits/test_fold_${FOLD}_with_cryptic.tsv

    #----PFAM ONLY MODEL----#

    # MODEL
    VOCAB=${BASEPATH}/models_multilabel_models/april_models/BERT_train.fold_0.unique_domains.min50.Pfam37.1.vocab.txt
    FINETUNED_MODEL=$(ls -t "${OUT_PFAM}/models_fold_${FOLD}"/*.pt 2>/dev/null | head -n 1)
    # DOMAINS
    DOMAINS_TEST=${PULPATH}/src/data/genecat_output/fold_${FOLD}/test.pfam.parquet

    python -m genecat.cli predict-cluster \
    -g ${GENES_TEST} -d ${DOMAINS_TEST} -c ${CLUSTERS_TEST} \
    --vocab ${VOCAB} -m ${FINETUNED_MODEL} -o ${OUT_PFAM}/fold_${FOLD} --batch-size 128

    #----PFAM+CAZY MODEL----# 

    # MODEL
    VOCAB=${BASEPATH}/models_multilabel_models/april_models/BERT_train.fold_0.unique_domains.min50.Pfam37.1_cazy_cayman_v0.12.vocab.txt
    FINETUNED_MODEL=$(ls -t "${OUT_CAZY}/models_fold_${FOLD}"/*.pt 2>/dev/null | head -n 1)
    # DOMAINS
    DOMAINS_TEST=${PULPATH}/src/data/genecat_output/fold_${FOLD}/test.dbcan.pfam.parquet

    python -m genecat.cli predict-cluster \
    -g ${GENES_TEST} -d ${DOMAINS_TEST} -c ${CLUSTERS_TEST} \
    --vocab ${VOCAB} -m ${FINETUNED_MODEL} -o ${OUT_CAZY}/fold_${FOLD} --batch-size 128

done
