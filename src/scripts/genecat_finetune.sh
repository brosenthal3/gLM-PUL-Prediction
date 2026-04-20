#!/bin/bash                                                                                                                                                                                                      

#SBATCH -t 24:00:00
#SBATCH -J genecat_extract_embs
#SBATCH --mail-user="benrosenthal03@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --array=0-6
#SBATCH -o genecat_%j.out
#SBATCH -e genecat_%j.err

source ~/.bashrc
module load system/python/3.12.6

mkdir -p $TMPDIR/genecat_env
python -m venv $TMPDIR/genecat_env --system-site-packages
source $TMPDIR/genecat_env/bin/activate

# set bash strict mode http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'


BASEPATH=/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/data_split_class_level
PULPATH=/exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction
VOCAB=${BASEPATH}/models_multilabel_models/jan_model/BERT_train.fold_0.all_domains.min50.vocab.txt
MODEL_NAME=model_gene_multilabel_untied_jan_za9lmkbs_v0
MODEL=${BASEPATH}/models_multilabel_models/jan_model/${MODEL_NAME}.pt

GENES=${PULPATH}/src/data/genecat_output/genome.genes.parquet
FEATURES=${PULPATH}/src/data/genecat_output/genome.features.parquet
CLUSTERS=${PULPATH}/src/data/splits/train_fold_${SLURM_ARRAY_TASK_ID}.tsv
CLUSTERS_TEST=${PULPATH}/src/data/splits/test_fold_${SLURM_ARRAY_TASK_ID}.tsv
OUT=${PULPATH}/src/data/results/genecat/fine_tuned

export PYTHONPATH='/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/src/:/exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/src/'
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

python $PULPATH/src/scripts/genecat_finetune.py -g ${GENES} -d ${FEATURES} -c ${CLUSTERS} --vocab ${VOCAB} -m ${MODEL} -o ${OUT}\\
 --batch-size 10 -j 1 --offline --name genecat_fold_${SLURM_ARRAY_TASK_ID} \\
 --test-gene-table ${GENES} --test-domain-table ${DOMAINS} --test-cluster-table ${CLUSTERS_TEST}