#!/bin/bash                                                                                                                                                                                                      

#SBATCH -t 24:00:00
#SBATCH -J genecat_extract_embs
#SBATCH --mail-user="benrosenthal03@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH -o genecat_%j.out
#SBATCH -e genecat_%j.err

source ~/.bashrc
module load system/python/3.12.6

mkdir -p $TMPDIR/genecat_env
python -m venv $TMPDIR/genecat_env --system-site-packages
source $TMPDIR/genecat_env/bin/activate
pip install wandb polars pysqlite3 rich-argparse sqlite-vec scikit-learn rich numpy pytorch_lightning pandas anndata hvplot pyarrow gb_io pyhmmer pyrodigal

# set bash strict mode http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

SLURM_ARRAY_TASK_ID=0
BASEPATH=/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/data_split_class_level
PULPATH=/exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction
# MODEL
VOCAB=${BASEPATH}/models_multilabel_models/jan_model/BERT_train.fold_0.all_domains.min50.vocab.txt
MODEL_NAME=model_gene_multilabel_untied_jan_za9lmkbs_v0
MODEL=${BASEPATH}/models_multilabel_models/jan_model/${MODEL_NAME}.pt

# GENES
GENES_TRAIN=${PULPATH}/src/data/genecat_output/fold_${SLURM_ARRAY_TASK_ID}/train.genes.parquet
GENES_TEST=${PULPATH}/src/data/genecat_output/fold_${SLURM_ARRAY_TASK_ID}/test.genes.parquet
# DOMAINS
DOMAINS_TRAIN=${PULPATH}/src/data/genecat_output/fold_${SLURM_ARRAY_TASK_ID}/train.pfam.parquet
DOMAINS_TEST=${PULPATH}/src/data/genecat_output/fold_${SLURM_ARRAY_TASK_ID}/test.pfam.parquet
# CLUSTERS
CLUSTERS_TRAIN=${PULPATH}/src/data/splits/train_fold_${SLURM_ARRAY_TASK_ID}.tsv
CLUSTERS_TEST=${PULPATH}/src/data/splits/test_fold_${SLURM_ARRAY_TASK_ID}.tsv
# OUTPUT
OUT=${PULPATH}/src/data/results/genecat_fine_tuned
mkdir -p ${OUT}

export PYTHONPATH='/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/src/:/exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/src/'
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

python $PULPATH/src/scripts/genecat_finetune.py\
 -g ${GENES_TRAIN} -d ${DOMAINS_TRAIN} -c ${CLUSTERS_TRAIN}\
 --vocab ${VOCAB} -m ${MODEL} -o ${OUT}/genecat_fine_tuned\
 --batch-size 10 -j 1 --offline --name fold_${SLURM_ARRAY_TASK_ID}\
 --test-gene-table ${GENES_TEST} --test-domain-table ${DOMAINS_TEST} --test-cluster-table ${CLUSTERS_TEST}\
 --middle-focus --epochs 30