#!/bin/bash                                                                                                                                                                                                      
#SBATCH -t 12:00:00
#SBATCH -J genecat_finetune
#SBATCH --mail-user="benrosenthal03@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH -o genecat_%j.out
#SBATCH -e genecat_%j.err

source ~/.bashrc
mamba activate genecat

# mkdir -p $TMPDIR/genecat_env
# python -m venv $TMPDIR/genecat_env --system-site-packages
# source $TMPDIR/genecat_env/bin/activate
# pip install wandb polars pysqlite3 rich-argparse sqlite-vec scikit-learn rich numpy pytorch_lightning pandas anndata hvplot pyarrow gb_io pyhmmer pyrodigal

# set bash strict mode http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

SLURM_ARRAY_TASK_ID=0
BASEPATH=/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/data_split_class_level
PULPATH=/exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction
# GENES
GENES_TRAIN=${PULPATH}/src/data/genecat_output/fold_${SLURM_ARRAY_TASK_ID}/train.genes.parquet
GENES_TEST=${PULPATH}/src/data/genecat_output/fold_${SLURM_ARRAY_TASK_ID}/test.genes.parquet
# CLUSTERS
CLUSTERS_TRAIN=${PULPATH}/src/data/splits/train_fold_${SLURM_ARRAY_TASK_ID}.tsv
CLUSTERS_TEST=${PULPATH}/src/data/splits/test_fold_${SLURM_ARRAY_TASK_ID}.tsv
# OUTPUT
OUT=${PULPATH}/src/data/results/genecat_fine_tuned
mkdir -p ${OUT}

export PYTHONPATH='/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/src/:/exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/src/'
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

#----TRAIN PFAM ONLY MODEL----#

# MODEL
VOCAB=${BASEPATH}/models_multilabel_models/april_models/BERT_train.fold_0.unique_domains.min50.Pfam37.1.vocab.txt
MODEL_NAME=model_gene_multilabel_untied_april_sriqcx3c_v0.pt
MODEL=${BASEPATH}/models_multilabel_models/april_models/${MODEL_NAME}
# DOMAINS
DOMAINS_TRAIN=${PULPATH}/src/data/genecat_output/fold_${SLURM_ARRAY_TASK_ID}/train.pfam.parquet
DOMAINS_TEST=${PULPATH}/src/data/genecat_output/fold_${SLURM_ARRAY_TASK_ID}/test.pfam.parquet

python $PULPATH/src/scripts/genecat_finetune.py\
 -g ${GENES_TRAIN} -d ${DOMAINS_TRAIN} -c ${CLUSTERS_TRAIN}\
 --vocab ${VOCAB} -m ${MODEL} -o ${OUT}/genecat_fine_tuned\
 --batch-size 128 -j 1 --offline --name pfam_fold_${SLURM_ARRAY_TASK_ID}\
 --test-gene-table ${GENES_TEST} --test-domain-table ${DOMAINS_TEST} --test-cluster-table ${CLUSTERS_TEST}\
 --middle-focus --epochs 30


#----TRAIN PFAM+CAZY MODEL----# 

# MODEL
VOCAB=${BASEPATH}/models_multilabel_models/april_models/BERT_train.fold_0.unique_domains.min50.Pfam37.1_cazy_cayman_v0.12.vocab.txt
MODEL_NAME=model_gene_multilabel_pfam_cazy_april_goycr91w_v0.pt
MODEL=${BASEPATH}/models_multilabel_models/april_models/${MODEL_NAME}
# DOMAINS
DOMAINS_TRAIN=${PULPATH}/src/data/genecat_output/fold_${SLURM_ARRAY_TASK_ID}/train.dbcan.pfam.parquet
DOMAINS_TEST=${PULPATH}/src/data/genecat_output/fold_${SLURM_ARRAY_TASK_ID}/test.dbcan.pfam.parquet

python $PULPATH/src/scripts/genecat_finetune.py\
 -g ${GENES_TRAIN} -d ${DOMAINS_TRAIN} -c ${CLUSTERS_TRAIN}\
 --vocab ${VOCAB} -m ${MODEL} -o ${OUT}/genecat_fine_tuned\
 --batch-size 128 -j 1 --offline --name pfam_cazy_fold_${SLURM_ARRAY_TASK_ID}\
 --test-gene-table ${GENES_TEST} --test-domain-table ${DOMAINS_TEST} --test-cluster-table ${CLUSTERS_TEST}\
 --middle-focus --epochs 30
