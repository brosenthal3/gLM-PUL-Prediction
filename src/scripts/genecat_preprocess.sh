#!/bin/bash

# best to run in interactive session
export PYTHONPATH='/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/src'
PFAM=/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/hmms/Pfam/Pfam-A.Pfam37.1.h3m
VOCAB=/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/data_split_class_level/models_multilabel_models/jan_model/BERT_train.fold_0.all_domains.min50.vocab.txt
OUT=/exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/src/data/genecat_output/
GENOMES=/exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/src/data/genomes/combined_genomes.gb

python -m genecat.cli preprocess -i $GENOMES --hmms $PFAM -v $VOCAB -o $OUT --write-tables --call-genes --index-db --write-tables --unique-domains
python -m genecat.cli call-genes -i $GENOMES -o $OUT --type faa+gff

#export PYTHONPATH='/exports/archive/lucid-grpzeller-primary/hackett/software/cayman'
#CUTOFFS=/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/hmms/CAZy/Cayman/cutoffs.csv
#CAZY_HMMS=/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/hmms/CAZy/Cayman/hmms_bin/cayman.v3.seed42_selected.h3m
#NUM_THREADS=16 # by default it will run just one thread
#PROTEINS=$OUT/genome.genes.faa

#python -m cayman annotate_proteome $CAZY_HMMS $PROTEINS --cutoffs $CUTOFFS -o $OUT/genome.features.cayman_cazy_v0.12.0.csv -t $NUM_THREADS
