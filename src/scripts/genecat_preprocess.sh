#!/bin/bash

# best to run in interactive session
export PYTHONPATH='/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/src'
PFAM=/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/hmms/Pfam/Pfam-A.Pfam37.1.h3m
CAZY=/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/hmms/CAZy/dbCAN/dbCAN-HMMdb-V14.h3m
#VOCAB=/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/data_split_class_level/models_multilabel_models/jan_model/BERT_train.fold_0.all_domains.min50.vocab.txt
OUT=/exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/src/data/genecat_output/
GENOMES=/exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/src/data/genomes/combined_genomes.gb

python -m genecat.cli preprocess -i $GENOMES --hmms $PFAM -o $OUT --write-tables --call-genes --hmm-type pfam --output-name pfam
python -m genecat.cli preprocess -i $GENOMES --hmms $CAZY -o $OUT --write-tables --call-genes --hmm-type cazy --output-name dbcan
python -m genecat.cli call-genes -i $GENOMES -o $OUT --type faa+gff
