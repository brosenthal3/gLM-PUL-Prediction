#!/bin/bash

# best to run in interactive session
mamba activate genecat
export PYTHONPATH='/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/src'
PFAM=/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/hmms/Pfam/Pfam-A.Pfam37.1.h3m
OUT=/exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/src/genecat_output/
GENOMES=/exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/src/data/genomes/combined_genomes.gb

python -m genecat.cli preprocess -i $GENOMES --hmms $PFAM -o $OUT
python -m genecat.cli call-genes -i $GENOMES -o $OUT --type faa+gff
