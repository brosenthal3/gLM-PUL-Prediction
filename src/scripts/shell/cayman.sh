#!/bin/bash

# to run in interactive session
export PYTHONPATH='/exports/archive/lucid-grpzeller-primary/hackett/software/cayman'

# create env and install dependencies
eval "$(mamba shell hook --shell bash)"
mamba activate
mamba env create -f /exports/archive/lucid-grpzeller-primary/hackett/software/cayman/environment.yml
mamba activate cayman
pip install -r /exports/archive/lucid-grpzeller-primary/hackett/software/cayman/requirements.txt

OUT=/exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/src/data/genecat_output/
CUTOFFS=/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/hmms/CAZy/Cayman/cutoffs.csv
CAZY_HMMS=/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/hmms/CAZy/Cayman/hmms_bin/cayman.v3.seed42_selected.h3m
NUM_THREADS=16 # by default it will run just one thread
PROTEINS=$OUT/genome.genes.faa

python -m cayman annotate_proteome $CAZY_HMMS $PROTEINS --cutoffs $CUTOFFS -o $OUT/genome.features.cayman_cazy_v0.12.0.csv -t $NUM_THREADS
