#!/bin/bash

#SBATCH -t 05:00:00
#SBATCH -J preprocessing
#SBATCH --mail-user="benrosenthal03@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 16
#SBATCH -o preprocess_%j.out
#SBATCH -e preprocess_%j.err

source ~/.bashrc
mamba activate genecat

# set bash strict mode http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

PFAM=/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/hmms/Pfam/Pfam-A.Pfam37.1.h3m
CAZY=/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/hmms/CAZy/dbCAN/dbCAN-HMMdb-V14.h3m
OUT=/exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/src/data/genecat_output/
GENOMES=/exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/src/data/genomes/combined_genomes.gb

export PYTHONPATH='/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/src'

# preprocess with pfam
python -m genecat.cli preprocess -i $GENOMES --hmms $PFAM -o $OUT --write-tables --call-genes --hmm-type pfam --output-name pfam
# preprocess with cazy
python -m genecat.cli preprocess -i $GENOMES --hmms $CAZY -o $OUT --write-tables --call-genes --hmm-type cazy --output-name dbcan
# call genes and save faa and gff files
# python -m genecat.cli call-genes -i $GENOMES -o $OUT --type faa+gff

# combine features
cd /exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/
python src/scripts/combine_features.py