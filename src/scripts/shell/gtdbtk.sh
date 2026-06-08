#!/bin/bash                                                                                                                                                                                                                                                              
#SBATCH -t 03:00:00
#SBATCH -J GTDB_TK_classification
#SBATCH --mail-user="benrosenthal03@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 12
#SBATCH -o slurm_output/gtdbtk_annotation_%j.out
#SBATCH -e slurm_output/gtdbtk_annotation_%j.err

source ~/.bashrc
# Activate the GTDB-Tk conda environment
mamba activate gtdbtk

# set bash strict mode http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

OUT_DIR="/exports/lucid-grpzeller-work/${USER}/gtdbtk_annotations/gtdbtk_output"
TEMP_DIR="${OUT_DIR}/tmp/"
MASH_DB="${OUT_DIR}/mash_db.msh"
GENOME_DIR="/exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/src/data/genomes/gtdb_genomes"

mkdir -p $OUT_DIR
mkdir -p $TEMP_DIR


# Set variable to the directory containing the UN-ARCHIVED GTDB-Tk reference data
export GTDBTK_DATA_PATH="/exports/archive/lucid-grpzeller-primary/SHARED/DATA/gene_catalogues/GTDBTK_R226/gtdbtk_r226_data"

# run classification
gtdbtk classify_wf --genome_dir $GENOME_DIR --tmpdir $TEMP_DIR --cpus 12 --out_dir $OUT_DIR -x fa

# copy result to gLM PUL Prediction dir
cp ${OUT_DIR}/gtdbtk.bac120.summary.tsv /exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction/src/data/data_collection/

# run script to process gtdb annotations
mamba activate genecat
cd /exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction
python src/scripts/process_gtdb_output.py