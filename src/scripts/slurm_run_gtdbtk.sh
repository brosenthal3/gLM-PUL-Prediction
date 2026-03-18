#!/bin/bash                                                                                                                                                                                                                                                              
#SBATCH -t 02:00:00
#SBATCH -J GTDB_TK_classification
#SBATCH --mail-user="benrosenthal03@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 12
#SBATCH -o gtdbtk_annotation_%j.out
#SBATCH -e gtdbtk_annotation_%j.err

source ~/.bashrc
# Activate the GTDB-Tk conda environment
conda activate gtdbtk

# set bash strict mode http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

OUT_DIR="/exports/lucid-grpzeller-work/${USER}/gtdbtk_annotations/gtdbtk_output"
mkdir -p $OUT_DIR

TEMP_DIR="${OUT_DIR}/tmp/"
mkdir -p $TEMP_DIR

MASH_DB="${OUT_DIR}/mash_db.msh"
GENOME_DIR="/exports/lucid-grpzeller-work/brosenthal/gtdbtk_annotations/genomes/gtdb_genomes"

# Set variable to the directory containing the UN-ARCHIVED GTDB-Tk reference data
export GTDBTK_DATA_PATH="/exports/archive/lucid-grpzeller-primary/SHARED/DATA/gene_catalogues/GTDBTK_R226/gtdbtk_r226_data"

gtdbtk classify_wf --genome_dir $GENOME_DIR --tmpdir $TEMP_DIR --cpus 12 --out_dir $OUT_DIR -x fa
