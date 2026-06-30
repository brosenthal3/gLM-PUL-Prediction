# PULpy
Open prediction of Polysaccharide Utilisation Loci (PUL)
This README is adapted from the PULpy github repository (https://github.com/WatsonLab/PULpy).

# Input files

PULpy is designed to process genomes downloaded from NCBI, therefore expects them to be in a directory called `genomes` in the following format:

```
"./genomes/{id}_genomic.fna.gz"
```

All genomes from the dataset were already transferred to this directory and zipped in the `data_collection.py` script.

# Custom script
The tool can be ran with a customly made script that was based on the original README: 
`run_pulpy.sh`.

The steps in the script are as follows:


## 1) Create conda env
```sh
conda env create -f envs/PULpy.yaml
source activate PULpy
```

## 2) Get Pfam and dbCAN data
```sh
# Pfam

mkdir pfam_data && cd pfam_data
wget --no-check-certificate ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam38.0/Pfam-A.hmm.gz
wget --no-check-certificate ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam38.0/Pfam-A.hmm.dat.gz
wget --no-check-certificate ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam38.0/active_site.dat.gz
gunzip Pfam-A.hmm.gz Pfam-A.hmm.dat.gz active_site.dat.gz
hmmpress Pfam-A.hmm
cd ..

# dbCAN 
mkdir dbcan_data && cd dbcan_data
wget http://bcb.unl.edu/dbCAN2/download/Databases/dbCAN-old@UGA/hmmscan-parser.sh
wget http://bcb.unl.edu/dbCAN2/download/Databases/dbCAN-old@UGA/dbCAN-fam-HMMs.txt
hmmpress dbCAN-fam-HMMs.txt
chmod 755 hmmscan-parser.sh
cd ..

```
## 3) Make scripts executable

```sh
chmod -R 755 scripts
```

## 4) Run using snakemake
```sh
snakemake --use-conda
```
