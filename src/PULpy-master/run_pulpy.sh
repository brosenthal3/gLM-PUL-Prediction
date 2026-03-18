#!/bin/bash
# FROM PULpy repository README.md

source ~/.bashrc
mamba env create -f envs/PULpy.yaml
mamba activate PULpy

mkdir pfam_data && cd pfam_data
wget ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz
wget ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.dat.gz
wget ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/active_site.dat.gz
gunzip Pfam-A.hmm.gz Pfam-A.hmm.dat.gz active_site.dat.gz
hmmpress Pfam-A.hmm
cd ..

mkdir dbcan_data && cd dbcan_data
wget http://bcb.unl.edu/dbCAN2/download/Databases/dbCAN-old@UGA/hmmscan-parser.sh
wget http://bcb.unl.edu/dbCAN2/download/Databases/dbCAN-old@UGA/dbCAN-fam-HMMs.txt
hmmpress dbCAN-fam-HMMs.txt
chmod 755 hmmscan-parser.sh
cd ..

# make scripts executable
chmod -R 755 scripts

# run scripts
snakemake --use-conda
