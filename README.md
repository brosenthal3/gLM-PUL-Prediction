# PUL dataset construction and preprocessing
All data is obtained from two main sources: dbCAN-PUL, and PULDB.

### Data Summary - Before filtering

|  | Total | from dbCAN | from PULDB | 
| - | ----- | ---------- | ---------- | 
| Genomes | 414 | 370 | 44 | 
| PULs | 991 | 633 | 358 |

### dbCAN-PUL
Accession IDs are processed by splitting on the first dot and removing the version number afterwards. 
One genome is downloaded manually from JGI (`Ga0139390_150`, link: https://genome.jgi.doe.gov/portal/pages/dynamicOrganismDownload.jsf?organism=IMG_2703719109). One PUL spans over two accession ids (`ADWO01000021.1, ADWO01000020.1`). Using BLAST on NCBI, the PUL can be found in the full genome with accession ID CP091800. However, this seems to be two separate PULs, where `ADWO01000021` maps to `811569-817528` and `ADWO01000020` maps to `49194-67389`. These are manually added to the dataframe as two PULs.

Then there is a total of: 370 genomes, 633 PULs. 


### PULDB
Data on app literature derived PULs was scraped from the website using a custom script. 
Out of these, 6 IDs are not valid NCBI identifiers. 3 of them are from one paper https://doi.org/10.1111/1462-2920.14414Digital
- 'FG27DRAFT_unitig_0_quiver_dupTrim_7536', from Salegentibacter sp. Hel_I_6. Data on DOE-JGI GOLD, analysis project ID Ga0040974, genbank ID NZ_JQNQ01000001.1
- 'P164DRAFT_scf7180000000008_quiver', from Flavimarina sp. Hel_I_48. Data on DOE-JGI GOLD, analysis project ID Ga0005363, genbank ID JPOL01000002.1 (scaffold ...8)
- 'P164DRAFT_scf7180000000009_quiver', from Flavimarina sp. Hel_I_48. Data on DOE-JGI GOLD, analysis project ID Ga0005363, genbank ID JPOL01000003.1 (scaffold ...9)

Three are from separate papers, with no provided data. These are manually removed from the dataset.
- 'SEQ15336-1', from Bacteroides thetaiotaomicron 7330 in paper https://www.science.org/doi/10.1126/science.aac5992. No data provided...
- 'Contig5_1_7083079', from Bacteroides cellulosilyticus WH2 in paper https://doi.org/10.1371/journal.pbio.1001637. No data provided...
- 'SEQ15336-2_ori', from Bacteroides ovatus ATCCC 8483 in paper https://doi.org/10.1371/journal.pbio.1001221. No data provided...

Then there are 358 PULs from 44 unique accession IDs.

## Preprocessing
Cluster tables from dbCAN and PULDB were merged to a single table. 
After observing that some parent sequences are very short (131 with `sequence_length < 100kbp`), and consist mainly of the desired PUL, we used BLAST with the MegaBlast setting to find these shorter sequences in full genomes or larger contigs. BLAST results were filtered based on self-hits, identity percentage (>99%) and sequence length (max taken). 48 longer sequences were identified, and 16 of these were merged with sequences in the original dataset. Resulting in a new total of 386 genomes.

All-vs-all **OrthoANI** was used to find any overly similar sequences, so they can be de-duplicated. The ANI table is also used for train/test splitting. 
98 sequences with ANI > 99% were de-duplicated, leaving 288 genomes in the dataset.

### PULpy
PULpy used as extra annotations for bacteriodata sequences. The tool did not run immediately and some adaptations were made to the code:
1. `Snakefile` and all files in `scripts/` contained mismatching indentations (spaces and tabs), so all tabs were replaced by spaces.
2. `run_pulpy.sh` script was written, including all preparation scripts from the original `README.md`. Changes: all wget commands were added a `--no-check-certificate` flag, download links for dbCAN HMMs was updated to the current domain where the data is hosted (http://pro.unl.edu instead of depracated http://bcb.unl.edu).
3. Dependencies (snakemake and misceallaneous perl dependencies) were added to the environment definition at `envs/PULpy.yaml`.

PULpy identified 1753 PULs (many overlapping with existing annotations).

### Cblaster
Cblaster used to generate additional annotations for homologous PULs across the sequences in the dataset. The search is done using the following filters:
``` 
-min_eval 1.0e-9 -min_identity 70 -min_coverage 75 --gap_size 5000 --min_hits 2 --unique 2
``` 
All hits were filtered based on the number of genes from the query that were also in the hit, requiring a mimimum of `70%` of the genes to be present. With this search, 376 new homologous PULs were identified and added to the dataset.

An additional, more liberal, Cblaster search was performed to identify any "cryptic PULs" - regions in the genome that can possibly constitute a PUL but do not have any annotations. The default Cblaster filters were used, except for two changes (to account for PULs being possibly 2 genes):
```
--min_hits 2 --unique 2"
```
A threshold of `55%` of genes being present in the hit was used, to enforce the 2 gene minimum, but also include any hits of 2 genes from longer query PULs.

### GTDB-tk annotations
GTDB-tk (version 226) was used to generate taxonomic annotations for all of the genomes in the dataset. 

### Train-test split
Train-test splits are created in the `train_test_split.py` script. StratifiedGroupedKFold on taxonomic annotations, grouped by genus. 

Two additional splits were created, to test the generalizability of the models from bacteroidota to other phyla. One with only bacteroidota in the training set vice versa. 

### Running preprocessing scripts:
Order of scripts is currently as follows:
```bash
1) data_collection.py
2) slurm_run_gtdbtk.sh # potentially configuration required for hpc
3) data_collection.py # again... still need to fix is so that these two can be ran sequentially
4) slurm_run_orthoANI.sh
5) genecat_preprocess.sh
6) run_cblaster.py -rc -po, run_cblaster.py --liberal_filters -rc -po -gene_threshold 0.55
7) PULpy-master # need to run snakemake inside directory
9) integrate_pulpy.py
10) train_test_split.py
```

## Classifiers
- Gecco as baseline
- GeneCAT zeroshot (logistic regression on embeddings)
- GeneCAT finetuned
- pLM embeddings
- Bacformer emnbeddings

# TODO:
- finish genecat end-to-end finetuning
- finish Bacformer and ESM-C embeddings stuff
- handle negatives in training: add weights for logistic regression, add masks for finetuning, remove from training in gecco (???).
