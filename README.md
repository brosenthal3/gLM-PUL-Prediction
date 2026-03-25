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
After observing that some parent sequences are very short, and consist mainly of the desired PUL, the dataset was filtered such that the sequence is larger than 100kbp:
```
sequence_length > 100000
```
This resulted in 131 sequences being filtered out. To potentially keep these PULs, we used BLAST with the MegaBlast setting to find these shorter sequences in full genomes or larger contigs. BLAST results were filtered based on self-hits, identity percentage (>99%) and sequence length (max taken). Replaced 67 sequences, to a new total of 384 sequences. 

All-vs-all OrthoANI was used to find any overly similar sequences, so they can be de-duplicated. The ANI table is also used for train/test splitting. 
92 sequences de-duplicated.

### PULpy
PULpy used as extra annotations for bacteriodata sequences. The tool did not run immediately and some adaptations were made to the code:
1. `Snakefile` and all files in `scripts/` contained mismatching indentations (spaces and tabs), so all tabs were replaced by spaces.
2. `run_pulpy.sh` script was written, including all preparation scripts from the original `README.md`. Changes: all wget commands were added a `--no-check-certificate` flag, download links for dbCAN HMMs was updated to the current domain where the data is hosted (http://pro.unl.edu instead of depracated http://bcb.unl.edu).
3. Dependencies (snakemake and misceallaneous perl dependencies) were added to the environment definition at `envs/PULpy.yaml`.