# Repository overview

This repository contains the code, data-processing steps, and evaluation scripts used to build and analyze models for predicting polysaccharide utilization loci (PULs) from genomic sequences.

## Main structure
- `envs/`: conda environment files (see below).
- `src/scripts/`: preprocessing, model training, inference, and analysis scripts. Subdirectories include:
    - `/analysis`: misc scripts for analyzing the results.
    - `/visualization`: visualization scripts to create plots
    - `/shell`: all bash and slurm scripts
- `src/data/`: intermediate inputs, annotations, splits, and model outputs.
- `src/PULpy-master/`: PULpy workflow used for additional annotations.
- `results/`: plots and other final analysis outputs.


## Requirements and environments
The `/envs` directory contains `.yaml` files for 4 environments:
- `genecat`: data collection, preprocessing, GeneCAT & GECCO training, and logistic regression.
- `gtdbtk`: GTDB-Tk taxonomic classification only.
- `bacformer`: embedding extraction for ESM-C and Bacformer.
- `viz`: plotting, cluster-level evaluation, and UMAP/analysis scripts.

The environments can be created with mamba as follows:
```bash
mamba env create -n genecat -f envs/environment_genecat.yaml
mamba env create -n gtdbtk -f envs/environment_gtdbtk.yaml
mamba env create -n bacformer -f envs/environment_bacformer.yaml
mamba env create -n viz -f envs/environment_viz.yaml
```
Note: PULpy also requires its own environment, but it is already specified and created when running the `src/PULpy-master/run_pulpy.sh` script.  

### Other requirements:
- BLAST+: Some scripts for data preprocessing rely on the blast+ binaries (ncbi-blast-2.17.0+). Make sure these are installed and added to `PATH`. 

- GeneCAT CLI: All shell scripts that start with `genecat_` make use of the genecat cli. In the script, this is handled by the line `export PYTHONPATH='/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/src'`, but if this path changes then the script will fail.

- gLM-bench: for generating embeddings for bacformer and esmc, the gLM-bench toolkit is used. In the script this is done in the following line: `export PYTHONPATH='/exports/archive/lucid-grpzeller-primary/hackett/glm_bench'`

- GTDB data: For GTDB-tk, the R226 external data is currently in the following path: `export GTDBTK_DATA_PATH="/exports/archive/lucid-grpzeller-primary/SHARED/DATA/gene_catalogues/GTDBTK_R226/gtdbtk_r226_data"`

### Project directory:
All shell scripts rely on the path of the project directory. Currently on SHARK  this is `/exports/lucid-grpzeller-work/brosenthal/gLM-PUL-Prediction`. When running the shell scripts in another context, this has to be changed manually.

# Running preprocessing scripts:
Order of scripts is currently as follows:
```bash
# run data collection script. This downloads and cleans initial data, runs blast search on short contigs and downloads all genomes from ncbi
python src/scripts/data_collection.py

# run gtdb-tk annotations on hpc, takes ~1-2 hours
sbatch src/scripts/shell/gtdbtk.sh 

# run orthoANI and deduplication on HPC, can take up to 48 hours 
sbatch src/scripts/shell/orthoANI.sh

# preprocess genomes to get gene and feature tabkes
sbatch src/scripts/shell/genecat_preprocess.sh

# run cblaster searches
python src/scripts/run_cblaster.py -rc -po
python src/scripts/run_cblaster.py --liberal_filters -rc -po -gene_threshold 0.55

# generate train test splits
python src/scripts/train_test_split.py
python src/scripts/split_genes_and_features.py

# run PULpy
# Note: there might be issues running this, the pulpy script itself is old and unmaintained.
cd src/PULpy-master
bash run_pulpy.sh
cd ../../
python src/scripts/save_PULpy_annotations.py

# make list of cryptic PULs for masking in model training and evaluation
python src/scripts/cryptic_puls.py
```

The following files and directories are important for model training and evaluation:
- `src/data/splits/` => contains cluster tables of train/test splits:
    - `train_fold_k.tsv` => regular cluster table of only train set for fold k, used for all models
    - `train_fold_k_with_cryptic.tsv` => contains additional cryptic PULs indicated by
    `origin` column. Used for masked finetuning GeneCAT 
    - (this could probably only be one file instead of two, but at the moment this works fine.)
    
- `src/data/genecat_output/pfam.genes.parquet` => Gene table used for genecat training and in all evaluations (identical to `dbcan.genes.parquet` and `genome.genes.parquet`)
- `src/data/genecat_output/pfam.features.parquet` & `src/data/genecat_output/dbcan.pfam.features.parquet`=> Feature tables used for genecat
- `src/data/data_collection/clusters_deduplicated_cblaster.tsv` => Cluster table containing all literature-derived PULs after preprocessing
- `src/data/data_collection/cryptic_puls_genes.tsv` => All genes from cryptic PULs, used for masking during training
- `src/data/genecat_output/genome.genes.faa` => Amino-acid sequences, used for ESM-C and Bacformer

<br>
<br>

# Model training
## GECCO
Training GECCO and generating predictions on the test sets can be both done using one script that takes approximately 1 hour. Currently folds are not ran in parallel, but that can be changed to improve performance.
```bash
sbatch src/scripts/shell/gecco.sh
```

Output = two directories: `src/data/results/gecco_pfam` and `src/data/results/gecco_cazy`, each containing for every fold:
- dir: `fold_k` with all predicted clusters.
- dir: `model_k` with the model, selected domains and transition weights.
- files: `labeled_results_test_k.tsv` with the predicted probabilities per gene and the cluster labels.

## Finetuning GeneCAT
This script trains and evaluates 3 genecat models on 7 folds. By default, cryptic PULs are masked. The script needs to be manually changed to include them in training. The models are:

- GeneCAT pretrained with Pfam only
- GeneCAT pretrained with Pfam+CAZy
- GeneCAT without pretraining (untrained)

Takes quite a long time, 25-35 hours and runs an array of 7 slurm jobs, one for each fold. After training the models, you need to process the output so it can be used in the vizualization scripts.
```bash
sbatch src/scripts/shell/genecat_finetune.sh

# process output so it can be evaluated
python src/scripts/process_genecat_finetuning_output.py
```

## Extracting embeddings: GeneCAT, ESM-C and Bacformer
For zero-shot classification, we first need to get embeddings before running logistic regression:

```bash
# GeneCAT:
sbatch src/scripts/shell/genecat_get_embeddings.sh

# ESM-C and Bacformer:
sbatch src/scripts/shell/esmc_bacformer_embeddings.sh
```

Output = four directories for model results in `src/data/results/`: `genecat_zeroshot_cazy`, `genecat_zeroshot_cazy`, `esmc`, `bacformer`. Embeddings are saved in multiple places:
- dir: `src/data/embeddings/genecat_embeddings` => zero-shot genecat embs
- dir: `src/data/embeddings/esmc_bacformer_embeddings` => bacformer and esmc embs
- files: `src/data/results/<model_name>/fold_data/fold_k_data.parquet`, where each gene is labeled and marked as 'train' or 'test'. **These files are used downstream for logistic regression**.

Also the following are created by these scripts, but only for visualization/analysis purposes (UMAP):
- `src/data/embeddings/genecat_finetuned_embeddings` => pretrained-finetuned genecat embs
- `src/data/embeddings/genecat_untrained_embeddings` => untrained-finetuned genecat embs

## Logistic regression
One script runs logistic regression for all zero-shot models. Creates an array of 2 slurm jobs, one for logreg with and one for without masking cryptic PULs. Due to gridsearch and 7 folds not running in parallel, this can take a while (10-20 hours). There is room for optimization for sure.
```bash
sbatch src/scripts/shell/logistic_regression.sh
```
Results are saved as `labeled_results_test_k.tsv` in each model's results directory.

<br>
<br>

# Analysis and Visualization

For evaluations on a cluster-level, we need to combine gene-basis predicted probabilities into clusters with the following script:
```bash
python src/scripts/analysis/generate_clusters.py
```
Saves for each model: `predicted_clusters.parquet`, `predicted_clusters_5.parquet`, `predicted_clusters_6.parquet`  

To compare all models:
```bash
mamba activate viz
python src/scripts/visualization/evaluate_predictions.py --model selected
```
Creates plots in `results/plots/aggregated`:
- `barplot_selected.png`
- `pr_curves_selected.png`
- `roc_curves_selected.png`
- `pul_length_barplot.png`

There are a few more scripts in this directory for making other plots for the thesis, but no use discussing them one-by-one.

<br>
<br>

# Specific Methods
All data is obtained from two main sources: dbCAN-PUL, and PULDB.

### dbCAN-PUL
Accession IDs are processed by splitting on the first dot and removing the version number afterwards. 
One genome is downloaded manually from JGI (`Ga0139390_150`, link: https://genome.jgi.doe.gov/portal/pages/dynamicOrganismDownload.jsf?organism=IMG_2703719109). One PUL spans over two accession ids (`ADWO01000021.1, ADWO01000020.1`). Using BLAST on NCBI, the PUL can be found in the full genome with accession ID CP091800. However, this seems to be two separate PULs, where `ADWO01000021` maps to `811569-817528` and `ADWO01000020` maps to `49194-67389`. These are manually added to the dataframe as two PULs.

### PULDB
Data on app literature derived PULs was scraped from the website using a custom script (`src/scripts/scrape_puldb.py`).
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

Cblaster used to generate additional annotations for homologous PULs across the sequences in the dataset. All hits were filtered based on the number of genes from the query that were also in the hit, requiring a mimimum of `70%` of the genes to be present. With this search, 376 new homologous PULs were identified and added to the dataset.

An additional, more liberal, Cblaster search was performed to identify any "cryptic PULs" - regions in the genome that can possibly constitute a PUL but do not have any annotations. The default Cblaster filters were used, except for two changes (to account for PULs being possibly 2 genes):
```
--min_hits 2 --unique 2"
```
A threshold of `55%` of genes being present in the hit was used, to enforce the 2 gene minimum, but also include any hits of 2 genes from longer query PULs.

GTDB-tk (version 226) was used to generate taxonomic annotations for all of the genomes in the dataset. 

PULpy used as extra annotations for bacteriodata sequences. The tool did not run immediately and some adaptations were made to the code:
1. `Snakefile` and all files in `scripts/` contained mismatching indentations (spaces and tabs), so all tabs were replaced by spaces.
2. `run_pulpy.sh` script was written, including all preparation scripts from the original `README.md`. Changes: all wget commands were added a `--no-check-certificate` flag, download links for dbCAN HMMs was updated to the current domain where the data is hosted (http://pro.unl.edu instead of depracated http://bcb.unl.edu).
3. Dependencies (snakemake and misceallaneous perl dependencies) were added to the environment definition at `envs/PULpy.yaml`.

### Gene calling and annotation
Using the `genecat` toolkit, we perform gene calling with Pyrodigal on all genomic sequences retrieved previously. 

Subsequently, all genes are annotated with Pfam and dbCAN HMMs.

### Train-test split
Train-test splits are created in the `train_test_split.py` script. StratifiedGroupedKFold on taxonomic annotations, grouped by genus. Two additional splits are created, to test the generalizability of the models from bacteroidota to other phyla. One with only bacteroidota in the training set vice versa. 
