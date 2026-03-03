import os
import subprocess
from pathlib import Path
import tempfile
import time
import polars
from Bio import Entrez, SeqIO
import numpy as np
import gb_io

EMAIL = 'b.rosenthal@lumc.com'
Entrez.email = EMAIL

def download_data_files():
    # check if required data files are present, if not download them
    data_files = os.listdir("../data")

    dbCAN_file = "data/dbCAN-PUL_Feb-2025.xlsx"
    img_genome_file = "data/IMG_2703719109.download.zip"
    puldb_scraped_file = "data/PULdb_data.parquet"

    if dbCAN_file not in data_files:
        cmd = "wget -O ../data/dbCAN-PUL_Feb-2025.xlsx https://pro.unl.edu/static/DBCAN-PUL/dbCAN-PUL_Feb-2025.xlsx"
        subprocess.run(cmd, shell=True, check=True)

    if img_genome_file not in data_files:
        print("IMG genome file for one PUL without genbank ID not found, cannot be downloaded automatically (I don't have an OrchidID)")
        # download at https://genome.jgi.doe.gov/portal/pages/dynamicOrganismDownload.jsf?organism=IMG_2703719109

    if puldb_scraped_file not in data_files:
        # run scraping script, might take a while
        cmd = "python src/scripts/scrape_puldb.py"
        subprocess.run(cmd, shell=True, check=True)

    return

def clean_cluster_table(cluster_path: str) -> polars.DataFrame:
    # read dbCAN PUL annotations from excel sheet
    pul_annotations = polars.read_excel(cluster_path).select(['ID', 'genomic_accession_number', 'nucleotide_position_range', 'ncbi_species_tax_id'])
    # add a column for the start and end positions of the PULs
    pul_annotations = pul_annotations.with_columns([
        polars.col('nucleotide_position_range').map_elements(lambda x: int(x.split(',')[0].split('-')[0]), int).alias('start'),
        polars.col('nucleotide_position_range').map_elements(lambda x: int(x.split(',')[0].split('-')[1]), int).alias('end'),
        polars.col('genomic_accession_number').map_elements(lambda x: x.split('.')[0].strip()).alias('sequence_id'),
        polars.col('ID').map_elements(lambda x: x.strip()).alias('cluster_id'),
        polars.col('ncbi_species_tax_id').alias('tax_id'),
    ]).sort('sequence_id')
    return pul_annotations.select(['sequence_id', 'cluster_id', 'start', 'end', 'tax_id'])

def replace_contig_spanning_pul(cluster_table: polars.DataFrame) -> polars.DataFrame:
    # TODO: finish this function, will not work
    # replace sequence id by "CP091800.1" and change start and end, for ADWO01000021
    cluster_table = cluster_table[cluster_table['cluster_id'] == 'PUL0456_1'].with_columns([
        polars.lit("CP091800.1").alias('sequence_id'),
        polars.lit(811569).alias('start'),
        polars.lit(817528).alias('end')
    ])
    # second entry for ADWO01000020
    cluster_table = cluster_table[cluster_table['cluster_id'] == 'PUL0456_2'].with_columns([
        polars.lit("CP091800.1").alias('sequence_id'),
        polars.lit(49194).alias('start'),
        polars.lit(67389).alias('end')
    ])

if __name__ == "__main__":
    download_data_files()
    cluster_table = clean_cluster_table("../data/dbCAN-PUL_Feb-2025.xlsx")