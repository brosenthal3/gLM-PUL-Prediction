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

# set path for genecat

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


def clean_dbcan(cluster_path: str) -> polars.DataFrame:
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
    
    cluster_table = pul_annotations.select(['sequence_id', 'cluster_id', 'start', 'end', 'tax_id'])
    # add row for PUL0456
    pul0456_1 = polars.DataFrame({
        'sequence_id': ['CP091800', 'CP091800'],
        'cluster_id': ['PUL0456_1', 'PUL0456_2'],
        'start': [811569, 49194],
        'end': [817528, 67389],
        'tax_id': [77095, 77095]
    })
    cluster_table = polars.concat([cluster_table, pul0456_1], how='vertical').sort('cluster_id')

    # remove ADWO01000020 and/or ADWO01000021
    cluster_table = cluster_table.filter(~polars.col('sequence_id').is_in(['ADWO01000020', 'ADWO01000021']))
    return cluster_table


def clean_puldb_data(puldb_path: str) -> polars.DataFrame:
    # clean puldb data
    puldb_data = polars.read_parquet(puldb_path)
    puldb_data = puldb_data.with_columns([
        polars.col('accession').map_elements(lambda x: x.split('.')[0].strip()).alias('sequence_id'),
        polars.col('pul_id').alias('cluster_id').cast(polars.Utf8),
        polars.col('start_bp').alias('start'),
        polars.col('end_bp').alias('end')  
    ]).sort('sequence_id')["cluster_id", "sequence_id", "start", "end", "status"]
    puldb_data_literature = puldb_data.filter(polars.col("status").eq("literature"))

    # replace invalid ids with genbank where available
    id_mapping = {
        "FG27DRAFT_unitig_0_quiver_dupTrim_7536": "NZ_JQNQ01000001",
        "P164DRAFT_scf7180000000008_quiver": "JPOL01000002",
        "P164DRAFT_scf7180000000009_quiver": "JPOL01000003"
    }
    puldb_data_literature = puldb_data_literature.with_columns(polars.col("sequence_id").replace(id_mapping))

    # remove other invalid ids that cannot be mapped to genbank accessions
    invalid_ids = ['SEQ15336-2_ori', 'Contig5_1_7083079', 'SEQ15336-1']
    puldb_data_literature = puldb_data_literature.filter(~polars.col("sequence_id").is_in(invalid_ids))
    # add tax_id column to puldb_data_literature so it has the same columns as the cluster table
    puldb_data_literature = puldb_data_literature.with_columns(polars.lit(None).alias("tax_id").cast(polars.Int64))
    puldb_data_literature = puldb_data_literature.select(['cluster_id', 'sequence_id', 'start', 'end', 'tax_id'])
    return puldb_data_literature
    

def get_sequence_lengths(unique_accessions: polars.DataFrame) -> list:
    lengths = []
    errors = []
    for acc in list(unique_accessions['sequence_id']):
        record = request_summary(acc)
        if 'error' in record.keys():
            # try getting full sequence and parsing length from there
            record = request_sequence(acc)
            length = record.split('\n')[0].split()[2]
        else:
            uid = record['result']['uids'][0]
            length = record['result'][uid]['slen']
        lengths.append({'sequence_id': acc, 'length': length})
        time.sleep(0.1)

    # Transform to int, handle string errors
    for row in lengths:
        try:
            row['length'] = int(row['length'])
        except ValueError:
            row['length'] = None

    return lengths


if __name__ == "__main__":
    download_data_files()
    # clean dbCAN data
    dbcan_clusters = clean_dbcan("../data/dbCAN-PUL_Feb-2025.xlsx")
    # save intermediate cleaned data
    dbcan_clusters.write_csv('../data/dbcan_clusters.tsv', separator='\t')

    # save intermediate cleaned data
    puldb_clusters = clean_puldb_data("../data/PULdb_data.parquet")
    puldb_clusters.write_csv('../data/puldb_clusters.tsv', separator='\t')

    # combine dbcan and puldb to one cluster table, with column of database origin (dbcan or puldb)
    combined_clusters = polars.concat([
        dbcan_clusters.select(['cluster_id', 'sequence_id', 'start', 'end', 'tax_id']).with_columns(polars.lit("dbcan").alias("database")),
        puldb_clusters.select(['cluster_id', 'sequence_id', 'start', 'end', 'tax_id']).with_columns(polars.lit("puldb").alias("database"))
    ], how='vertical').sort('cluster_id')
    display(combined_clusters)

    # get sequence lengths for all unique accessions in cluster table using Entrez esummary
    unique_accessions = combined_clusters.select('sequence_id').unique()
    lengths = get_sequence_lengths(unique_accessions)

    # merge lengths with cluster table
    lengths_df = polars.DataFrame(lengths, schema={'sequence_id': polars.Utf8, 'length': polars.Int64}) 
    cluster_table_with_length = combined_clusters.join(lengths_df, on='sequence_id', how='left')


    def get_percentage_bp_in_puls(cluster_table, length_df):
        # get total length of PULs per genome
        pul_lengths = cluster_table.group_by('sequence_id').agg(
            (polars.col('end') - polars.col('start')).sum().alias('pul_length')).join(length_df, on='sequence_id', how='left')
        pul_lengths = pul_lengths.with_columns(
            (polars.col('pul_length') / polars.col('length') * 100).alias('percentage_in_puls')
        )
        return pul_lengths

    percentage_in_puls = get_percentage_bp_in_puls(combined_clusters, lengths_df)
    percentage_in_puls.write_csv('../data/percentage_bp_in_puls.tsv', separator='\t')

    truncated_genomes = percentage_in_puls.filter(polars.col('percentage_in_puls') > 20, polars.col('length')<1000000)
    truncated_genomes_puls = combined_clusters.filter(polars.col('sequence_id').is_in(truncated_genomes['sequence_id']))
