import os
import subprocess
from pathlib import Path
import tempfile
import time
import polars
from Bio import Entrez, SeqIO
import numpy as np
import gb_io
from utility_scripts import request_summary, request_sequence
from tqdm import tqdm

EMAIL = 'b.rosenthal@lumc.com'
Entrez.email = EMAIL

def download_data_files(data_dir: str):
    dbCAN_file = f"{data_dir}/dbCAN-PUL_Feb-2025.xlsx"
    img_genome_file_zip = f"{data_dir}/genomes/IMG_2703719109.download.zip"
    img_genome_file = f"{data_dir}/genomes/Ga0139390_150.gb"
    puldb_scraped_file = f"{data_dir}/puldb_data.parquet"

    if not Path(dbCAN_file).exists():
        print("dbCAN PUL cluster file not found, downloading...")
        cmd = f"wget -O {dbCAN_file} https://pro.unl.edu/static/DBCAN-PUL/dbCAN-PUL_Feb-2025.xlsx"
        subprocess.run(cmd, shell=True, check=True)

    if not Path(puldb_scraped_file).exists():
        print("PULdb scraped data file not found, scraping...")
        # run scraping script, might take a while
        cmd = f"python src/scripts/scrape_puldb.py"
        subprocess.run(cmd, shell=True, check=True)

    if not Path(img_genome_file).exists() and not Path(img_genome_file_zip).exists():
        print("IMG genome file for one PUL without genbank ID not found, cannot be downloaded automatically (I don't have an OrchidID)")
        # download at https://genome.jgi.doe.gov/portal/pages/dynamicOrganismDownload.jsf?organism=IMG_2703719109

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
    

def merge_overlapping_puls(df):
    merged_puls = polars.DataFrame(schema=df.schema)

    for sequence_id, group in df.group_by('sequence_id'):
        if group.shape[0] == 1:
            merged_puls = merged_puls.vstack(group[0])
            continue

        # sort by start position
        group = group.sort('start')
        current_pul = None
        for row in group.iter_rows(named=True):
            if current_pul is None:
                current_pul = row
            else:
                # check if there is an overlap with the current PUL
                if row['start'] <= current_pul['end']:
                    # merge the PULs by updating the end position to the maximum end position
                    current_pul['end'] = max(current_pul['end'], row['end'])
                    # merge cluster_id by concatenating with an underscore
                    current_pul['cluster_id'] = f"{current_pul['cluster_id']}_{row['cluster_id']}"
                else:
                    merged_puls = merged_puls.vstack(polars.DataFrame([current_pul]))
                    current_pul = row

        # add the last PUL after processing all rows for this sequence_id
        if current_pul is not None:
            merged_puls = merged_puls.vstack(polars.DataFrame([current_pul]))
            
    return polars.DataFrame(merged_puls)


def get_length(acc):
    # get esummary from ncbi
    record = request_summary(acc)

    if 'error' in record.keys():
        # try getting full sequence and parsing length from there
        record = request_sequence(acc)
        length = record.split('\n')[0].split()[2]
    else:
        uid = record['result']['uids'][0]
        length = record['result'][uid]['slen']

    try:
        length = int(length) 
    except ValueError:
        length = None

    return length


def get_sequence_lengths(unique_accessions: polars.DataFrame) -> list:
    lengths = []
    for acc in tqdm(list(unique_accessions['sequence_id']), desc="Fetching sequence lengths"):
        try:
            length = get_length(acc)
        except Exception as e:
            print(f"Error fetching length for {acc}, skipping.")
            length = None

        lengths.append({'sequence_id': acc, 'length': length})
        time.sleep(0.1)

    return lengths


def merge_blast_hits(combined_clusters, blast_output):
    # format output
    blast_output_valid = (
        blast_output
        .filter(~polars.col('subject_accession').eq('NO_HIT'))
        .with_columns(
            polars.col('subject_start').cast(polars.Int64),
            polars.col('subject_end').cast(polars.Int64)
        )
        .rename({
            'subject_accession': 'sequence_id', 
            'subject_start': 'start', 
            'subject_end': 'end'
        })
        .select('cluster_id', 'sequence_id', 'start', 'end')
    )
    # get lengths and percentage in PULs for blast hits
    blast_output_with_lengths = merge_with_lengths(blast_output_valid, data_dir, lengths_path=f"{data_dir}/results/blast_sequence_lengths")
    # rename again to prepare for merging
    blast_output_with_lengths = blast_output_with_lengths.rename({
        'sequence_id': 'new_sequence_id', 
        'start': 'new_start', 
        'end': 'new_end', 
        'length': 'new_length', 
        'pul_length_sum': 'new_pul_length_sum',
        'percentage_in_puls': 'new_percentage_in_puls'
    })
    clusters_with_blast = combined_clusters.join(blast_output_with_lengths, on='cluster_id', how='left')
    print(f"Found {blast_output_with_lengths.shape[0]} valid blast hits")

    return clusters_with_blast

    # total_replaced = 0
    # # replace accessions in combined cluster table with blast hits where available
    # for row in blast_output_valid.iter_rows(named=True):
    #     pul = row["cluster_id"]
    #     new_accession = row["subject_accession"]
    #     new_pul_range = (int(row["subject_start"]), int(row["subject_end"]))
    #     old_pul_range = (int(row["query_start"]), int(row["query_end"]))
    #     new_genome_length = get_length(new_accession)

    #     # replace all necessary values 
    #     combined_clusters_adapted = combined_clusters_adapted.with_columns(
    #         polars.when(polars.col("cluster_id") == pul)
    #         .then(polars.lit(new_accession))
    #         .otherwise(polars.col("sequence_id"))
    #         .alias("sequence_id"),

    #         polars.when(polars.col("cluster_id") == pul)
    #         .then(polars.lit(new_pul_range[0]))
    #         .otherwise(polars.col("start"))
    #         .alias("start"),

    #         polars.when(polars.col("cluster_id") == pul)
    #         .then(polars.lit(new_pul_range[1]))
    #         .otherwise(polars.col("end"))
    #         .alias("end"),
    #     )
    #     total_replaced += 1
    
    # print(f"Replaced {total_replaced} PULs with blast hits.")


def get_non_genbank_genome():
    # check if exists
    path = f"{data_dir}/genomes/Ga0139390_150.gb"
    if Path(path).exists():
        return path

    # get the one non-genbank genome into a single genbank file
    jgi_genome = f"{data_dir}/genomes/IMG_2703719109/IMG Data/99440.assembled.gbk"
    contig_id = "Ga0139390_150 Ga0139390_150"
    # save contig as genbank file
    for record in gb_io.iter(jgi_genome):
        if record.definition == contig_id:
            gb_io.dump(record, path)


def get_dbcan_clusters(data_dir):
    dbcan_clusters_path = f"{data_dir}/results/dbcan_clusters.tsv"
    if Path(dbcan_clusters_path).exists():
        print(f"dbCAN clusters file already exists at {dbcan_clusters_path}, loading from file.")
        return polars.read_csv(dbcan_clusters_path, separator='\t')
    else:
        dbcan_clusters = clean_dbcan(f"{data_dir}/dbCAN-PUL_Feb-2025.xlsx")
        dbcan_clusters.write_csv(dbcan_clusters_path, separator='\t')


def get_puldb_clusters(data_dir):
    puldb_clusters_path = f"{data_dir}/results/puldb_clusters.tsv"
    if Path(puldb_clusters_path).exists():
        print(f"PULdb clusters file already exists at {puldb_clusters_path}, loading from file.")
        puldb_clusters = polars.read_csv(puldb_clusters_path, separator='\t')
    else:
        puldb_clusters = clean_puldb_data(f"{data_dir}/puldb_data.parquet")
        puldb_clusters.write_csv(puldb_clusters_path, separator='\t')

    puldb_clusters = puldb_clusters.with_columns(polars.col("tax_id").cast(polars.Int64), polars.col("cluster_id").cast(polars.Utf8))
    return puldb_clusters


def merge_with_lengths(cluster_table, data_dir, lengths_path):
    # merge lengths with cluster table
    if Path(lengths_path).exists():
        print(f"Sequence lengths file already exists at {lengths_path}, loading from file.")
        lengths_df = polars.read_csv(lengths_path, separator='\t')
    else:
        print(f"Calculating length of PULs, this may take a few minutes...\n")
        # get length of all genomes
        unique_accessions = cluster_table.select('sequence_id').unique()
        lengths = get_sequence_lengths(unique_accessions)
        lengths_df = polars.DataFrame(lengths, schema={'sequence_id': polars.Utf8, 'length': polars.Int64}) 
        # save intermediate lengths
        lengths_df.write_csv(lengths_path, separator='\t')

    # get total length of PULs per genome and calculate percentage of genome in PULs
    pul_lengths = (
        cluster_table
        .group_by('sequence_id')
        .agg((polars.col('end') - polars.col('start')).sum().alias('pul_length_sum')) # length of all puls in sequence
        .join(lengths_df, on='sequence_id', how='left') # add length of genome
        .with_columns((100 * polars.col('pul_length_sum') / polars.col('length')).alias('percentage_in_puls')) # % of puls in genome
        .select('sequence_id', 'length', 'pul_length_sum', 'percentage_in_puls') # select only relevant columns
    )
    # merge back with cluster table
    cluster_table_With_length = cluster_table.join(pul_lengths, on='sequence_id', how='left').sort("cluster_id")

    assert cluster_table.shape[0] == cluster_table_With_length.shape[0], "Length table has different number of rows than cluster table, something went wrong."
    return cluster_table_With_length


def get_genomes(data_dir, ids):
    output_path = f"{data_dir}/genomes/combined_genomes.gb"
    output_ids_path = f"{data_dir}/genomes/combined_genomes.ids.txt"
    if Path(output_path).exists() and Path(output_ids_path).exists():
        with open(output_ids_path, "r") as id_handle:
            fetched_ids = set(id_handle.read().splitlines())
        missing_ids = [acc for acc in ids if acc not in fetched_ids]
        if missing_ids:
            print(f"File exists but some IDs are missing, fetching {len(missing_ids)} missing records...")
            run_genomes_fetcher(data_dir, output_path)
        else:
            print(f"GenBank records file already exists at {output_path} with no missing IDs, skipping fetching step.")
        return
    else:
        run_genomes_fetcher(data_dir, output_path)


def run_genomes_fetcher(data_dir, output_path):
    cmd = f"python src/scripts/ncbi_record_fetcher.py -i {data_dir}/results/unique_sequence_ids.tsv -o {output_path} --email {EMAIL} --type genbank"
    subprocess.run(cmd, shell=True, check=True)


def main(data_dir, filter_truncated):
    download_data_files(data_dir)

    # get cluster tables
    dbcan_clusters = get_dbcan_clusters(data_dir)
    puldb_clusters = get_puldb_clusters(data_dir)

    # combine dbcan and puldb to one cluster table, with column of database origin (dbcan or puldb)
    combined_clusters = polars.concat([
        dbcan_clusters.select(['cluster_id', 'sequence_id', 'start', 'end', 'tax_id']).with_columns(polars.lit("dbcan").alias("database")),
        puldb_clusters.select(['cluster_id', 'sequence_id', 'start', 'end', 'tax_id']).with_columns(polars.lit("puldb").alias("database"))
    ], how='vertical')
    combined_clusters = merge_overlapping_puls(combined_clusters).sort('cluster_id')
    # get length and percentage of genome in PULs
    combined_clusters = merge_with_lengths(combined_clusters, data_dir, lengths_path=f"{data_dir}/results/sequence_lengths.tsv")
    combined_clusters.write_csv(f"{data_dir}/results/combined_clusters.tsv", separator='\t')

    # find which genomes are likely to be truncated
    truncated_genomes = combined_clusters.filter(polars.col('percentage_in_puls') > 50, polars.col('length')<1000000)
    # get all puls that are in these truncated genomes
    truncated_genomes_puls = combined_clusters.join(truncated_genomes.select('sequence_id'), left_on='sequence_id', right_on='sequence_id', how='semi')
    truncated_genomes_puls.write_csv(f"{data_dir}/results/truncated_genomes.tsv", separator='\t')

    # check if blast results for truncated genomes already exist, if not run blast for all truncated genomes
    if not Path(f"{data_dir}/results/blast_results.tsv").exists():
        cmd = f"python src/scripts/blast_truncated_sequences.py -i {data_dir}/results/truncated_genomes.tsv -o {data_dir}/results/blast_results.tsv --email {EMAIL}"
        subprocess.run(cmd, shell=True, check=True)
    else:
        print(f"Blast results file already exists at {data_dir}/results/blast_results.tsv, skipping blast step.\n")
    
    # open blast results
    blast_output = polars.read_csv(f"{data_dir}/results/blast_results.tsv", separator='\t')
    # replace short PULs with blast hits where possible
    combined_clusters_blasted = merge_blast_hits(combined_clusters, blast_output).sort('cluster_id')
    # merge again
    len_before = combined_clusters_blasted.shape[0]
    combined_clusters_blasted = merge_overlapping_puls(combined_clusters_blasted)
    print(f"Merging again after blasting reduced PULs from {len_before} to {combined_clusters_blasted.shape[0]}.")
    combined_clusters_blasted.write_csv(f"{data_dir}/results/combined_clusters_blasted.tsv", separator='\t')

    # if filter_truncated:
    #     blasted_percentage_in_puls = get_percentage_bp_in_puls_df(combined_clusters_blasted, data_dir, path=f"{data_dir}/results/percentage_in_puls_blasted.tsv")
    #     blasted_percentage_in_puls.write_csv(f"{data_dir}/results/percentage_in_puls_blasted.tsv", separator='\t')

    #     blasted_truncated_genomes = blasted_percentage_in_puls.filter(polars.col('percentage_in_puls') > 50, polars.col('length')<1000000)
    #     print(f"\nAfter blasting, there are {blasted_truncated_genomes.shape[0]} truncated genomes left, down from {truncated_genomes.shape[0]} before blasting.")
    #     blasted_truncated_genomes_puls = combined_clusters_blasted.join(blasted_truncated_genomes.select('sequence_id'), left_on='sequence_id', right_on='sequence_id', how='semi')
    #     # filter out truncated genomes
    #     combined_clusters_filtered = combined_clusters_blasted.join(blasted_truncated_genomes.select('sequence_id'), left_on='sequence_id', right_on='sequence_id', how='anti')
    #     combined_clusters_filtered.write_csv(f"{data_dir}/results/combined_clusters_filtered.tsv", separator='\t')
    # else:
    #     combined_clusters_filtered = combined_clusters_blasted

    raise NotImplementedError

    # create file of unique accession ids from cluster tables
    unique_accessions = combined_clusters_filtered['sequence_id'].unique().to_frame(name='sequence_id')
    unique_accessions.write_csv(f'{data_dir}/results/unique_sequence_ids.tsv', separator='\t')
    print(f"There are {len(unique_accessions)} unique sequence ids in the cluster table.")

    # run genecat script for fetching ncbi genomes.
    print("Fetching GenBank records, might take a while...")
    get_genomes(data_dir, unique_accessions['sequence_id'].to_list())

    # check if need to add the one non-genbank genome
    cmd = f"grep 'Ga0139390_150' {data_dir}/genomes/combined_genomes.gb"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if "Ga0139390_150" in result.stdout:
        print("Ga0139390_150 is already in the .gb file.")
    else:
        print("Ga0139390_150 is not in the .gb file, adding it now.")
        non_genbank_genome_path = get_non_genbank_genome()
        cmd = f"cat {non_genbank_genome_path} >> {data_dir}/genomes/combined_genomes.gb"
        subprocess.run(cmd, shell=True, check=True)

    print(f"Data collection and cleaning complete! Full genomes file in {data_dir}/genomes/combined_genomes.gb")

if __name__ == "__main__":
    # TODO set path for genecat

    data_dir = "src/data"
    filter_truncated = True
    main(data_dir, filter_truncated)
