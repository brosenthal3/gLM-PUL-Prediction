import json
import polars
from Bio import Entrez
Entrez.email = "b.rosenthal@lumc.nl"

# Entrez utility functions
def request_summary(acc, db="nuccore"):
    handle = Entrez.esummary(
        db=db,
        id=acc,
        retmode="json"
    )
    record = handle.read()
    handle.close()
    # parse the JSON response to get the sequence length
    record = json.loads(record)
    return record

def request_sequence(acc, db="nuccore"):
    handle = Entrez.efetch(
        db=db,
        id=acc,
        rettype="gb",
        retmode="json",
        complexity=1
    )
    record = handle.read()
    handle.close()
    return record


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


def reset_start_end(table: polars.DataFrame) -> polars.DataFrame:
    return table.with_columns(
        polars.when(polars.col("start") < polars.col("end")).then(polars.col("start")).otherwise(polars.col("end")).alias("start"),
        polars.when(polars.col("start") < polars.col("end")).then(polars.col("end")).otherwise(polars.col("start")).alias("end"),
    )


def join_gene_and_PUL_table(gene_table: polars.DataFrame, cluster_table: polars.DataFrame, buffer: int = 100,) -> polars.DataFrame:
    gene_table = reset_start_end(gene_table)
    cluster_table = reset_start_end(cluster_table)

    labled_gene_table = (
        cluster_table
        .rename({"start": "pul_start", "end": "pul_end"}) # avoid column name conflicts
        .join(
            gene_table,
            on="sequence_id",
            how="inner",
            validate="m:m",
        )
        .with_columns(
            polars.when(
                polars.col("start") >= polars.col("pul_start") - buffer, # allow for some buffer around the PUL boundaries
                polars.col("end") <= polars.col("pul_end") + buffer,
            )
            .then(polars.col("cluster_id"))
            .otherwise(None)
            .alias("cluster_id"),
            polars.when(
                polars.col("start") >= polars.col("pul_start") - buffer,
                polars.col("end") <= polars.col("pul_end") + buffer,
            )
            .then(True)
            .otherwise(False)
            .cast(polars.Boolean)
            .alias("is_PUL")
        )
        # aggregate by protein_id to determine if protein is in any PUL
        .group_by("protein_id")
        .agg(
            polars.col("is_PUL").any().alias("is_PUL"),
            polars.col("sequence_id").first().alias("sequence_id"),
            polars.col("start").first().alias("start"),
            polars.col("end").first().alias("end"),
            polars.col("strand").first().alias("strand"),
            polars.col("cluster_id").drop_nulls().first().alias("cluster_id")
        )
        .sort(by=["sequence_id", "start", "end"])
        .with_row_index(name="gene_id", offset=0)  # important
        .select(["sequence_id", "protein_id", "start", "end", "strand", "is_PUL", "cluster_id"])
    )

    return labled_gene_table


def recompute_length_percentage(cluster_table: polars.DataFrame) -> polars.DataFrame:
    # recompute length and percentage in PUL for all clusters, since we added new ones
    # recalculate sum of length of PULs per genome and percentage of genome in PULs
    pul_lengths = (
        cluster_table
        .group_by('sequence_id')
        .agg(
            (polars.col('end') - polars.col('start')).sum().alias('pul_length_sum'),
            polars.col('length').first(),
        ) # length of all puls in sequence, full sequence length
        .with_columns((100 * polars.col('pul_length_sum') / polars.col('length')).alias('percentage_in_puls')) # % of puls in genome
        .select('sequence_id', 'length', 'pul_length_sum', 'percentage_in_puls') # select only relevant columns
    )
    # merge back with cluster table
    cluster_table = (
        cluster_table
        .drop(['length', 'pul_length_sum', 'percentage_in_puls'])
        .join(pul_lengths, on='sequence_id', how='left')
    )

    return cluster_table