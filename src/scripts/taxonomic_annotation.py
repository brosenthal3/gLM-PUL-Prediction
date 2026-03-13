from pathlib import Path
import polars
from data_collection import merge_overlapping_puls

def filter_clusters_table():
    clusters_table = polars.read_csv("src/data/results/combined_clusters_blasted_gtdb.tsv", separator='\t', infer_schema_length=600)
    # remove original sequences that were later merged
    clusters_table = (clusters_table.filter((polars.col("merged") == "merged") | polars.col("merged").is_null()))

    # check that annotations are the same for all species
    for rank in ["species"]:
        mismatch = clusters_table.filter(polars.col(rank) != polars.col(f"{rank}_new"), (polars.col("blast_status") == True))
        print(f"There are {mismatch.shape[0]} mismatches between original and new sequence annotations on {rank} level.")
        print(mismatch.select("sequence_id", "length", "new_sequence_id", "new_length", rank, rank+"_new"))

    original_cols = [col for col in clusters_table.columns if not "new" in col]
    fixed_cols = ["cluster_id", "tax_id", "database", "merged", "blast_status"]
    new_cols = [col for col in clusters_table.columns if "new" in col]
    rename_map = {old: old.replace("new", "").strip("_") for old in new_cols if "new" in old}

    # get rows where blast_status is null or False, and select original columns
    clusters_table_original = (
        clusters_table
        .filter(polars.col("blast_status") == False)
        .select(original_cols)
    )
    print(f"To not replace: {clusters_table_original.select('sequence_id').n_unique()} unique sequences.")

    # get rows to replace by blast results, replace with new columns
    clusters_table_blasted = (
        clusters_table
        .filter(polars.col("blast_status") == True)
        .select(fixed_cols + new_cols)
        .rename(rename_map)
        .select(original_cols)
    )
    print(f"To replace left: {clusters_table.filter(polars.col("blast_status") == True).select('sequence_id').n_unique()} unique sequences.")
    print(f"To replace right: {clusters_table_blasted.select('sequence_id').n_unique()} unique sequences.\n\n")


    clusters_table_full = clusters_table_original.vstack(clusters_table_blasted)
    clusters_table_full = merge_overlapping_puls(clusters_table_full, blast=True, keep_original=False).sort('merged')

    pul_lengths = (
        clusters_table_full
        .group_by('sequence_id')
        .agg(
            (polars.col('end') - polars.col('start')).sum().alias('pul_length_sum'),
            polars.col('length').first(),
        ) # length of all puls in sequence, full sequence length
        .with_columns((100 * polars.col('pul_length_sum') / polars.col('length')).alias('percentage_in_puls')) # % of puls in genome
        .select('sequence_id', 'length', 'pul_length_sum', 'percentage_in_puls') # select only relevant columns
    )
    # merge back with cluster table
    clusters_table_filtered = (
        clusters_table_full
        .drop(['length', 'pul_length_sum', 'percentage_in_puls'])
        .join(pul_lengths, on='sequence_id', how='left')
        .sort("cluster_id")
        .sort("blast_status")
        .sort("merged")
        .select(original_cols)
    )

    clusters_table_grouped = clusters_table_filtered.select("sequence_id", "blast_status").filter(polars.col('blast_status') == True)
    print(f"Selected {clusters_table_grouped['sequence_id'].unique().shape[0]} sequences with blast results")


    print(f"{clusters_table_filtered.filter(polars.col("class").is_null()).shape[0]} rows with no taxonomic information from GTDB")
    print(f"Total of {clusters_table_filtered.select('sequence_id').unique().shape[0]} unique sequences in the filtered table")
    clusters_table_filtered.write_csv("src/data/results/combined_clusters_blasted_gtdb_filtered.tsv", separator='\t')
    return clusters_table_filtered


def separate_classification(classification: polars.Expr, index) -> polars.Expr:
    return classification.str.split(by=";").list.get(index, null_on_oob=True).str.split(by="__").list.get(1, null_on_oob=True)

def main():
    taxonomic_annotation_path = Path("src/data/results/gtdbtk.bac120.summary.tsv")
    taxonomic_annotation = polars.read_csv(taxonomic_annotation_path, separator="\t").select('user_genome', 'classification')
    taxonomic_annotation = (
        taxonomic_annotation
        .with_columns(
            separate_classification(polars.col("classification"), 0).alias("domain"),
            separate_classification(polars.col("classification"), 1).alias("phylum"),
            separate_classification(polars.col("classification"), 2).alias("class"),
            separate_classification(polars.col("classification"), 3).alias("order"),
            separate_classification(polars.col("classification"), 4).alias("family"),
            separate_classification(polars.col("classification"), 5).alias("genus"),
            separate_classification(polars.col("classification"), 6).alias("species"),
        )
        .rename({'user_genome': 'sequence_id'})
        .drop("classification")
    )
    null_count = taxonomic_annotation.filter(polars.col("domain").is_null()).select("sequence_id").to_series()
    print(f"There are {null_count.shape[0]}/{taxonomic_annotation.shape[0]} genomes that did not get annotated.")
    taxonomic_annotation.write_csv("src/data/results/taxonomic_annotation.tsv", separator="\t")

    # merge into clusters table
    clusters_table = polars.read_csv("src/data/results/combined_clusters_blasted.tsv", separator="\t", infer_schema_length=300)
    # merge taxonomic annotation into clusters table, both on sequence_id and new_sequence_id
    clusters_table = (
        clusters_table
        .join(taxonomic_annotation, on="sequence_id", how="left")
        .join(taxonomic_annotation, left_on="new_sequence_id", right_on="sequence_id", how="left", suffix="_new")
    )
    clusters_table.write_csv("src/data/results/combined_clusters_blasted_gtdb.tsv", separator="\t")
    filter_clusters_table()


if __name__ == "__main__":
    main()