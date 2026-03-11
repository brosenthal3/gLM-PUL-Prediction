import polars

def filter_clusters_table():
    clusters_table = polars.read_csv("src/data/results/combined_clusters_blasted_gtdb.tsv", separator='\t', infer_schema_length=600)
    clusters_table = (
        # add whether to select blast results or original data
        clusters_table
        .with_columns((polars.col("new_length").gt(polars.col("length")) & polars.col("new_percentage_in_puls").lt(polars.col("percentage_in_puls"))).alias("blast_status"))
        .filter((polars.col("merged") == "merged") | polars.col("merged").is_null())
    )

    original_cols = [col for col in clusters_table.columns if not "new" in col]
    fixed_cols = ["cluster_id", "tax_id", "database", "merged", "blast_status"]
    new_cols = [col for col in clusters_table.columns if "new" in col]
    rename_map = {old: old.replace("new", "").strip("_") for old in new_cols if "new" in old}

    # get rows where blast_status is null or False, and select original columns
    clusters_table_original = (
        clusters_table
        .filter(polars.col("blast_status").is_null())
        .select(original_cols)
        .vstack(
            clusters_table
            .filter(polars.col("blast_status") == False)
            .select(original_cols)
        )
    )
    # get rows to replace by blast results, replace with new columns
    clusters_table_blasted = (
        clusters_table
        .filter(polars.col("blast_status") == True)
        .select(fixed_cols + new_cols)
        .rename(rename_map)
        .select(original_cols)
    )
    clusters_table_full = clusters_table_original.vstack(clusters_table_blasted)

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
        .select(original_cols)
        .filter(polars.col("percentage_in_puls")<50)
        .filter(polars.col("length")>50000)
    )
    clusters_table_filtered.write_csv("src/data/results/combined_clusters_blasted_gtdb_filtered.tsv", separator='\t')
    return clusters_table_filtered


if __name__ == "__main__":
    filter_clusters_table()