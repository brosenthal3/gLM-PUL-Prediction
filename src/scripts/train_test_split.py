import polars
from sklearn.model_selection import GroupKFold
from pathlib import Path
#from data_collection import merge_overlapping_puls

def merge_overlapping_puls(df, group_col='sequence_id', start_col='start', end_col='end', blast=False, keep_original=True):
    merged_puls = polars.DataFrame(schema=df.schema)
    merged_ids = []

    for sequence_id, group in df.group_by(group_col):
        if group.shape[0] == 1 or sequence_id[0] is None:
            merged_puls = merged_puls.vstack(polars.DataFrame(group))
            continue

        # sort by start position
        group = group.sort(start_col)
        current_pul = None
        for row in group.iter_rows(named=True):
            if current_pul is None:
                current_pul = row
            else:
                # check if there is an overlap with the current PUL
                if row[start_col] <= current_pul[end_col]:
                    # merge the PULs by updating the end position to the maximum end position
                    current_pul[end_col] = max(current_pul[end_col], row[end_col])
                    # merge cluster_id by concatenating with an underscore
                    current_pul['cluster_id'] = f"{current_pul['cluster_id']}_{row['cluster_id']}"
                    # add taxonomic id if exists
                    current_pul['tax_id'] = current_pul['tax_id'] if current_pul['tax_id'] is not None else row['tax_id']
                    # merge database column by concatenating with an underscore if different
                    current_pul['database'] = f"{current_pul['database']}_{row['database']}" if current_pul['database'] not in row['database'] else current_pul['database']
                    merged_ids.append({'cluster_id': current_pul['cluster_id'], 'merged': "merged_blast" if blast else "merged"})                        
                else:
                    merged_puls = merged_puls.vstack(polars.DataFrame([current_pul]))
                    current_pul = row

        # add the last PUL after processing all rows for this sequence_id
        if current_pul is not None:
            merged_puls = merged_puls.vstack(polars.DataFrame([current_pul]))

    # add column for which puls are merged
    merged_puls = (
        merged_puls
        .join(polars.DataFrame(merged_ids), on="cluster_id", how="left")
    )
    if keep_original:
        # add all puls from original table that were originally unmerged
        previously_merged = {"cluster_id": [item for sublist in [cluster_id['cluster_id'].split("_") for cluster_id in merged_ids] for item in sublist]}
        merged_puls = merged_puls.vstack(
            df
            .join(polars.DataFrame(previously_merged).unique(), on='cluster_id', how='semi')
            .with_columns(polars.lit("original").alias('merged'))
        )

    return merged_puls.sort('cluster_id').sort('merged')
def filter_clusters_table():
    clusters_table = polars.read_csv("src/data/results/combined_clusters_blasted_gtdb.tsv", separator='\t', infer_schema_length=600)
    clusters_table = (
        # add whether to select blast results or original data
        clusters_table
        .with_columns((polars.col("new_length").gt(polars.col("length")+1000)).alias("blast_status"))
        .filter((polars.col("merged") == "merged") | polars.col("merged").is_null())
    )

    # check that annotations are the same
    for rank in ["genus", "species"]:
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


def split_dataset(clusters_table, k):
    group_kfold = GroupKFold(n_splits=k)
    X = clusters_table.select("cluster_id").to_series()
    clusters_table = clusters_table.with_columns(polars.col("class").fill_null("unknown"))
    groups = (
        clusters_table
        .with_columns(polars.col("class").cast(polars.Categorical).alias("class_cat"))
        .select(polars.col("class_cat").to_physical())
        .to_series()
    )
    for i, (train_index, test_index) in enumerate(group_kfold.split(X, groups=groups)):
        print(f"Fold {i}:")
        print(f"  Train groups={groups[train_index].unique()}")
        print(f"  Test groups={groups[test_index].unique()}")

if __name__ == "__main__":
    # clusters_table_path = "src/data/results/combined_clusters_blasted_gtdb_filtered.tsv"
    # if Path(clusters_table_path).exists():
    #     clusters_table = polars.read_csv(clusters_table_path, separator='\t')
    # else:
    clusters_table = filter_clusters_table()
    k = 5 # later as argument
#    split_dataset(clusters_table, k)

    
# def report_stats_on_gene_table(
#     gene_table: polars.DataFrame,
#     label: str,
#     stat_file: TextIO,
#     full_table_counts: Dict[str, int],
# ) -> None:

#     stat_file.write(
#         f"{len(gene_table)} genes in {label} set. Percentage: {round(gene_table['protein_id'].n_unique()*100/full_table_counts['genes'], 2)}\n"
#     )
#     stat_file.write(
#         f"{gene_table['sequence_id'].n_unique()} contigs in {label} set. Percentage: {round(gene_table['sequence_id'].n_unique()*100/full_table_counts['contigs'],2)}\n"
#     )

#     stat_file.write(
#         f"{gene_table['genome_id'].n_unique()} genomes in {label} set. Percentage: {round(gene_table['genome_id'].n_unique()*100/full_table_counts['genomes'],2)}\n"
#     )

#     if "taxonomy" in gene_table.columns:
#         stat_file.write(
#             f"{gene_table['taxonomy'].n_unique()} taxa in {label} set. Percentage: {round(gene_table['taxonomy'].n_unique()*100/full_table_counts['taxa'],2)}\n"
#         )

#     # annotated_genes = gene_table.filter(polars.col("domain").is_not_null())[
#     #     "protein_id"
#     # ].n_unique()
#     # stat_file.write(
#     #     f"Number of Annotated Genes: {annotated_genes} in {label} set. Percentage: {round(annotated_genes*100/gene_table['protein_id'].n_unique(),2)}\n\n"
#     # )

