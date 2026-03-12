import polars
from sklearn.model_selection import GroupKFold
from pathlib import Path

def filter_clusters_table():
    clusters_table = polars.read_csv("src/data/results/combined_clusters_blasted_gtdb.tsv", separator='\t', infer_schema_length=600)
    clusters_table = (
        # add whether to select blast results or original data
        clusters_table
        .with_columns((polars.col("new_length").gt(polars.col("length")+1000)).alias("blast_status"))
        .filter((polars.col("merged") == "merged") | polars.col("merged").is_null())
    )
    clusters_table_grouped = clusters_table.group_by("sequence_id").agg(polars.col("blast_status").any().alias("sequence_blast_status"))
    print(f"Selected {clusters_table_grouped.filter(polars.col('sequence_blast_status') == True).shape[0]} sequences with blast results")

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
#        .filter(polars.col("length")>100000)
    )

#    print(f"After filtering, there are {clusters_table_filtered.shape[0]}")
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

