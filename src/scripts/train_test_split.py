import polars
from sklearn.model_selection import GroupKFold
from pathlib import Path


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
    clusters_table_path = "src/data/results/combined_clusters_blasted_gtdb_filtered.tsv"
    if Path(clusters_table_path).exists():
        clusters_table = polars.read_csv(clusters_table_path, separator='\t')
    else:
        return
    k = 5 # later as argument
    split_dataset(clusters_table, k)

    
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

