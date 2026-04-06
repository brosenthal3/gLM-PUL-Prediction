from pathlib import Path
import argparse
import polars
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
from utility_scripts import join_gene_and_PUL_table


class DatasetSplitter:
    def __init__(self, clusters_table: polars.DataFrame, gene_table: polars.DataFrame, ani_table: polars.DataFrame, ani_threshold: float):
        self.clusters_table = clusters_table
        self.gene_table = gene_table
        self.ani_table = self._process_ani_table(ani_table)
        self.ani_threshold = ani_threshold


    def filter_on_genes(self, gene_threshold: int):
        labeled_table = (
            join_gene_and_PUL_table(gene_table=self.gene_table, cluster_table=self.clusters_table)
            .group_by("sequence_id")
            .agg(polars.col("sequence_id").count().alias("gene_count"))
            .filter(polars.col("gene_count") < gene_threshold)
        )
        print(f"Filtering out {labeled_table.shape[0]} sequences with less than {gene_threshold} genes...")
        self.clusters_table = self.clusters_table.join(labeled_table.select("sequence_id"), on="sequence_id", how="anti")


    def _process_ani_table(self, ani_table):
        # rename col names and remove version numbers from sequence ids
        ani_table = (
            ani_table
            .rename({"column_1": "query", "column_2": "reference", "column_3": "ani"})
            .with_columns(
                polars.col("query").str.split(".").list.first().alias("query"),
                polars.col("reference").str.split(".").list.first().alias("reference"),
            ))

        # only keep genomes that are in the clusters table        
        valid_sequences = self.clusters_table.select("sequence_id").unique()
        filtered_ani_table =(
            ani_table
            .join(valid_sequences, left_on="query", right_on="sequence_id", how="semi")
            .join(valid_sequences, left_on="reference", right_on="sequence_id", how="semi")
            .sort(["query", "reference"])
        )
        # also filter the clusters table to only keep sequences that are in the ani table
        self.clusters_table = self.clusters_table.join(filtered_ani_table, left_on="sequence_id", right_on="query", how="semi")

        # transform to matrix format with query_id as index, subject_id as columns, and ani as values
        ani_matrix = (
            filtered_ani_table
            .pivot(values="ani", index="query", on="reference")
            .sort("query")
        )

        return ani_matrix


    def _cluster_on_ANI(self):
        ani_queries = self.ani_table.select("query").to_series()
        ani_matrix = self.ani_table.drop("query").to_numpy()
        distance = 1 - (ani_matrix / 100)

        clustering = HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
            metric='precomputed',
            cluster_selection_epsilon = 1 - (self.ani_threshold / 100),
            copy=False
        )

        labels = clustering.fit_predict(distance)
        labels = polars.Series(labels)
        result = polars.DataFrame({"sequence_id": ani_queries, "ani_cluster_id": labels}).sort("sequence_id")
        return result


    def split_dataset(self, k, output_dir):
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        self.filter_on_genes(10)
        labels = self._cluster_on_ANI()
        clusters_with_labels = self.clusters_table.join(labels, on="sequence_id", how="left").sort(["ani_cluster_id", "cluster_id"])
        groups = clusters_with_labels.select("ani_cluster_id").to_series()
        X = clusters_with_labels.select("cluster_id").to_series()

        group_kfold = GroupKFold(n_splits=k)
        for i, (train_index, test_index) in enumerate(group_kfold.split(X, groups=groups)):
            print(f"Fold {i}:")
            train_groups = polars.DataFrame({ "ani_cluster_id": groups[train_index].unique() })
            test_groups = polars.DataFrame({ "ani_cluster_id": groups[test_index].unique() })
            train_data = clusters_with_labels.join(train_groups, left_on="ani_cluster_id", right_on="ani_cluster_id", how="semi").drop("ani_cluster_id")
            test_data = clusters_with_labels.join(test_groups, left_on="ani_cluster_id", right_on="ani_cluster_id", how="semi").drop("ani_cluster_id")
            print(f"Train set: {len(train_data)} PULs, Test set: {len(test_data)} PULs")

            # save the train and test sets for this fold
            train_data.write_csv(f"{output_dir}/train_fold_{i}.tsv", separator='\t')
            test_data.write_csv(f"{output_dir}/test_fold_{i}.tsv", separator='\t')


def main(clusters_table_path: str, gene_table_path: str, k: int, ani_threshold: float):
    clusters_table = polars.read_csv(clusters_table_path, separator='\t', infer_schema_length=600)
    gene_table = polars.read_parquet(gene_table_path)
    ani_table = polars.read_csv("src/data/results/orthoANI_output.txt", separator='\t', has_header=False)
    splitter = DatasetSplitter(clusters_table, gene_table, ani_table, ani_threshold)
    splitter.split_dataset(k=k, output_dir=Path("src/data/splits/"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train and test sets based on ANI clustering")
    parser.add_argument("-i", "--input", type=str, default="src/data/results/cblaster_results.tsv", help="Path to the clusters table")
    parser.add_argument("--genes", type=str, default="src/data/genecat_output/genome.genes.parquet", help="Path to the gene table")
    parser.add_argument("--k", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--ani_threshold", type=float, default=90.0, help="ANI threshold for clustering")

    args = parser.parse_args()
    main(args.input, args.genes, args.k, args.ani_threshold)



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

