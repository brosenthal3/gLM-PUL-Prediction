from pathlib import Path
import argparse
import polars
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
from utility_scripts import join_gene_and_PUL_table


class DatasetSplitter:
    def __init__(
            self, 
            clusters_table: polars.DataFrame, 
            gene_table: 
            polars.DataFrame, 
            ani_table: polars.DataFrame, 
            rank: str,
            ani_split: bool,
            ani_threshold: float,
            gene_threshold: int
        ):

        self.clusters_table = clusters_table
        self.gene_table = gene_table
        self.gene_threshold = gene_threshold
        self.rank = rank
        self.ani_split = ani_split

        if ani_split:
            self.ani_table = self._process_ani_table(ani_table)
            self.ani_threshold = ani_threshold


    def filter_on_genes(self):
        labeled_table = (
            join_gene_and_PUL_table(gene_table=self.gene_table, cluster_table=self.clusters_table)
            .group_by("sequence_id")
            .agg(polars.col("sequence_id").count().alias("gene_count"))
            .filter(polars.col("gene_count") < self.gene_threshold)
        )
        print(f"Filtering out {labeled_table.shape[0]} sequences with less than {self.gene_threshold} genes...")
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

    def split_on_ani(self, k, output_dir, stratify):
        labels = self._cluster_on_ANI()
        clusters_with_labels = self.clusters_table.join(labels, on="sequence_id", how="left").sort(["ani_cluster_id", "cluster_id"])
        groups = clusters_with_labels.select("ani_cluster_id").to_series()
        X = clusters_with_labels.select("cluster_id").to_series()

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


    def split_dataset(self, k, output_dir, stratify, split_bacteroidata):
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        self.filter_on_genes()
        if stratify:
            group_kfold = StratifiedGroupKFold(n_splits=k)
        else:
            group_kfold = GroupKFold(n_splits=k)

        if self.ani_split:
            print("Splitting dataset based on ANI clustering...")
            self.split_on_ani(k, output_dir, stratify)

        else:
            clusters_table = (
                join_gene_and_PUL_table(
                    self.gene_table, 
                    self.clusters_table.with_columns(polars.col(self.rank).fill_null("unknown"))
                )
                .join(
                    self.clusters_table.select("sequence_id", self.rank).unique(),
                    on="sequence_id",
                    how="left"
                )
                .with_columns(polars.col(self.rank).fill_null("unknown"))
                .with_row_index("index")
            )

            X = clusters_table.select("index").to_series()
            y = clusters_table.select("is_PUL").to_series()
            groups = clusters_table.select(self.rank).to_series()
            groups_cat = (
                clusters_table
                .with_columns(polars.col(self.rank).cast(polars.Categorical).alias("rank_cat"))
                .sort("index")
                .select(polars.col("rank_cat").to_physical())
                .to_series()
            )

            for i, (train_index, test_index) in enumerate(group_kfold.split(X, groups=groups, y=y if stratify else None)):
                print(f"Fold {i}:")
                # print(f"\tTrain groups={groups[train_index].unique()}")
                # print(f"\tTest groups={groups[test_index].unique()}")
                train_df = (clusters_table.filter(polars.col("index").is_in(train_index)).drop("index"))
                test_df = (clusters_table.filter(polars.col("index").is_in(test_index)).drop("index"))
                
                # report fold stats
                print(f"\tTrain set: {len(train_df)} Genes, Test set: {len(test_df)} Genes")
                print(f"\tTrain set: {train_df.select('cluster_id').n_unique()} PULs, Test set: {test_df.select('cluster_id').n_unique()} PULs")
                print(f"\tPositives in train {train_df.filter(polars.col('is_PUL') == 1).shape[0]/len(train_df)*100:.2f}%, Positives in test {test_df.filter(polars.col('is_PUL') == 1).shape[0]/len(test_df)*100:.2f}%")

                # check for overlap in taxa between train and test sets
                overlap_in_taxa = set(
                    train_df.select(self.rank).unique().to_series().to_list()
                ).intersection(
                    set(test_df.select(self.rank).unique().to_series().to_list())
                )
                assert len(overlap_in_taxa) == 0, f"Overlap in taxa between train and test sets in fold {i}: {overlap_in_taxa}"

                # save clusters
                train_groups = (
                    self.clusters_table
                    .join(train_df.select("sequence_id").unique(), on="sequence_id", how="semi")
                    .write_csv(f"{output_dir}/train_fold_{i}.tsv", separator='\t')
                )
                test_groups = (
                    self.clusters_table
                    .join(test_df.select("sequence_id").unique(), on="sequence_id", how="semi")
                    .write_csv(f"{output_dir}/test_fold_{i}.tsv", separator='\t')
                )

        if split_bacteroidata:
            print("Splitting dataset based on Bacteroidata phylum...")
            bacteroidata_sequences = (
                self.clusters_table.filter(
                    polars.col("phylum").eq("Bacteroidota") | polars.col("phylum").eq("Bacteroidota_A")
                )
                .select("sequence_id").unique()
            )

            bacteroidata_clusters = (
                self.clusters_table
                .join(bacteroidata_sequences, on="sequence_id", how="semi")
            )
            non_bacteroidata_clusters = (
                self.clusters_table
                .join(bacteroidata_sequences, on="sequence_id", how="anti")
            )
            bacteroidata_clusters.write_csv(f"{output_dir}/train_fold_{k}.tsv", separator='\t')
            bacteroidata_clusters.write_csv(f"{output_dir}/test_fold_{k+1}.tsv", separator='\t')

            non_bacteroidata_clusters.write_csv(f"{output_dir}/test_fold_{k}.tsv", separator='\t')
            non_bacteroidata_clusters.write_csv(f"{output_dir}/train_fold_{k+1}.tsv", separator='\t')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train and test sets based on ANI clustering")
    parser.add_argument("-i", "--input", type=str, default="src/data/results/cblaster_results.tsv", help="Path to the clusters table")
    parser.add_argument("--genes", type=str, default="src/data/genecat_output/genome.genes.parquet", help="Path to the gene table")
    parser.add_argument("--k", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--rank", type=str, default="genus", help="Taxonomic rank to use for splitting (if not using ANI)")
    parser.add_argument("--gene_threshold", type=int, default=10, help="Minimum number of genes for a sequence to be included")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--stratify", action="store_true", help="Whether to stratify splits based on PUL presence")
    parser.add_argument("--split_bacteroidata", action="store_true", help="Whether to additionally split the dataset based on Bacteroidata phylum")
    # ANI arguments, no longer used...
    parser.add_argument("--ani_split", action="store_true", help="Whether to split based on ANI instead of Taxonomy")
    parser.add_argument("--ani_threshold", type=float, default=90.0, help="ANI threshold for clustering")
    args = parser.parse_args()

    clusters_table = polars.read_csv(args.input, separator='\t', infer_schema_length=600)
    gene_table = polars.read_parquet(args.genes)
    ani_table = polars.read_csv("src/data/results/orthoANI_output.txt", separator='\t', has_header=False)

    splitter = DatasetSplitter(clusters_table, gene_table, ani_table, args.rank, args.ani_split, args.ani_threshold, args.gene_threshold)
    splitter.split_dataset(k=args.k, output_dir=Path("src/data/splits/"), stratify=args.stratify, split_bacteroidata=args.split_bacteroidata)



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

