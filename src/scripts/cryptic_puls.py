"""
Script to combine PULpy output with liberal cblaster output, merge with gene table and save the gene ids.
Also creates additional train/test cluster tables that contain cryptic puls with an additional "origin" col.
"""

import polars
from utility_scripts import join_gene_and_PUL_table
from data_collection import merge_overlapping_puls

liberal_cblaster = polars.read_csv("src/data/data_collection/cblaster_results_liberal.tsv", separator="\t")
pulpy = (
    polars.read_csv("src/data/data_collection/pulpy_annotations.tsv", separator="\t")
    .rename({"genome": "sequence_id", "pulid": "cluster_id"})
    .select(liberal_cblaster.columns)
)
cryptic_puls = polars.concat([liberal_cblaster, pulpy])

def save_cryptic_pul_genes():
    # load all data
    experimental_puls = polars.read_csv("src/data/data_collection/clusters_deduplicated_cblaster.tsv", separator="\t")
    genes = polars.read_parquet("src/data/genecat_output/genome.genes.parquet")

    # merge clusters and cryptic puls with genes
    cryptic_puls_genes = join_gene_and_PUL_table(genes, cryptic_puls).select("protein_id", "is_PUL").rename({"is_PUL": "is_cryptic_PUL"})
    experimental_puls_genes = join_gene_and_PUL_table(genes, experimental_puls).select("protein_id", "is_PUL")
    joined_genes = (
        experimental_puls_genes
        .join(cryptic_puls_genes, on="protein_id", how="inner", validate="1:1")
        .filter(
            (polars.col("is_PUL") == False) & (polars.col("is_cryptic_PUL") == True)
        )
        .select("protein_id")
        .write_csv("src/data/data_collection/cryptic_puls_genes.tsv", separator="\t")
    )


def add_cryptic_puls_to_splits():
    for k in range(7):
        for split in ["train", "test"]:
            clusters_table = polars.read_csv(f"src/data/splits/{split}_fold_{k}.tsv", separator="\t").with_columns(polars.lit("experimental").alias("origin"))

            cryptic_puls_filtered = cryptic_puls.join(clusters_table.select("sequence_id").unique(), on="sequence_id", how="semi").with_columns(
                polars.lit("predicted").alias("database")
            )
            only_predicted_puls = (
                merge_overlapping_puls(polars.concat([clusters_table, cryptic_puls_filtered], how="diagonal"), blast=False)
                .with_columns(polars.col("database").str.split("_").list.unique().alias("databases"))
                .filter(polars.col("databases").eq(["predicted"]))
                .drop("databases")
                .with_columns(polars.lit("predicted").alias("origin"))
                .sort(by="sequence_id")
            )
            all_puls = polars.concat([clusters_table, only_predicted_puls])
            all_puls.write_csv(f"src/data/splits/{split}_fold_{k}_with_cryptic.tsv", separator="\t")


if __name__ == "__main__":
    save_cryptic_pul_genes()
    add_cryptic_puls_to_splits()
