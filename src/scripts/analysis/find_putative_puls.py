import polars
import os
import argparse

model_name = "gecco_pfam"
clusters_table_path = "src/data/data_collection/clusters_deduplicated_cblaster.tsv"
cryptic_puls_path = "src/data/data_collection/cryptic_puls_genes.tsv"
#labeled_results_path = f"src/data/results/{model_name}/labeled_results_test"
predicted_clusters_path = f"src/data/results/{model_name}/predicted_clusters.parquet"
species = ["uniformis", "phocaicola", "vulgatus", "dorei", "stercoris", "fragilis"]
genus = ["Faecalibacterium", "Eubacterium", "Dorea", "Roseburia", "Blautia", "Ruminococcus", "Clostridium"]


clusters_table = polars.read_csv(clusters_table_path, separator="\t", infer_schema_length=700)

predicted_clusters = polars.read_parquet(predicted_clusters_path)
predicted_clusters_filtered = (
    predicted_clusters
    .join(
        clusters_table.select("sequence_id", "phylum", "genus", "species").unique(),
        on="sequence_id",
        how="left"
    )
    .filter(
        polars.col("average_p").ge(0.4),
        (polars.col("species").str.contains("|".join(species)) | polars.col("genus").str.contains("|".join(genus))),
        polars.col("gene_count").ge(3)
    )
    .select(
        "sequence_id",
        "phylum",
        "genus",
        "species",
        "start",
        "end",
        "average_p",
        "gene_count",
    )
    .sort(by=["sequence_id", "start"])
)

print(predicted_clusters_filtered)
predicted_clusters_filtered.write_csv(f"src/data/analysis/high_confidence_predictions_{model_name}.tsv", separator="\t")