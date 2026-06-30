"""
Script used to automatically find predicted clusters with high confidence and save them to "src/data/results/{model_name}/predicted_clusters.parquet".
Requires the output files of generate_clusters.py
"""

import polars

def find_putative_puls(predicted_clusters, clusters_table, genus, species, model_name, prob_threshold=0.4):
    predicted_clusters_filtered = (
        predicted_clusters
        .join(
            clusters_table.select("sequence_id", "phylum", "genus", "species").unique(),
            on="sequence_id",
            how="left"
        )
        .filter(
            polars.col("average_p").ge(prob_threshold),
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


if __name__ == "__main__":
    # model_name = "genecat_finetuned_cazy_masked"
    model_name = "gecco_pfam"

    species = ["uniformis", "phocaicola", "vulgatus", "dorei", "stercoris", "fragilis"]
    genus = ["Faecalibacterium", "Eubacterium", "Dorea", "Roseburia", "Blautia", "Ruminococcus", "Clostridium"]
    clusters_table = polars.read_csv("src/data/data_collection/clusters_deduplicated_cblaster.tsv", separator="\t", infer_schema_length=700)
    predicted_clusters_path = f"src/data/results/{model_name}/predicted_clusters.parquet"
    predicted_clusters = polars.read_parquet(predicted_clusters_path)

    find_putative_puls(predicted_clusters, clusters_table, genus, species, model_name)

