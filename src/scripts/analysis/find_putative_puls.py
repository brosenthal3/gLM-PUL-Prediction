import polars
import os
import argparse

model_name = "genecat_finetuned_cazy_masked"
clusters_table_path = "src/data/data_collection/clusters_deduplicated_cblaster.tsv"
cryptic_puls_path = "src/data/data_collection/cryptic_puls_genes.tsv"
labeled_results_path = f"src/data/results/{model_name}/labeled_results_test"
species = ["thetaiotaomicron", "uniformis", "phocaicola", "vulgatus", "dorei", "stercoris", "fragilis"]
genus = ["Faecalibacterium", "Eubacterium", "Dorea", "Roseburia", "Blautia", "Ruminococcus", "Clostridium"]


clusters_table = polars.read_csv(clusters_table_path, separator="\t", infer_schema_length=700)
cryptic_puls = polars.read_csv(cryptic_puls_path, separator="\t")
labeled_results = []
k = 5
for i in range(k):
    labeled_results.append(polars.read_csv(f"{labeled_results_path}_{i}.tsv", separator='\t'))

labeled_results = polars.concat(labeled_results)
labeled_results_filtered = (
    labeled_results
    .join(cryptic_puls, on="protein_id", how="anti") # remove cryptic puls
    .join(
        clusters_table.select("sequence_id", "phylum", "genus", "species").unique(),
        on="sequence_id",
        how="left"
    )
    .filter(
        ~polars.col("is_PUL"), # remove experimental puls
        polars.col("average_p").ge(0.8),
        (polars.col("species").str.contains("|".join(species)) | polars.col("genus").str.contains("|".join(genus)))
    )
    .select(
        "sequence_id",
        "phylum",
        "genus",
        "species",
        "start",
        "end",
        "average_p"
    )
    .sort(by=["sequence_id", "start"])
)

print(labeled_results_filtered)
labeled_results_filtered.write_csv(f"src/data/analysis/high_confidence_predictions_{model_name}.tsv", separator="\t")