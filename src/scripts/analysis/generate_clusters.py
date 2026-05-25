import polars
from tqdm import tqdm

def combine_pul_genes(model_name, output_path, threshold=0.25):
    labeled_results_path = f"src/data/results/{model_name}/labeled_results_test"

    # combine all labeled results into one dataframe
    labeled_results = []
    for i in range(5):
        labeled_results.append(polars.read_csv(f"{labeled_results_path}_{i}.tsv", separator='\t', infer_schema_length=8000))

    labeled_results = polars.concat(labeled_results).with_columns(
        polars.col("average_p").fill_null(0.0),
        polars.col("average_p").ge(threshold).alias("predicted_PUL")
    )

    # group by sequence_id, combine all adjacent genes with average_p >= threshold into clusters
    clusters = []
    previous_gene_in_PUL = False
    current_genome = None
    for row in labeled_results.iter_rows(named=True):
        if previous_gene_in_PUL and row["sequence_id"] != current_genome:
            previous_gene_in_PUL = False

        if row["predicted_PUL"]:
            # if previous gene was in pul then extend the current cluster, otherwise start a new cluster
            if previous_gene_in_PUL:
                gene_count = clusters[-1]["gene_count"]
                clusters[-1]["end"] = row["end"]
                clusters[-1]["genes"].append(row["protein_id"])
                clusters[-1]["gene_count"] = gene_count + 1
                clusters[-1]["average_p"] = (clusters[-1]["average_p"] * (gene_count - 1) + row["average_p"]) / clusters[-1]["gene_count"]
            else:
                clusters.append({
                    "sequence_id": row["sequence_id"],
                    "start": row["start"],
                    "end": row["end"],
                    "genes": [row["protein_id"]],
                    "gene_count": 1,
                    "average_p": row["average_p"]
                })
                previous_gene_in_PUL = True
        else:
            previous_gene_in_PUL = False

        current_genome = row["sequence_id"]


    # make clusters into a dataframe and save to tsv
    clusters_df = polars.DataFrame(clusters)
    print(clusters_df)
    clusters_df.write_parquet(output_path)


if __name__ == "__main__":
    model_names = {
        "gecco_pfam": 0.3,
        "gecco_cazy": 0.3,
        "genecat_zeroshot_pfam_masked": 0.276,
        "genecat_zeroshot_cazy_masked": 0.237,
        "genecat_finetuned_pfam_masked": 0.5,
        "genecat_finetuned_cazy_masked": 0.5,
        "esmc_masked": 0.158,
        "bacformer_masked": 0.24,
        "genecat_untrained": 0.55,
    }

    for model_name, threshold in tqdm(model_names.items(), desc="Generating clusters for models", total=len(model_names)):
        output_path = f"src/data/results/{model_name}/predicted_clusters.parquet"
        combine_pul_genes(model_name, output_path, threshold=threshold)