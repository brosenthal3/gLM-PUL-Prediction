import polars
import os

def filter_on_clusters(clusters, table):
    table_filtered =  table.join(clusters, on="sequence_id", how="semi")
    assert table_filtered.shape[0] > 0

    return table_filtered

genes = polars.read_parquet("src/data/genecat_output/genome.genes.parquet")
pfam_features = polars.read_parquet("src/data/genecat_output/pfam.features.parquet")
dbcan_features = polars.read_parquet("src/data/genecat_output/dbcan.pfam.features.parquet")

splits_path = "src/data/splits"
for file_path in os.listdir(splits_path):
    fold_num = file_path.split(".")[0].split("_")[-1]
    split = file_path.split("_")[0]

    output_path = f"src/data/genecat_output/fold_{fold_num}"
    os.makedirs(output_path, exist_ok=True)
    clusters = polars.read_csv(f"{splits_path}/{file_path}", separator="\t")

    filter_on_clusters(clusters, genes).write_parquet(f"{output_path}/{split}.genes.parquet")
    filter_on_clusters(clusters, pfam_features).write_parquet(f"{output_path}/{split}.pfam.parquet")
    filter_on_clusters(clusters, dbcan_features).write_parquet(f"{output_path}/{split}.dbcan.pfam.parquet")
