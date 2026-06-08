import polars
import os
import anndata as ad
from utility_scripts import join_gene_and_PUL_table

def save_pul_predictions(h5ad_path, save_path):
    adata = ad.io.read_h5ad(h5ad_path)

    # get predicted probabilities and labels
    probas = polars.DataFrame(data=adata.X, schema=["average_p"])
    genecat_results = polars.DataFrame(adata.obs)
    genecat_results = polars.concat([genecat_results, probas], how="horizontal")

    # get cluster IDs
    cols = ["protein_id", "sequence_id", "cluster_id", "is_PUL", "start", "end"]
    if "test_cluster_table" in adata.uns.keys():
        clusters_path = adata.uns["test_cluster_table"]
        genes_path = adata.uns["test_gene_table"]
    else:
        clusters_path = adata.uns["cluster_table"]
        genes_path = adata.uns["gene_table"]

    clusters = polars.read_csv(clusters_path, separator='\t')
    # filter out predicted clusters, if they are included
    if "origin" in clusters.columns:
        clusters = clusters.filter(~polars.col("origin").eq("predicted"))

    genes = polars.read_parquet(genes_path)
    labeled_test_genes = join_gene_and_PUL_table(genes, clusters).select(cols)


    # combine both
    labeled_table = (
        labeled_test_genes
        .join(genecat_results.select("protein_id", "average_p"), on="protein_id", how="inner")
        .with_columns(
            polars.when(polars.col("is_PUL").is_null()).then(False).otherwise(polars.col("is_PUL")).alias("is_PUL"),
            polars.when(polars.col("average_p").ge(0.5)).then(True).otherwise(False).alias("is_PUL_pred"),
        )
        .sort("sequence_id")
    )

    labeled_table.write_csv(save_path, separator='\t')


def main():
    for k in range(7):
        for features in ["pfam", "cazy"]:
            # predictions = f"src/data/results/genecat_finetuned_{features}_masked/logs_fold_{k}/wandb/latest-run/files/pul_predictions.h5ad"
            predictions = f"src/data/results/genecat_finetuned_{features}_masked/fold_{k}.h5ad" 
            save_path = f"src/data/results/genecat_finetuned_{features}_masked/labeled_results_test_{k}.tsv"
            if not os.path.exists(predictions):
                print("Could not find file at ", predictions)
                continue

            save_pul_predictions(predictions, save_path)

        predictions_untrained = f"src/data/results/genecat_untrained/logs_fold_{k}/wandb/latest-run/files/pul_predictions.h5ad"
        save_path_untrained = f"src/data/results/genecat_untrained/labeled_results_test_{k}.tsv"
        if not os.path.exists(predictions_untrained):
            print("Could not find file at ", predictions_untrained)
            continue

        save_pul_predictions(predictions_untrained, save_path_untrained)
        


if __name__ == "__main__":
    main()
