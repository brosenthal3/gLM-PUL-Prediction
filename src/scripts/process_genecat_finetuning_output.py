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
    clusters = polars.read_csv(adata.uns["test_cluster_table"], separator='\t')
    genes = polars.read_parquet(adata.uns["test_gene_table"])
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
            predictions = f"src/data/results/genecat_finetuned_{features}/logs_fold_{k}/wandb/latest-run/files/pul_predictions.h5ad"
            save_path = f"src/data/results/genecat_finetuned_{features}/labeled_results_test_{k}.tsv"
            if not os.path.exists(predictions):
                continue

            save_pul_predictions(predictions, save_path)


if __name__ == "__main__":
    main()