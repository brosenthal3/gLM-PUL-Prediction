import polars
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


if __name__ == "__main__":
    pfam = "src/data/results/genecat_fine_tuned/logs_genecat_fine_tuned/wandb/offline-run-20260429_150219-gtv4ep3n/files/pul_predictions.h5ad"    
    save_path_pfam = "src/data/results/genecat_fine_tuned/pfam_labeled_results_test_0.tsv"  
    save_pul_predictions(pfam, save_path_pfam)

    pfam_cazy = "src/data/results/genecat_fine_tuned/logs_genecat_fine_tuned/wandb/offline-run-20260429_221043-bgckiapc/files/pul_predictions.h5ad"
    save_path_pfam_cazy = "src/data/results/genecat_fine_tuned/pfam_cazy_labeled_results_test_0.tsv"
    save_pul_predictions(pfam_cazy, save_path_pfam_cazy)
