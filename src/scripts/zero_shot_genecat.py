# APPROACH:
# 1. Load the trained model and the test data.
# 2. Generate embeddings for each gene in the test set using the trained model.
# 3. Save all embeddings in a single file, with protein_id and label (is_PUL) for each gene.
# 4. Use these embeddings to train a simple classifier (e.g., logistic regression)
# 5. Save classifier predictions and probabilities for each gene.
# 6. use evaluate_predictions.py 


import polars
import subprocess
import os
from pathlib import Path
import urllib.request
import tempfile
import argparse
from utility_scripts import join_gene_and_PUL_table


class GenecatHandler:
    def __init__(self, genes, features, clusters_dir, output_dir, hmms):
        self.genes = self._validate_table(genes)
        self.features = self._validate_table(features)
        self.clusters_dir = clusters_dir
        self.output_dir = output_dir


    def _validate_table(self, table_path):
        file_path = Path(table_path)
        if file_path.suffix == ".csv":
            table = polars.read_csv(table_path, separator='\t')
            return table
        elif file_path.suffix == ".parquet":
            table = polars.read_parquet(table_path)
            return table
        else:
            raise ValueError(f"Invalid file format for {table_path}. Expected .csv or .parquet")


    def _extract_embeddings(self):

    
    def _predict(self, genome_path, model_path, output_path):


    def _evaluate(self, predictions_path, fold, test_clusters):
        pred_clusters = []
        pred_genes = []
        # collect all predicted clusters and genes from the predictions directory
        for output_dir in os.listdir(predictions_path):
            for output_file in os.listdir(os.path.join(predictions_path, output_dir)):
                if output_file.endswith(".clusters.tsv"):
                    pred_path = os.path.join(predictions_path, output_dir, output_file)
                    pred_clusters.append(polars.read_csv(pred_path, separator='\t'))
                elif output_file.endswith(".genes.tsv"):
                    pred_path = os.path.join(predictions_path, output_dir, output_file)
                    pred_genes.append(polars.read_csv(pred_path, separator='\t'))

        # concat both into single tables, remove version num from sequence_id
        pred_clusters = polars.concat(pred_clusters).with_columns(
            polars.col("sequence_id").map_elements(lambda x: x.split('.')[0]).alias("sequence_id")
        )
        pred_genes = polars.concat(pred_genes).with_columns(
            polars.col("sequence_id").map_elements(lambda x: x.split('.')[0]).alias("sequence_id")
        )
        pred_clusters.write_csv(f"{self.output_dir}/predicted_clusters_{fold}.tsv", separator='\t')

        test_clusters = polars.read_csv(test_clusters, separator='\t')
        print(f"Total predicted clusters: {pred_clusters.shape[0]}")
        print(f"Total test clusters: {test_clusters.shape[0]}")

        # get all genes in test set
        test_genes = (pred_genes.join(test_clusters, on="sequence_id", how="semi"))

        # join genes with test clusters and predicted clusters
        cols = ["protein_id", "sequence_id", "cluster_id", "is_PUL", "start", "end"]
        labeled_test_genes = join_gene_and_PUL_table(test_genes, test_clusters).select(cols) 
        labeled_prediction_genes = join_gene_and_PUL_table(test_genes, pred_clusters).select(cols)

        # join gene tables of predicted clusters with test clusters
        labeled_table = (
            labeled_test_genes
            .join(labeled_prediction_genes, on="protein_id", how="full", suffix="_pred")
            .with_columns(
                polars.when(polars.col("is_PUL").is_null()).then(False).otherwise(polars.col("is_PUL")).alias("is_PUL"),
                polars.when(polars.col("is_PUL_pred").is_null()).then(False).otherwise(polars.col("is_PUL_pred")).alias("is_PUL_pred"),
            )
            .join(test_genes.select("protein_id", "average_p"), on="protein_id", how="left") # add predicted probability for each gene (for PR curve)
            .sort("protein_id")
            .sort("sequence_id")
        )
        labeled_table.write_csv(f"{self.output_dir}/labeled_results_{fold}.tsv", separator='\t')


    def _save_temp_table(self, table, clusters):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tsv")
        genes_table = (
            table
            .join(clusters, on="sequence_id", how="semi")
            .write_csv(temp_file.name, separator='\t')
        )
        return temp_file


    def run_fold(self, fold):


    def get_training_data(self, fold):
        # get the training data for this fold, which is all clusters that are not in the test set
        test_clusters = f"{self.clusters_dir}/test_fold_{fold}.tsv"
        train_clusters = f"{self.clusters_dir}/train_fold_{fold}.tsv"
        return train_clusters, test_clusters


    def run_cross_validation(self, k):
        for fold in range(k):
            print(f"Running fold {fold}...")
            results = self.run_fold(fold)


def main():
    parser = argparse.ArgumentParser(description="Run GECCO cross-validation")
    parser.add_argument("--genes", type=str, default="src/data/genecat_output/genome.genes.parquet", help="Path to genes table")
    parser.add_argument("--features", type=str, default="src/data/genecat_output/genome.features.parquet", help="Path to features table")
    parser.add_argument("--clusters_dir", type=str, default="src/data/splits", help="Directory containing train/test cluster splits")
    parser.add_argument("--output_dir", type=str, default="src/data/results/gecco", help="Directory to save results")
    parser.add_argument("--hmms", type=str, default="src/data/hmms/Pfam35.hmm.gz", help="Path to HMM file (will be downloaded if not found)")
    parser.add_argument("-k", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--run_fold", type=int, help="Run a specific fold instead of cross-validation")
    args = parser.parse_args()



if __name__ == "__main__":
    main()
