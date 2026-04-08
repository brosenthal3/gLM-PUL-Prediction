import polars
import subprocess
import os
from pathlib import Path
import argparse
from utility_scripts import join_gene_and_PUL_table


class GenecatHandler:
    def __init__(self, genes, clusters_dir, embeddings, output_dir):
        self.genes = self._validate_table(genes)
        self.embeddings = self._validate_table(embeddings)
        self.clusters_dir = clusters_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)


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


    def save_fold_data(self, fold):
        train_clusters, test_clusters = self.get_training_data(fold)
        train_df = polars.read_csv(train_clusters, separator='\t')
        test_df = polars.read_csv(test_clusters, separator='\t')

        # join with genes
        train_genes = join_gene_and_PUL_table(self.genes, train_df).with_columns(
            polars.lit("train").alias("label")
        )
        test_genes = join_gene_and_PUL_table(self.genes, test_df).with_columns(
            polars.lit("test").alias("label")
        )
        fold_data = polars.concat([train_genes, test_genes])
        print(fold_data)
        fold_data = fold_data.join(self.embeddings, on="protein_id", how="left").select("protein_id", "sequence_id", "embedding", "label") # add embeddings for each gene
        print(fold_data.head())
        fold_output_path = f"{self.output_dir}/fold_{fold}_data.parquet"
        fold_data.write_parquet(fold_output_path)
        return fold_output_path


    def get_training_data(self, fold):
        # get the training data for this fold, which is all clusters that are not in the test set
        test_clusters = f"{self.clusters_dir}/test_fold_{fold}.tsv"
        train_clusters = f"{self.clusters_dir}/train_fold_{fold}.tsv"
        return train_clusters, test_clusters


    def save_folds(self, k):
        for fold in range(k):
            print(f"Saving data for fold {fold}...")
            results = self.save_fold_data(fold)


def main():
    parser = argparse.ArgumentParser(description="Run Zero-shot Genecat cross-validation")
    parser.add_argument("--genes", type=str, default="src/data/genecat_output/genome.genes.parquet", help="Path to genes table")
    parser.add_argument("--clusters_dir", type=str, default="src/data/splits", help="Directory containing train/test cluster splits")
    parser.add_argument("--embeddings", type=str, default="src/data/results/genecat/PUL_embs/model_gene_multilabel_untied_march_s4spvlec_v0_context_embedding.embeddings.parquet", help="Path to trained model embeddings")
    parser.add_argument("-k", type=int, default=1, help="Number of folds for cross-validation")
    args = parser.parse_args()

    handler = GenecatHandler(args.genes, args.clusters_dir, args.embeddings, output_dir="src/data/results/genecat/fold_data")
    handler.save_folds(args.k)



if __name__ == "__main__":
    main()
