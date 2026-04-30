import polars
import subprocess
import os
from pathlib import Path
import argparse
from utility_scripts import join_gene_and_PUL_table


class EmbeddingsHandler:
    def __init__(self, genes, clusters_dir, embeddings, output_dir, embedding_col="embedding", dir=False):
        self.dir = dir
        self.embedding_col = embedding_col
        self.genes = self._validate_table(genes)
        self.embeddings = self._validate_table(embeddings).select("protein_id", "embedding")
        self.clusters_dir = clusters_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)


    def _validate_table(self, table_path):
        file_path = Path(table_path)
        if file_path.suffix == ".csv":
            table = polars.read_csv(table_path, separator='\t')
        elif file_path.suffix == ".parquet":
            table = polars.read_parquet(table_path)
        elif self.dir:
            print("Input in directory format, processing to one dataframe")
            embs = []
            for embs_file in os.listdir(table_path):
                embs.append(polars.read_parquet(f"{table_path}/{embs_file}"))
            table = polars.concat(embs)
            print(f"Found embeddings for {len(table)} genes.")
        else:
            raise ValueError(f"Invalid file format for {table_path}. Expected .csv or .parquet. If the embeddings are in a directory, specify the --dir flag.")

        return table.rename({self.embedding_col: "embedding"}, strict=False)


    def save_fold_data(self, fold):
        train_clusters, test_clusters = self.get_training_data(fold)
        train_df = polars.read_csv(train_clusters, separator='\t')
        test_df = polars.read_csv(test_clusters, separator='\t')

        # join with genes
        train_genes = join_gene_and_PUL_table(self.genes, train_df).with_columns(
            polars.lit("train").alias("split"),
            polars.col("is_PUL").alias("label")
        )
        test_genes = join_gene_and_PUL_table(self.genes, test_df).with_columns(
            polars.lit("test").alias("split"),
            polars.col("is_PUL").alias("label")
        )
        fold_data = polars.concat([train_genes, test_genes])
        fold_data = fold_data.join(self.embeddings, on="protein_id", how="left").select("protein_id", "sequence_id", "embedding", "label", "split") # add embeddings for each gene

        fold_output_path = f"{self.output_dir}/fold_{fold}_data.parquet"
        print(f"Saving table to {fold_output_path}")
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
    parser = argparse.ArgumentParser(description="Process encoder embeddings output, by saving train and test tables of the embeddings for each fold.")
    parser.add_argument("--genes", type=str, default="src/data/genecat_output/genome.genes.parquet", help="Path to genes table")
    parser.add_argument("--clusters_dir", type=str, default="src/data/splits", help="Directory containing train/test cluster splits")
    parser.add_argument("--embeddings", "-e", type=str, default="src/data/results/genecat/PUL_embs/model_gene_multilabel_untied_march_s4spvlec_v0_context_embedding.embeddings.parquet", help="Path to trained model embeddings")
    parser.add_argument("-k", type=int, default=7, help="Number of folds for cross-validation")
    parser.add_argument("--output_dir", "-o", type=str, default="src/data/results/genecat/fold_data", help="Directory to save fold data")
    parser.add_argument("--dir", action='store_true', help="Whether the input embeddings are a directory containing .parquet files")
    parser.add_argument("--embedding_col", type=str, default="embedding", help="Column containing the embeddings")
    args = parser.parse_args()

    handler = EmbeddingsHandler(args.genes, args.clusters_dir, args.embeddings, output_dir=args.output_dir, embedding_col=args.embedding_col, dir=args.dir)
    handler.save_folds(args.k)


if __name__ == "__main__":
    main()

"""
For genecat pfam:
python src/scripts/process_embeddings_output.py -e src/data/results/genecat/PUL_embs/model_gene_multilabel_untied_april_sriqcx3c_v0_context_embedding.embeddings.parquet -o src/data/results/genecat/fold_data_pfam

For genecat pfam+cazy:
python src/scripts/process_embeddings_output.py -e src/data/results/genecat/PUL_embs/model_gene_multilabel_pfam_cazy_april_goycr91w_v0_context_embedding.embeddings.parquet -o src/data/results/genecat/fold_data_cazy

For esmc:
python src/scripts/process_embeddings_output.py -e src/data/results/esmc/esmc_bacformer_embeddings -o src/data/results/esmc/fold_data --dir --embedding_col embedding_esmc

For bacformer
python src/scripts/process_embeddings_output.py -e src/data/results/esmc/esmc_bacformer_embeddings -o src/data/results/bacformer/fold_data --dir --embedding_col embedding_bacformer

"""