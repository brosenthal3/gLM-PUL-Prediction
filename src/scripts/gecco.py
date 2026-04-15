from __future__ import annotations
import polars
import subprocess
import os
from pathlib import Path
import urllib.request
import tempfile
import argparse
from utility_scripts import join_gene_and_PUL_table
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


class GECCOHandler:
    def __init__(self, genes, features, clusters_dir, output_dir, hmms):
        self.genes = self._validate_table(genes)
        self.features = self._validate_table(features)
        self.clusters_dir = clusters_dir
        self.output_dir = output_dir
        self.hmms = Path(hmms) # directory containing HMM files


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


    def _train(self, train_clusters, genes, features, model_path):
        print(f"Starting training for fold {model_path.split('_')[-1]}...")
        if os.path.exists(model_path):
            print(f"Model path {model_path} already exists, skipping training.")
            return
        # example: gecco -vv train --genes genes.tsv --features features.tsv --clusters clusters.tsv -o model
        cmd = f"gecco -vv train --genes {genes} --features {features} --clusters {train_clusters} -o {model_path} --select 0.5"
        subprocess.run(cmd, shell=True, check=True)

    
    def _predict(self, test_clusters, genes, features, model_path):
        # print(f"Starting prediction for genome {Path(genome_path).stem}...")
        # # check output path
        # if os.path.exists(output_path):
        #     print(f"Predictions for genome {Path(genome_path).stem} already exist, skipping prediction.\n")
        #     return

        # # example: gecco run --model model --hmm Pfam35.hmm.gz --genome genome.fa -o ./predictions/
        # cmd = f"gecco run --model {model_path} --hmm {model_path}/features.h3m --genome {genome_path} -o {output_path}"
        # subprocess.run(cmd, shell=True, check=True)

        fold = model_path.split('_')[-1]
        all_genomes = "src/data/genomes/combined_genomes.gb"
        print(f"Starting prediction for fold {fold}...")
        cmd = f"gecco -vv predict --genes {genes} --features {features} --clusters {test_clusters} --model {model_path} -o {self.output_dir}/fold_{fold} --genome {all_genomes}"
        subprocess.run(cmd, shell=True, check=True)


    def _evaluate(self, predictions_path, fold, test_clusters, train_clusters):
        # pred_clusters = []
        # pred_genes = []
        # # collect all predicted clusters and genes from the predictions directory
        # for output_dir in os.listdir(predictions_path):
        #     for output_file in os.listdir(os.path.join(predictions_path, output_dir)):
        #         if output_file.endswith(".clusters.tsv"):
        #             pred_path = os.path.join(predictions_path, output_dir, output_file)
        #             pred_clusters.append(polars.read_csv(pred_path, separator='\t'))
        #         elif output_file.endswith(".genes.tsv"):
        #             pred_path = os.path.join(predictions_path, output_dir, output_file)
        #             pred_genes.append(polars.read_csv(pred_path, separator='\t'))

        # concat both into single tables, remove version num from sequence_id
        pred_clusters = polars.read_csv(pred_clusters, separator='\t').with_columns(
            polars.col("sequence_id").map_elements(lambda x: x.split('.')[0]).alias("sequence_id")
        )
        pred_genes = polars.read_csv(pred_genes, separator='\t').with_columns(
            polars.col("sequence_id").map_elements(lambda x: x.split('.')[0]).alias("sequence_id")
        )

        test_clusters = polars.read_csv(test_clusters, separator='\t')
        train_clusters = polars.read_csv(train_clusters, separator='\t')
        self.save_labeled_table(test_clusters, pred_clusters, pred_genes, fold, split="test")
        self.save_labeled_table(train_clusters, pred_clusters, pred_genes, fold, split="train")

        return pred_clusters, pred_genes


    # get all genes in test set
    def save_labeled_table(self, test_clusters, pred_clusters, pred_genes, fold, split):
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
        labeled_table.write_csv(f"{self.output_dir}/labeled_results_{split}_{fold}.tsv", separator='\t')


    def _save_temp_table(self, table, clusters):
        # save temporary file containing only sequences from clusters table (either test or train)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tsv")
        genes_table = (
            table
            .join(clusters, on="sequence_id", how="semi")
            .write_csv(temp_file.name, separator='\t')
        )
        return temp_file
    

    # def _save_hmm_file(self, model_path):
    #     # filter hmms to only include those in the model
    #     hmm_out_file = f"{model_path}/features"
    #     model_features = polars.read_csv(f"{model_path}/domains.tsv", separator='\t').to_series().to_list()
    #     # open and save hmms
    #     hmms_to_save = HMMLoader.read_hmms(hmmdb_path=self.hmms, whitelist=model_features)
    #     hmms_to_save.write_to_h3m_file(hmm_out_file)

    #     return hmm_out_file + ".selected.h3m"


    def run_fold(self, fold):
        train_clusters, test_clusters = self.get_training_data(fold)
        model_path = f"{self.output_dir}/model_{fold}"
        # filter gene and feature tables to only include genes in training set
        train_genes = polars.read_csv(train_clusters, separator='\t').select("sequence_id").unique()

        temp_genes_file = self._save_temp_table(self.genes, train_genes)
        temp_features_file = self._save_temp_table(self.features, train_genes)
        self._train(train_clusters, temp_genes_file.name, temp_features_file.name, model_path)

        test_genes = polars.read_csv(test_clusters, separator='\t').select("sequence_id").unique()
        temp_test_genes_file = self._save_temp_table(self.genes, test_genes)
        temp_test_features_file = self._save_temp_table(self.features, test_genes)
        self._predict(test_clusters, temp_test_genes_file.name, temp_test_features_file.name, model_path) # predict on test set
        self._predict(train_clusters, temp_genes_file.name, temp_features_file.name, model_path) # predict on train set

        temp_features_file.unlink()
        temp_genes_file.unlink()
        temp_test_genes_file.unlink()
        temp_test_features_file.unlink()

        # # get all genomes in test set, which are the sequences in the test_clusters table
        # for test_genome in test_genomes.to_series().to_list():
        #     genome_path = f"src/data/genomes/selected_genomes/{test_genome}.fa"
        #     output_path = f"{self.output_dir}/fold_{fold}/{test_genome}"
        #     self._predict(genome_path, model_path, output_path)

        # train_genomes = polars.read_csv(train_clusters, separator='\t').select("sequence_id").unique()
        # self._save_hmm_file(model_path)
        # for train_genome in train_genomes.to_series().to_list():
        #     genome_path = f"src/data/genomes/selected_genomes/{train_genome}.fa"
        #     output_path = f"{self.output_dir}/fold_{fold}/{train_genome}"
        #     self._predict(genome_path, model_path, output_path)
        raise NotImplementedError("Evaluation not implemented yet")
        results = self._evaluate(f"{self.output_dir}/fold_{fold}", fold, test_clusters, train_clusters)
        return results


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
    parser.add_argument("--hmms", type=str, default="src/data/hmms", help="Path to directory containing HMM files used to annotate data")
    parser.add_argument("-k", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--run_fold", type=int, help="Run a specific fold instead of cross-validation")
    args = parser.parse_args()

    handler = GECCOHandler(
        genes=args.genes,
        features=args.features,
        clusters_dir=args.clusters_dir,
        output_dir=args.output_dir,
        hmms=args.hmms
    )
    if args.run_fold is not None:
        handler.run_fold(args.run_fold)
    else:
        handler.run_cross_validation(k=args.k)


if __name__ == "__main__":
    main()
