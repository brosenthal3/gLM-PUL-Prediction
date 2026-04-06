import polars
import subprocess
import os
from pathlib import Path
import urllib.request
import tempfile
from utility_scripts import join_gene_and_PUL_table
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


class GECCOHandler:
    def __init__(self, genes, features, clusters_dir, output_dir, hmms):
        self.genes = self._validate_table(genes)
        self.features = self._validate_table(features)
        self.clusters_dir = clusters_dir
        self.output_dir = output_dir
        self.hmms = self._validate_hmm_dir(hmms)


    def _validate_hmm_dir(self, hmms):
        if not os.path.exists(hmms):
            print("HMM file not found, downloading...")
            # create dir 
            os.makedirs(os.path.dirname(hmms), exist_ok=True)
            # dowload hmms
            url = "ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam35.0/Pfam-A.hmm.gz"
            try:
                urllib.request.urlretrieve(url, hmms)
            except Exception as e:
                raise Exception(f"Failed to download HMMs from {url}: {str(e)}")
        else:
            print("HMM file found, skipping download.")
    
        return hmms


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
        cmd = f"gecco -vv train --genes {genes} --features {features} --clusters {train_clusters} -o {model_path}"
        subprocess.run(cmd, shell=True, check=True)

    
    def _predict(self, genome_path, model_path, output_path):
        print(f"Starting prediction for genome {Path(genome_path).stem}...")
        # check output path
        if os.path.exists(output_path):
            return

        # gecco run --model model --hmm Pfam35.hmm.gz --genome genome.fa -o ./predictions/
        cmd = f"gecco run --model {model_path} --hmm {self.hmms} --genome {genome_path} -o {output_path}"
        subprocess.run(cmd, shell=True, check=True)


    def _evaluate(self, predictions_path, fold, test_clusters):
        pred_clusters = []
        pred_genes = []
        for output_dir in os.listdir(predictions_path):
            for output_file in os.listdir(os.path.join(predictions_path, output_dir)):
                if output_file.endswith(".clusters.tsv"):
                    pred_path = os.path.join(predictions_path, output_dir, output_file)
                    pred_clusters.append(polars.read_csv(pred_path, separator='\t'))
                elif output_file.endswith(".genes.tsv"):
                    pred_path = os.path.join(predictions_path, output_dir, output_file)
                    pred_genes.append(polars.read_csv(pred_path, separator='\t'))

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

        # get all genes of test set
        test_genes = (pred_genes.join(test_clusters, on="sequence_id", how="semi"))
        # label test genes
        cols = ["protein_id", "sequence_id", "cluster_id", "is_PUL", "start", "end"]
        labeled_test_genes = join_gene_and_PUL_table(test_genes, test_clusters).select(cols)
        labeled_prediction_genes = join_gene_and_PUL_table(test_genes, pred_clusters).select(cols)
        # join predicted clusters with test clusters
        labeled_table = (
            labeled_test_genes
            .join(labeled_prediction_genes, on="protein_id", how="full", suffix="_pred")
            .with_columns(
                polars.when(polars.col("is_PUL").is_null()).then(False).otherwise(polars.col("is_PUL")).alias("is_PUL"),
                polars.when(polars.col("is_PUL_pred").is_null()).then(False).otherwise(polars.col("is_PUL_pred")).alias("is_PUL_pred"),
            )
            .join(test_genes.select("protein_id", "average_p"), on="protein_id", how="left")
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


    def run_fold(self, train_clusters, test_clusters, fold):
        model_path = f"{self.output_dir}/model_{fold}"
        # filter gene and feature tables to only include genes in training set
        train_genes = polars.read_csv(train_clusters, separator='\t').select("sequence_id").unique()

        temp_genes_file = self._save_temp_table(self.genes, train_genes)
        temp_features_file = self._save_temp_table(self.features, train_genes)
        self._train(train_clusters, temp_genes_file.name, temp_features_file.name, model_path)

        # get all genomes in test set, which are the sequences in the test_clusters table
        test_genomes = polars.read_csv(test_clusters, separator='\t').select("sequence_id").unique()
        for test_genome in test_genomes.to_series().to_list():
            genome_path = f"src/data/genomes/selected_genomes/{test_genome}.fa"
            output_path = f"{self.output_dir}/fold_{fold}/{test_genome}"
            self._predict(genome_path, model_path, output_path)

        results = self._evaluate(f"{self.output_dir}/fold_{fold}", fold, test_clusters)
        return results


    def get_training_data(self, fold):
        # get the training data for this fold, which is all clusters that are not in the test set
        test_clusters = f"{self.clusters_dir}/test_fold_{fold}.tsv"
        train_clusters = f"{self.clusters_dir}/train_fold_{fold}.tsv"

        # adapt train clusters to fit gecco input format
        train_clusters_df = polars.read_csv(train_clusters, separator='\t').select("sequence_id", "cluster_id", "start", "end")
        return train_clusters, test_clusters


    def run_cross_validation(self, k):
        for fold in range(k):
            print(f"Running fold {fold}...")
            train_clusters, test_clusters = self.get_training_data(fold)
            results = self.run_fold(train_clusters, test_clusters, fold)


def main():
    handler = GECCOHandler(
        genes="src/data/genecat_output/preprocess_output/genome.genes.parquet",
        features="src/data/genecat_output/preprocess_output/genome.features.parquet",
        clusters_dir="src/data/splits",
        output_dir="src/data/results/gecco",
        hmms="src/data/hmms/Pfam35.hmm.gz" # make sure hmms are downloaded and path is correct
    )
    handler.run_cross_validation(k=5)


if __name__ == "__main__":
    main()
