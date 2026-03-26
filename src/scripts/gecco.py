import polars
import subprocess
import os
import pathlib


class GECCOHandler:
    def __init__(self, genes, features, clusters_dir, output_dir, hmm_dir):
        self.genes = genes
        self.features = features
        self.clusters_dir = clusters_dir
        self.output_dir = output_dir
        self.hmm_dir = hmm_dir


    def _train(self, train_clusters, model_path):
        # example: gecco -vv train --genes genes.tsv --features features.tsv --clusters clusters.tsv -o model
        cmd = f"gecco -vv train --genes {self.genes} --features {self.features} --clusters {train_clusters} -o {model_path}"
        subprocess.run(cmd, shell=True, check=True)

    
    def _predict(self, genome_path, model_path, output_path):
        # gecco run --model model --hmm Pfam35.hmm.gz --genome genome.fa -o ./predictions/
        cmd = f"gecco run --model {model_path} --hmm {self.hmm_dir} --genome {genome_path} -o {output_path}"
        subprocess.run(cmd, shell=True, check=True)


    def _evaluate(self, predictions_path, test_clusters):
        # open all predictions
        # concat to one dataframe
        # compare with test_clusters; for each gene check pred vs true label, calculate metrics
        pass


    def run_fold(self, train_clusters, test_clusters, fold):
        model_path = f"{self.output_dir}/model_{fold}"
        self._train(train_clusters, model_path)

        # get all genomes in test set, which are the sequences in the test_clusters table
        test_genomes = polars.read_csv(test_clusters, separator='\t').select("sequence_id").unique().to_series().to_list()
        for test_genome in test_genomes:
            genome_path = f"src/data/genomes/selected_genomes/{test_genome}.fa"
            output_path = f"{self.output_dir}/fold_{fold}/{test_genome}"
            self._predict(genome_path, model_path, output_path)

        results = self._evaluate(f"{self.output_dir}/fold_{fold}", test_clusters)
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
        output_dir="src/results/gecco",
        hmm_dir="src/data/hmm/Pfam35.hmm.gz" # make sure hmms are downloaded and path is correct
    )
    handler.run_cross_validation(k=5)


if __name__ == "__main__":
    main()