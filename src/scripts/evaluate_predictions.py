import polars
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
import seaborn as sns
from utility_scripts import join_gene_and_PUL_table

class PredictionEvaluator:
    """
    Evaluator class for evaluating the predictions of GECCO against experimental data and PULpy annotations.
    Currently aggregates predictions across all folds.
    """

    def __init__(self, labeled_results_path, clusters_table_path, pulpy_annotations_path, k, model_name, split):
        self.model_name = model_name
        self.split = split
        self.classification_reports = []
        self.average_precision_scores = []
        labeled_results_list = []

        for i in range(k):
            labeled_results = polars.read_csv(f"{labeled_results_path}_{i}.tsv", separator='\t')
            labeled_results_list.append(labeled_results)
            self.labeled_results = labeled_results
            self.get_pulpy_annotations(pulpy_annotations_path)
            self.clusters_table = polars.read_csv(clusters_table_path, separator='\t', infer_schema_length=600)
            self.set_evaluation_data()
            self.filter = None
            print("Fold ", i)
            self.confusion_matrix()
            self.classification_reports.append(classification_report(self.true, self.pred, output_dict=True))
            self.average_precision_scores.append(average_precision_score(self.true, self.p_pred))

        self.labeled_results = polars.concat(labeled_results_list)
        self.get_pulpy_annotations(pulpy_annotations_path)
        self.clusters_table = polars.read_csv(clusters_table_path, separator='\t', infer_schema_length=600)
        self.set_evaluation_data()
        self.filter = None


    def set_evaluation_data(self):
        self.true = self.labeled_results.select(polars.col("is_PUL")).fill_null(False).to_series().to_list()
        self.pred = self.labeled_results.select(polars.col("is_PUL_pred")).fill_null(False).to_series().to_list()
        self.p_pred = self.labeled_results.select(polars.col("average_p")).fill_null(0.0).to_series().to_list()
        self.pulpy_pred = self.labeled_results.select(polars.col("is_PUL_pulpy")).fill_null(False).to_series().to_list()


    def get_pulpy_annotations(self, pulpy_annotations_path):
        pulpy_annotations = (
            polars.read_csv(pulpy_annotations_path, separator='\t')
            .select("genome", "pulid", "start", "end")
            .rename({"genome": "sequence_id", "pulid": "cluster_id"})
        )
        pulpy_annotations = (
            join_gene_and_PUL_table(self.labeled_results, pulpy_annotations)
            .select("protein_id", "is_PUL", "cluster_id").rename({"is_PUL": "is_PUL_pulpy", "cluster_id": "cluster_id_pulpy"})
        )
        self.labeled_results = self.labeled_results.join(
            pulpy_annotations,
            on="protein_id",
            how="left"
        )

    
    def filter_phylum(self, phylum):
        self.labeled_results = (
            self.labeled_results
            .join(
                self.clusters_table.select("sequence_id", "phylum").unique(),
                on="sequence_id",
                how="left"
            )
            .filter(polars.col("phylum") == phylum)
        )
        self.set_evaluation_data()
        self.filter = phylum

    
    def confusion_matrix(self):
        cm = confusion_matrix(self.true, self.pred)
        print(cm)


    def evaluate(self):
        print("True vs Predicted:")
        print(classification_report(self.true, self.pred))

        print("True vs PULpy:")
        print(classification_report(self.true, self.pulpy_pred))

        print("PULpy vs Predicted:")
        print(classification_report(self.pulpy_pred, self.pred))


    def plot_pr(self, true, pred, label, color):
        precision, recall, thresholds = precision_recall_curve(true, pred)
        auc = average_precision_score(true, pred)
        plt.plot(recall, precision, label=label + " (AUC: {:.2f})".format(auc), color=color)


    def precision_recall_curve(self):
        # standardize predicted probabilities to be between 0 and 1
        self.p_pred = (self.p_pred - np.min(self.p_pred)) / (np.max(self.p_pred) - np.min(self.p_pred)) 

        # use similar colors for associated curves
        colors = plt.cm.tab20.colors
        # for true vs pred
        self.plot_pr(self.true, self.p_pred, "True vs " + self.model_name, colors[0])
        # for pulpy vs pred
        self.plot_pr(self.pulpy_pred, self.p_pred, "PULpy vs " + self.model_name, colors[1])

        # then filter by phylum and plot again
        self.filter_phylum("Bacteroidota")
        self.plot_pr(self.true, self.p_pred, "True vs " + self.model_name + " (Bacteroidota)", colors[2])
        self.plot_pr(self.pulpy_pred, self.p_pred, "PULpy vs " + self.model_name + " (Bacteroidota)", colors[3])

        # add labels and legend
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="upper right")
        plt.title(f"Precision-Recall Curve for {self.model_name} (on {self.split} set)")
        plt.savefig(f"src/data/plots/pr_curve_{self.model_name}_{self.filter}_{self.split}.png")
        plt.clf()


    def get_prediction_lengths(self, table, label_col, cluster_col):
        lengths = (
            table
            .filter(polars.col(label_col) == True)
            .group_by(cluster_col)
            .agg(polars.count("protein_id").alias("length"))
            .select("length").to_series().to_list()
        )
        return lengths


    def lengths_histogram(self):
        predicted_lengths = self.get_prediction_lengths(self.labeled_results, "is_PUL_pred", "cluster_id_pred")
        true_lengths = self.get_prediction_lengths(self.labeled_results, "is_PUL", "cluster_id")
        pulpy_lengths = self.get_prediction_lengths(self.labeled_results, "is_PUL_pulpy", "cluster_id_pulpy")
        lengths = [predicted_lengths, true_lengths, pulpy_lengths]

        plt.figure()
        for data, label in zip(lengths, ["Predicted PULs", "True PULs", "PULpy PULs"]):
            sns.kdeplot(data=data, fill=True, label=label, cut=0, common_norm=False, bw_adjust=0.7)
        plt.xlim(0, 100)
        plt.xlabel('PUL Length (number of genes)')
        plt.ylabel('Density (KDE)')
        plt.title(f'PUL Lengths distributions {"(filtered by " + self.filter + ")" if self.filter else ""}') 
        plt.legend()
        plt.savefig(f"src/data/plots/pul_length_kde_{self.model_name}_{self.filter}_{self.split}.png")
        plt.clf()

    
    def visualize_predictions(self, sequence_id):
        # get all genes of this sequence
        genes = self.labeled_results.filter(polars.col("sequence_id") == sequence_id)
        sequence_length = self.clusters_table.filter(polars.col("sequence_id") == sequence_id).select("length").to_series().to_list()[0]
        fig, axs = plt.subplots(figsize=(10, 3))
        features = []
        # top ax: called genes
        for row in genes.iter_rows(named=True):
            if row["is_PUL"]:
                features.append((row["start"], row["end"], 0, "Experimental"))
            if row["is_PUL_pred"]:
                features.append((row["start"], row["end"], 1, "Predicted"))
            if row["is_PUL_pulpy"]:
                features.append((row["start"], row["end"], 2, "PULpy"))

        colors = {
            "Experimental": "blue",
            "Predicted": "orange",
            "PULpy": "green"
        }
        for start, end, y, label in features:
            axs.fill_betweenx([y, y + 0.9], start, end, color=colors[label], alpha=1)

        axs.set_ylim(0, 3)
        axs.set_yticks([0.25, 1.25, 2.25], ["Experimental", "Predicted", "PULpy"])
        plt.suptitle(f"PUL predictions for sequence {sequence_id}")
        plt.tight_layout()
        plt.savefig(f"src/data/plots/predictions_{sequence_id}_{self.split}.png")
        plt.clf()


    def f1_per_genome(self):
        all_labels = self.labeled_results
        unique_genomes = all_labels.select("sequence_id").unique().to_series().to_list()
        for genome in unique_genomes:
            genome_labels = all_labels.filter(polars.col("sequence_id") == genome)
            true = genome_labels.select(polars.col("is_PUL")).fill_null(False).to_series().to_list()
            pred = genome_labels.select(polars.col("is_PUL_pred")).fill_null(False).to_series().to_list()
            report = classification_report(true, pred, output_dict=True, zero_division=0)
            f1_score_false = report["False"]["f1-score"]
            f1_score_true = report["True"]["f1-score"]
            print(f"Genome: {genome}, F1 Score (False): {f1_score_false:.2f}, F1 Score (True): {f1_score_true:.2f}")
            print(f"Total predicted: {sum(pred)}, Total true: {sum(true)}\n")

    
    def f1_per_fold(self):
        f1_scores_per_fold = []
        for i, report in enumerate(self.classification_reports):
            f1_score_false = report["False"]["f1-score"]
            f1_score_true = report["True"]["f1-score"]
            f1_scores_per_fold.append((f1_score_false, f1_score_true))

        # plot the F1 scores per fold
        folds = np.arange(len(self.classification_reports))
        f1_false = [score[0] for score in f1_scores_per_fold]
        f1_true = [score[1] for score in f1_scores_per_fold]
        plt.figure()
        plt.bar(folds - 0.2, self.average_precision_scores, width=0.4, label="Average Precision Score", color='purple')
        plt.bar(folds + 0.2, f1_true, width=0.4, label="F1 Score (True)")
        plt.xlabel("Fold")
        plt.ylabel("Score")
        plt.title(f"F1 and RC-AUC Scores per fold (on {self.split} set)")
        plt.legend()
        plt.savefig(f"src/data/plots/f1_scores_per_fold_{self.model_name}_{self.split}.png")
        plt.clf()


# def process_genecat_zeroshot():
#     genecat_results = polars.read_parquet("src/data/results/genecat/zero_shot_results/linmodel_results_pfam_embeddings_0.parquet")
#     genes = polars.read_parquet("src/data/genecat_output/genome.genes.parquet")
#     test_clusters = polars.read_csv("src/data/splits/test_fold_0.tsv", separator='\t')

#     # get all genes in test set
#     test_genes = (genes.join(test_clusters, on="sequence_id", how="semi"))

#     # join genes with test clusters and predicted clusters
#     cols = ["protein_id", "sequence_id", "cluster_id", "is_PUL", "start", "end"]
#     labeled_test_genes = join_gene_and_PUL_table(test_genes, test_clusters).select(cols)

#     # join gene tables of predicted clusters with test clusters
#     labeled_table = (
#         labeled_test_genes
#         .join(genecat_results.select("protein_id", "probas").rename({"probas": "average_p"}), on="protein_id", how="left")
#         .with_columns(
#             polars.when(polars.col("is_PUL").is_null()).then(False).otherwise(polars.col("is_PUL")).alias("is_PUL"),
#             polars.when(polars.col("average_p").ge(0.5)).then(True).otherwise(False).alias("is_PUL_pred"),
#         )
#         .sort("protein_id")
#         .sort("sequence_id")
#     )
#     labeled_table.write_csv("src/data/results/genecat/zero_shot_results/labeled_results_0.tsv", separator='\t')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate predictions of GECCO against experimental data and PULpy annotations"
    )
    parser.add_argument("--model", type=str, help="Name of model to evaluate", required=True)
    parser.add_argument("--split", type=str, default="test", help="Whether to evaluate on test or train set")
    parser.add_argument("-k", type=int, default=1, help="Number of folds to evaluate")
    args = parser.parse_args()
    model_name = args.model

    if model_name == "genecat":
        results_path = "src/data/results/genecat/zero_shot_results/labeled_results_" + args.split
    elif model_name == "gecco":
        results_path = "src/data/results/gecco/labeled_results_" + args.split
    else:
        raise ValueError("Invalid model name.")

    evaluator = PredictionEvaluator(
        f"{results_path}",
        "src/data/results/cblaster_results.tsv",
        "src/data/results/pulpy_annotations.tsv",
        k=args.k,
        model_name=model_name,
        split=args.split
    )
    evaluator.f1_per_fold()
    evaluator.precision_recall_curve()
    # evaluator.visualize_predictions("NC_008261")