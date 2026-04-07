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

    def __init__(self, labeled_results_path, clusters_table_path, pulpy_annotations_path):
        labeled_results_list = []
        for i in range(5):
            labeled_results = polars.read_csv(f"{labeled_results_path}_{i}.tsv", separator='\t')
            labeled_results_list.append(labeled_results)

        self.labeled_results = polars.concat(labeled_results_list)
        self.clusters_table = polars.read_csv(clusters_table_path, separator='\t', infer_schema_length=600)
        self.get_pulpy_annotations(pulpy_annotations_path)
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


    def precision_recall_curve(self):
        # for true vs pred
        precision, recall, thresholds = precision_recall_curve(self.true, self.p_pred)
        auc = average_precision_score(self.true, self.p_pred)
        plt.plot(recall, precision, label="Experimental vs Gecco (AUC: {:.2f})".format(auc))

        # for pulpy vs pred
        precision, recall, thresholds = precision_recall_curve(self.pulpy_pred, self.p_pred)
        auc = average_precision_score(self.pulpy_pred, self.p_pred)
        plt.plot(recall, precision, label="PULpy vs Gecco (AUC: {:.2f})".format(auc))

        # add labels and legend
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="upper right")
        plt.title(f"Precision-Recall Curve {'(filtered by ' + self.filter + ')' if self.filter else ''}")
        plt.savefig(f"src/data/plots/pr_curve{self.filter}.png")
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
        plt.savefig(f"src/data/plots/pul_length_kde{self.filter}.png")
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
            axs.plot([start, end], [y, y+0.9], label=label, color=colors[label], linewidth=3)

        axs.set_ylim(0, 3)
        axs.set_yticks([0.25, 1.25, 2.25], ["Experimental", "Predicted", "PULpy"])
        plt.suptitle(f"PUL predictions for sequence {sequence_id}")
        plt.tight_layout()
        plt.savefig(f"src/data/plots/predictions_{sequence_id}.png")
        plt.clf()



if __name__ == "__main__":
    results_path = "src/data/results/gecco"
    pulpy_annotations_path = "src/data/results/pulpy_annotations.tsv"

    evaluator = PredictionEvaluator(
        f"{results_path}/labeled_results",
        "src/data/results/cblaster_results.tsv",
        f"{pulpy_annotations_path}"
    )
    evaluator.filter_phylum("Bacteroidota")
    evaluator.lengths_histogram()
    evaluator.precision_recall_curve()
    evaluator.evaluate()
    #evaluator.visualize_predictions("FP476056")