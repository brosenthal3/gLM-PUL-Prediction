import polars
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score,  roc_curve, auc, matthews_corrcoef
import seaborn as sns
from utility_scripts import join_gene_and_PUL_table
from matplotlib_venn import venn3
from tqdm import tqdm

class PredictionEvaluator:
    """
    Evaluator class for evaluating the predictions of GECCO against experimental data and PULpy annotations.
    Currently aggregates predictions across all folds.
    """

    def __init__(self, labeled_results_path, clusters_table_path, pulpy_annotations_path, k, model_name, split, output_path):
        self.model_name = model_name
        self.split = split
        self.output_path = output_path
        self.labeled_results = []

        for i in range(k):
            labeled_results = polars.read_csv(f"{labeled_results_path}_{i}.tsv", separator='\t')
            self.labeled_results.append(labeled_results)

        self.get_pulpy_annotations(pulpy_annotations_path)
        self.clusters_table = polars.read_csv(clusters_table_path, separator='\t', infer_schema_length=600)
        self.filter = None


    def set_evaluation_data(self, fold):
        self.true = self.labeled_results[fold].select(polars.col("is_PUL")).fill_null(False).to_series().to_list()
        self.pred = self.labeled_results[fold].select(polars.col("is_PUL_pred")).fill_null(False).to_series().to_list()
        self.p_pred = self.labeled_results[fold].select(polars.col("average_p")).fill_null(0.0).to_series().to_list()
        self.pulpy_pred = self.labeled_results[fold].select(polars.col("is_PUL_pulpy")).fill_null(False).to_series().to_list()


    def recompute_predictions(self, fold, threshold):
        self.labeled_results[fold] = self.labeled_results[fold].with_columns(
            polars.when(polars.col("average_p") >= threshold).then(True).otherwise(False).alias("is_PUL_pred")
        )
        self.set_evaluation_data(fold)


    def aggregate_all_folds(self):
        # aggregate all folds into one table for overall evaluation
        print("Aggregating all folds for overall evaluation...")
        all_labeled_tables = []
        for fold in range(len(self.labeled_results)):
            all_labeled_tables.append(
                self.labeled_results[fold]
                .join(self.clusters_table.select("sequence_id", "phylum", "species").unique(), on="sequence_id", how="left")
                .with_columns(
                    polars.col("start_pred").cast(polars.Int64, strict=False),
                    polars.col("end_pred").cast(polars.Int64, strict=False),
                )
            )
        self.labeled_results = [polars.concat(all_labeled_tables)] # keep as list
        self.get_pulpy_annotations("src/data/results/pulpy_annotations.tsv") # re-join with pulpy annotations after concatenation


    def get_pulpy_annotations(self, pulpy_annotations_path):
        pulpy_annotations = (
            polars.read_csv(pulpy_annotations_path, separator='\t')
            .select("genome", "pulid", "start", "end")
            .rename({"genome": "sequence_id", "pulid": "cluster_id"})
        )
        for fold in range(len(self.labeled_results)):
            self.labeled_results[fold] = self.labeled_results[fold].join(
                (
                    join_gene_and_PUL_table(self.labeled_results[fold], pulpy_annotations)
                    .select("protein_id", "is_PUL", "cluster_id").rename({"is_PUL": "is_PUL_pulpy", "cluster_id": "cluster_id_pulpy"})
                ),
                on="protein_id",
                how="left"
            )

    
    def filter_phylum(self, phylum, fold):
        self.labeled_results[fold] = (
            self.labeled_results[fold]
            .join(
                self.clusters_table.select("sequence_id", "phylum").unique(),
                on="sequence_id",
                how="left"
            )
            .filter(polars.col("phylum") == phylum)
        )
        self.set_evaluation_data(fold)
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


    def calculate_mmc(self, true, pred, thresholds):
        mmc_scores = []
        for threshold in tqdm(thresholds, desc="Calculating MCC for thresholds"):
            binary_pred = [1 if p >= threshold else 0 for p in pred]
            mmc = matthews_corrcoef(true, binary_pred)
            mmc_scores.append(mmc)

        best_mmc_idx = np.argmax(mmc_scores)
        best_threshold = thresholds[best_mmc_idx]
        binary_pred = [1 if p >= best_threshold else 0 for p in pred]
        pr_mmc = precision_recall_curve(true, binary_pred, drop_intermediate=True)
        return mmc_scores[best_mmc_idx], best_threshold, (pr_mmc[1][1], pr_mmc[0][1])


    def plot_pr(self, true, pred, label, color, ax, plot_mcc=False):
        if len(true) == 0 or len(pred) == 0:
            print(f"Warning: No data to plot for {label}. Skipping PR curve.")
            return

        precision, recall, thresholds = precision_recall_curve(true, pred, drop_intermediate=True)
        auc = average_precision_score(true, pred)
        ax.plot(recall, precision, label=label + " (AP: {:.2f})".format(auc), color=color, alpha=0.8)

        if plot_mcc:
            # Matthews correlation coefficient
            mcc_thresholds = np.linspace(0, 1, num=20)
            mcc, threshold, pr_point = self.calculate_mmc(true, pred, mcc_thresholds)        
            plt.scatter(pr_point[0], pr_point[1], label=f"Best MCC: {round(mcc, 2)}, threshold {round(threshold, 2)}", color=color, marker='X', s=100)


    def plot_pr_dot(self, true, pred, label, color, ax):
        if len(true) == 0 or len(pred) == 0:
            print(f"Warning: No data to plot for {label}. Skipping PR curve.")
            return

        precision, recall, thresholds = precision_recall_curve(true, pred, drop_intermediate=True)
        auc = average_precision_score(true, pred)
        ax.scatter(recall[1], precision[1], label=label + " (AP: {:.2f})".format(auc), color=color)


    def precision_recall_curve(self, fold):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        if fold == "all":
            self.aggregate_all_folds()
            self.set_evaluation_data(0)

        # standardize predicted probabilities to be between 0 and 1
        self.p_pred = (self.p_pred - np.min(self.p_pred)) / (np.max(self.p_pred) - np.min(self.p_pred)) 

        # use similar colors for associated curves
        colors = plt.cm.tab20.colors

        # for true vs pred
        self.plot_pr(self.true, self.p_pred, "True vs " + self.model_name, colors[0], ax, plot_mcc=True)
        # for pulpy vs pred
        self.plot_pr(self.pulpy_pred, self.p_pred, "PULpy vs " + self.model_name, colors[1], ax)
        self.plot_pr_dot(self.true, self.pulpy_pred, "True vs PULpy", colors[4], ax)
        baseline = sum(self.true) / len(self.true) if len(self.true) > 0 else 0

        # then filter by phylum and plot again
        self.filter_phylum("Bacteroidota", fold if fold != "all" else 0)
        self.plot_pr(self.true, self.p_pred, "True vs " + self.model_name + " (Bacteroidota)", colors[2], ax)
        self.plot_pr(self.pulpy_pred, self.p_pred, "PULpy vs " + self.model_name + " (Bacteroidota)", colors[3], ax)

        # plot baseline
        ax.plot([0, 1], [baseline, baseline], linestyle='--', label="Baseline", color='gray')

        # add labels and legend
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="upper right")
        ax.set_title(f"Precision-Recall Curve for {self.model_name} (on {self.split} set, fold {fold})")
        plt.savefig(f"{self.output_path}/pr_curve_{self.model_name}_{self.split}_{fold}.png")
        plt.clf()


    def roc_curve(self, true, p_pred, label, color):
        if len(true) == 0 or len(p_pred) == 0:
            print(f"Warning: No data to plot for ROC curve. Skipping.")
            return

        fpr, tpr, thresholds = roc_curve(true, p_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, label=f'{label} (AUC: {round(roc_auc, 2)})')

    def plot_roc_curves(self, fold):
        self.set_evaluation_data(fold)
        self.roc_curve(self.true, self.p_pred, "True vs " + self.model_name, 'blue')
        self.roc_curve(self.pulpy_pred, self.p_pred, "PULpy vs " + self.model_name, 'green')

        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {self.model_name} (on {self.split} set, fold {fold})')
        plt.legend(loc="lower right")
        plt.savefig(f"{self.output_path}/roc_curve_{self.model_name}_{self.split}_{fold}.png")
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


    def lengths_histogram(self, fold):
        predicted_lengths = self.get_prediction_lengths(self.labeled_results[fold], "is_PUL_pred", "cluster_id_pred")
        true_lengths = self.get_prediction_lengths(self.labeled_results[fold], "is_PUL", "cluster_id")
        pulpy_lengths = self.get_prediction_lengths(self.labeled_results[fold], "is_PUL_pulpy", "cluster_id_pulpy")
        lengths = [predicted_lengths, true_lengths, pulpy_lengths]

        plt.figure()
        for data, label in zip(lengths, ["Predicted PULs", "True PULs", "PULpy PULs"]):
            sns.kdeplot(data=data, fill=True, label=label, cut=0, common_norm=False, bw_adjust=0.7)
        plt.xlim(0, 100)
        plt.xlabel('PUL Length (number of genes)')
        plt.ylabel('Density (KDE)')
        plt.title(f'PUL Lengths distributions {"(filtered by " + self.filter + ")" if self.filter else ""}') 
        plt.legend()
        plt.savefig(f"{self.output_path}/pul_length_kde_{self.model_name}_{self.filter}_{fold}.png")
        plt.clf()

    
    def visualize_predictions_in_genome(self, sequence_id, fold, threshold):
        self.recompute_predictions(fold, threshold)

        # get all genes of this sequence
        genes = self.labeled_results[fold].filter(polars.col("sequence_id") == sequence_id)
        sequence_length = self.clusters_table.filter(polars.col("sequence_id") == sequence_id).select("length").to_series().to_list()[0]
        phylum = self.clusters_table.filter(polars.col("sequence_id") == sequence_id).select("phylum").to_series().to_list()[0]
        species = self.clusters_table.filter(polars.col("sequence_id") == sequence_id).select("species").to_series().to_list()[0]
        # create fig
        fig, axs = plt.subplots(figsize=(10, 3))
        features = []
        # top ax: called genes
        for row in genes.iter_rows(named=True):
            if row["is_PUL"]:
                features.append((row["start"], row["end"], 0, "Experimental"))
            if row["is_PUL_pulpy"]:
                features.append((row["start"], row["end"], 1, "PULpy"))
            if row["is_PUL_pred"]:
                features.append((row["start"], row["end"], 2, "Predicted"))

        colors = {
            "Experimental": "blue",
            "Predicted": "orange",
            "PULpy": "green"
        }
        for start, end, y, label in features:
            axs.fill_betweenx([y, y + 0.9], start, end, color=colors[label], alpha=1)

        axs.set_ylim(0, 3)
        axs.set_yticks([0.25, 1.25, 2.25], ["Experimental", "PULpy", f"{self.model_name} (Threshold: {threshold})"])
        plt.suptitle(f"PUL predictions for sequence {sequence_id} (species: {species}, phylum: {phylum})")
        plt.tight_layout()
        os.makedirs(f"{self.output_path}/predictions_in_genome/", exist_ok=True)
        plt.savefig(f"{self.output_path}/predictions_in_genome/{sequence_id}_{self.split}_{fold}.png")
        plt.clf()


    def venn_diagram(self):
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        for i in range(7):
            ax = axs[i//4, i%4]
            true_set = set(self.labeled_results[i].filter(polars.col("is_PUL") == True).select("protein_id").to_series().to_list())
            pred_set = set(self.labeled_results[i].filter(polars.col("is_PUL_pred") == True).select("protein_id").to_series().to_list())
            pulpy_set = set(self.labeled_results[i].filter(polars.col("is_PUL_pulpy") == True).select("protein_id").to_series().to_list())
            venn3([true_set, pred_set, pulpy_set], ("Experimental", self.model_name, "PULpy"), ax=ax)
            ax.set_title(f"PUL predictions ({self.split} set, fold {i})")

        axs[1, 3].axis('off')
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/venn_diagram_{self.model_name}_{self.split}.png")
        plt.clf()

    
    def f1_per_fold(self):
        f1_scores_per_fold = []
        average_precision_scores = []
        # get f1 score and AP for each fold
        for i in range(len(self.labeled_results)):
            self.set_evaluation_data(i)
            average_precision_scores.append(average_precision_score(self.true, self.p_pred))
            report = classification_report(self.true, self.pred, output_dict=True)
            f1_score_false = report["False"]["f1-score"]
            f1_score_true = report["True"]["f1-score"]
            f1_scores_per_fold.append((f1_score_false, f1_score_true))

        # plot the F1 scores per fold
        folds = np.arange(len(self.labeled_results))
        f1_false = [score[0] for score in f1_scores_per_fold]
        f1_true = [score[1] for score in f1_scores_per_fold]
        plt.figure()
        plt.bar(folds - 0.2, average_precision_scores, width=0.4, label="Average Precision Score", color='purple')
        plt.bar(folds + 0.2, f1_true, width=0.4, label="F1 Score (True)")
        plt.xlabel("Fold")
        plt.ylabel("Score")
        plt.title(f"F1 and PR-AUC Scores per fold (on {self.split} set)")
        plt.legend()
        plt.savefig(f"{self.output_path}/f1_scores_per_fold_{self.model_name}_{self.split}.png")
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate predictions of GECCO against experimental data and PULpy annotations"
    )
    parser.add_argument("--model", type=str, help="Name of model to evaluate", required=True)
    parser.add_argument("--split", type=str, default="test", help="Whether to evaluate on test or train set")
    parser.add_argument("-k", type=int, default=1, help="Number of folds to evaluate")
    args = parser.parse_args()
    model_name = args.model
    output_path = f"src/data/plots/{model_name}"
    os.makedirs(output_path, exist_ok=True)

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
        split=args.split,
        output_path=output_path
    )
    #evaluator.venn_diagram()
    #evaluator.f1_per_fold()

    # for fold in range(args.k):
    #     evaluator.precision_recall_curve(fold)
    #     evaluator.plot_roc_curves(fold)
    #evaluator.precision_recall_curve("all")

    evaluator.visualize_predictions_in_genome("AE015928", 0, 0.21)


    """
    python src/scripts/evaluate_predictions.py --model genecat --split test -k 5
    python src/scripts/evaluate_predictions.py --model gecco --split test -k 5
    """