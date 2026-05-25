import polars
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score,  roc_curve, auc, matthews_corrcoef
import seaborn as sns
from matplotlib_venn import venn3
from tqdm import tqdm
from viz_data import model_names


def reset_start_end(table: polars.DataFrame) -> polars.DataFrame:
    return table.with_columns(
        polars.when(polars.col("start") < polars.col("end")).then(polars.col("start")).otherwise(polars.col("end")).alias("start"),
        polars.when(polars.col("start") < polars.col("end")).then(polars.col("end")).otherwise(polars.col("start")).alias("end"),
    )

def join_gene_and_PUL_table(gene_table: polars.DataFrame, cluster_table: polars.DataFrame, buffer: int = 100,) -> polars.DataFrame:
    gene_table = reset_start_end(gene_table)
    cluster_table = reset_start_end(cluster_table)

    labled_gene_table = (
        cluster_table
        .rename({"start": "pul_start", "end": "pul_end"}) # avoid column name conflicts
        .join(
            gene_table,
            on="sequence_id",
            how="inner",
            validate="m:m",
        )
        .with_columns(
            polars.when(
                polars.col("start") >= polars.col("pul_start") - buffer, # allow for some buffer around the PUL boundaries
                polars.col("end") <= polars.col("pul_end") + buffer,
            )
            .then(polars.col("cluster_id"))
            .otherwise(None)
            .alias("cluster_id"),
            polars.when(
                polars.col("start") >= polars.col("pul_start") - buffer,
                polars.col("end") <= polars.col("pul_end") + buffer,
            )
            .then(True)
            .otherwise(False)
            .cast(polars.Boolean)
            .alias("is_PUL")
        )
        # aggregate by protein_id to determine if protein is in any PUL
        .group_by("protein_id")
        .agg(
            polars.col("is_PUL").any().alias("is_PUL"),
            polars.col("sequence_id").first().alias("sequence_id"),
            polars.col("start").first().alias("start"),
            polars.col("end").first().alias("end"),
            polars.col("cluster_id").drop_nulls().first().alias("cluster_id")
        )
        .sort(by=["sequence_id", "start", "end"])
        .with_row_index(name="gene_id", offset=0)  # important
        .select(["sequence_id", "protein_id", "start", "end", "is_PUL", "cluster_id"])
    )

    return labled_gene_table


class PredictionEvaluator:
    """
    Evaluator class for evaluating the predictions of GECCO against experimental data and PULpy annotations.
    Currently aggregates predictions across all folds.
    """
    def __init__(self, labeled_results_path, 
                clusters_table_path="src/data/data_collection/clusters_deduplicated_cblaster.tsv", 
                pulpy_annotations_path="src/data/data_collection/pulpy_annotations.tsv",
                cblaster_annotations_path="src/data/data_collection/cblaster_results_liberal.tsv", 
                cryptic_puls_path="src/data/data_collection/cryptic_puls_genes.tsv",
                k=7, model_name="gecco_pfam", split="test", output_path="results/plots", weight=1.0, aggregate=False):

        self.model_name = model_name
        self.split = split
        self.output_path = output_path
        self.labeled_results_raw = []
        self.labeled_results = []
        self.weight = weight

        for i in range(k):
            labeled_results = polars.read_csv(f"{labeled_results_path}_{i}.tsv", separator='\t')
            self.labeled_results.append(labeled_results)
            self.labeled_results_raw.append(labeled_results) # keep a copy of the raw results before joining with annotations

        self.get_pulpy_annotations(pulpy_annotations_path)
        self.cryptic_puls = polars.read_csv(cryptic_puls_path, separator="\t")
        self.clusters_table = polars.read_csv(clusters_table_path, separator='\t', infer_schema_length=600)
        self.filter = None
        self.aggregated = False
        if aggregate and k >= 5:
            self.aggregate_all_folds()

        self.threshold = self.set_threshold()
        os.makedirs(self.output_path, exist_ok=True)


    def set_threshold(self):
        thresholds_df_path = f"src/data/results/{self.model_name}/thresholds.tsv"
        if os.path.exists(thresholds_df_path):
            thresholds_df = polars.read_csv(thresholds_df_path, separator='\t').select("threshold")[:5].median()
            return thresholds_df.item()
        else:
            return 0.25


    def get_evaluation_data(self, labeled_results_df, mask_cryptic=False):
        """
        Returns tuple of (true_labels, predictions, predicted_probs, pulpy_predictions)
        Removes genes labeled as cryptic if mask_cryptic is set to True
        """

        if mask_cryptic:
            labeled_results_df = labeled_results_df.join(self.cryptic_puls.select("protein_id"), how="anti", on="protein_id")

        true = labeled_results_df.select(polars.col("is_PUL")).fill_null(False).fill_nan(False).to_series().to_list()
        pred = labeled_results_df.select(polars.col("is_PUL_pred")).fill_null(False).fill_nan(False).to_series().to_list()
        p_pred = labeled_results_df.select(polars.col("average_p")).fill_null(0.0).fill_nan(0.0).to_series().to_list()
        pulpy_pred = labeled_results_df.select(polars.col("is_PUL_pulpy")).fill_null(False).fill_nan(False).to_series().to_list()
        return true, pred, p_pred, pulpy_pred


    def recompute_predictions(self, fold, threshold=None):
        if threshold is None:
            threshold = self.threshold

        self.labeled_results[fold] = self.labeled_results[fold].with_columns(
            polars.when(polars.col("average_p") >= threshold).then(True).otherwise(False).alias("is_PUL_pred")
        )


    def aggregate_all_folds(self, k=5):
        # aggregate all folds into one table for overall evaluation, saves as labeled_results[0]
        if not self.aggregated:
            print("Aggregating all folds for overall evaluation...")
            all_labeled_tables = []
            for fold in range(k):
                df = (
                    self.labeled_results[fold]
                    .join(
                        self.clusters_table.select("sequence_id", "phylum", "species").unique(), 
                        on="sequence_id", 
                        how="left"
                    )
                )
                # cast types to prevent issues
                if "start_pred" in df.columns: 
                    df = df.with_columns(
                        polars.col("start_pred").cast(polars.Int64, strict=False),
                        polars.col("end_pred").cast(polars.Int64, strict=False),
                    )
                all_labeled_tables.append(df)

            self.labeled_results = [polars.concat(all_labeled_tables)] # keep as list
            self.labeled_results_raw = [polars.concat(all_labeled_tables)]
            self.get_pulpy_annotations("src/data/data_collection/pulpy_annotations.tsv") # re-join with pulpy annotations after concatenation
            self.aggregated = True


    def get_pulpy_annotations(self, pulpy_annotations_path):
        pulpy_annotations = (
            polars.read_csv(pulpy_annotations_path, separator='\t')
            .select("genome", "pulid", "start", "end")
            .rename({"genome": "sequence_id", "pulid": "cluster_id"})
        )
        for fold in range(len(self.labeled_results)):
            self.labeled_results[fold] = self.labeled_results[fold].join(
                (
                    join_gene_and_PUL_table(self.labeled_results_raw[fold], pulpy_annotations)
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
            .drop("phylum")
        )
        self.filter = phylum


    def plot_pr(self, true, pred, label, color, ax, weights=None, thresholds_to_mark=[]):
        """
        Plots PR curve based on true and pred. Applies plot to ax specified in arguments.
        Returns: AUPRC score
        """

        if len(true) == 0 or len(pred) == 0:
            print(f"Warning: No data to plot for {label}. Skipping PR curve.")
            return

        precision, recall, thresholds = precision_recall_curve(true, pred, drop_intermediate=True, sample_weight=weights)
        auc = average_precision_score(true, pred, sample_weight=weights)
        ax.plot(recall, precision, label=label + " (AUC: {:.2f})".format(auc), color=color)

        # show cutoffs on the plot
        if len(thresholds_to_mark) > 0:
            for t in thresholds_to_mark:
                idx = (np.abs(thresholds - t)).argmin()
                r = recall[idx + 1]
                p = precision[idx + 1]
                ax.scatter(r, p, color=color, s=5, alpha=0.5)
                ax.text(r, p, f"{t:.1f}", fontsize=8, color=color)

        return auc


    def plot_pr_dot(self, true, pred, color, ax):
        if len(true) == 0 or len(pred) == 0:
            print(f"Warning: No data to plot. Skipping PR dot.")
            return

        precision, recall, thresholds = precision_recall_curve(true, pred, drop_intermediate=True)
        auc = average_precision_score(true, pred)
        ax.scatter(recall[1], precision[1], color=color)


    def plot_baseline(self, true, ax):
        baseline = sum(true) / len(true) if len(true) > 0 else 0
        ax.plot([0, 1], [baseline, baseline], linestyle='--', color='gray')


    def roc_curve(self, true, p_pred, label, color, ax):
        if len(true) == 0 or len(p_pred) == 0:
            print(f"Warning: No data to plot for ROC curve. Skipping.")
            return

        fpr, tpr, thresholds = roc_curve(true, p_pred)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, label=f'{label} (AUC: {round(roc_auc, 2)})')


    # def precision_recall_curve(self, fold):
    #     fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    #     if fold == "all":
    #         self.aggregate_all_folds()
    #         self.set_evaluation_data(0)
    #     else:
    #         self.set_evaluation_data(fold)

    #     colors = plt.cm.tab20.colors
    #     # for true vs pred
    #     self.plot_pr(self.true, self.p_pred, "All taxa", colors[0], ax[0])

    #     # for pulpy vs pred
    #     self.plot_pr(self.pulpy_pred, self.p_pred, "All taxa", colors[1], ax[1])
    #     # dot for pulpy vs experimental
    #     self.plot_pr_dot(self.true, self.pulpy_pred, colors[4], ax[0])
    #     # compute baselines
    #     baseline = sum(self.true) / len(self.true) if len(self.true) > 0 else 0
    #     baseline_pulpy = sum(self.pulpy_pred) / len(self.pulpy_pred) if len(self.pulpy_pred) > 0 else 0

    #     # then filter by phylum and plot again
    #     self.filter_phylum("Bacteroidota", fold if fold != "all" else 0)
    #     self.plot_pr(self.true, self.p_pred, "Bacteroidota", colors[2], ax[0])
    #     self.plot_pr(self.pulpy_pred, self.p_pred, "Bacteroidota", colors[3], ax[1])

    #     # plot baselines
    #     ax[0].plot([0, 1], [baseline, baseline], linestyle='--', label="Baseline", color='gray')
    #     ax[1].plot([0, 1], [baseline_pulpy, baseline_pulpy], linestyle='--', label="Baseline", color='gray')

    #     # add labels and legend
    #     for i in range(2):
    #         ax[i].set_xlabel("Recall")
    #         ax[i].set_ylabel("Precision")
    #         ax[i].legend(loc="upper right")
    #     ax[0].set_title(self.model_name + " tested on experimental annotations")
    #     ax[1].set_title(self.model_name + " tested on PULpy annotations")

    #     fig.suptitle(f"Precision-Recall Curve for {self.model_name} (on {self.split} set, fold {fold})")
    #     plt.tight_layout()
    #     plt.savefig(f"{self.output_path}/pr_curve_{self.model_name}_{self.split}_{fold}.png")
    #     plt.clf()

    # def plot_roc_curves(self, fold):
    #     fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    #     self.set_evaluation_data(fold)
    #     self.roc_curve(self.true, self.p_pred, "True vs " + self.model_name, 'blue', ax[0])
    #     self.roc_curve(self.pulpy_pred, self.p_pred, "PULpy vs " + self.model_name, 'green', ax[1])

    #     plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title(f'ROC Curve for {self.model_name} (on {self.split} set, fold {fold})')
    #     plt.legend(loc="lower right")
    #     plt.savefig(f"{self.output_path}/roc_curve_{self.model_name}_{self.split}_{fold}.png")
    #     plt.clf()
    

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
        plt.close()


    def test_cryptic_puls(self, fold='all'):
        """
        Generates three PR curves, with either only experimental puls, cryptic puls or both as positives.
        """
        if fold == "all":
            self.aggregate_all_folds()
            df = self.labeled_results[0]
        else:
            df = self.labeled_results[fold]

        # add cryptic label
        cryptic_df = self.cryptic_puls.select("protein_id").with_columns(polars.lit(True).alias("is_cryptic"))
        df = df.join(cryptic_df, on="protein_id", how="left").with_columns(polars.col("is_cryptic").fill_null(False))

        # make binary labels
        df = df.with_columns([
            polars.col("is_PUL").fill_null(False).alias("y_exp"),
            polars.col("is_cryptic").alias("y_cryptic"),
            (polars.col("is_PUL").fill_null(False) | polars.col("is_cryptic")).alias("y_both"),
            polars.col("average_p").fill_nan(0.0)
        ])
        # test on ONLY experimental, remove cryptic from evaluation
        df_exp = df.filter(~polars.col("is_cryptic"))
        y_exp = df_exp.select("y_exp").to_series().to_list()
        p_pred_exp = df_exp.select("average_p").fill_null(0.0).to_series().to_list()
        # test on cryptic only, removing experimental from evaluation
        df_cryptic = df.filter(~polars.col("is_PUL"))
        y_cryptic = df_cryptic.select("y_cryptic").to_series().to_list()
        p_pred_cryptic = df_cryptic.select("average_p").fill_null(0.0).to_series().to_list()
        # test on both, including all in evaluation
        y_both = df.select("y_both").to_series().to_list()
        p_pred_both = df.select("average_p").fill_null(0.0).to_series().to_list()

        # --- Plot ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

        colors = plt.cm.tab10.colors
        thresholds_to_mark = [0.1, 0.3, 0.5, 0.7, 0.9]
        # --- Top: PR curves ---
        auprc_exp = self.plot_pr(y_exp, p_pred_exp, "Experimental", colors[0], ax1, thresholds_to_mark=thresholds_to_mark)
        auprc_cryptic = self.plot_pr(y_cryptic, p_pred_cryptic, "Cryptic", colors[1], ax1, thresholds_to_mark=thresholds_to_mark)
        auprc_both = self.plot_pr(y_both, p_pred_both, "Experimental + Cryptic", colors[2], ax1, thresholds_to_mark=thresholds_to_mark)

        ax1.set_xlabel("Recall")
        ax1.set_ylabel("Precision")
        ax1.set_title(f"Cryptic PUL evaluation ({model_names.get(self.model_name)}, fold {fold})")
        ax1.legend()

        # --- Bottom: histogram of prediction scores ---
        # get only probabilities of positives/negatives
        p_pred_exp = df_exp.filter(polars.col("y_exp")).select("average_p").fill_null(0.0).to_series().to_list()
        p_pred_cryptic = df_cryptic.filter(polars.col("y_cryptic")).select("average_p").fill_null(0.0).to_series().to_list()
        p_pred_negatives = df.filter(~polars.col("y_both")).select("average_p").fill_null(0.0).to_series().to_list()

        ax2.hist(p_pred_negatives, bins=30, density=False, alpha=0.2, label="Negatives", color=colors[2])
        ax2.hist(p_pred_exp, bins=30, density=False, alpha=0.5,label="Experimental", color=colors[0])
        ax2.hist(p_pred_cryptic, bins=30, density=False, alpha=0.5, label="Cryptic", color=colors[1])
        ax2.set_yscale("log")

        ax2.set_xlabel("Predicted probability (p_pred)")
        ax2.set_ylabel("Counts")
        ax2.set_title("Distribution of prediction scores")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f"{self.output_path}/cryptic_pr_{self.model_name}_{self.split}_{fold}.png")
        plt.close()

        return auprc_exp, auprc_cryptic, auprc_both


        def venn_diagram(self, fold):
            if fold == "all":
                self.aggregate_all_folds()
                df = self.labeled_results[0]
            else:
                df = self.labeled_results[fold]

            fig, ax = plt.subplots(figsize=(8, 4))
            true_set = set(df.filter(polars.col("is_PUL") == True).select("protein_id").to_series().to_list())
            pred_set = set(df.filter(polars.col("is_PUL_pred") == True).select("protein_id").to_series().to_list())
            pulpy_set = set(df.filter(polars.col("is_PUL_pulpy") == True).select("protein_id").to_series().to_list())
            venn3([true_set, pred_set, pulpy_set], ("Experimental", model_names.get(self.model_name), "PULpy"), ax=ax)
            ax.set_title(f"PUL predictions overlap ({self.split} set, fold {i})")
            ax.axis('off')
            fig.tight_layout()
            fig.savefig(f"{self.output_path}/venn_{fold}.png")
            plt.close()
