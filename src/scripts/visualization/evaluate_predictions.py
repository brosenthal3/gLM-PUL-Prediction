import polars
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score,  roc_curve, auc, matthews_corrcoef
import seaborn as sns
#from utility_scripts import join_gene_and_PUL_table
from matplotlib_venn import venn3
from tqdm import tqdm


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
                k=7, model_name="gecco_pfam", split="test", output_path="results/plots", weight=1.0):

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
        self.get_cblaster_annotations(cblaster_annotations_path)
        self.clusters_table = polars.read_csv(clusters_table_path, separator='\t', infer_schema_length=600)
        self.filter = None


    def set_evaluation_data(self, fold):
        self.true = self.labeled_results[fold].select(polars.col("is_PUL")).fill_null(False).to_series().to_list()
        self.pred = self.labeled_results[fold].select(polars.col("is_PUL_pred")).fill_null(False).to_series().to_list()
        self.p_pred = self.labeled_results[fold].select(polars.col("average_p")).fill_null(0.0).to_series().to_list()
        self.pulpy_pred = self.labeled_results[fold].select(polars.col("is_PUL_pulpy")).fill_null(False).to_series().to_list()
        # weight down genes with PULpy or cblaster annotations but no experimental data (likely cryptic puls)
        self.sample_weights = (
            self.labeled_results[fold]
            .with_columns(
                polars.when(((polars.col("is_PUL_pulpy") == True) | (polars.col("is_PUL_cblaster") == True)) & (polars.col("is_PUL") == False))
                .then(self.weight)
                .otherwise(1.0)
                .alias("sample_weight")
            )
            .select("sample_weight").to_series().to_list()
        )

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
            df = (
                self.labeled_results[fold]
                .join(self.clusters_table.select("sequence_id", "phylum", "species").unique(), on="sequence_id", how="left")
            )
            # cast types to prevent issues
            if "start_pred" in df.columns: 
                df = df.with_columns(
                    polars.col("start_pred").cast(polars.Int64, strict=False),
                    polars.col("end_pred").cast(polars.Int64, strict=False),
                )
            all_labeled_tables.append(df)


        self.labeled_results = [polars.concat(all_labeled_tables)] # keep as list
        self.get_pulpy_annotations("src/data/data_collection/pulpy_annotations.tsv") # re-join with pulpy annotations after concatenation


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

    def get_cblaster_annotations(self, cblaster_annotations_path):
        cblaster_annotations = (
            polars.read_csv(cblaster_annotations_path, separator='\t')
            .select("sequence_id", "start", "end")
        )
        for fold in range(len(self.labeled_results)):
            self.labeled_results[fold] = self.labeled_results[fold].join(
                (
                    join_gene_and_PUL_table(self.labeled_results_raw[fold], cblaster_annotations)
                    .select("protein_id", "is_PUL", "cluster_id").rename({"is_PUL": "is_PUL_cblaster", "cluster_id": "cluster_id_cblaster"})
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


    def plot_pr(self, true, pred, label, color, ax, plot_mcc=False, weights=None):
        if len(true) == 0 or len(pred) == 0:
            print(f"Warning: No data to plot for {label}. Skipping PR curve.")
            return

        precision, recall, thresholds = precision_recall_curve(true, pred, drop_intermediate=True, sample_weight=weights)
        auc = average_precision_score(true, pred, sample_weight=weights)
        ax.plot(recall, precision, label=label + " (AP: {:.2f})".format(auc), color=color, alpha=0.8)

        if plot_mcc:
            # Matthews correlation coefficient
            mcc_thresholds = np.linspace(0, 1, num=20)
            mcc, threshold, pr_point = self.calculate_mmc(true, pred, mcc_thresholds)        
            plt.scatter(pr_point[0], pr_point[1], label=f"Best MCC: {round(mcc, 2)}, threshold {round(threshold, 2)}", color=color, marker='X', s=100)


    def plot_pr_dot(self, true, pred, color, ax):
        if len(true) == 0 or len(pred) == 0:
            print(f"Warning: No data to plot. Skipping PR dot.")
            return

        precision, recall, thresholds = precision_recall_curve(true, pred, drop_intermediate=True)
        auc = average_precision_score(true, pred)
        ax.scatter(recall[1], precision[1], color=color)


    def precision_recall_curve(self, fold):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        if fold == "all":
            self.aggregate_all_folds()
            self.set_evaluation_data(0)
        else:
            self.set_evaluation_data(fold)

        # use similar colors for associated curves
        colors = plt.cm.tab20.colors

        # for true vs pred
        self.plot_pr(self.true, self.p_pred, "All taxa", colors[0], ax[0])
        #self.plot_pr(self.true, self.p_pred, f"All taxa, weighted {self.weight}", colors[5], ax[0], weights=self.sample_weights)

        # for pulpy vs pred
        self.plot_pr(self.pulpy_pred, self.p_pred, "All taxa", colors[1], ax[1])
        # dot for pulpy vs experimental
        self.plot_pr_dot(self.true, self.pulpy_pred, colors[4], ax[0])
        # compute baselines
        baseline = sum(self.true) / len(self.true) if len(self.true) > 0 else 0
        baseline_pulpy = sum(self.pulpy_pred) / len(self.pulpy_pred) if len(self.pulpy_pred) > 0 else 0

        # then filter by phylum and plot again
        self.filter_phylum("Bacteroidota", fold if fold != "all" else 0)
        self.plot_pr(self.true, self.p_pred, "Bacteroidota", colors[2], ax[0])
        self.plot_pr(self.pulpy_pred, self.p_pred, "Bacteroidota", colors[3], ax[1])

        # plot baselines
        ax[0].plot([0, 1], [baseline, baseline], linestyle='--', label="Baseline", color='gray')
        ax[1].plot([0, 1], [baseline_pulpy, baseline_pulpy], linestyle='--', label="Baseline", color='gray')

        # add labels and legend
        for i in range(2):
            ax[i].set_xlabel("Recall")
            ax[i].set_ylabel("Precision")
            ax[i].legend(loc="upper right")
        ax[0].set_title(self.model_name + " tested on experimental annotations")
        ax[1].set_title(self.model_name + " tested on PULpy annotations")

        fig.suptitle(f"Precision-Recall Curve for {self.model_name} (on {self.split} set, fold {fold})")
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/pr_curve_{self.model_name}_{self.split}_{fold}.png")
        plt.clf()


    def roc_curve(self, true, p_pred, label, color, ax=None):
        if len(true) == 0 or len(p_pred) == 0:
            print(f"Warning: No data to plot for ROC curve. Skipping.")
            return

        fpr, tpr, thresholds = roc_curve(true, p_pred)
        roc_auc = auc(fpr, tpr)

        if ax:
            ax.plot(fpr, tpr, color=color, label=f'{label} (AUC: {round(roc_auc, 2)})')
        else:
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


def compare_all_models(all_models, model_class):
    # comparison of all models
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig_roc, ax_roc = plt.subplots(1, 2, figsize=(12, 6))
    colors = plt.cm.tab20.colors

    # list of evaluators for all models
    evaluators = [
        PredictionEvaluator(
            labeled_results_path = f"src/data/results/{model_name}/labeled_results_test",
            model_name=model_name,
            k=5
        )
        for model_name in all_models
    ]

    for i, model_evaluator in enumerate(evaluators):
        print(f"Plotting for {all_models[i]}")
        model_evaluator.aggregate_all_folds()
        model_evaluator.set_evaluation_data(0)

        # for true vs pred
        model_evaluator.plot_pr(model_evaluator.true, model_evaluator.p_pred, all_models[i], colors[i], ax[0])
        model_evaluator.roc_curve(model_evaluator.true, model_evaluator.p_pred, all_models[i], colors[i], ax_roc[0])
        # for pulpy vs pred
        model_evaluator.plot_pr(model_evaluator.pulpy_pred, model_evaluator.p_pred, all_models[i], colors[i], ax[1])
        model_evaluator.roc_curve(model_evaluator.pulpy_pred, model_evaluator.p_pred, all_models[i], colors[i], ax_roc[1])

        # plot baselines only once
        if i == len(all_models)-1:
            baseline = sum(model_evaluator.true) / len(model_evaluator.true) if len(model_evaluator.true) > 0 else 0
            baseline_pulpy = sum(model_evaluator.pulpy_pred) / len(model_evaluator.pulpy_pred) if len(model_evaluator.pulpy_pred) > 0 else 0
            ax[0].plot([0, 1], [baseline, baseline], linestyle='--', label="Baseline", color='gray')
            ax[1].plot([0, 1], [baseline_pulpy, baseline_pulpy], linestyle='--', label="Baseline", color='gray')


    # add labels and legend
    for j in range(2):
        ax[j].set_xlabel("Recall")
        ax[j].set_ylabel("Precision")
        ax[j].legend(loc="upper right")

        ax_roc[j].set_xlabel('False Positive Rate')
        ax_roc[j].set_ylabel("True Positive Rate")
        ax_roc[j].legend(loc="lower right")

    ax[0].set_title("Models tested on experimental annotations")
    ax[1].set_title("Models tested on PULpy annotations")
    ax_roc[0].set_title("Models tested on experimental annotations")
    ax_roc[1].set_title("Models tested on PULpy annotations")

    fig.suptitle("Precision-Recall Curves of all tested models (all folds)")
    fig_roc.suptitle("ROC Curves of all tested models (all folds)")

    fig.tight_layout()
    fig.savefig(f"results/plots/pr_curves_{model_class}.png")
    fig_roc.tight_layout()
    fig_roc.savefig(f"results/plots/roc_curves_{model_class}.png")
    plt.close()


def main(args):
    model_name = args.model
    if model_name == "all":
        all_models = ["gecco_pfam", "gecco_cazy", "genecat_zeroshot_pfam", "genecat_zeroshot_cazy", "esmc", "bacformer"]
        compare_all_models(all_models, model_name)
        return

    if model_name == "masked":
        all_models = ["genecat_zeroshot_pfam", "genecat_zeroshot_pfam_masked", "genecat_zeroshot_cazy", "genecat_zeroshot_cazy_masked", "esmc", "esmc_masked", "bacformer", "bacformer_masked"]
        compare_all_models(all_models, model_name)
        return

    # path where results are saved
    results_path = f"src/data/results/{model_name}/labeled_results_{args.split}"
    if not os.path.exists(results_path+"_0.tsv"):
        raise ValueError("Invalid model name, or no results found.")

    # output path to save plots
    output_path = f"results/plots/{model_name}"
    os.makedirs(output_path, exist_ok=True)

    evaluator = PredictionEvaluator(
        f"{results_path}",
        k=args.k,
        model_name=f"{model_name}",
        split=args.split,
        output_path=output_path,
        weight=args.weight
    )

    if args.k == 7:
        evaluator.venn_diagram()

    evaluator.f1_per_fold()
    for fold in range(args.k):
         evaluator.precision_recall_curve(fold)
         evaluator.plot_roc_curves(fold)


    # new evaluator class for aggregating 5 folds instead of 7
    if args.k >= 5:
        evaluator = PredictionEvaluator(
            f"{results_path}",
            k=5,
            model_name=f"{model_name}",
            split=args.split,
            output_path=output_path,
            weight=args.weight
        )
        evaluator.precision_recall_curve("all")

    # two heaily annoated bacteroidetes
    # evaluator.visualize_predictions_in_genome("AE015928", 0, 0.25)
    # evaluator.visualize_predictions_in_genome("JH724241", 0, 0.25)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate predictions of GECCO against experimental data and PULpy annotations"
    )
    parser.add_argument("--model", type=str, help="Name of model to evaluate", required=True)
    parser.add_argument("--split", type=str, default="test", help="Whether to evaluate on test or train set")
    parser.add_argument("-k", type=int, default=7, help="Number of folds to evaluate")
    parser.add_argument("--weight", type=float, default=0.01, help="Weight for uncertain negative examples.")
    parser.add_argument("--features", type=str, default=None, help="Feature representation used to train model (only applicable to genecat and gecco)")
    args = parser.parse_args()

    main(args)

"""
--GECCO--
python src/scripts/visualization/evaluate_predictions.py --model gecco_pfam --split test -k 7
python src/scripts/visualization/evaluate_predictions.py --model gecco_cazy --split test -k 7


--GENECAT ZEROSHOT--
python src/scripts/visualization/evaluate_predictions.py --model genecat_zeroshot_pfam --split test -k 7
python src/scripts/visualization/evaluate_predictions.py --model genecat_zeroshot_cazy --split test -k 7
python src/scripts/visualization/evaluate_predictions.py --model genecat_zeroshot_cazy_masked --split test -k 7
python src/scripts/visualization/evaluate_predictions.py --model genecat_zeroshot_pfam_masked --split test -k 7


--GENECAT FINETUNED--
python src/scripts/visualization/evaluate_predictions.py --model genecat_finetuned_pfam --split test -k 7
python src/scripts/visualization/evaluate_predictions.py --model genecat_finetuned_cazy --split test -k 7


--ESMC & BACFORMER--
python src/scripts/visualization/evaluate_predictions.py --model esmc --split test -k 7
python src/scripts/visualization/evaluate_predictions.py --model bacformer --split test -k 7


"""
