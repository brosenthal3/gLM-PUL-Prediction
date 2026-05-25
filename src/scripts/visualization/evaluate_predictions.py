import polars
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score,  roc_curve, auc, matthews_corrcoef
import seaborn as sns
from matplotlib_venn import venn3
from tqdm import tqdm
from visualization_utilities import PredictionEvaluator, get_bins
import altair_upset as au
from viz_data import model_names, model_names_masked, model_colors, model_names_features

def get_evaluators(all_models, k=7, aggregate=False):
    return [
        PredictionEvaluator(
            labeled_results_path = f"src/data/results/{model_name}/labeled_results_test",
            model_name=model_name,
            k=k,
            output_path=f"results/plots/{model_name}",
            aggregate=aggregate
        )
        for model_name in all_models
    ]


def generate_upset_plot(all_models, model_class):
    evaluators = get_evaluators(all_models, k=5, aggregate=True)
    gene_table = evaluators[0].labeled_results[0].select("protein_id", "is_PUL", "is_PUL_pulpy").rename({"is_PUL": "experimental", "is_PUL_pulpy": "pulpy"})
    for model_evaluator in evaluators:
        if not "gecco" in model_evaluator.model_name:
            model_evaluator.recompute_predictions(0)

        gene_table = gene_table.join(
            model_evaluator.labeled_results[0].select("protein_id", "is_PUL_pred").rename({"is_PUL_pred": model_evaluator.model_name}),
            on="protein_id",
            how="left"
        ).fill_null(False)
    gene_table = gene_table.drop("protein_id").cast(polars.Int8)

    # Create UpSet plot
    chart = au.UpSetAltair(
        data=gene_table.to_pandas(),
        sets=gene_table.columns,
#        sort_by="frequency",
#        sort_order="descending",
        width=800,
        height=400,
        vertical_bar_label_size=10,
        abbre=["experimental", "pulpy", "gecco", "genecat"],
#        title="Predicted and experimental PUL genes overlap"
    )
    chart.save(f"results/plots/aggregated/upset_plot_{model_class}.svg")


def barplot_features(all_models, model_names_dict=model_names_features):
    # list of evaluators for all models
    evaluators = get_evaluators(all_models)
    fig_bar, ax_bar = plt.subplots(1, 1, figsize=(6, 6))
    ax_bar.set_xlabel("Model")
    ax_bar.set_ylabel("AUPRC (Area Under Pecision-Recall Curve)")
    ax_bar.set_title("AUPRC per model")
    auprc_exp = []

    for i, model_evaluator in enumerate(evaluators):
        current_model_name = model_names_dict.get(all_models[i])
        # aggregate all folds
        model_evaluator.aggregate_all_folds()
        # evaluate predictions on cryptic puls
        auprc_e, auprc_cr, auprc_b = model_evaluator.test_cryptic_puls("all")
        # save auprc scores
        auprc_exp.append(auprc_e)

    # plot bar plot of auprc scores
    models = [model_names_dict.get(e.model_name) for e in evaluators]
    colors = [model_colors.get(e.model_name) for e in evaluators]
    n = len(models)
    auprc = auprc_exp
    # group size = 2 models per group
    group_size = 2
    gap = 0.8  # visual spacing between groups
    x = []
    group_centers = []
    pos = 0
    for i in range(0, n, group_size):
        group = list(range(i, min(i + group_size, n)))
        group_pos = [pos + j for j in range(len(group))]

        x.extend(group_pos)
        group_centers.append(np.mean(group_pos))

        pos += len(group) + gap

    x = np.array(x)
    bars = ax_bar.bar(x, auprc, color=colors)
    ax_bar.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax_bar.set_ylim(0, 0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(models, rotation=45, ha="right")

    fig_bar.tight_layout()
    fig_bar.savefig("results/plots/aggregated/barplot_features.png")


def barplot_pul_length(all_models, model_names_dict=model_names):
    pul_length_distributions = {}
    # get gene count distributions for all models from predicted clusters
    for model_name in all_models:
        clusters_table_path = f"src/data/results/{model_name}/predicted_clusters.parquet"
        clusters_table = polars.read_parquet(clusters_table_path)
        pul_length_distributions[model_name] = clusters_table["gene_count"].to_list()

    # plot distributions as barplot
    fig, ax = plt.subplots(figsize=(8, 6))
    # get bins and labels for barplot
    bins, labels = get_bins(10, start=0, stop=15)
    n_models = len(all_models)
    width = 0.12
    for model_name, lengths in pul_length_distributions.items():
        labeled_table = polars.DataFrame({"n_genes": lengths})
        binned = labeled_table.with_columns(
            polars.col("n_genes")
            .cut(breaks=bins.tolist(), include_breaks=False, labels=labels)
            .alias("gene_bin")
        )
        counts = (
            binned.group_by("gene_bin")
            .len()
            .sort("gene_bin")
        )
        # plot bar for current model
        counts_dict = dict(zip(counts["gene_bin"].to_list(), counts["len"].to_list()))
        y = [counts_dict.get(label, 0)/len(lengths) for label in labels]
        x = np.arange(len(labels))
        idx = all_models.index(model_name)
        # center grouped bars: compute offset so bars for each model are centered around each xtick
        offset = (idx - (n_models - 1) / 2) * width
        ax.bar(x + offset, y, edgecolor="black", width=width, align="center", label=model_names_dict.get(model_name), color=model_colors.get(model_name))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.margins(x=0.02)
#    ax.set_xlim(-0.5, len(labels)-0.5)    
    ax.set_xlabel("PUL length in genes")
    ax.set_ylabel("Proportion of predicted clusters")
    ax.set_title("PUL length distribution in predicted clusters")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/plots/aggregated/pul_length_barplot.png")



def compare_all_models(all_models, model_class, model_names_dict=model_names):
    # list of evaluators for all models
    evaluators = get_evaluators(all_models)

    # comparison of all models, 3 separate plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig_roc, ax_roc = plt.subplots(1, 2, figsize=(12, 6))
    fig_bac, ax_bac = plt.subplots(1, 2, figsize=(12, 6))
    fig_bar, ax_bar = plt.subplots(1, 1, figsize=(8, 8))
    colors = model_colors

    # add labels and legend
    for j in range(2):
        ax[j].set_xlabel("Recall")
        ax[j].set_ylabel("Precision")

        ax_roc[j].set_xlabel('False Positive Rate')
        ax_roc[j].set_ylabel("True Positive Rate")

        ax_bac[j].set_xlabel('Recall')
        ax_bac[j].set_ylabel("Precision")

    ax_bar.set_xlabel("Model")
    ax_bar.set_ylabel("AUPRC (Area Under Pecision-Recall Curve)")

    # add titles and stuff
    ax[0].set_title("Models tested on experimental annotations")
    ax[1].set_title("Models tested on PULpy annotations")
    ax_roc[0].set_title("Models tested on experimental annotations")
    ax_roc[1].set_title("Models tested on PULpy annotations")
    ax_bac[0].set_title("Trained on Bacteroidota, tested on other phyla")
    ax_bac[1].set_title("Trained on other phyla, tested on Bacteroidota")
    ax_bar.set_title("Overall model comparison (all folds)")

    fig.suptitle("Precision-Recall Curves of all tested models (all folds)")
    fig_roc.suptitle("ROC Curves of all tested models (all folds)")
    fig_bac.suptitle("Precision-Recall Curves for Bacteroidota generalization test (folds 5 and 6)")

    auprc_exp = []
    auprc_cryptic = []
    auprc_both = []
    for i, model_evaluator in enumerate(evaluators):
        print(f"Plotting for {all_models[i]}")
        current_model_name = model_names_dict.get(all_models[i])
        # before aggregating, plot for folds 5 and 6 separately
        true_5, _, p_pred_5, _ = model_evaluator.get_evaluation_data(model_evaluator.labeled_results[5], mask_cryptic=True)
        model_evaluator.plot_pr(true_5, p_pred_5, current_model_name, colors[all_models[i]], ax_bac[0])
        true_6, _, p_pred_6, _ = model_evaluator.get_evaluation_data(model_evaluator.labeled_results[6], mask_cryptic=True)
        model_evaluator.plot_pr(true_6, p_pred_6, current_model_name, colors[all_models[i]], ax_bac[1])

        # aggregate all folds
        model_evaluator.aggregate_all_folds()

        # evaluate predictions on cryptic puls
        auprc_e, auprc_cr, auprc_b = model_evaluator.test_cryptic_puls("all")
        # save auprc scores
        auprc_exp.append(auprc_e)
        auprc_cryptic.append(auprc_cr)
        auprc_both.append(auprc_b)
        
        # for true vs pred
        true_masked, _, p_pred_masked, _ = model_evaluator.get_evaluation_data(model_evaluator.labeled_results[0], mask_cryptic=True)
        model_evaluator.plot_pr(true_masked, p_pred_masked, current_model_name, colors[all_models[i]], ax[0])
        model_evaluator.roc_curve(true_masked, p_pred_masked, current_model_name, colors[all_models[i]], ax_roc[0])

        # for pulpy vs pred
        _, _, p_pred, pulpy_pred = model_evaluator.get_evaluation_data(model_evaluator.labeled_results[0], mask_cryptic=False) # don't mask cryptic, since we only consider PULpy
        model_evaluator.plot_pr(pulpy_pred, p_pred, current_model_name, colors[all_models[i]], ax[1])
        model_evaluator.roc_curve(pulpy_pred, p_pred, current_model_name, colors[all_models[i]], ax_roc[1])

        # plot baselines only once at the end
        if i == len(all_models)-1:
            model_evaluator.plot_baseline(true_masked, ax[0])
            model_evaluator.plot_baseline(pulpy_pred, ax[1])
            model_evaluator.plot_baseline(true_5, ax_bac[0])
            model_evaluator.plot_baseline(true_6, ax_bac[1])


        # add legends
        for j in range(2):
            ax[j].legend(loc="upper right")
            ax_roc[j].legend(loc="lower right")
            ax_bac[j].legend(loc="upper right")

    # plot bar plot of auprc scores
    models = [model_names_dict.get(e.model_name) for e in evaluators]
    x = np.arange(len(models))   # group positions
    width = 0.25
    bars_exp= ax_bar.bar(x - width, auprc_exp, width, label="Experimental", color="steelblue")
    bars_cryptic = ax_bar.bar(x, auprc_cryptic, width, label="Cryptic", color="orange")
    bars_both= ax_bar.bar(x + width,auprc_both,width,label="Both",color="green")
    ax_bar.bar_label(bars_exp, fmt="%.2f", padding=3, fontsize=8)
    ax_bar.bar_label(bars_cryptic, fmt="%.2f", padding=3, fontsize=8)
    ax_bar.bar_label(bars_both, fmt="%.2f", padding=3, fontsize=8)
    ax_bar.set_ylim(0, 0.8)
    
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(models, rotation=45, ha="right")
    ax_bar.set_ylabel("AUPRC")
    ax_bar.set_title("AUPRC per model")
    ax_bar.legend()

    fig.tight_layout()
    fig.savefig(f"results/plots/aggregated/pr_curves_{model_class}.png")
    fig_roc.tight_layout()
    fig_roc.savefig(f"results/plots/aggregated/roc_curves_{model_class}.png")
    fig_bac.tight_layout()
    fig_bac.savefig(f"results/plots/aggregated/pr_curves_bacteroidota_{model_class}.png")
    fig_bar.tight_layout()
    fig_bar.savefig(f"results/plots/aggregated/barplot_{model_class}.png")

    plt.close()


def evaluate_model(args, model_name):
    # path where results are saved
    results_path = f"src/data/results/{model_name}/labeled_results_{args.split}"
    if not os.path.exists(results_path+"_0.tsv"):
        raise ValueError("Invalid model name, or no results found.")
    # output path to save plots
    output_path = f"results/plots/{model_name}"

    # evaluator class for aggregating 5 folds instead of 7
    if args.k >= 5:
        evaluator = PredictionEvaluator(
            f"{results_path}",
            k=5,
            model_name=f"{model_name}",
            split=args.split,
            output_path=output_path,
            weight=args.weight
        )
        for k in range(5):
            evaluator.test_cryptic_puls(k)

        evaluator.test_cryptic_puls("all")


def main(args):
    model_name = args.model
    if model_name == "all":
        all_models = ["gecco_pfam", "gecco_cazy", "genecat_zeroshot_pfam_masked", "genecat_zeroshot_cazy_masked", "genecat_finetuned_pfam_masked", "genecat_finetuned_cazy_masked", "esmc", "bacformer"]
        compare_all_models(all_models, model_name)
        return

    if model_name == "masked":
        all_models = ["genecat_zeroshot_cazy", "genecat_zeroshot_cazy_masked", "genecat_finetuned_cazy", "genecat_finetuned_cazy_masked", "esmc", "esmc_masked", "bacformer", "bacformer_masked"]
        compare_all_models(all_models, model_name, model_names_masked)
        return

    if model_name == "features":
        all_models = ["gecco_pfam", "gecco_cazy", "genecat_zeroshot_pfam_masked", "genecat_zeroshot_cazy_masked", "genecat_finetuned_pfam_masked", "genecat_finetuned_cazy_masked"]
        barplot_features(all_models)
        return

    if model_name == "selected":
        all_models = ["gecco_pfam", "genecat_zeroshot_cazy_masked", "genecat_finetuned_cazy_masked", "genecat_untrained", "esmc_masked", "bacformer_masked"]
        barplot_pul_length(all_models)
        #compare_all_models(all_models, model_name)
        return

    if model_name == "upset":
        all_models = ["gecco_pfam", "genecat_zeroshot_cazy_masked"]
        generate_upset_plot(all_models, model_name)
        return

    else:
        evaluate_model(args, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate predictions against experimental data and PULpy/cblaster annotations"
    )
    parser.add_argument("--model", type=str, help="Name of model to evaluate", required=True)
    parser.add_argument("--split", type=str, default="test", help="Whether to evaluate on test or train set")
    parser.add_argument("-k", type=int, default=7, help="Number of folds to evaluate")
    parser.add_argument("--weight", type=float, default=0.01, help="Weight for uncertain negative examples.")
    args = parser.parse_args()

    main(args)

"""
--GECCO--
python src/scripts/visualization/evaluate_predictions.py --model gecco_pfam --split test -k 7
python src/scripts/visualization/evaluate_predictions.py --model gecco_cazy --split test -k 7


--GENECAT ZEROSHOT--
python src/scripts/visualization/evaluate_predictions.py --model genecat_zeroshot_pfam --split test -k 7
python src/scripts/visualization/evaluate_predictions.py --model genecat_zeroshot_cazy --split test -k 7
python src/scripts/visualization/evaluate_predictions.py --model genecat_zeroshot_pfam_masked --split test -k 7
python src/scripts/visualization/evaluate_predictions.py --model genecat_zeroshot_cazy_masked --split test -k 7


--GENECAT FINETUNED--
python src/scripts/visualization/evaluate_predictions.py --model genecat_finetuned_pfam --split test -k 7
python src/scripts/visualization/evaluate_predictions.py --model genecat_finetuned_cazy --split test -k 7


--ESMC & BACFORMER--
python src/scripts/visualization/evaluate_predictions.py --model esmc --split test -k 7
python src/scripts/visualization/evaluate_predictions.py --model bacformer --split test -k 7
"""
