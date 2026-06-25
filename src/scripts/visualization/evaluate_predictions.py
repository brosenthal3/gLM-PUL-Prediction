import polars
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score,  roc_curve, auc, matthews_corrcoef
import seaborn as sns
from matplotlib_venn import venn3
from tqdm import tqdm
from visualization_utilities import PredictionEvaluator, get_bins, get_pul_lengths
import altair_upset as au
from viz_data import model_names, model_names_masked, model_colors, model_names_features, Cork_7, Bold_10, Bilbao_5, Buda_4, model_colors_selected, model_names_selected
from matplotlib.patches import Patch

# get list of evaluator class instances from a list of model names
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

# sort handles/labels together by score
def sort_legend_labels(ax, scores, loc="upper right"):
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    sorted_handles = [handles[i] for i in order]
    sorted_labels = [labels[i] for i in order]
    ax.legend(sorted_handles, sorted_labels, loc=loc)


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
        width=800,
        height=400,
        vertical_bar_label_size=10,
        abbre=["experimental", "pulpy", "gecco", "genecat"],
        title="Predicted and experimental PUL genes overlap"
    )
    chart.save(f"results/plots/aggregated/upset_plot_{model_class}.svg")


def barplot_features(all_models, model_names_dict=model_names_features):
    # list of evaluators for all models
    evaluators = get_evaluators(all_models)
    fig_bar, ax_bar = plt.subplots(1, 1, figsize=(4.5, 5))
    ax_bar.set_xlabel("Model")
    ax_bar.set_ylabel("AUPRC (Area Under Pecision-Recall Curve)")
    ax_bar.set_title("AUPRC per model")
    auprc_exp = []

    for i, model_evaluator in enumerate(evaluators):
        current_model_name = model_names_dict.get(all_models[i])
        # aggregate all folds
        model_evaluator.aggregate_all_folds()
        # evaluate predictions
        auprc_e, auprc_cr, auprc_b = model_evaluator.test_cryptic_puls("all")
        # save auprc scores
        auprc_exp.append(auprc_e)

    models = ["GECCO", "GeneCAT 0-Shot", "GeneCAT Fine-tuned"]
    features = ["Pfam features", "Pfam+CAZy features"]
    colors = [Cork_7[0], Cork_7[2]]

    for i in range(len(features)):
        scores = auprc_exp[i::2]  # get scores for current feature set
        x = np.arange(len(models))
        bars = ax_bar.bar(x + (i-0.5)*0.3, scores, width=0.28, label=features[i], color=colors[i], edgecolor="black", alpha=0.9)
        ax_bar.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(models, rotation=45, ha="right")
    ax_bar.set_ylim(0, 0.8)
    ax_bar.legend(loc="upper left")

    fig_bar.tight_layout()
    fig_bar.savefig("results/plots/aggregated/barplot_features.png")


def barplot_masked(all_models, model_names_dict=model_names_features):
    # list of evaluators for all models
    evaluators = get_evaluators(all_models)
    fig_bar, ax_bar = plt.subplots(1, 1, figsize=(8, 4))
    ax_bar.set_xlabel("Model")
    ax_bar.set_ylabel("AUPRC (Area Under Pecision-Recall Curve)")
    ax_bar.set_title("AUPRC per model evaluated on experimental and cryptic PULs")

    # save auprc scores
    auprc_exp = []
    auprc_cryptic = []
    auprc_both = []
    for i, model_evaluator in enumerate(evaluators):
        current_model_name = model_names_dict.get(all_models[i])
        model_evaluator.aggregate_all_folds()
        auprc_e, auprc_cr, auprc_b = model_evaluator.test_cryptic_puls("all")
        auprc_exp.append(auprc_e)
        auprc_cryptic.append(auprc_cr)
        auprc_both.append(auprc_b)

    # plot bar plot of auprc scores
    models = [model_names_dict.get(e.model_name) for e in evaluators]

    n_groups = len(models)
    width = 0.3
    base_x = np.arange(n_groups)
    group_gap = 0.4
    # add extra gap for every pair, so the same model with/without masking remain close to each other
    extra = (base_x // 2) * group_gap
    x = base_x + extra

    bars_exp= ax_bar.bar(x - width, auprc_exp, width, label="Experimental", color=Cork_7[-1])
    bars_cryptic = ax_bar.bar(x, auprc_cryptic, width, label="Cryptic", color=Cork_7[0])
    bars_both = ax_bar.bar(x + width,auprc_both,width,label="Both",color="#808889")

    is_masked_model = (base_x % 2 == 1)
    for bars in (bars_exp, bars_cryptic, bars_both):
        for bar, second in zip(bars, is_masked_model):
            if second:
                bar.set_hatch("///")
                bar.set_edgecolor("black")
                bar.set_linewidth(0.6)
    
    # only one model name per pair
    pair_centers = []
    pair_labels = []
    for i in range(0, n_groups, 2):
        # average the x-position of the two groups in this pair
        center = (x[i] + x[i + 1]) / 2
        label = models[i] 
        pair_centers.append(center)
        pair_labels.append(label)

    ax_bar.set_xticks(pair_centers)
    ax_bar.set_xticklabels(pair_labels)

    ax_bar.bar_label(bars_exp, fmt="%.2f", padding=3, fontsize=8)
    ax_bar.bar_label(bars_cryptic, fmt="%.2f", padding=3, fontsize=8)
    ax_bar.bar_label(bars_both, fmt="%.2f", padding=3, fontsize=8)
    ax_bar.set_ylim(0, 0.8)

    handles, labels = ax_bar.get_legend_handles_labels()
    handles += [Patch(facecolor="white", edgecolor="black", hatch="///", label="Masked in training")]
    ax_bar.legend(handles=handles, labels=labels + ["Masked in training"])
    fig_bar.tight_layout()

    fig_bar.savefig("results/plots/aggregated/barplot_masked.png")


def barplot_pul_length(all_models, model_names_dict=model_names):
    pul_length_distributions = {}
    # get gene count distributions for all models from predicted clusters
    for model_name in all_models:
        clusters_table_path = f"src/data/results/{model_name}/predicted_clusters.parquet"
        clusters_table = polars.read_parquet(clusters_table_path)
        pul_length_distributions[model_name] = clusters_table["gene_count"].to_list()
    
    all_models.append("experimental")
    pul_length_distributions["experimental"] = get_pul_lengths(
        polars.read_csv("src/data/data_collection/clusters_deduplicated_cblaster.tsv", separator="\t", infer_schema_length=700),
        polars.read_parquet("src/data/genecat_output/genome.genes.parquet")
    )["pul_length"].to_list()

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
        ax.bar(x + offset, y, edgecolor="black", width=width, align="center", label=model_names_dict.get(model_name), color=model_colors_selected.get(model_name))

    ax.set_xticks(x)
    labels[0] = "1"
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.margins(x=0.02)
#    ax.set_xlim(-0.5, len(labels)-0.5)    
    ax.set_xlabel("PUL length in genes")
    ax.set_ylabel("Proportion of predicted clusters")
    ax.set_title("PUL length distribution in predicted clusters")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/plots/aggregated/pul_length_barplot.png")

    plt.close()
    all_models.pop(-1)



def compare_all_models(all_models, model_class, model_names_dict=model_names):
    # list of evaluators for all models
    evaluators = get_evaluators(all_models)

    # comparison of all models, 3 separate plots
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig_roc, ax_roc = plt.subplots(1, 1, figsize=(5, 5))
    fig_bac, ax_bac = plt.subplots(1, 2, figsize=(12, 6))
    fig_bar, ax_bar = plt.subplots(1, 1, figsize=(8, 5))
    if model_class == "selected":
        colors = model_colors_selected
    else:
        colors = model_colors

    # add labels and legend
    for j in range(2):
        ax_bac[j].set_xlabel('Recall')
        ax_bac[j].set_ylabel("Precision")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel("True Positive Rate")
    ax_bar.set_xlabel("Model")
    ax_bar.set_ylabel("AUPRC (Area Under Precision-Recall Curve)")

    # add titles and stuff
    ax.set_title("PR-curve of models tested on experimental PULs")
    ax_roc.set_title("ROC-curves of models tested on experimental PULs")
    ax_bac[0].set_title("Trained on Bacteroidota, tested on other phyla")
    ax_bac[1].set_title("Trained on other phyla, tested on Bacteroidota")
    ax_bar.set_title("AUPRC per model evaluated on experimental and cryptic PULs")

    fig_bac.suptitle("Precision-Recall Curves for Bacteroidota generalization test (folds 5 and 6)")

    auprc_exp = []
    auprc_cryptic = []
    auprc_both = []
    aurocs = []
    bac_scores = [[], []]
    for i, model_evaluator in enumerate(evaluators):
        print(f"Plotting for {all_models[i]}")
        current_model_name = model_names_dict.get(all_models[i])
        # before aggregating, plot for folds 5 and 6 separately
        true_5, _, p_pred_5, _ = model_evaluator.get_evaluation_data(model_evaluator.labeled_results[5], mask_cryptic=True)
        bac_score_5 = model_evaluator.plot_pr(true_5, p_pred_5, current_model_name, colors[all_models[i]], ax_bac[0])
        true_6, _, p_pred_6, _ = model_evaluator.get_evaluation_data(model_evaluator.labeled_results[6], mask_cryptic=True)
        bac_score_6 = model_evaluator.plot_pr(true_6, p_pred_6, current_model_name, colors[all_models[i]], ax_bac[1])
        # save scores for these folds
        bac_scores[0].append(bac_score_5)
        bac_scores[1].append(bac_score_6)

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
        model_evaluator.plot_pr(true_masked, p_pred_masked, current_model_name, colors[all_models[i]], ax)
        auroc = model_evaluator.roc_curve(true_masked, p_pred_masked, current_model_name, colors[all_models[i]], ax_roc)
        aurocs.append(auroc)

        # plot baselines only once at the end
        if i == len(all_models)-1:
            model_evaluator.plot_baseline(true_masked, ax)
            model_evaluator.plot_baseline(true_5, ax_bac[0])
            model_evaluator.plot_baseline(true_6, ax_bac[1])
    
    # add legends
    for j in range(2):
        sort_legend_labels(ax_bac[j], bac_scores[j], loc="upper right")

    sort_legend_labels(ax, auprc_exp, loc="upper right")
    sort_legend_labels(ax_roc, aurocs, loc="lower right")

    # plot bar plot of auprc scores
    models = [model_names_dict.get(e.model_name) for e in evaluators]
    x = np.arange(len(models))   # group positions
    width = 0.25
    bars_exp= ax_bar.bar(x - width, auprc_exp, width, label="Experimental", color=Cork_7[-1])
    bars_cryptic = ax_bar.bar(x, auprc_cryptic, width, label="Cryptic", color=Cork_7[0])
    bars_both = ax_bar.bar(x + width,auprc_both,width,label="Both",color="#808889")
    ax_bar.bar_label(bars_exp, fmt="%.2f", padding=3, fontsize=8)
    ax_bar.bar_label(bars_cryptic, fmt="%.2f", padding=3, fontsize=8)
    ax_bar.bar_label(bars_both, fmt="%.2f", padding=3, fontsize=8)
    ax_bar.set_ylim(0, 0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(models, rotation=40, ha="right")
    ax_bar.legend()

    fig.tight_layout()
    fig.savefig(f"results/plots/aggregated/pr_curves_{model_class}.png", dpi=300)
    fig_roc.tight_layout()
    fig_roc.savefig(f"results/plots/aggregated/roc_curves_{model_class}.png", dpi=300)
    fig_bac.tight_layout()
    fig_bac.savefig(f"results/plots/aggregated/pr_curves_bacteroidota_{model_class}.png", dpi=300)
    fig_bar.tight_layout()
    fig_bar.savefig(f"results/plots/aggregated/barplot_{model_class}.png", dpi=300)

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

        evaluator.test_cryptic_puls("all")


def main(args):
    model_name = args.model
    if model_name == "all":
        all_models = ["gecco_pfam", "gecco_cazy", "genecat_zeroshot_pfam_masked", "genecat_zeroshot_cazy_masked", "genecat_finetuned_pfam_masked", "genecat_finetuned_cazy_masked", "esmc", "bacformer"]
        compare_all_models(all_models, model_name)
        return

    if model_name == "masked":
        all_models = ["genecat_zeroshot_cazy", "genecat_zeroshot_cazy_masked", "genecat_finetuned_cazy", "genecat_finetuned_cazy_masked", "esmc", "esmc_masked", "bacformer", "bacformer_masked"]
        barplot_masked(all_models, model_names_masked)
        #compare_all_models(all_models, model_name, model_names_masked)

        return

    if model_name == "features":
        all_models = ["gecco_pfam", "gecco_cazy", "genecat_zeroshot_pfam_masked", "genecat_zeroshot_cazy_masked", "genecat_finetuned_pfam_masked", "genecat_finetuned_cazy_masked"]
        barplot_features(all_models)
        return

    if model_name == "selected":
        all_models = ["gecco_pfam", "genecat_zeroshot_cazy_masked", "genecat_finetuned_cazy_masked", "genecat_untrained", "esmc_masked", "bacformer_masked"]
        #barplot_pul_length(all_models)
        compare_all_models(all_models, model_name, model_names_selected)
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
