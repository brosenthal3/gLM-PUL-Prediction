import polars
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score,  roc_curve, auc, matthews_corrcoef
import seaborn as sns
from matplotlib_venn import venn3
from tqdm import tqdm
from visualization_utilities import PredictionEvaluator

def compare_all_models(all_models, model_class):
    # list of evaluators for all models
    evaluators = [
        PredictionEvaluator(
            labeled_results_path = f"src/data/results/{model_name}/labeled_results_test",
            model_name=model_name,
            k=7,
            output_path=f"results/plots/{model_name}"
        )
        for model_name in all_models
    ]

    # comparison of all models
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig_roc, ax_roc = plt.subplots(1, 2, figsize=(12, 6))
    fig_bac, ax_bac = plt.subplots(1, 2, figsize=(12, 6))
    colors = plt.cm.tab20.colors

    for i, model_evaluator in enumerate(evaluators):
        print(f"Plotting for {all_models[i]}")
        # before aggregating, plot for folds 5 and 6 separately
        model_evaluator.set_evaluation_data(5)
        model_evaluator.plot_pr(model_evaluator.true, model_evaluator.p_pred, all_models[i], colors[i], ax_bac[0])
        model_evaluator.set_evaluation_data(6)
        model_evaluator.plot_pr(model_evaluator.true, model_evaluator.p_pred, all_models[i], colors[i], ax_bac[1])

        # for masked models, also evaluate cryptic puls
        print("testing on cryptic puls")
        cryptic_df = polars.read_csv("src/data/data_collection/cryptic_puls_genes.tsv", separator="\t").unique()
        model_evaluator.aggregate_all_folds()
        model_evaluator.test_cryptic_puls(cryptic_df, "all")
        model_evaluator.set_evaluation_data(0)        

        # for true vs pred
        model_evaluator.plot_pr(model_evaluator.true, model_evaluator.p_pred, all_models[i], colors[i], ax[0])
        model_evaluator.roc_curve(model_evaluator.true, model_evaluator.p_pred, all_models[i], colors[i], ax_roc[0])
        # for pulpy vs pred
        model_evaluator.plot_pr(model_evaluator.pulpy_pred, model_evaluator.p_pred, all_models[i], colors[i], ax[1])
        model_evaluator.roc_curve(model_evaluator.pulpy_pred, model_evaluator.p_pred, all_models[i], colors[i], ax_roc[1])

        # plot baselines only once at the end
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

        ax_bac[j].set_xlabel('Recall')
        ax_bac[j].set_ylabel("Precision")
        ax_bac[j].legend(loc="upper right")


    ax[0].set_title("Models tested on experimental annotations")
    ax[1].set_title("Models tested on PULpy annotations")
    ax_roc[0].set_title("Models tested on experimental annotations")
    ax_roc[1].set_title("Models tested on PULpy annotations")
    ax_bac[0].set_title("Trained on Bacteroidota, tested on other phyla")
    ax_bac[1].set_title("Trained on other phyla, tested on Bacteroidota")

    fig.suptitle("Precision-Recall Curves of all tested models (all folds)")
    fig_roc.suptitle("ROC Curves of all tested models (all folds)")
    fig_bac.suptitle("Precision-Recall Curves for Bacteroidota generalization test (folds 5 and 6)")

    fig.tight_layout()
    fig.savefig(f"results/plots/aggregated/pr_curves_{model_class}.png")
    fig_roc.tight_layout()
    fig_roc.savefig(f"results/plots/aggregated/roc_curves_{model_class}.png")
    fig_bac.tight_layout()
    fig_bac.savefig(f"results/plots/aggregated/pr_curves_bacteroidota_{model_class}.png")

    plt.close()


def evaluate_model(args, model_name):
    # path where results are saved
    results_path = f"src/data/results/{model_name}/labeled_results_{args.split}"
    if not os.path.exists(results_path+"_0.tsv"):
        raise ValueError("Invalid model name, or no results found.")

    # output path to save plots
    output_path = f"results/plots/{model_name}"

    evaluator = PredictionEvaluator(
        f"{results_path}",
        k=args.k,
        model_name=f"{model_name}",
        split=args.split,
        output_path=output_path,
        weight=args.weight
    )

    evaluator.f1_per_fold()

    for fold in range(args.k):
        evaluator.precision_recall_curve(fold)
        evaluator.plot_roc_curves(fold)
        evaluator.test_cryptic_puls(fold)

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
        evaluator.test_cryptic_puls("all")


def main(args):
    model_name = args.model
    if model_name == "all":
        all_models = ["gecco_pfam", "gecco_cazy", "genecat_zeroshot_pfam_masked", "genecat_zeroshot_cazy_masked", "genecat_finetuned_pfam", "genecat_finetuned_cazy", "esmc", "bacformer"]
        compare_all_models(all_models, model_name)
        return

    if model_name == 'logistic_regression':
        all_models = ["genecat_zeroshot_pfam", "genecat_zeroshot_pfam_masked", "genecat_zeroshot_cazy", "genecat_zeroshot_cazy_masked", "esmc", "esmc_masked", "bacformer", "bacformer_masked"]
        compare_all_models(all_models, model_name)
        return

    if model_name == "masked":
        all_models = ["genecat_zeroshot_pfam_masked", "genecat_zeroshot_cazy_masked", "esmc_masked", "bacformer_masked"]
        compare_all_models(all_models, model_name)
        return

    if model_name == "selected":
        all_models = ["gecco_pfam", "genecat_zeroshot_cazy_masked", "genecat_finetuned_cazy", "esmc_masked", "bacformer_masked"]
        compare_all_models(all_models, model_name)
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
