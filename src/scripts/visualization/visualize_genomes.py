from evaluate_predictions import PredictionEvaluator
import matplotlib.pyplot as plt
import polars
import numpy as np
import os
import gb_io
import argparse
from dna_features_viewer import GraphicFeature, GraphicRecord

def visualize_predictions_in_genome(evaluators, sequence_ids, threshold=None, model_name=None):
    # perform some steps on all evaluators
    for model_evaluator in evaluators:
        model_evaluator.aggregate_all_folds()
        model_evaluator.set_evaluation_data(0)
        # compute optimal threshold TODO: ensure this is done on train/val set and not test set...
        if threshold == None:
            mcc_thresholds = np.linspace(0, 0.5, num=10)
            _, threshold, _ = model_evaluator.calculate_mmc(model_evaluator.true, model_evaluator.p_pred, mcc_thresholds)

        # get predictions for given threshold
        model_evaluator.recompute_predictions(0, threshold)

    # ensure sequence_ids is list
    if not isinstance(sequence_ids, list):
        sequence_ids = [sequence_ids]
    # visualize for each sequence id passed
    for sequence_id in sequence_ids:
        phylum = evaluators[0].clusters_table.filter(polars.col("sequence_id") == sequence_id).select("phylum").to_series().to_list()[0]
        species = evaluators[0].clusters_table.filter(polars.col("sequence_id") == sequence_id).select("species").to_series().to_list()[0]

        # create fig
        colors_tab10 = plt.cm.tab10.colors
        fig, axs = plt.subplots(figsize=(10, len(evaluators)+2))
        for i, model_evaluator in enumerate(evaluators):
            # get all genes of this sequence
            genes = model_evaluator.labeled_results[0].filter(polars.col("sequence_id") == sequence_id)
            sequence_length = model_evaluator.clusters_table.filter(polars.col("sequence_id") == sequence_id).select("length").to_series().to_list()[0]

            features = []
            # top ax: called genes
            for row in genes.iter_rows(named=True):
                # add experimental and pulpy only at the beginning
                if i == 0:
                    if row["is_PUL"]:
                        features.append((row["start"], row["end"], 0, "Experimental"))
                    if row["is_PUL_pulpy"]:
                        features.append((row["start"], row["end"], 1, "PULpy"))
                # add predictions
                if row["is_PUL_pred"]:
                    features.append((row["start"], row["end"], i+2, "Predicted"))

            colors = {
                "Experimental": "black",
                "PULpy": "grey",
                "Predicted": colors_tab10[i],
            }
            for start, end, y, label in features:
                axs.fill_betweenx([y, y + 0.9], start, end, color=colors[label], alpha=1)

        axs.set_ylim(0, len(evaluators)+2)
        axs.set_yticks(
            [0.25+i for i in range(len(evaluators)+2)],
            ["Experimental", "PULpy"] + [f"{evaluator.model_name} (Threshold: {threshold})" for evaluator in evaluators]
        )
        plt.suptitle(f"PUL predictions across models for {sequence_id} (species: {species}, phylum: {phylum})")
        plt.tight_layout()
        if model_name == None:
            model_name = "all_models"
        os.makedirs(f"results/plots/aggregated/predictions_in_genome_{model_name}/", exist_ok=True)
        plt.savefig(f"results/plots/aggregated/predictions_in_genome_{model_name}/{sequence_id}.png")
        plt.close()


def visualize_genes(model_evaluator, sequence_id, start=0, end=10000):
    model_evaluator.aggregate_all_folds()
    model_evaluator.set_evaluation_data(0)
    model_evaluator.recompute_predictions(0, threshold=0.15)

    # genes from genbank file
    gb_file = f"src/data/genomes/genbank_genomes/{sequence_id}.gb"
    # write annotations df with cols: protein_id, start, end, product, note
    features = []
    for record in gb_io.iter(gb_file):
        length = min(record.length, end-start)
        for feature in filter(lambda feat: feat.kind == "CDS" and int(feat.location.start) >= start and int(feat.location.end) <= end, record.features):
            qualifiers = {q.key:q.value for q in feature.qualifiers}
            location = feature.location
            if isinstance(feature.location, gb_io.Complement):
                strand = -1
            else:
                strand = +1

            features.append(
                GraphicFeature(
                    start=int(location.start)-start,
                    end=int(location.end)-start,
                    strand=strand,
                    label=qualifiers.get("product", "") + " " + qualifiers.get("note", "")
                )
            )

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 5), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1]}
    )

    # PLOT THE RECORD MAP
    graphic_record = GraphicRecord(sequence_length=length, features=features)
    graphic_record.plot(ax=ax1, with_ruler=False, strand_in_label_threshold=7)

    # # PLOT THE PUL annotations
    genes = (
        model_evaluator.labeled_results[0]
        .filter(
            polars.col("sequence_id") == sequence_id,
            polars.col("start") >= start,
            polars.col("end") <= end,
        )
    )

    predicted = []
    labels = []
    colors_tab10 = plt.cm.tab10.colors
    # top ax: called genes
    for row in genes.iter_rows(named=True):
        if row["is_PUL"]:
            labels.append((row["start"]-start, row["end"]-start, "Experimental"))
        # add predictions
        if row["is_PUL_pred"]:
            predicted.append((row["start"]-start, row["end"]-start, "Predicted", row["average_p"]))

    for start, end, label, p in predicted:
        ax2.fill_betweenx([0, 1], start, end, color=colors_tab10[0], alpha=min(0.1+p, 1))

    for start, end, label in labels:
        ax3.fill_betweenx([0, 1], start, end, color=colors_tab10[1], alpha=1)
    ax2.set_ylim(0, 1)
    ax3.set_ylim(0, 1)
    ax2.set_yticks([0.45], ["Predicted"])
    ax3.set_yticks([0.45], ["Experimental"])
    fig.suptitle(f"Gene map and PUL predictions for {sequence_id}")
    fig.tight_layout()
    fig.savefig(f"results/plots/temp_{sequence_id}.png")
    plt.close()
    return


def compare_all_models(all_models, model_class):
    # list of evaluators for all models
    evaluators = [
        PredictionEvaluator(
            labeled_results_path = f"src/data/results/{model_name}/labeled_results_test",
            model_name=model_name,
            k=5,
            output_path=f"results/plots/{model_name}"
        )
        for model_name in all_models
    ]

    # visualize predictions in genome for all models on some species
    genome_ids = ["AE015928", "AP006841", "JH724241", "NZ_AP022379"]
    visualize_predictions_in_genome(evaluators, genome_ids, 0.15)
    return


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
        model_evaluator = PredictionEvaluator(
            labeled_results_path = f"src/data/results/{model_name}/labeled_results_test",
            model_name=model_name,
            k=5,
            output_path=f"results/plots/{model_name}"
        )
        visualize_genes(model_evaluator, "ABJL02000008", start=1216000, end=1229882)
        visualize_genes(model_evaluator, "RHLG01000001", start=108000, end=116900)
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize predictions in genomes"
    )
    parser.add_argument("--model", type=str, help="Name of model to evaluate", required=True)
    args = parser.parse_args()
    main(args)


"""
python src/scripts/visualization/visualize_genomes.py --model gecco_pfam
"""

    # model_evaluator.labeled_results[0].sort(by=["sequence_id", "average_p"], descending=[False, True])
    # print(model_evaluator.labeled_results[0].select(
    #     "protein_id",
    #     "sequence_id",
    #     "start",
    #     "end",
    #     "is_PUL",
    #     "is_PUL_pulpy",
    #     "average_p"
    # ))
