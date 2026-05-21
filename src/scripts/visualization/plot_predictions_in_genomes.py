from visualization_utilities import PredictionEvaluator
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


def visualize_genes(model_evaluators, sequence_id, start=0, end=10000, model_class="predictions"):
    cryptic_puls = polars.read_csv("src/data/data_collection/cryptic_puls_genes.tsv", separator='\t')
    # aggregate model predictions across folds
    for model_evaluator in model_evaluators:
        model_evaluator.aggregate_all_folds()
    
    # genes from genbank file
    gb_file = f"src/data/genomes/genbank_genomes/{sequence_id}.gb"
    # write annotations df with cols: protein_id, start, end, product, note
    features = []
    for record in gb_io.iter(gb_file):
        record_definition = record.definition
        length = min(record.length, end-start)
        for feature in filter(lambda feat: feat.kind == "CDS" and int(feat.location.start) >= start and int(feat.location.end) <= end, record.features):
            location = feature.location
            qualifiers = {q.key:q.value for q in feature.qualifiers}            
            strand_direction = -1 if type(location) is gb_io.Complement else 1

            product = qualifiers.get("product", None)
            note = qualifiers.get("note", None)
            if product == "hypothetical_protein" and note is not None:
                product = None

            label = f"{product if product else ""} {note if note else ""}"
            feature_start = int(location.start)-start if strand_direction == 1 else int(location.end)-start
            feature_end = int(location.end)-start if strand_direction == 1 else int(location.start)-start
            graphic_feature = GraphicFeature(
                    start=feature_start,
                    end=feature_end,
                    strand=strand_direction,
                    label=label
                )
            features.append(graphic_feature)

    fig, axes = plt.subplots(
        len(model_evaluators)+3, 1, figsize=(16, 6), sharex=True, gridspec_kw={"height_ratios": [8]+[1]*(len(model_evaluators)+2)}
    )

    # PLOT THE RECORD MAP
    graphic_record = GraphicRecord(sequence_length=length, features=features)
    graphic_record.plot(ax=axes[0], with_ruler=False, strand_in_label_threshold=4, max_label_length=55)

    # PLOT THE PUL annotations
    genes_dfs = []
    for model_evaluator in model_evaluators:
        genes = (
            model_evaluator.labeled_results[0]
            .filter(
                polars.col("sequence_id") == sequence_id,
                polars.col("start") >= start,
                polars.col("end") <= end,
            )
            .join(cryptic_puls.with_columns(polars.lit(True).alias("is_cryptic")), on="protein_id", how="left")
        )
        genes_dfs.append(genes)

    labels = []
    cryptic = []
    colors_tab10 = plt.cm.tab10.colors
    # top ax: called genes
    for row in genes_dfs[0].iter_rows(named=True):
        if row["is_PUL"]:
            labels.append((row["start"]-start, row["end"]-start, "Experimental"))
        # add cryptic puls
        if row["is_cryptic"]:
            cryptic.append((row["start"]-start, row["end"]-start, "Cryptic PULs"))

    for start_gene, end_gene, label in labels:
        axes[-2].fill_betweenx([0, 1], start_gene, end_gene, color=colors_tab10[1], alpha=1)
    for start_gene, end_gene, label in cryptic:
        axes[-1].fill_betweenx([0, 1], start_gene, end_gene, color=colors_tab10[2])


    # add prediction probabilities for every model
    for i, genes in enumerate(genes_dfs):
        predicted = []
        for row in genes.iter_rows(named=True):
            predicted.append((row["start"]-start, row["end"]-start, "Predicted", row["average_p"]))

        for start_gene, end_gene, label, p in predicted:
            axes[i+1].fill_betweenx([0, 1], start_gene, end_gene, color=colors_tab10[0], alpha=min(0.1+p, 1))

    for i, ax in enumerate(axes):
        if i == 0:
            continue
        if i > 0 and i <= len(model_evaluators):
            ax.set_yticks([0.45], [f"{model_evaluators[i-1].model_name}"])

        ax.set_ylim(0, 1)

    axes[-2].set_yticks([0.45], ["Experimental PULs"])
    axes[-1].set_yticks([0.45], ["Cryptic PULs"])

    tick_skips = round((end-start)*0.1)
    axes[-1].set_xticks(ticks=range(0, end-start, tick_skips), labels=range(start, end, tick_skips))
    axes[-1].set_xlabel("Location in genome (bp)")
    fig.suptitle(f"{record_definition} ({sequence_id})")
    fig.tight_layout()
    fig.savefig(f"results/plots/high_confidence_predictions/{model_class}_{sequence_id}_{start}_{end}.png")
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

    # vulgatus
#    visualize_genes(evaluators, "CP000139", start=2317000, end=2331000, model_class=model_class)
    # fragilis
#    visualize_genes(evaluators, "CR626927", start=354000, end=363000, model_class=model_class)
    # clostridium 1 
#    visualize_genes(evaluators, "NZ_CP010086", start=2883500, end=2896500, model_class=model_class)
    # clostridium 2 
    visualize_genes(evaluators, "NZ_CP010086", start=4787882, end=4803326, model_class=model_class)  
    visualize_genes(evaluators, "NZ_CP010086", start=6310368, end=6318315, model_class=model_class)  # experimental PUL
    # clostridium 3 
#    visualize_genes(evaluators, "NZ_CP010086", start=5728000, end=5734500, model_class=model_class)
    # dorei
#    visualize_genes(evaluators, "NZ_CP046176", start=3858000, end=3879000, model_class=model_class)
    # rosubria
#    visualize_genes(evaluators, "NZ_LR027880", start=4277000, end=4297500, model_class=model_class)

    return


def main(args):
    model_name = args.model
    if model_name == "all":
        all_models = ["gecco_pfam", "gecco_cazy", "genecat_zeroshot_pfam_masked", "genecat_zeroshot_cazy_masked", "genecat_finetuned_pfam", "genecat_finetuned_cazy", "esmc", "bacformer"]
        compare_all_models(all_models, model_name)
        return

    if model_name == "selected":
        all_models = ["gecco_pfam", "genecat_finetuned_cazy_masked", "bacformer_masked"]
        compare_all_models(all_models, model_name)
        return


    else:
        model_evaluator = PredictionEvaluator(
            labeled_results_path = f"src/data/results/{model_name}/labeled_results_test",
            model_name=model_name,
            k=5,
            output_path=f"results/plots/{model_name}"
        )
        model_evaluator = [model_evaluator]
        # vulgatus
        visualize_genes(model_evaluator, "CP000139", start=2317000, end=2331000)
        # fragilis
        visualize_genes(model_evaluator, "CR626927", start=354000, end=363000)
        # clostridium 1 
        visualize_genes(model_evaluator, "NZ_CP010086", start=2883500, end=2896500)
        # clostridium 2 
        visualize_genes(model_evaluator, "NZ_CP010086", start=4787882, end=4803326)
        # clostridium 3 
        visualize_genes(model_evaluator, "NZ_CP010086", start=5728000, end=5734500)
        # dorei
        visualize_genes(model_evaluator, "NZ_CP046176", start=3858000, end=3879000)
        # rosubria
        visualize_genes(model_evaluator, "NZ_LR027880", start=4277000, end=4297500)

        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize predictions in genomes"
    )
    parser.add_argument("--model", type=str, help="Name of model to evaluate", required=True)
    args = parser.parse_args()
    main(args)


"""
python src/scripts/visualization/plot_predictions_in_genomes.py --model gecco_pfam
"""