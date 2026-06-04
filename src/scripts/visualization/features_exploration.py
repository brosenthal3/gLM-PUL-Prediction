import polars
from matplotlib_venn import venn2
import matplotlib.pyplot as plt
from viz_data import Cork_7, model_names_selected
from visualization_utilities import join_gene_and_PUL_table
import numpy as np

cols = ["protein_id", "domain", "sequence_id"]
features_pfam = polars.read_parquet("src/data/genecat_output/pfam.features.parquet").select(cols).with_columns(polars.lit("pfam").alias("feature"))
features_cazy = polars.read_parquet("src/data/genecat_output/dbcan.features.parquet").select(cols).with_columns(polars.lit("cazy").alias("feature"))
selected_sequences = (
    polars.read_csv("src/data/data_collection/clusters_deduplicated_cblaster.tsv", separator="\t", infer_schema_length=600)
    .select("sequence_id")
    .unique()
)
all_puls = polars.read_csv("src/data/data_collection/clusters_deduplicated_cblaster.tsv", separator="\t")
genes = polars.read_parquet("src/data/genecat_output/genome.genes.parquet")
genes_with_puls = join_gene_and_PUL_table(gene_table=genes, cluster_table=all_puls).filter("is_PUL").select("protein_id").unique()

genes_cazy = set(features_cazy.join(selected_sequences, on="sequence_id", how="semi").select("protein_id").to_series())
genes_pfam = set(features_pfam.join(selected_sequences, on="sequence_id", how="semi").select("protein_id").to_series())
only_cazy = polars.DataFrame({"protein_id":list(genes_cazy - genes_pfam)})



def get_unique_cazy_domains():
    cazy_domains_with_pfam_equivalent = features_cazy.join(polars.DataFrame({"protein_id": list(genes_pfam)}), on="protein_id", how="semi").select("domain")
    cazy_only_domains = features_cazy.join(only_cazy, on="protein_id", how="semi")
    print(
        (
            cazy_only_domains
            .join(cazy_domains_with_pfam_equivalent, how="anti", on="domain")
            # .with_columns(
            #     polars.col("domain").str.split("_").list.first().alias("domain")
            # )
            .group_by("domain")
            .agg(polars.col("domain").count().alias("count"))
            .sort(by="count", descending=True)
        )
    )


susc = ["PF00593", "PF07715"]
susd = ["PF07980", "PF12741", "PF12771", "PF14322"]
suscd = susc+susd
cazy_features_selected = features_cazy["domain"].unique()

def plot_features_venn():
    fig, ax = plt.subplots(figsize=(5, 5))
    # plot overlap
    v = venn2(
        [genes_pfam, genes_cazy], 
        set_labels=(f'Pfam\n(n={len(genes_pfam)})', f'CAZy\n(n={len(genes_cazy)})'),
        set_colors=(Cork_7[0], Cork_7[1]),
        ax=ax,
        alpha=0.9
    )
    for patch in v.patches:
        patch.set_edgecolor("white")
        patch.set_linewidth(0.5)

    for t in v.set_labels + v.subset_labels:
        if t:
            t.set_fontsize(10)
            if t.get_text() == "1304":
                t.set_position((t.get_position()[0]+0.065, t.get_position()[1]))
            if "CAZy" in t.get_text():
                t.set_position((t.get_position()[0]+0.1, t.get_position()[1]))

    ax.set_title("Overlap in proteins with Pfam and/or CAZy annotations", fontsize=12, pad=2)
    fig.tight_layout()
    fig.savefig("results/plots/feature_venn.png")


def get_feature_counts(labeled_results, genes, features, selected_features, groups="cluster_id"):
    predicted_puls_total = labeled_results.height
    # create unique cluster id
    labeled_results = labeled_results.with_columns(
        polars.concat_str([polars.col("sequence_id"), polars.col("start")]).alias("cluster_id")
    )
    selected_features = polars.DataFrame({"domain": selected_features})
    all_pul_genes = join_gene_and_PUL_table(genes, labeled_results).filter("is_PUL")
    # genes with features
    genes_with_features = features.join(selected_features, on="domain", how="semi").join(labeled_results.select("sequence_id"), on="sequence_id", how="semi")
    # find clusters/genes with selected domains
    labeled_results = (
        all_pul_genes
        .join(
            genes_with_features,
            on="protein_id",
            how="left"
        )
        .filter(polars.col("domain").is_not_null())
        .group_by(groups).agg()
    )
    return labeled_results.height / all_pul_genes.group_by(groups).agg().height


def plot_functional_composition():
    all_models = ["gecco_pfam", "genecat_zeroshot_cazy_masked", "genecat_finetuned_cazy_masked", "genecat_untrained", "esmc_masked", "bacformer_masked"]

    all_folds_suscd_values = []
    all_folds_cazy_values = []
    fold_5_cazy_values = []
    fold_5_suscd_values = []
    fold_6_cazy_values = []
    fold_6_suscd_values = []

    for model_name in all_models:
        # get predicted puls for only bacteroidota species
        bacteroidetes = all_puls.filter(polars.col("phylum").eq("Bacteroidota")).select("sequence_id").unique()
        all_folds = polars.read_parquet(f"src/data/results/{model_name}/predicted_clusters.parquet").join(bacteroidetes, on="sequence_id", how="semi")
        fold_6 = polars.read_parquet(f"src/data/results/{model_name}/predicted_clusters_6.parquet").join(bacteroidetes, on="sequence_id", how="semi")
        fold_5 = polars.read_parquet(f"src/data/results/{model_name}/predicted_clusters_5.parquet").join(bacteroidetes, on="sequence_id", how="anti") # in this case only for non-bacteroidetes!

        # get frequency of susC and SusD domains in predicted clusters
        all_folds_suscd_values.append(get_feature_counts(all_folds, genes, features_pfam, suscd))
        fold_6_suscd_values.append(get_feature_counts(fold_6, genes, features_pfam, suscd))
        fold_5_suscd_values.append(get_feature_counts(fold_5, genes, features_pfam, suscd))

        # get proportion of cazy domains in predicted clusters
        all_folds_cazy_values.append(get_feature_counts(all_folds, genes, features_cazy, cazy_features_selected, groups="cluster_id"))
        fold_6_cazy_values.append(get_feature_counts(fold_6, genes, features_cazy, cazy_features_selected, groups="cluster_id"))
        fold_5_cazy_values.append(get_feature_counts(fold_5, genes, features_cazy, cazy_features_selected, groups="cluster_id"))


    # # add one bar for experimental
    experimental_bacteroidetes_puls = all_puls.filter(polars.col("phylum").eq("Bacteroidota"))
    experimental_suscd = get_feature_counts(experimental_bacteroidetes_puls, genes, features_pfam, suscd)
    experimental_cazy = get_feature_counts(experimental_bacteroidetes_puls, genes, features_cazy, cazy_features_selected)

    folds_susc = [all_folds_suscd_values, fold_6_suscd_values, fold_5_suscd_values]
    folds_cazy = [all_folds_cazy_values, fold_6_cazy_values, fold_5_cazy_values]

    # values for testing
#    folds_susc = [[0.4364141765114663, 0.5076287349014622, 0.8129130655821047, 0.9097525473071325, 0.3785290415392915, 0.5492160278745645], [0.0429553264604811, 0.038058466629895205, 0.1327683615819209, 0.12899106002554278, 0.03585817888799355, 0.03257437261093568], [0.466839378238342, 0.4112690889942075, 0.7065843621399177, 0.7906446092413006, 0.29806959613919226, 0.46333454943451297]] 
#    folds_cazy = [[0.12881578015440187, 0.1147035348547774, 0.40081506773873776, 0.351681537405628, 0.09762204465849388, 0.10426185926725928], [0.09456595047692457, 0.4731404958677686, 0.38374178123132097, 0.39692540322580644, 0.05138368405880657, 0.057838222017326496], [0.1079622444843361, 0.11368445802962615, 0.3601140955550273, 0.3452198789169051, 0.09029398925771982, 0.10679391047489208]]
#    experimental_suscd = 0.9018987341772152
#    experimental_cazy = 0.7468354430379747

    fold_labels = ["5-fold cross-validation", "Trained on non-Bacteroidetes"]
    colors = [Cork_7[0], Cork_7[2]]
    x = np.arange(len(all_models))
    baseline_x = len(all_models) + 0.6

    width = 0.3
    n_bars = 2
    fig, (ax1, ax2), = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    # SusC/SusD
    # model predictions
    for i in range(n_bars):
        bars = ax1.bar(
            x + (i - 0.5) * width,
            folds_susc[i],
            width=width,
            label=fold_labels[i],
            color=colors[i],
            edgecolor="black",
            alpha=0.9
        )
        ax1.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    # experimental baseline
    baseline_bar = ax1.bar(
        baseline_x,
        experimental_suscd,
        width=0.45,
        color=Cork_7[-1],
        edgecolor="black",
        label="Experimental baseline"
    )
    ax1.bar_label(baseline_bar, fmt="%.2f", padding=3, fontsize=8)

    ax1.set_xticks(list(x)+[baseline_x])
    ax1.set_xticklabels([model_names_selected[m] for m in all_models] + ["Experimental"], rotation=45, ha="right")
    ax1.set_title("Predicted PULs with SusC/SusD homologs")
    ax1.set_ylabel("Proportion")

    # CAZy
    for i in range(n_bars):
        bars = ax2.bar(
            x + (i - 0.5) * width,
            folds_cazy[i],
            width=width,
#            label=fold_labels[i],
            color=colors[i],
            edgecolor="black",
            alpha=0.9
        )
        ax2.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    # experimental baseline
    baseline_bar = ax2.bar(
        baseline_x,
        experimental_cazy,
        width=0.4,
        color=Cork_7[-1],
        edgecolor="black",
#        label="Experimental baseline"
    )
    ax2.bar_label(baseline_bar, fmt="%.2f", padding=3, fontsize=8)
    ax2.set_xticks(list(x)+[baseline_x])
    ax2.set_xticklabels([model_names_selected[m] for m in all_models] + ["Experimental"], rotation=45, ha="right")
    ax2.set_ylim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.set_title("Predicted PULs with CAZymes")

    fig.suptitle("Functional composition of predicted Bacteroidetes PULs", fontsize=16)
    fig.legend(loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(
        "results/plots/susc_susd_and_cazy_frequency.png",
        dpi=300,
        bbox_inches="tight"
    )


def compute_gecco_weights(weights_df):
    return (
        weights_df
        .join(polars.DataFrame({"attr": suscd}), on="attr", how="semi")
        .filter(polars.col("label") == 1)
        .select("attr", "weight")
        .with_columns(
            polars.col("attr").is_in(susc).alias("susc"),            
            polars.col("attr").is_in(susd).alias("susd")
        )
        .group_by("susc").agg(polars.col("weight").mean().alias("weight"))
        .sort("susc", descending=True)
    )

def plot_gecco_weights():
    # aggregate gecco weights across all folds
    all_folds_weights = []
    for i in range(5):
        weights = (
            polars.read_csv(f"src/data/results/gecco_pfam/model_{i}/model.state.tsv", separator="\t")
            .join(polars.DataFrame({"attr": suscd}), on="attr", how="semi")
            .filter(polars.col("label") == 1)
            .select("attr", "label", "weight")
        )
        all_folds_weights.append(weights)

    all_folds_weights = compute_gecco_weights(
        polars.concat(all_folds_weights, how="align")
        .group_by("attr")
        .agg(polars.col("weight").mean().alias("weight"), polars.lit(1).alias("label"))
    )

    # get gecco weights for folds 5 and 6
    fold_5_weights = compute_gecco_weights(polars.read_csv("src/data/results/gecco_pfam/model_5/model.state.tsv", separator="\t"))
    fold_6_weights = compute_gecco_weights(polars.read_csv("src/data/results/gecco_pfam/model_6/model.state.tsv", separator="\t"))

    # plott
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    folds_susc = [all_folds_weights["weight"].to_list(), fold_5_weights["weight"].to_list(), fold_6_weights["weight"].to_list()]
    fold_labels = ["5-fold cross-validation", "Trained on Bacteroidetes", "Trained on non-Bacteroidetes"]
    feature_labels = ["SusC domains", "SusD domains"]
    colors = [Cork_7[0], Cork_7[4], Cork_7[2]]
    x = np.arange(len(feature_labels))
    width = 0.25

    for i in range(3):
        bars = ax.bar(
            x + (i - 0.5) * width,
            folds_susc[i],
            width=width,
            label=fold_labels[i],
            color=colors[i],
            edgecolor="black",
            alpha=0.9
        )
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, rotation=0, ha="center")
    ax.set_title("GECCO feature weights of SusC/SusD domains")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 2.25)
    fig.tight_layout()
    fig.savefig("results/plots/gecco_feature_weights.png", dpi=300)

if __name__ == "__main__":
    # plot_features_venn()
    # plot_functional_composition()
    # plot_gecco_weights()

    get_unique_cazy_domains()