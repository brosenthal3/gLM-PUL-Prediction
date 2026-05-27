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
cazy_only_domains = features_cazy.join(only_cazy, on="protein_id", how="semi")

only_cazy_puls = only_cazy.join(genes_with_puls, on="protein_id", how="inner").select("protein_id").n_unique()
print(f"Number of proteins with only CAZy annotations that are in PULs: {only_cazy_puls}")


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


cutoffs = {
    "gecco_pfam": 0.3,
    "genecat_zeroshot_cazy_masked": 0.237,
    "genecat_finetuned_cazy_masked": 0.5,
    "esmc_masked": 0.158,
    "bacformer_masked": 0.24,
    "genecat_untrained": 0.55,
}

all_models = ["gecco_pfam", "genecat_zeroshot_cazy_masked", "genecat_finetuned_cazy_masked", "genecat_untrained", "esmc_masked", "bacformer_masked"]
suscd = ["PF00593", "PF07715", "PF07980", "PF12741", "PF12771", "PF14322"]

def get_feature_counts(labeled_results, features_pfam, selected_features, cutoff):
    labeled_results = labeled_results.filter(polars.col("average_p").ge(cutoff))
    predicted_pul_genes = labeled_results.height
    labeled_results = (
        labeled_results
        .join(
            features_pfam.filter(polars.col("domain").is_in(selected_features)),
            on="protein_id",
            how="inner"
        )
        .group_by("protein_id").agg()
    )
    return labeled_results.height / predicted_pul_genes

all_folds_suscd_values = []
fold_6_suscd_values = []
all_folds_cazy_values = []
fold_6_cazy_values = []

for model_name in all_models:
    # get predicted probabilities of genes for all folds, only for bacteroidota species
    bacteroidetes = all_puls.filter(polars.col("phylum").eq("Bacteroidota")).select("sequence_id").unique()
    all_folds = (
        polars.concat([polars.read_csv(f"src/data/results/{model_name}/labeled_results_test_{k}.tsv", separator="\t", infer_schema_length=10000) for k in range(5)])
        .join(bacteroidetes, on="sequence_id", how="semi")
    )
    # fold fold trained on non-bac and tested on bac
    fold_6 = polars.read_csv(f"src/data/results/{model_name}/labeled_results_test_6.tsv", separator="\t").join(bacteroidetes, on="sequence_id", how="semi")

    # get frequency of susC and SusD domains in predicted clusters
    cutoff = cutoffs[model_name]
    all_folds_suscd_values.append(get_feature_counts(all_folds, features_pfam, suscd, cutoff))
    fold_6_suscd_values.append(get_feature_counts(fold_6, features_pfam, suscd, cutoff))

    # get proportion of cazy domains in predicted clusters
    cazy_features_selected = features_cazy["domain"].unique()
    all_folds_cazy_values.append(get_feature_counts(all_folds, features_cazy, cazy_features_selected, cutoff))
    fold_6_cazy_values.append(get_feature_counts(fold_6, features_cazy, cazy_features_selected, cutoff))


folds_susc = [all_folds_suscd_values, fold_6_suscd_values]
folds_cazy = [all_folds_cazy_values, fold_6_cazy_values]
fold_labels = ["5-fold cross-validation", "Trained on non-Bacteroidetes"]
colors = [Cork_7[0], Cork_7[2]]
x = np.arange(len(all_models))
width = 0.32
fig = plt.figure(figsize=(12, 6), constrained_layout=True)
subfig_left, subfig_right = fig.subfigures(1, 2, wspace=0.05)

# SusC/SusD
ax1 = subfig_left.subplots()

for i in range(2):
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

ax1.set_xticks(x)
ax1.set_xticklabels([model_names_selected[m] for m in all_models], rotation=45, ha="right")
ax1.set_ylim(0, 0.6)
ax1.set_title("Proportion of SusC/SusD in predicted PUL genes")
ax1.set_ylabel("Proportion")

# CAZy
ax2 = subfig_right.subplots()

for i in range(2):
    bars = ax2.bar(
        x + (i - 0.5) * width,
        folds_cazy[i],
        width=width,
        label=fold_labels[i],
        color=colors[i],
        edgecolor="black",
        alpha=0.9
    )
    ax2.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

ax2.set_xticks(x)
ax2.set_xticklabels([model_names_selected[m] for m in all_models], rotation=45, ha="right")
ax2.set_ylim(0, 0.6)
ax2.set_title("Proportion of CAZymes in predicted PUL genes")
ax2.legend(loc="upper right")

fig.suptitle("Functional composition of predicted PUL genes for Bacteroidetes", fontsize=14)

fig.savefig(
    "results/plots/susc_susd_and_cazy_frequency.png",
    dpi=300,
    bbox_inches="tight"
)
