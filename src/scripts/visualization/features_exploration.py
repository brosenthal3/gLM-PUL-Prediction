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

all_models = ["gecco_pfam", "genecat_zeroshot_cazy_masked", "genecat_finetuned_cazy_masked", "genecat_untrained", "esmc_masked", "bacformer_masked"]
suscd = ["PF00593", "PF07715", "PF07980", "PF12741", "PF12771", "PF14322"]

def get_feature_counts(labeled_results, genes, features, selected_features):
    predicted_puls_total = labeled_results.height
    # create unique cluster id
    labeled_results = labeled_results.with_columns(
        polars.concat_str([polars.col("sequence_id"), polars.col("start")]).alias("cluster_id")
    )
    selected_features = polars.DataFrame({"domain": selected_features})
    # find clusters with selected domains
    labeled_results = (
        join_gene_and_PUL_table(genes, labeled_results).filter("is_PUL")
        .join(
            features.join(selected_features, on="domain", how="semi"),
            on="protein_id",
            how="inner"
        )
        .group_by("cluster_id").agg()
    )
    return labeled_results.height / predicted_puls_total

all_folds_suscd_values = []
fold_6_suscd_values = []
all_folds_cazy_values = []
fold_6_cazy_values = []

for model_name in all_models:
    # get predicted puls for only bacteroidota species
    bacteroidetes = all_puls.filter(polars.col("phylum").eq("Bacteroidota")).select("sequence_id").unique()
    all_folds = polars.read_parquet(f"src/data/results/{model_name}/predicted_clusters.parquet").join(bacteroidetes, on="sequence_id", how="semi")
    fold_6 = polars.read_parquet(f"src/data/results/{model_name}/predicted_clusters_6.parquet").join(bacteroidetes, on="sequence_id", how="semi")

    # get frequency of susC and SusD domains in predicted clusters
    all_folds_suscd_values.append(get_feature_counts(all_folds, genes, features_pfam, suscd))
    fold_6_suscd_values.append(get_feature_counts(fold_6, genes, features_pfam, suscd))

    # get proportion of cazy domains in predicted clusters
    cazy_features_selected = features_cazy["domain"].unique()
    all_folds_cazy_values.append(get_feature_counts(all_folds, genes, features_cazy, cazy_features_selected))
    fold_6_cazy_values.append(get_feature_counts(fold_6, genes, features_cazy, cazy_features_selected))

# add one bar for experimental
experimental_bacteroidetes_puls = all_puls.filter(polars.col("phylum").eq("Bacteroidota"))
experimental_suscd = get_feature_counts(experimental_bacteroidetes_puls, genes, features_pfam, suscd)
experimental_cazy = get_feature_counts(experimental_bacteroidetes_puls, genes, features_cazy, cazy_features_selected)

folds_susc = [all_folds_suscd_values, fold_6_suscd_values]
folds_cazy = [all_folds_cazy_values, fold_6_cazy_values]
fold_labels = ["5-fold cross-validation", "Trained on non-Bacteroidetes"]
colors = [Cork_7[0], Cork_7[2]]
x = np.arange(len(all_models))
baseline_x = len(all_models) + 1

width = 0.32
fig = plt.figure(figsize=(12, 6), constrained_layout=True)
subfig_left, subfig_right = fig.subfigures(1, 2, wspace=0.05)
ax1 = subfig_left.subplots()
ax2 = subfig_right.subplots()

# SusC/SusD
# model predictions
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
ax1.set_ylim(0, 1)
ax1.set_title("Predicted PULs with SusC/SusD homologs")
ax1.set_ylabel("Proportion")

# CAZy
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

# experimental baseline
baseline_bar = ax2.bar(
    baseline_x,
    experimental_cazy,
    width=0.45,
    color=Cork_7[-1],
    edgecolor="black",
    label="Experimental baseline"
)
ax2.bar_label(baseline_bar, fmt="%.2f", padding=3, fontsize=8)


ax2.set_xticks(list(x)+[baseline_x])
ax2.set_xticklabels([model_names_selected[m] for m in all_models] + ["Experimental"], rotation=45, ha="right")
ax2.set_ylim(0, 1)
ax2.set_title("Predicted PULs with CAZymes")

ax2.legend(loc="upper right")

fig.suptitle("Functional composition of predicted PUL genes for Bacteroidetes", fontsize=14)
fig.savefig(
    "results/plots/susc_susd_and_cazy_frequency.png",
    dpi=300,
    bbox_inches="tight"
)
