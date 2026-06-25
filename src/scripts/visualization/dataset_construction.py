import polars
import matplotlib.pyplot as plt 
from matplotlib_venn import venn3, venn2
import seaborn as sns
from cryptic_puls_plots import join_gene_and_PUL_table
import numpy as np
from visualization_utilities import get_bins, get_pul_lengths
from viz_data import model_colors, Cork_7, Bold_10, Bilbao_5, Buda_4


# KDE PUL LENGTHS PLOT #
# def get_pul_lengths(puls_table):
#     return puls_table.with_columns(abs(polars.col("end") - polars.col("start")).alias("pul_length"))

# def plot_length_distributions(ax, experimental_puls, cblaster_results_liberal, cblaster_results_strict, pulpy):
#     experimental = get_pul_lengths(experimental_puls)
#     cblaster_liberal_puls = get_pul_lengths(cblaster_results_liberal)
#     cblaster_strict_puls = get_pul_lengths(cblaster_results_strict)
#     pulpy_puls = get_pul_lengths(pulpy)

#     # Extract series
#     exp_lengths = experimental.select("pul_length").to_series()
#     lib_lengths = cblaster_liberal_puls.select("pul_length").to_series()
#     strict_lengths = cblaster_strict_puls.select("pul_length").to_series()
#     pulpy_lengths = pulpy_puls.select("pul_length").to_series()

#     # KDE plots
#     sns.kdeplot(exp_lengths, ax=ax, label="Experimental", clip=(0, 100000))
#     sns.kdeplot(lib_lengths, ax=ax, label="Liberal Cblaster", clip=(0, 100000))
#     sns.kdeplot(strict_lengths, ax=ax, label="Strict Cblaster", clip=(0, 100000))
#     sns.kdeplot(pulpy_lengths, ax=ax, label="PULpy", clip=(0, 100000))

#     ax.set_title("PUL length distributions (KDE)")
#     ax.set_xlabel("PUL length (bp)")
#     ax.set_ylabel("Density")
#     ax.legend()


# TAXONOMY PLOT #
def get_taxonomic_counts(table=None, rank="phylum", cutoff=10):
    return (
        table
        .group_by(rank)
        .len()
        .rename({"len": "count"})
        .with_columns(
            polars.when(polars.col("count") < cutoff)
            .then(polars.lit("other"))
            .otherwise(polars.col(rank))
            .alias(f"{rank}_group")
        )
        .group_by(f"{rank}_group")
        .agg(polars.col("count").sum().alias("count"))
        .fill_null("Unknown")
        .sort("count", descending=True)
    )

def donut_chart(ax, counts, rank, title):
    x = counts.select(f"{rank.lower()}_group").to_series()
    heights = counts.select("count").to_series()

    labels = [f"{name}\n(n={count})" if count>20 else name for name, count in zip(x, heights)]

    ax.pie(
        heights,
        labels=labels,
        radius=1,
        wedgeprops=dict(width=0.35, edgecolor='w'),
        textprops={'fontsize': 9},
        colors=Cork_7[:len(counts)]
    )

def plot_taxonomy_phylum(ax, all_puls):
    phylum_counts = get_taxonomic_counts(all_puls, rank="phylum", cutoff=10)
    donut_chart(ax, phylum_counts, "Phylum", title="Taxonomic distribution of PULs on a phylum level")

def plot_taxonomy_class(ax, all_puls):
    class_counts = get_taxonomic_counts(all_puls, rank="class", cutoff=10)
    donut_chart(ax, class_counts, "Class", title="Taxonomic distribution of PULs on a class level")


# GENE COUNT PLOT #
def plot_pul_gene_count(ax, genes, labeled_table):
    bins_num = 15
    labeled_table_with_length = get_pul_lengths(labeled_table, genes)
    bins, labels = get_bins(bins_num)
    print(f"median PUL length: {labeled_table_with_length["pul_length"].median()}")
    print(f"mean PUL length: {labeled_table_with_length["pul_length"].mean()}")
    print(labeled_table_with_length.sort(by="pul_length", descending=True))
    print(labeled_table_with_length.sort(by="pul_length", descending=True).join(labeled_table.select("sequence_id", "species").unique(), how="left", on="sequence_id"))

    binned = labeled_table_with_length.with_columns(
        polars.col("pul_length")
        .cut(breaks=bins.tolist(), include_breaks=False, labels=labels)
        .alias("gene_bin")
    )

    counts = (
        binned.group_by("gene_bin")
        .len()
        .sort("gene_bin")
    )

    # ensure correct order (important!)
    counts_dict = dict(zip(counts["gene_bin"].to_list(), counts["len"].to_list()))
    y = [counts_dict.get(label, 0) for label in labels]
    x = np.arange(len(labels))

    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.bar(x, y, edgecolor="black", width=1.0, align="center", color="#739CBE")
    ax.set_xticks(x)
    labels[0] = "1"
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.margins(x=0.02)
    ax.set_xlabel("PUL length in genes")
    ax.set_ylabel("Number of PULs")


# DATABASE VENN DIAGRAM #
def plot_venn_diagram_database(ax, experimental_puls):
    # filter on database
    dbcan = experimental_puls.filter(polars.col("database").str.contains("dbcan"))
    puldb = experimental_puls.filter(polars.col("database").str.contains("puldb"))
    cblaster = experimental_puls.filter(polars.col("database").str.contains("cblaster"))

    # make sets of sequence ids
    dbcan_sequences = set(dbcan.select("cluster_id").to_series())
    puldb_sequences = set(puldb.select("cluster_id").to_series())
    cblaster_sequences = set(cblaster.select("cluster_id").to_series())
    
    # plot overlap
    v = venn3(
        [dbcan_sequences, puldb_sequences, cblaster_sequences], 
        set_labels=(f'dcCAN-PUL (n={len(dbcan_sequences)})', f'PULDB (n={len(puldb_sequences)})', f'Cblaster-strict (n={len(cblaster_sequences)})'), 
        ax=ax,
        set_colors=(Bold_10[2], Bold_10[0], Bold_10[1]),
        alpha=0.8
    )
    for patch in v.patches:
        patch.set_edgecolor("white")
        patch.set_linewidth(0.5)

    for t in v.set_labels + v.subset_labels:
        if t:
            t.set_fontsize(9)

def get_info_on_data(clusters_table, gene_table):
    labeled_table = join_gene_and_PUL_table(gene_table=gene_table, cluster_table=clusters_table)

    print("Number of contigs: ", clusters_table.select("sequence_id").n_unique())
    print("Number of PULs: ", clusters_table.select("cluster_id").n_unique())
    print("Number of genes: ", labeled_table.select("protein_id").n_unique())
    print("Number of genes in PULs: ", labeled_table.filter("is_PUL").select("protein_id").n_unique())
    print("Percentage: ", labeled_table.filter("is_PUL").select("protein_id").n_unique() / labeled_table.select("protein_id").n_unique() * 100)


def main():
    experimental_puls = polars.read_csv("src/data/data_collection/clusters_deduplicated.tsv", separator="\t")
    all_puls = polars.read_csv("src/data/data_collection/clusters_deduplicated_cblaster.tsv", separator="\t")
    cblaster_results_liberal = polars.read_csv("src/data/data_collection/cblaster_results_liberal.tsv", separator='\t', infer_schema_length=600)
    cblaster_results_strict = (polars.read_csv("src/data/data_collection/cblaster_results_strict.tsv", separator='\t'))
    pulpy = (
        polars.read_csv("src/data/data_collection/pulpy_annotations.tsv", separator="\t")
        .rename({"genome": "sequence_id", "pulid": "cluster_id"})
        .select(cblaster_results_liberal.columns)
    )
    genes = polars.read_parquet("src/data/genecat_output/genome.genes.parquet")
    original_clusters = polars.read_csv("src/data/data_collection/combined_clusters.tsv", separator="\t", infer_schema_length=600)

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    fig2, ax_tax = plt.subplots(figsize=(6, 6))
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    fig4, ax_tax_2 = plt.subplots(figsize=(6, 6))

    plot_venn_diagram_database(ax1, all_puls)
    plot_taxonomy_phylum(ax_tax, all_puls)
    plot_taxonomy_class(ax_tax_2, all_puls)
    plot_pul_gene_count(ax3, genes, all_puls)

    titles = {
        ax1: "Origin of literature-derived PUL annotations",
        ax3: "Length distribution of literature-derived PULs",
        ax_tax: "Phylum-level distributions of literature-derived PULs",
        ax_tax_2: "Class-level distributions of literature-derived PULs"
    }
    for ax, title in titles.items():
        ax.set_title(title, pad=12)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 300
    })
    fig1.tight_layout()
    fig1.savefig("results/plots/dataset_venn.png", dpi=300)
    fig2.tight_layout()
    fig2.savefig("results/plots/dataset_tax.png", dpi=300)
    fig3.tight_layout()
    fig3.savefig("results/plots/dataset_pul_lengths.png", dpi=300)
    fig4.tight_layout()
    fig4.savefig("results/plots/dataset_tax_class.png", dpi=300)

    get_info_on_data(all_puls, genes)

if __name__ == "__main__":
    main()
