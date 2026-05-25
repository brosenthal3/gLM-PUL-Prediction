import polars
import matplotlib.pyplot as plt 
from matplotlib_venn import venn3, venn2
import seaborn as sns
from cryptic_puls_plots import join_gene_and_PUL_table
import numpy as np
from visualization_utilities import get_bins


# KDE PUL LENGTHS PLOT #
def get_pul_lengths(puls_table):
    return puls_table.with_columns(abs(polars.col("end") - polars.col("start")).alias("pul_length"))

def plot_length_distributions(ax, experimental_puls, cblaster_results_liberal, cblaster_results_strict, pulpy):
    experimental = get_pul_lengths(experimental_puls)
    cblaster_liberal_puls = get_pul_lengths(cblaster_results_liberal)
    cblaster_strict_puls = get_pul_lengths(cblaster_results_strict)
    pulpy_puls = get_pul_lengths(pulpy)

    # Extract series
    exp_lengths = experimental.select("pul_length").to_series()
    lib_lengths = cblaster_liberal_puls.select("pul_length").to_series()
    strict_lengths = cblaster_strict_puls.select("pul_length").to_series()
    pulpy_lengths = pulpy_puls.select("pul_length").to_series()

    # KDE plots
    sns.kdeplot(exp_lengths, ax=ax, label="Experimental", clip=(0, 100000))
    sns.kdeplot(lib_lengths, ax=ax, label="Liberal Cblaster", clip=(0, 100000))
    sns.kdeplot(strict_lengths, ax=ax, label="Strict Cblaster", clip=(0, 100000))
    sns.kdeplot(pulpy_lengths, ax=ax, label="PULpy", clip=(0, 100000))

    ax.set_title("PUL length distributions (KDE)")
    ax.set_xlabel("PUL length (bp)")
    ax.set_ylabel("Density")
    ax.legend()


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
        textprops={'fontsize': 9}
    )

def plot_taxonomy(ax, all_puls):
    phylum_counts = get_taxonomic_counts(all_puls, rank="phylum", cutoff=10)
    donut_chart(ax, phylum_counts, "Phylum", title="Taxonomic distribution of PULs on a phylum level")


# GENE COUNT PLOT #
def get_n_genes(genes, labeled_table):
    print(labeled_table.with_columns((polars.col("end")-polars.col("start")).alias("PUL_length")).sort(by="PUL_length").select("cluster_id", "sequence_id", "species", "PUL_length"))
    labeled_table = join_gene_and_PUL_table(gene_table=genes, cluster_table=labeled_table, buffer=0)
    labeled_table = labeled_table.group_by("cluster_id").agg(polars.col("is_PUL").sum().alias("n_genes")).sort("n_genes", descending=False).filter(polars.col("cluster_id").is_not_null())
    return labeled_table

def plot_pul_gene_count(ax, genes, labeled_table):
    bins_num = 15
    labeled_table = get_n_genes(genes, labeled_table)
    bins, labels = get_bins(bins_num)

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

    # ensure correct order (important!)
    counts_dict = dict(zip(counts["gene_bin"].to_list(), counts["len"].to_list()))
    y = [counts_dict.get(label, 0) for label in labels]
    x = np.arange(len(labels))

    ax.bar(x, y, edgecolor="black", width=1.0, align="center")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.margins(x=0.02)
    ax.set_xlabel("PUL length in genes")
    ax.set_ylabel("Count")


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
        set_labels=('dcCAN-PUL', 'PULDB', "Cblaster-strict"), 
        ax=ax
    )
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

    get_info_on_data(all_puls, genes)

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1, 1],
        height_ratios=[1, 1]
    )
    # left column
    ax1 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[1, 0])
    # right column spanning both rows
    ax_tax = fig.add_subplot(gs[:, 1])

    plot_venn_diagram_database(ax1, all_puls)
    plot_taxonomy(ax_tax, all_puls)
    plot_pul_gene_count(ax3, genes, all_puls)

    titles = {
        ax1: "PUL origin in dataset",
        ax3: "PUL length distribution",
        ax_tax: "Taxonomic distributions of PULs on a phylum level"
    }
    for ax, title in titles.items():
        ax.set_title(title, loc='left', pad=12)
    fig.align_ylabels([ax1, ax3, ax_tax])

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 300
    })
    fig.tight_layout()
    fig.savefig("results/plots/dataset_construction.svg")


if __name__ == "__main__":
    main()
