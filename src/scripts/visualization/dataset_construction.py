import polars
import matplotlib.pyplot as plt 
from matplotlib_venn import venn3, venn2
import seaborn as sns
from cryptic_puls_plots import join_gene_and_PUL_table
import numpy as np


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

def donut_chart(ax, counts, rank, title="Taxonomic distribution"):
    x = counts.select(f"{rank.lower()}_group").to_series()
    heights = counts.select("count").to_series()
    ax.pie(heights, labels=x, radius=1, wedgeprops=dict(width=0.3, edgecolor='w'))
    ax.set_title(f"{title}")

def plot_taxonomy(ax, all_puls):
    phylum_counts = get_taxonomic_counts(all_puls, rank="phylum", cutoff=15)
    class_counts = get_taxonomic_counts(all_puls, rank="class", cutoff=15)

    # # draw two donut charts inside a single parent axis
    # ax.axis("off")
    # ax_left = ax.inset_axes([0.00, 0.0, 0.48, 1.0])
    # ax_right = ax.inset_axes([0.52, 0.0, 0.48, 1.0])

    donut_chart(ax, phylum_counts, "Phylum", title="Phylum distribution")
#    donut_chart(ax_right, class_counts, "Class", title="Class distribution")


# GENE COUNT PLOT #
def get_bins(labeled_table):
    bins = np.unique(
        np.logspace(
            start=1,
            stop=np.log2(80),
            base=2,
            num=20,
        ).astype(int)
    )
    labels = [f'<{bins[0]}'] + [f'{bins[i]}-{bins[i + 1]}' for i in range(len(bins[:-1]))] + [f'>{bins[-1]}']
    return bins, labels

def get_n_genes(genes, labeled_table):
    labeled_table = join_gene_and_PUL_table(gene_table=genes, cluster_table=labeled_table)
    labeled_table = labeled_table.group_by("cluster_id").agg(polars.col("is_PUL").sum().alias("n_genes")).sort("n_genes", descending=False).filter(polars.col("cluster_id").is_not_null())
    return labeled_table

def plot_pul_gene_count(ax, genes, labeled_table):
    # get bins 
    labeled_table = get_n_genes(genes, labeled_table)
#    bins, labels = get_bins(labeled_table)
#    binned_labeled_table = labeled_table.with_columns(polars.col('n_genes').cut(breaks=bins.tolist(), include_breaks=False, labels=labels).alias('gene_bin'))

    # plot histogram
    ax.hist(labeled_table.select("n_genes").to_series(), bins=20, edgecolor="black")
    ax.set_xlabel("Number of genes in PUL")
    ax.set_ylabel("Count")
    ax.set_title("Gene counts in experimental annotations")
    ax.tick_params(axis='x', rotation=45)


# DATABASE VENN DIAGRAM #
def plot_venn_diagram_database(ax, experimental_puls):
    # filter on database
    dbcan = experimental_puls.filter(polars.col("database").str.contains("dbcan"))
    puldb = experimental_puls.filter(polars.col("database").str.contains("puldb"))

    # make sets of sequence ids
    dbcan_sequences = set(dbcan.select("sequence_id").to_series())
    puldb_sequences = set(puldb.select("sequence_id").to_series())
    
    # plot overlap
    venn2([dbcan_sequences, puldb_sequences], set_labels = ('DBCAN', 'PULDB'), ax=ax)
    ax.set_title("Genomes annotated by each database")



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

    fig, axes = plt.subplots(
        2, 2, 
        figsize=(12, 8), 
        width_ratios=[2, 3], 
        height_ratios=[1, 1]
    )
    ax1, ax2, ax3, ax4 = axes.flatten()
    plot_venn_diagram_database(ax1, original_clusters)
    plot_taxonomy(ax2, all_puls)
    plot_pul_gene_count(ax3, genes, experimental_puls)
    plot_length_distributions(ax4, experimental_puls, cblaster_results_liberal, cblaster_results_strict, pulpy)
    fig.tight_layout()
    fig.savefig("results/plots/dataset_construction.png", dpi=300)


if __name__ == "__main__":
    main()
