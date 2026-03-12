import polars
import matplotlib.pyplot as plt

def plot_percentage_in_puls_over_genome_length(clusters_table_filtered, save="src/data/plots/temp.png", blast=False):
    figure, axs = plt.subplots(1, figsize=(8, 6))
    
    axs.scatter(x=clusters_table_filtered.select('length'), y=clusters_table_filtered.select('percentage_in_puls'))
    if blast:
        axs.scatter(x=clusters_table_filtered.filter(polars.col("blast_status") == True).select('length'), y=clusters_table_filtered.filter(polars.col("blast_status") == True).select('percentage_in_puls'), color="orange", label="Blasted sequences")

    axs.set_xscale("log")
    axs.set_xlim(1e4, 1e7)
    axs.set_title("Percentage of genome in PULs over genome length")
    axs.set_xlabel("Genome Length (bp)")
    axs.set_ylabel("Percentage of genome in PULs")

    # vertical line at 50.000kb
    plt.vlines(100000, ymin=0, ymax=100, color="red", linestyle="dashed", label="100kb")
    plt.legend()
    plt.savefig(save, dpi=300)


def get_taxonomic_counts(clusters_table_filtered, rank="phylum", cutoff=10, save="src/data/plots/temp.png"):
    return (
        clusters_table_filtered
        .group_by(rank)
        .len()  # or .count() on older Polars
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

def plot_taxonomic_distributions(clusters_table_filtered, save="src/data/plots/temp.png"):
    phylum_counts = get_taxonomic_counts(clusters_table_filtered, rank="phylum", cutoff=5)
    class_counts = get_taxonomic_counts(clusters_table_filtered, rank="class", cutoff=5)
    order_counts = get_taxonomic_counts(clusters_table_filtered, rank="order", cutoff=5)
    family_counts = get_taxonomic_counts(clusters_table_filtered, rank="family", cutoff=5)

    figure, axs = plt.subplots(2, 2, figsize=(16, 12))
    for ax, counts, rank in zip(axs.flatten(), [phylum_counts, class_counts, order_counts, family_counts], ["Phylum", "Class", "Order", "Family"]):
        x = counts.select(f"{rank.lower()}_group").to_series()
        heights = counts.select("count").to_series()
        ax.pie(heights, labels=x, radius=1, wedgeprops=dict(width=0.3, edgecolor='w'))
        ax.set_title(f"Taxonomic distribution on {rank} level")

    plt.tight_layout()
    plt.savefig(save, dpi=300)
    
    # DRAFT OF NESTED PIE CHARTS FOR PHYLUM AND CLASS, BUT SORTING FUCKS IT UP A BIT
    # fig, ax = plt.subplots()
    # size = 0.3
    # ax.pie(phylum_counts.select("count").to_series(), radius=1,
    #     wedgeprops=dict(width=size, edgecolor='w'),
    #     labels=phylum_counts.select(f"phylum_group").to_series())

    # ax.pie(class_counts.select("count").to_series(), radius=1-size,
    #     wedgeprops=dict(width=size, edgecolor='w'),)
    # plt.tight_layout()
    # plt.savefig("src/data/plots/temp.png")


if __name__ == "__main__":
    clusters_table_filtered = polars.read_csv("src/data/results/combined_clusters_blasted_gtdb_filtered.tsv", separator='\t')
    clusters_table = polars.read_csv("src/data/results/combined_clusters.tsv", separator='\t', infer_schema_length=600).filter((polars.col("merged") == "merged") | polars.col("merged").is_null())

    plot_taxonomic_distributions(clusters_table_filtered, save="src/data/plots/taxonomy.png")
    plot_percentage_in_puls_over_genome_length(clusters_table_filtered, save="src/data/plots/scatter_post_blast.png", blast=True)
    plot_percentage_in_puls_over_genome_length(clusters_table, save="src/data/plots/scatter_pre_blast.png")