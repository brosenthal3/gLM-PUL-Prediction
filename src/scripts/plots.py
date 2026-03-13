import polars
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

def plot_percentage_in_puls_over_genome_length(clusters_table_filtered, replaced_sequences, save="src/data/plots/temp.png", blast=False):
    figure, axs = plt.subplots(1, figsize=(8, 6))
    
    axs.scatter(x=clusters_table_filtered.select('length'), y=clusters_table_filtered.select('percentage_in_puls'))
    if blast:
        color = "orange"
    else:
        color = "red"
    axs.scatter(x=replaced_sequences.select('length'), y=replaced_sequences.select('percentage_in_puls'), color=color, label="Replaced by BLAST hit")

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

def plot_venn_diagram_database(save="src/data/plots/temp.png"):
    plt.figure(figsize=(5, 5))

    dbcan = polars.read_csv("src/data/results/dbcan_clusters.tsv", separator='\t', infer_schema_length=600)
    puldb = polars.read_csv("src/data/results/puldb_clusters.tsv", separator='\t', infer_schema_length=600)
    dbcan_sequences = set(dbcan.select("sequence_id").to_series())
    puldb_sequences = set(puldb.select("sequence_id").to_series())
    venn2([dbcan_sequences, puldb_sequences], set_labels = ('DBCAN', 'PULDB'))
    plt.title("Overlap between PULDB and DBCAN sequences")
    plt.savefig(save, dpi=300)

def plot_venn_diagram_blast(save="src/data/plots/temp.png"):
    clusters_table = polars.read_csv("src/data/results/combined_clusters_blasted_gtdb.tsv", separator='\t', infer_schema_length=600)
    old_sequences = set(clusters_table.select("sequence_id").unique().drop_nulls().to_series())
    new_sequences = set(clusters_table.select("new_sequence_id").unique().drop_nulls().to_series())
    print(len(new_sequences))

    venn2([old_sequences, new_sequences], set_labels = ('Original', 'BLASTed'))
    plt.title("Overlap between original and BLASTed sequences")
    plt.savefig(save, dpi=300)


def plot_venn_diagram_blast_filtered(save="src/data/plots/temp.png"):
    clusters_table = polars.read_csv("src/data/results/combined_clusters_blasted_gtdb.tsv", separator='\t', infer_schema_length=600)
    from_blast = set(clusters_table.filter(polars.col("blast_status") == True).select("new_sequence_id").to_series())
    original = set(clusters_table.filter(polars.col("blast_status") == False).select("sequence_id").to_series())

    venn2([original, from_blast], set_labels = ('Original', 'BLASTed'))
    plt.title("Overlap between original and sequences replaced by BLAST hits")
    plt.savefig(save, dpi=300)



if __name__ == "__main__":
    # clusters_table_filtered = polars.read_csv("src/data/results/combined_clusters_blasted_gtdb_filtered.tsv", separator='\t')
    # clusters_table = polars.read_csv("src/data/results/combined_clusters.tsv", separator='\t', infer_schema_length=600).filter((polars.col("merged") == "merged") | polars.col("merged").is_null())
    # replaced_PULs = clusters_table_filtered.filter(polars.col("blast_status") == True)
    # replaced_PULs_original = clusters_table.join(replaced_PULs, on="cluster_id", how="semi")

    # plot_taxonomic_distributions(clusters_table_filtered, save="src/data/plots/taxonomy.png")
    # plot_percentage_in_puls_over_genome_length(clusters_table_filtered, replaced_PULs, save="src/data/plots/scatter_post_blast.png", blast=True)
    # plot_percentage_in_puls_over_genome_length(clusters_table,replaced_PULs_original, save="src/data/plots/scatter_pre_blast.png")
    plot_venn_diagram_blast()