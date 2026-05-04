import polars
import matplotlib.pyplot as plt 
from matplotlib_venn import venn3, venn2
import seaborn as sns

def reset_start_end(table: polars.DataFrame) -> polars.DataFrame:
    return table.with_columns(
        polars.when(polars.col("start") < polars.col("end")).then(polars.col("start")).otherwise(polars.col("end")).alias("start"),
        polars.when(polars.col("start") < polars.col("end")).then(polars.col("end")).otherwise(polars.col("start")).alias("end"),
    )

def join_gene_and_PUL_table(gene_table: polars.DataFrame, cluster_table: polars.DataFrame, buffer: int = 100,) -> polars.DataFrame:
    gene_table = reset_start_end(gene_table)
    cluster_table = reset_start_end(cluster_table)

    labled_gene_table = (
        cluster_table
        .rename({"start": "pul_start", "end": "pul_end"}) # avoid column name conflicts
        .join(
            gene_table,
            on="sequence_id",
            how="inner",
            validate="m:m",
        )
        .with_columns(
            polars.when(
                polars.col("start") >= polars.col("pul_start") - buffer, # allow for some buffer around the PUL boundaries
                polars.col("end") <= polars.col("pul_end") + buffer,
            )
            .then(polars.col("cluster_id"))
            .otherwise(None)
            .alias("cluster_id"),
            polars.when(
                polars.col("start") >= polars.col("pul_start") - buffer,
                polars.col("end") <= polars.col("pul_end") + buffer,
            )
            .then(True)
            .otherwise(False)
            .cast(polars.Boolean)
            .alias("is_PUL")
        )
        # aggregate by protein_id to determine if protein is in any PUL
        .group_by("protein_id")
        .agg(
            polars.col("is_PUL").any().alias("is_PUL"),
            polars.col("sequence_id").first().alias("sequence_id"),
            polars.col("start").first().alias("start"),
            polars.col("end").first().alias("end"),
            polars.col("cluster_id").drop_nulls().first().alias("cluster_id")
        )
        .sort(by=["sequence_id", "start", "end"])
        .with_row_index(name="gene_id", offset=0)  # important
        .select(["sequence_id", "protein_id", "start", "end", "is_PUL", "cluster_id"])
    )

    return labled_gene_table


cblaster_results_liberal = polars.read_csv("src/data/data_collection/cblaster_results_liberal.tsv", separator='\t', infer_schema_length=600)
cblaster_results_strict = (
    polars.read_csv("src/data/data_collection/cblaster_results_strict.tsv", separator='\t')
)
pulpy = (
    polars.read_csv("src/data/data_collection/pulpy_annotations.tsv", separator="\t")
    .rename({"genome": "sequence_id", "pulid": "cluster_id"})
    .select(cblaster_results_liberal.columns)
)
experimental_puls = polars.read_csv("src/data/data_collection/clusters_deduplicated.tsv", separator="\t")
genes = polars.read_parquet("src/data/genecat_output/genome.genes.parquet")


def get_protein_ids_in_clusters(cluster_table):
    return (
        join_gene_and_PUL_table(genes, cluster_table)
        .filter(polars.col("is_PUL") == True)
        .join(experimental_puls.select('sequence_id'), on="sequence_id", how="semi")
        .select("protein_id")
        .to_series()
        .to_list()
    )

# venn diagrams
def plot_venn_diagram_cblaster(save="results/plots/venn_cblaster.png"):
    experimental_genes = set(get_protein_ids_in_clusters(experimental_puls))
    cblaster_liberal_genes = set(get_protein_ids_in_clusters(cblaster_results_liberal))
    cblaster_strict_genes = set(get_protein_ids_in_clusters(cblaster_results_strict))
    pulpy_genes = set(get_protein_ids_in_clusters(pulpy))

    # --- Intersections with experimental ---
    pulpy_exp = pulpy_genes & experimental_genes
    liberal_exp = cblaster_liberal_genes & experimental_genes
    strict_exp = cblaster_strict_genes & experimental_genes

    # Create figure with two rows
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

    # --- Top: all genes ---
    venn3(
        [pulpy_genes, cblaster_liberal_genes, cblaster_strict_genes],
        set_labels=('PULpy', "Liberal Cblaster", 'Strict Cblaster'),
        ax=ax1
    )
    ax1.set_title("All identified genes")

    # --- Bottom: only genes also in experimental ---
    venn2(
        [(cblaster_liberal_genes | cblaster_strict_genes | pulpy_genes), experimental_genes],
        set_labels=('PULpy+Cblaster', 'Experimental'),
        ax=ax2
    )
    ax2.set_title("Genes overlapping with experimental annotations")

    plt.tight_layout()
    plt.savefig(save, dpi=300)
    plt.close()


def get_pul_lengths(puls_table):
    return puls_table.with_columns(abs(polars.col("end") - polars.col("start")).alias("pul_length"))


def plot_length_distributions():
    experimental = get_pul_lengths(experimental_puls)
    cblaster_liberal_puls = get_pul_lengths(cblaster_results_liberal)
    cblaster_strict_puls = get_pul_lengths(cblaster_results_strict)
    pulpy_puls = get_pul_lengths(pulpy)

    # Extract series once (cleaner + reusable)
    exp_lengths = experimental.select("pul_length").to_series()
    lib_lengths = cblaster_liberal_puls.select("pul_length").to_series()
    strict_lengths = cblaster_strict_puls.select("pul_length").to_series()
    pulpy_lengths = pulpy_puls.select("pul_length").to_series()

    # Create figure with two rows
    fig, (ax1, ax2) = plt.subplots(
        2, 1, 
        figsize=(6, 6), 
        sharex=False,
        gridspec_kw={"height_ratios": [1, 1]}
    )

    # Boxplot
    ax1.boxplot(
        [exp_lengths, lib_lengths, strict_lengths, pulpy_lengths],
        tick_labels=["Experimental", "Liberal Cblaster", "Strict Cblaster", "PULpy"]
    )
    ax1.set_ylabel("PUL length (bp)")
    ax1.set_title("PUL lengths of PULpy, Cblaster and Experimental annotations")

    # KDE plots
    sns.kdeplot(exp_lengths, ax=ax2, label="Experimental", clip=(0, 100000))
    sns.kdeplot(lib_lengths, ax=ax2, label="Liberal Cblaster", clip=(0, 100000))
    sns.kdeplot(strict_lengths, ax=ax2, label="Strict Cblaster", clip=(0, 100000))
    sns.kdeplot(pulpy_lengths, ax=ax2, label="PULpy", clip=(0, 100000))

    ax2.set_xlabel("PUL length (bp)")
    ax2.set_ylabel("Density")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("results/plots/pulpy_pul_lengths.png", dpi=300)

plot_venn_diagram_cblaster()