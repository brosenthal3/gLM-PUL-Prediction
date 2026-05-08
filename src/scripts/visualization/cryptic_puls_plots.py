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


def merge_overlapping_puls(df, group_col='sequence_id', start_col='start', end_col='end', blast=False, keep_original=True):
    merged_puls = polars.DataFrame(schema=df.schema)
    merged_ids = []

    for sequence_id, group in df.group_by(group_col):
        if group.shape[0] == 1 or sequence_id[0] is None:
            merged_puls = merged_puls.vstack(polars.DataFrame(group))
            continue

        # sort by start position
        group = group.sort(start_col)
        current_pul = None
        for row in group.iter_rows(named=True):
            if current_pul is None:
                current_pul = row
            else:
                # check if there is an overlap with the current PUL
                if row[start_col] <= current_pul[end_col]:
                    # merge the PULs by updating the end position to the maximum end position
                    current_pul[end_col] = max(current_pul[end_col], row[end_col])
                    # merge cluster_id by concatenating with an underscore
                    current_pul['cluster_id'] = f"{current_pul['cluster_id']}_{row['cluster_id']}"
                    # merge database column by concatenating with an underscore if different
                    current_pul['database'] = f"{current_pul['database']}_{row['database']}" if current_pul['database'] not in row['database'] else current_pul['database']
                    if blast:
                        current_pul['blast_status'] = current_pul['blast_status'] or row['blast_status']
                        merged_ids.append({'cluster_id': current_pul['cluster_id'], 'merged': "merged_blast"})                        
                    else:
                        merged_ids.append({'cluster_id': current_pul['cluster_id'], 'merged': "merged"})                        
                else:
                    merged_puls = merged_puls.vstack(polars.DataFrame([current_pul]))
                    current_pul = row

        # add the last PUL after processing all rows for this sequence_id
        if current_pul is not None:
            merged_puls = merged_puls.vstack(polars.DataFrame([current_pul]))

    # add column for which puls are merged
    merged_puls = (
        merged_puls
        .join(polars.DataFrame(merged_ids), on="cluster_id", how="left", suffix="_new")
    )
    # handle when merged col already exists
    merged_puls = (
        merged_puls.with_columns(polars.coalesce(polars.col("merged_new"), polars.col("merged")).alias("merged"))
        .drop("merged_new") 
        if "merged_new" in merged_puls.columns 
        else merged_puls
    )

    return merged_puls.sort('cluster_id').sort('merged')


def plot_pul_overlap(all_puls, pulpy, cblaster_results_liberal):
    all_puls = all_puls.select(["sequence_id", "start", "end", "cluster_id"]).with_columns(
        polars.lit("experimental").alias("database")
    )
    pulpy_puls = pulpy.select(["sequence_id", "start", "end", "cluster_id"]).with_columns(
        polars.lit("pulpy").alias("database")
    ).join(
        all_puls.select("sequence_id").unique(), on="sequence_id", how="semi"
    ) # only keep cblaster puls in sequences where we have experimental annotations, for fair comparison

    liberal_cblaster_puls = cblaster_results_liberal.select(["sequence_id", "start", "end", "cluster_id"]).with_columns(
        polars.lit("cblaster").alias("database")
    ).join(
        all_puls.select("sequence_id").unique(), on="sequence_id", how="semi"
    ) # only keep cblaster puls in sequences where we have experimental annotations, for fair comparison

    all_puls_combined = polars.concat([all_puls, pulpy_puls, liberal_cblaster_puls])
    merged_puls = merge_overlapping_puls(all_puls_combined, blast=False).with_columns(
        polars.col("database").str.split("_").list.unique().alias("databases")
    )
    merged_puls = merged_puls.with_columns(
        polars.lit('pulpy').is_in(polars.col("databases")).alias("has_pulpy"),
        polars.lit('cblaster').is_in(polars.col("databases")).alias("has_cblaster"),
        polars.lit('experimental').is_in(polars.col("databases")).alias("has_experimental"),
    )
    # plot Venn diagram of overlap between databases
    pulpy_puls = set(merged_puls.filter(polars.col("has_pulpy") == True).select("cluster_id").to_series().to_list())
    cblaster_liberal_puls = set(merged_puls.filter(polars.col("has_cblaster") == True).select("cluster_id").to_series().to_list())
    experimental_puls = set(merged_puls.filter(polars.col("has_experimental") == True).select("cluster_id").to_series().to_list())

    fig, ax1 = plt.subplots(figsize=(8, 4))
    venn3(
        [pulpy_puls, cblaster_liberal_puls, experimental_puls],
        set_labels=(f'PULpy (total: {len(pulpy_puls)})', f'Liberal Cblaster (total: {len(cblaster_liberal_puls)})', f'Experimental + Strict Cblaster (total: {len(experimental_puls)})'),
        ax=ax1
    )
#    ax1.set_title("PULs identified by PULpy, Liberal Cblaster and Experimental annotations")
    plt.tight_layout()
    plt.savefig("results/plots/cryptic_pul_overlap_venn.png", dpi=300)
    plt.close()


def main():
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
    all_puls = polars.read_csv("src/data/data_collection/clusters_deduplicated_cblaster.tsv", separator="\t")
    genes = polars.read_parquet("src/data/genecat_output/genome.genes.parquet")

    plot_pul_overlap(all_puls, pulpy, cblaster_results_liberal)
    #plot_venn_diagram_cblaster()


if __name__ == "__main__":
    main()
