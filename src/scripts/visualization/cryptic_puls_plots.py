import polars
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib_venn import venn3, venn2
import seaborn as sns
from viz_data import Cork_7, Bilbao_5, Buda_4, Bold_10, model_colors
from visualization_utilities import get_bins, join_gene_and_PUL_table, get_pul_lengths

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


def plot_length_distributions(genes, experimental_puls, cblaster_results_liberal, cblaster_results_strict, pulpy):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    bins, labels = get_bins(10, start=0, stop=100)
    experimental = get_pul_lengths(experimental_puls, genes)
    cblaster_liberal_puls = get_pul_lengths(cblaster_results_liberal, genes)
    cblaster_strict_puls = get_pul_lengths(cblaster_results_strict, genes)
    pulpy_puls = get_pul_lengths(pulpy, genes)

    # Extract series
    exp_lengths = experimental.select("pul_length").to_series()
    lib_lengths = cblaster_liberal_puls.select("pul_length").to_series()
    strict_lengths = cblaster_strict_puls.select("pul_length").to_series()
    pulpy_lengths = pulpy_puls.select("pul_length").to_series()

    n_datasets = 4
    pul_length_distributions = {
        "Experimental": exp_lengths,
        "PULpy": pulpy_lengths,
        "Liberal Cblaster": lib_lengths,
        "Strict Cblaster": strict_lengths,
    }
    all_datasets = list(pul_length_distributions.keys())
    colors = [Cork_7[-1], Cork_7[0], Cork_7[2], Cork_7[-2]]
    width = 0.2
    for dataset_name, lengths in pul_length_distributions.items():
        labeled_table = polars.DataFrame({"n_genes": lengths})
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
        # plot bar for current model
        counts_dict = dict(zip(counts["gene_bin"].to_list(), counts["len"].to_list()))
        y = [counts_dict.get(label, 0)/len(lengths) for label in labels]
        x = np.arange(len(labels))
        idx = all_datasets.index(dataset_name)
        # center grouped bars: compute offset so bars for each model are centered around each xtick
        offset = (idx - (n_datasets - 1) / 2) * width
        ax.bar(x + offset, y, edgecolor="black", width=width, align="center", label=dataset_name, color=colors[idx], alpha=0.9)

    ax.set_xticks(x)
    labels[0] = "1"
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.margins(x=0.02)
    ax.set_xlabel("PUL length in genes")
    ax.set_ylabel("Proportion of identified clusters")
    ax.set_title("PUL length distribution in identified clusters")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/plots/cryptic_pul_length_barplot.png")



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
        polars.concat_str([polars.col("cluster_id"), polars.col("sequence_id")]).alias("cluster_id_unique") # concat ids to make unique labels
    )

    # plot Venn diagram of overlap between databases
    pulpy_puls = set(merged_puls.filter(polars.col("has_pulpy") == True).select("cluster_id_unique").to_series().to_list())
    cblaster_liberal_puls = set(merged_puls.filter(polars.col("has_cblaster") == True).select("cluster_id_unique").to_series().to_list())
    experimental_puls = set(merged_puls.filter(polars.col("has_experimental") == True).select("cluster_id_unique").to_series().to_list())

    fig, ax1 = plt.subplots(figsize=(7, 4))
    venn3(
        [pulpy_puls, cblaster_liberal_puls, experimental_puls],
        set_labels=(f'PULpy (total: {len(pulpy_puls)})', f'Liberal Cblaster (total: {len(cblaster_liberal_puls)})', f'Experimental + Strict Cblaster (total: {len(experimental_puls)})'),
        ax=ax1,
        set_colors=(Cork_7[0], Cork_7[2], Cork_7[-1]),
        alpha=0.8
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
    plot_length_distributions(genes, experimental_puls, cblaster_results_liberal, cblaster_results_strict, pulpy)

if __name__ == "__main__":
    main()
