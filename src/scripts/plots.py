import polars
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import numpy as np
from utility_scripts import join_gene_and_PUL_table


class Plotter:
    def __init__(self, clusters_table: polars.DataFrame, gene_table: polars.DataFrame, save_path: str = "src/data/plots"):
        self.clusters_table = clusters_table
        self.gene_table = gene_table
        self.save_path = save_path


    def plot_percentage_in_puls_over_genome_length(self, blast=False):
        replaced_sequences = self.clusters_table.filter(polars.col("blast_status") == True)
        color = "orange" if blast else "red"

        figure, axs = plt.subplots(1, figsize=(8, 6))
        axs.scatter(x=self.clusters_table.select('length'), y=self.clusters_table.select('percentage_in_puls'))
        axs.scatter(x=replaced_sequences.select('length'), y=replaced_sequences.select('percentage_in_puls'), color=color, label="Replaced by BLAST hit")

        axs.set_xscale("log")
        axs.set_xlim(1e4, 1e7)
        axs.set_title("Percentage of genome in PULs over genome length")
        axs.set_xlabel("Genome Length (bp)")
        axs.set_ylabel("Percentage of genome in PULs")

        # vertical line at 50.000kb
        plt.vlines(100000, ymin=0, ymax=100, color="red", linestyle="dashed", label="100kb")
        plt.legend()
        plt.savefig(f"{self.save_path}/percentage_in_puls_over_genome_length.png", dpi=300)


    def get_taxonomic_counts(self, rank="phylum", cutoff=10):
        return (
            self.clusters_table
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


    def donut_chart(self, ax, counts, rank, title="Taxonomic distribution"):
        x = counts.select(f"{rank.lower()}_group").to_series()
        heights = counts.select("count").to_series()
        ax.pie(heights, labels=x, radius=1, wedgeprops=dict(width=0.3, edgecolor='w'))
        ax.set_title(f"{title} ({rank})")


    def plot_taxonomic_distributions(self, save="src/data/plots/temp.png"):
        phylum_counts = self.get_taxonomic_counts(rank="phylum", cutoff=5)
        class_counts = self.get_taxonomic_counts(rank="class", cutoff=5)
        order_counts = self.get_taxonomic_counts(rank="order", cutoff=5)
        family_counts = self.get_taxonomic_counts(rank="family", cutoff=5)

        figure, axs = plt.subplots(2, 2, figsize=(16, 12))
        for ax, counts, rank in zip(axs.flatten(), [phylum_counts, class_counts, order_counts, family_counts], ["Phylum", "Class", "Order", "Family"]):
            self.donut_chart(ax, counts, rank)

        plt.tight_layout()
        plt.savefig(save, dpi=300)


    def plot_PULs_in_genome(self, sequence_id: str):
        # add gene labels to cluster table
        labeled_table = join_gene_and_PUL_table(gene_table=self.gene_table, cluster_table=self.clusters_table)
        # Filter the table for the given sequence_id
        subset_puls = self.clusters_table.filter(polars.col("sequence_id") == sequence_id)
        subset = labeled_table.filter(polars.col("sequence_id") == sequence_id)
        # get range of genes in PULs for this sequence_id
        contig_range = np.zeros(subset_puls.select("length").to_series()[0], dtype=int)
        # set locations where genes are to 1
        gene_ranges = [[row[2]+1, row[3]+1] for row in subset.iter_rows()] # NOTE: unsure about off by one errors here, need to check with actual data
        pul_ranges = [[row[2], row[3], row[5]] for row in subset_puls.iter_rows()]

        for range in gene_ranges:
            contig_range[range[0]:range[1]] = 1

        for range in pul_ranges:
            if range[2] == 'dbcan':
                contig_range[range[0]:range[1]] = 2
            elif range[2] == 'puldb':
                contig_range[range[0]:range[1]] = 2
            elif "_" in range[2]: 
                contig_range[range[0]:range[1]] = 2
            else:
                contig_range[range[0]:range[1]] = 2

        plt.figure(figsize=(12, 2))
        plt.imshow([contig_range], aspect='auto', cmap='viridis', vmin=0, vmax=3)
        plt.title(f"Genes in {sequence_id} colored by PUL membership")
        plt.xlabel("bp in genome")
        # add legend
        plt.legend(
            handles=[
                plt.Line2D([0], [0], color='turquoise', lw=4, label='Gene'),
                plt.Line2D([0], [0], color='yellow', lw=4, label='Gene in PUL'),
                plt.Line2D([0], [0], color='purple', lw=4, label='No gene'),
            ],
            loc='upper right'
        )

        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/{sequence_id}.png", dpi=300)


    def plot_gene_counts(self, gene_table):
        gene_table = (
            gene_table.group_by("sequence_id").agg(
                polars.col("protein_id").n_unique().alias("gene_count"),
                (polars.col("is_PUL").sum()/polars.col("protein_id").n_unique()).alias("percentage_in_PUL")
            )
        )

        plt.figure(figsize=(6, 4))
        plt.scatter(x=gene_table.select("gene_count"), y=gene_table.select("percentage_in_PUL"), alpha=0.7, marker="o", edgecolors="black")
        plt.xlabel("Total gene count in sequence")
        plt.ylabel("Percentage of genes in PULs")
        plt.title("Percentage of genes in PULs over total gene count in sequence")
        plt.savefig(f"{self.save_path}/genes_in_puls_over_total_genes.png", dpi=300)

        plt.cla()

        plt.hist(gene_table.select("gene_count"), bins=25, alpha=0.7, edgecolor="black")
        plt.xlabel("Number of genes")
        plt.ylabel("Frequency")
        plt.title("Distribution of gene counts in genomes")
        plt.savefig(f"{self.save_path}/gene_count_distribution.png", dpi=300)


    def visualize_train_test_split(self, train_data, test_data):
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs = axs.flatten()
        # plot taxonomic distributions for train and test sets 
        rank = "phylum"
        phylum_counts_train = get_taxonomic_counts(train_data, rank=rank, cutoff=5)
        phylum_counts_test = get_taxonomic_counts(test_data, rank=rank, cutoff=5)
        x_train = phylum_counts_train.select(f"{rank.lower()}_group").to_series()
        x_test = phylum_counts_test.select(f"{rank.lower()}_group").to_series()
        heights_train = phylum_counts_train.select("count").to_series()
        heights_test = phylum_counts_test.select("count").to_series()
        axs[0].pie(heights_train, labels=x_train, radius=1, wedgeprops=dict(width=0.3, edgecolor='w'))
        axs[0].set_title(f"Train set {rank} distribution")
        axs[1].pie(heights_test, labels=x_test, radius=1, wedgeprops=dict(width=0.3, edgecolor='w'))
        axs[1].set_title(f"Test set {rank} distribution")

        # plot genome length distributions for train and test sets
        train_data = train_data.group_by("sequence_id").agg(polars.col("length").first(),
        polars.col("cluster_id").count().alias("pul_count"))
        test_data = test_data.group_by("sequence_id").agg(polars.col("length").first(),
        polars.col("cluster_id").count().alias("pul_count"))

        axs[2].scatter(
            x=train_data.select("length"), y=train_data.select("pul_count"), 
            alpha=0.7, marker="o", edgecolors="black",
        )
        axs[2].set_title("Genome length distribution in train set")
        axs[2].set_xlabel("Genome length (bp)")
        axs[2].set_ylabel("PUL count")
        axs[2].set_xscale("log")
        axs[3].scatter(
            x=test_data.select("length"), y=test_data.select("pul_count"), 
            alpha=0.7, marker="o", edgecolors="black",
        )
        axs[3].set_title("Genome length distribution in test set")
        axs[3].set_xlabel("Genome length (bp)")
        axs[3].set_ylabel("PUL count")
        axs[3].set_xscale("log")

        plt.tight_layout()
        plt.savefig(f"{self.save_path}/train_test_split.png", dpi=300)


    def get_pul_lengths(self, puls_table):
        return puls_table.with_columns(abs(polars.col("end") - polars.col("start")).alias("pul_length"))


    def plot_pulpy_puls(self):
        pulpy_puls = self.clusters_table.filter(polars.col("database").str.contains("PULpy"))
        experimental_puls = self.clusters_table.filter(~polars.col("database").str.contains("PULpy"))

        pulpy_pul_lengths = self.get_pul_lengths(pulpy_puls)
        experimental_pul_lengths = self.get_pul_lengths(experimental_puls)
        
        plt.figure(figsize=(6, 4))
        # plot box plots of pul lengths for pulpy and experimental puls
        plt.boxplot([pulpy_pul_lengths.select("pul_length").to_series(), experimental_pul_lengths.select("pul_length").to_series()], tick_labels=["PULpy", "Experimental"])
        plt.ylabel("PUL length (bp)")
        plt.title("PUL lengths of PULpy and experimental annotations")
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/pulpy_pul_lengths.png", dpi=300)


    def get_bins(self, labeled_table):
        bins = np.unique(
            np.logspace(
                start=np.log2(labeled_table['n_genes'].min()),
                stop=np.log2(80),
                base=2,
                num=20,
            ).astype(int)
        )
        labels = [f'<{bins[0]}'] + [f'{bins[i]}-{bins[i + 1]}' for i in range(len(bins[:-1]))] + [f'>{bins[-1]}']
        return bins, labels

    def get_n_genes(self, labeled_table):
        labeled_table = join_gene_and_PUL_table(gene_table=self.gene_table, cluster_table=labeled_table)
        labeled_table = labeled_table.group_by("cluster_id").agg(polars.col("is_PUL").sum().alias("n_genes")).sort("n_genes", descending=False).filter(polars.col("cluster_id").is_not_null())
        return labeled_table


    def plot_pul_gene_count(self):
        pulpy_table = self.clusters_table.filter(polars.col("cluster_id").str.contains("PULpy"))
        experimental_table = self.clusters_table.filter(~polars.col("cluster_id").str.contains("PULpy"))

        pulpy_table = self.get_n_genes(pulpy_table)
        experimental_table = self.get_n_genes(experimental_table)

        # Get bins from combined table to ensure consistency
        combined_table = polars.concat([pulpy_table, experimental_table])
        bins, labels = self.get_bins(combined_table)
        
        # Reapply bins to individual tables
        labeled_table_experimental = experimental_table.with_columns(polars.col('n_genes').cut(breaks=bins.tolist(), include_breaks=False, labels=labels).alias('gene_bin'))
        labeled_table_pulpy = pulpy_table.with_columns(polars.col('n_genes').cut(breaks=bins.tolist(), include_breaks=False, labels=labels).alias('gene_bin'))

        labels_pulpy = labels
        labels_experimental = labels

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].hist(labeled_table_pulpy.select("gene_bin").to_series(), bins=len(labels_pulpy), edgecolor="black")
        axs[0].set_xlabel("Number of genes in PUL")
        axs[0].set_ylabel("Count")
        axs[0].set_title("Distribution of gene counts in PULpy annotations")
        axs[0].tick_params(axis='x', rotation=45)

        axs[1].hist(labeled_table_experimental.select("gene_bin").to_series(), bins=len(labels_experimental), edgecolor="black")
        axs[1].set_xlabel("Number of genes in PUL")
        axs[1].set_ylabel("Count")
        axs[1].set_title("Distribution of gene counts in experimental annotations")
        axs[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f"{self.save_path}/pul_gene_count_distribution.png", dpi=300)


if __name__ == "__main__":
    # get clusters and gene table
    clusters_table = polars.read_csv("src/data/results/clusters_with_pulpy.tsv", separator='\t', infer_schema_length=600)
    gene_table = polars.read_parquet("src/data/genecat_output/preprocess_output/genome.genes.parquet")
    plotter = Plotter(clusters_table, gene_table)
    plotter.plot_pul_gene_count()
    
    # train_data = polars.read_csv("src/data/splits/train_fold_0.tsv", separator='\t', infer_schema_length=600)
    # test_data = polars.read_csv("src/data/splits/test_fold_0.tsv", separator='\t', infer_schema_length=600)
    # plotter.visualize_train_test_split(train_data, test_data)


## VENN DIAGRAMS, not used but code might be useful
# def plot_venn_diagram_database(save="src/data/plots/temp.png"):
#     plt.figure(figsize=(5, 5))

#     dbcan = polars.read_csv("src/data/results/dbcan_clusters.tsv", separator='\t', infer_schema_length=600)
#     puldb = polars.read_csv("src/data/results/puldb_clusters.tsv", separator='\t', infer_schema_length=600)
#     dbcan_sequences = set(dbcan.select("sequence_id").to_series())
#     puldb_sequences = set(puldb.select("sequence_id").to_series())
#     venn2([dbcan_sequences, puldb_sequences], set_labels = ('DBCAN', 'PULDB'))
#     plt.title("Overlap between PULDB and DBCAN sequences")
#     plt.savefig(save, dpi=300)

# def plot_venn_diagram_blast(save="src/data/plots/temp.png"):
#     clusters_table = polars.read_csv("src/data/results/combined_clusters_blasted_gtdb.tsv", separator='\t', infer_schema_length=600)
#     old_sequences = set(clusters_table.select("sequence_id").unique().drop_nulls().to_series())
#     new_sequences = set(clusters_table.select("new_sequence_id").unique().drop_nulls().to_series())
#     print(len(new_sequences))

#     venn2([old_sequences, new_sequences], set_labels = ('Original', 'BLASTed'))
#     plt.title("Overlap between original and BLASTed sequences")
#     plt.savefig(save, dpi=300)


# def plot_venn_diagram_blast_filtered(save="src/data/plots/temp.png"):
#     clusters_table = polars.read_csv("src/data/results/combined_clusters_blasted_gtdb.tsv", separator='\t', infer_schema_length=600)
#     from_blast = set(clusters_table.filter(polars.col("blast_status") == True).select("new_sequence_id").to_series())
#     original = set(clusters_table.filter(polars.col("blast_status") == False).select("sequence_id").to_series())

#     venn2([original, from_blast], set_labels = ('Original', 'BLASTed'))
#     plt.title("Overlap between original and sequences replaced by BLAST hits")
#     plt.savefig(save, dpi=300)
