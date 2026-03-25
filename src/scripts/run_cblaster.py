# requires: 
    # conda install -c bioconda -c conda-forge diamond=2.1.24
    # pip install cblaster

# stepwise approach
    # create 1 fasta file with all PUL sequences
    # make cblaster db from genomes/combined_genomes.gb
    # make diamond db
    # run cblaster with PUL sequences as query against diamond db, output in tabular format
    # parse cblaster output and integrate into clusters table

    # cblaster makedb src/data/genomes/combined_genomes.gb -n cblasterdb

import polars
from utility_scripts import join_gene_and_PUL_table
from Bio import SeqIO
import os
from tqdm import tqdm
import subprocess


class CblasterProcessor:
    def __init__(self, clusters_table_path: str, gene_table_path: str, cblaster_output_path: str):
        self.clusters_table = polars.read_csv(clusters_table_path, separator='\t', infer_schema_length=600)
        self.gene_table = polars.read_parquet(gene_table_path)
        self.cblaster_output_path = cblaster_output_path
        self.pul_genes_path = "src/data/puls_genes"


    def write_genes_fasta(self):
        os.makedirs(self.pul_genes_path, exist_ok=True)
        # check if needed:
        if len(os.listdir(self.pul_genes_path)) > 0:
            print("PUL genes already extracted, skipping...")
            return

        labeled_table = join_gene_and_PUL_table(gene_table=self.gene_table, cluster_table=self.clusters_table).group_by("cluster_id")
        all_genes = SeqIO.index("src/data/genecat_output/call_genes/genome.genes.faa", "fasta")
        for cluster_id, group in tqdm(labeled_table, desc="Extracting PUL genes", ):
            out_file = f"{self.pul_genes_path}/{cluster_id[0]}.fasta"
            genes = []
            for row in group.select("protein_id", "cluster_id").iter_rows():
                gene = all_genes[row[0]]
                genes.append(gene)

            with open(out_file, "w") as handle:
                SeqIO.write(genes, handle, "fasta")

    
    def run_cblaster(self, filename: str, cluster_id: str):
        cmd = f"cblaster search -m local -db cblasterdb.dmnd -qf {filename} -b {self.cblaster_output_path}_{cluster_id}.tsv -bde '\t'"
        subprocess.run(cmd, shell=True, check=True)

    
    def make_cblaster_db(self):
        if os.path.exists("cblasterdb.dmnd"):
            print("Cblaster database already exists, skipping...")
            return

        cmd = "cblaster makedb src/data/genomes/combined_genomes.gb -n cblasterdb --force"
        subprocess.run(cmd, shell=True, check=True)

    
    def run_cblaster_on_all_genes(self):
        os.makedirs(self.cblaster_output_path, exist_ok=True)
        for filename in tqdm(os.listdir(self.pul_genes_path)[:5], desc="Running cblaster on PUL genes"):
            if not filename.endswith(".fasta"):
                continue

            cluster_id = filename.split("/")[-1].split(".")[0]
            self.run_cblaster(filename, cluster_id)


    def process_cblaster_output(self):
        # read all cblaster output files and concatenate into one dataframe
        cblaster_results = []
        for filename in os.listdir(self.cblaster_output_path):
            df = polars.read_csv(f"{self.cblaster_output_path}/{filename}", separator='\t')
            num_genes = len(df.columns) - 4
            df = (
                df.with_columns(
                    polars.sum_horizontal(df.columns[4:]).alias("total_hits"),
                    polars.lit(filename.split("/")[-1].split(".")[0]).alias("cluster_id"), 
                )
                .rename({"Scaffold": "sequence_id", "Start": "start", "End": "end"})
                .filter(polars.col("total_hits") == num_genes)
                .select("sequence_id", "cluster_id", "start", "end")
            )
            cblaster_results.append(df)

        cblaster_results_df = polars.concat(cblaster_results)
        print(f"Total hits: {cblaster_results_df.shape[0]}")

        sequence_info = (
            self.clusters_table
            .select("sequence_id", "tax_id", "database", "merged", "length", "pul_length_sum", "percentage_in_puls", "blast_status", "domain", "phylum", "class", "order", "family", "genus", "species")
            .unique(subset="sequence_id")
        )
        cblaster_results_df = (
            cblaster_results_df
            .join(sequence_info, on="sequence_id", how="inner").select(self.clusters_table.columns)
            .with_columns(polars.lit("cblaster").alias("database"))
        )
        integrated_table = polars.concat([self.clusters_table, cblaster_results_df], how="vertical")
        # merge overlapping puls
        print(f"Before merging overlapping PULs: {integrated_table.shape[0]} PULs")
        integrated_table = merge_overlapping_puls(integrated_table, keep_original=False)
        print(f"After merging overlapping PULs: {integrated_table.shape[0]} PULs\n")

        integrated_table.write_csv("src/data/results/cblaster_results.tsv", separator='\t')


if __name__ == "__main__":
    cblaster_processor = CblasterProcessor(
        clusters_table_path="src/data/results/clusters_deduplicated.tsv",
        gene_table_path="src/data/genecat_output/preprocess_output/genome.genes.parquet",
        cblaster_output_path="src/data/cblaster_output"
    )
    cblaster_processor.write_genes_fasta()
    cblaster_processor.make_cblaster_db()
    cblaster_processor.run_cblaster_on_all_genes()
    cblaster_processor.process_cblaster_output()