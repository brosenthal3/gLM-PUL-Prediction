# requires: 
    # conda install -c bioconda -c conda-forge diamond=2.1.24
    # pip install cblaster

# stepwise approach
    # create 1 fasta file with all PUL sequences
    # make cblaster db from genomes/combined_genomes.gb
    # make diamond db
    # run cblaster with PUL sequences as query against diamond db, output in tabular format
    # parse cblaster output and integrate into clusters table

    # cblaster config --email b.rosenthal@lumc.nl
    # cblaster makedb src/data/genomes/combined_genomes.gb -n cblasterdb

import polars
from utility_scripts import join_gene_and_PUL_table, recompute_length_percentage
from data_collection import merge_overlapping_puls
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
        self.database = "./src/data/cblasterdb"


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
        filters = "-me 1.0e-9 -mi 70 -mc 75 -g 5000 -mh 2"
        cmd = f"cblaster search -m local -db {self.database}.dmnd -qf {filename} -b {self.cblaster_output_path}/{cluster_id}.csv -bde ',' " + filters
        subprocess.run(cmd, shell=True, check=True)

    
    def make_cblaster_db(self):
        if os.path.exists(f"{self.database}.dmnd"):
            print("Cblaster database already exists, skipping...")
            return

        cmd = f"cblaster config --email b.rosenthal@lumc.nl && cblaster makedb src/data/genomes/genbank_genomes/*.gb -n {self.database} --force -b 20"
        subprocess.run(cmd, shell=True, check=True)

    
    def run_cblaster_on_all_genes(self):
        os.makedirs(self.cblaster_output_path, exist_ok=True)
        for filename in tqdm(os.listdir(self.pul_genes_path), desc="Running cblaster on PUL genes"):
            if not filename.endswith(".fasta"):
                continue

            cluster_id = filename.split("/")[-1].split(".")[0]
            self.run_cblaster(f"{self.pul_genes_path}/{filename}", cluster_id)


    def process_cblaster_output(self):
        # read all cblaster output files and concatenate into one dataframe
        cblaster_results = []
        for filename in os.listdir(self.cblaster_output_path):
            df = polars.read_csv(f"{self.cblaster_output_path}/{filename}", separator=',')
            num_genes = len(df.columns) - 5
            query_id = self.clusters_table.filter(polars.col("cluster_id") == filename.split(".")[0]).select("sequence_id")
            df = (
                df.with_columns(
                    polars.sum_horizontal(df.columns[5:]).cast(polars.Int64).alias("total_hits"),
                    polars.lit(f"cblaster_{filename.split("/")[-1].split(".")[0]}").alias("cluster_id"), 
                )
                .rename({"Organism": "sequence_id", "Start": "start", "End": "end"})
                .filter(polars.col("total_hits").ge(num_genes * 0.7))
                .select("sequence_id", "cluster_id", "start", "end")
                .join(query_id, on="sequence_id", how="anti") # remove self-hits
            )
            if df.shape[0] > 0:
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
        print(f"Added total of {integrated_table.filter(polars.col("database").str.contains("cblaster")).shape[0]} PULs from cblaster")
        integrated_table = recompute_length_percentage(integrated_table).sort("start").sort("sequence_id")
        integrated_table.write_csv("src/data/results/cblaster_results.tsv", separator='\t')


if __name__ == "__main__":
    cblaster_processor = CblasterProcessor(
        clusters_table_path="src/data/results/clusters_deduplicated.tsv",
        gene_table_path="src/data/genecat_output/preprocess_output/genome.genes.parquet",
        cblaster_output_path="src/data/cblaster_output"
    )
    #cblaster_processor.write_genes_fasta()
    #cblaster_processor.make_cblaster_db()
    #cblaster_processor.run_cblaster_on_all_genes()
    cblaster_processor.process_cblaster_output()
