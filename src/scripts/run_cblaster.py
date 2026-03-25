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

clusters_table = polars.read_csv("src/data/results/clusters_deduplicated.tsv", separator='\t', infer_schema_length=600)
gene_table = polars.read_parquet("src/data/genecat_output/preprocess_output/genome.genes.parquet")
labeled_table = join_gene_and_PUL_table(gene_table=gene_table, cluster_table=clusters_table).group_by("cluster_id")

for cluster_id, group in labeled_table:
    print(f"Cluster {cluster_id}: {group['is_PUL'].sum()} genes in PUL, {len(group) - group['is_PUL'].sum()} genes not in PUL")
