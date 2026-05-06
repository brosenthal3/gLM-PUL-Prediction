import polars
from utility_scripts import join_gene_and_PUL_table

# load all data
liberal_cblaster = polars.read_csv("src/data/data_collection/cblaster_results_liberal.tsv", separator="\t")
pulpy = (
    polars.read_csv("src/data/data_collection/pulpy_annotations.tsv", separator="\t")
    .rename({"genome": "sequence_id", "pulid": "cluster_id"})
    .select(liberal_cblaster.columns)
)
cryptic_puls = polars.concat([liberal_cblaster, pulpy])
experimental_puls = polars.read_csv("src/data/data_collection/clusters_deduplicated_cblaster.tsv", separator="\t")
genes = polars.read_parquet("src/data/genecat_output/genome.genes.parquet")

# merge clusters and cryptic puls with genes
cryptic_puls_genes = join_gene_and_PUL_table(genes, cryptic_puls).select("protein_id", "is_PUL").rename({"is_PUL": "is_cryptic_PUL"})
experimental_puls_genes = join_gene_and_PUL_table(genes, experimental_puls).select("protein_id", "is_PUL")
joined_genes = (
    experimental_puls_genes
    .join(cryptic_puls_genes, on="protein_id", how="inner", validate="1:1")
    .filter(
        (polars.col("is_PUL") == False) & (polars.col("is_cryptic_PUL") == True)
    )
    .select("protein_id")
    .write_csv("src/data/data_collection/cryptic_puls_genes.tsv", separator="\t")
)
