import polars

def join_gene_and_PUL_table(
    gene_table: polars.DataFrame,
    cluster_table: polars.DataFrame,
    buffer: int = 100,
) -> polars.DataFrame:

    """
    Join a gene table with a PUL annotation table.
    """

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
            polars.col("strand").first().alias("strand"),
        )
        .sort(by=["sequence_id", "start", "end"])
        .with_row_index(name="gene_id", offset=0)  # important
        .select(["sequence_id", "protein_id", "start", "end", "strand", "is_PUL"])
    )

    return labled_gene_table


genes_table = polars.read_parquet("src/data/genecat_output/preprocess_output/genome.genes.parquet")
cluster_table = polars.read_csv("src/data/results/clusters_deduplicated.tsv", separator='\t', infer_schema_length=700)
genes_with_puls = join_gene_and_PUL_table(genes_table, cluster_table)
genes_with_puls.write_csv("src/data/results/genes_with_puls.tsv", separator='\t')
print(f"Total of {genes_with_puls['is_PUL'].sum()}/{genes_with_puls.shape[0]} genes are in PULs, which is {genes_with_puls['is_PUL'].sum() / genes_with_puls.shape[0] * 100}%.")