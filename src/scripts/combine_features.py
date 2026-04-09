import polars

cazy = (
    polars.read_csv(
        "src/data/genecat_output/genome.features.cayman_cazy_v0.12.0.csv",
        separator=",",
        has_header=True,
        schema_overrides={
            "sequenceID": polars.Utf8,
            "domain_start": polars.UInt32,
            "domain_end": polars.UInt32,
            "family": polars.Utf8,
            "pvalue": polars.Float64,
            "start": polars.UInt32,
            "end": polars.UInt32
        }
    )
    .rename({
        "sequenceID": "protein_id",
        "family": "domain",
    })
    .with_columns(
        polars.lit("Cayman_v0.12.0").alias("hmm"),
        polars.lit(0).cast(dtype=polars.Float64).alias("i_evalue"),
        polars.col("protein_id").str.split("_").list.get(0).alias("sequence_id"),
        polars.lit("+").alias("strand")
    )
    .select([
        "sequence_id",
        "protein_id",
        "start",
        "end",
        "strand",
        "domain",
        "hmm",
        "i_evalue",
        "pvalue",
        "domain_start",
        "domain_end"
    ])
)

cazy.write_parquet("src/data/genecat_output/genome.features.cayman.parquet")

pfam_features = polars.read_parquet("src/data/genecat_output/genome.features.parquet")
joined = pfam_features.vstack(cazy).sort(by=["sequence_id", "domain_start"])
joined.write_parquet("src/data/genecat_output/genome.features.cayman.pfam.parquet")


# cayman_schema = {
#     'sequenceID': String, 
#     'start': Int64, 
#     'end': Int64, 
#     'pvalue': Float64, 
#     'family': String, 
#     'annotLength': Int64, 
#     'domain_start': Int64, 
#     'domain_end': Int64}

# pfam_schema = {
#     'sequence_id': String, 
#     'protein_id': String, 
#     'start': UInt32, 
#     'end': UInt32, 
#     'strand': String, 
#     'domain': String, 
#     'hmm': String, 
#     'i_evalue': Float64, 
#     'pvalue': Float64, 
#     'domain_start': UInt32, 
#     'domain_end': UInt32
# }
# add: sequence_id, strand (+), hmm, i_evalue
# drop: annotLength
