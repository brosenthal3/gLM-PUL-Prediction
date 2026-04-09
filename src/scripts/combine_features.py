import polars

df = polars.read_parquet("PUL.features.Pfam37.p1e-9.parquet")

cazy = (
    polars.scan_csv(
        "PUL.features.cazy_cayman_v0.12.parquet",
        separator=",",
        has_header=True,
        schema_overrides={
            "sequenceID": polars.Utf8,
            "domain_start": polars.UInt32,
            "domain_end": polars.UInt32,
            "family": polars.Utf8,
            "pvalue": polars.Float64,
        }
    )
    .rename({
        "sequenceID": "protein_id",
        "family": "domain",
    })
    .drop(["start", "end"])
    .with_columns(
        hmm=polars.lit("Cayman_v0.12"),
        i_evalue=polars.lit(0).cast(dtype=polars.Float64),
    )
)
cazy.write_parquet("PUL.features.cazy_cayman_v0.12.parquet")

joined = df.vstack(cazy).sort(by=["sequence_id", "domain_start"])
joined.write_parquet("PUL.features.Pfam37.1_cazy_cayman_v0.12.p1e-9.parquet")