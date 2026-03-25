import json
import polars
from Bio import Entrez
Entrez.email = "b.rosenthal@lumc.nl"

# Entrez utility functions
def request_summary(acc, db="nuccore"):
    handle = Entrez.esummary(
        db=db,
        id=acc,
        retmode="json"
    )
    record = handle.read()
    handle.close()
    # parse the JSON response to get the sequence length
    record = json.loads(record)
    return record

def request_sequence(acc, db="nuccore"):
    handle = Entrez.efetch(
        db=db,
        id=acc,
        rettype="gb",
        retmode="json",
        complexity=1
    )
    record = handle.read()
    handle.close()
    return record


def join_gene_and_PUL_table(gene_table: polars.DataFrame, cluster_table: polars.DataFrame, buffer: int = 100,) -> polars.DataFrame:
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
