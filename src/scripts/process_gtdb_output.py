import polars
from pathlib import Path

def separate_classification(classification: polars.Expr, index) -> polars.Expr:
    return classification.str.split(by=";").list.get(index, null_on_oob=True).str.split(by="__").list.get(1, null_on_oob=True)

def get_taxonomic_annotation(gtdb_summary_path):
    if not Path(gtdb_summary_path).exists():
        print(f"GTDB-Tk taxonomic annotation file not found at {gtdb_summary_path}, cannot proceed with taxonomic annotation step.")
        print("Please run GTDB-Tk classification on the fasta file of genomes and place the resulting summary file at the specified path.")
        raise FileNotFoundError(f"GTDB-Tk summary file not found at {gtdb_summary_path}")

    taxonomic_annotation = polars.read_csv(gtdb_summary_path, separator="\t").select('user_genome', 'classification')
    taxonomic_annotation = (
        taxonomic_annotation
        .with_columns(
            separate_classification(polars.col("classification"), 0).alias("domain"),
            separate_classification(polars.col("classification"), 1).alias("phylum"),
            separate_classification(polars.col("classification"), 2).alias("class"),
            separate_classification(polars.col("classification"), 3).alias("order"),
            separate_classification(polars.col("classification"), 4).alias("family"),
            separate_classification(polars.col("classification"), 5).alias("genus"),
            separate_classification(polars.col("classification"), 6).alias("species"),
        )
        .rename({'user_genome': 'sequence_id'})
        .drop("classification")
    )
    return taxonomic_annotation


if __name__ == "__main__":
    # get cluster table
    combined_clusters_blasted = polars.read_csv("src/data/data_collection/combined_clusters_blasted.tsv")
    # merge taxonomic annotation into clusters table, both on sequence_id and new_sequence_id
    taxonomic_annotation = get_taxonomic_annotation("src/data/data_collection/gtdbtk.bac120.summary.tsv")
    combined_clusters_gtdb = (
        combined_clusters_blasted
        .join(taxonomic_annotation, on="sequence_id", how="left")
    )
    combined_clusters_gtdb.write_csv("src/data/data_collection/combined_clusters_blasted_gtdb.tsv", separator='\t')
