from pathlib import Path
import polars

def separate_classification(classification: polars.Expr, index) -> polars.Expr:
    return classification.str.split(by=";").list.get(index, null_on_oob=True).str.split(by="__").list.get(1, null_on_oob=True)

def main():
    taxonomic_annotation_path = Path("src/data/results/gtdbtk.bac120.summary.tsv")
    taxonomic_annotation = polars.read_csv(taxonomic_annotation_path, separator="\t").select('user_genome', 'classification')
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
    null_count = taxonomic_annotation.filter(polars.col("domain").is_null()).select("sequence_id").to_series()
    print(f"There are {null_count.shape[0]}/{taxonomic_annotation.shape[0]} genomes that did not get annotated.")
    taxonomic_annotation.write_csv("src/data/results/taxonomic_annotation.tsv", separator="\t")

    # merge into clusters table
    clusters_table = polars.read_csv("src/data/results/combined_clusters_blasted.tsv", separator="\t", infer_schema_length=300)
    # merge taxonomic annotation into clusters table, both on sequence_id and new_sequence_id
    clusters_table = (
        clusters_table
        .join(taxonomic_annotation, on="sequence_id", how="left")
        .join(taxonomic_annotation, left_on="new_sequence_id", right_on="sequence_id", how="left", suffix="_new")
    )
    clusters_table.write_csv("src/data/results/combined_clusters_blasted_gtdb.tsv", separator="\t")


if __name__ == "__main__":
    main()