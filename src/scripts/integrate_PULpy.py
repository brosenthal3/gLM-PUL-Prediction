import polars
from pathlib import Path
import os
from data_collection import merge_overlapping_puls

def integrate_PULpy_annotations(clusters_table: polars.DataFrame, pulpy_annotations: polars.DataFrame):
    # get pulpy annotations in shape of clusters table

    # cluster_id	sequence_id	start	end	tax_id	database	merged	length	pul_length_sum	percentage_in_puls	blast_status	domain	phylum	class	order	family	genus	species
    # genome	pulid	contigid	start	end	pattern
    sequence_info = (
        clusters_table
        .select("sequence_id", "tax_id", "database", "merged", "length", "pul_length_sum", "percentage_in_puls", "blast_status", "domain", "phylum", "class", "order", "family", "genus", "species")
        .unique(subset="sequence_id")
    )
    pulpy_annotations = (
        pulpy_annotations
        .select("genome", "pulid", "start", "end").rename({"genome": "sequence_id", "pulid": "cluster_id"})
        .join(sequence_info, on="sequence_id", how="inner").select(clusters_table.columns)
        .with_columns(polars.lit("PULpy").alias("database"))
    )
    print(f"Found {pulpy_annotations.shape[0]} PULpy annotations that could be integrated into the cluster table")
    # concat both
    integrated_table = polars.concat([clusters_table, pulpy_annotations], how="vertical")
    # merge overlapping puls
    print(f"Before merging overlapping PULs: {integrated_table.shape[0]} PULs")
    integrated_table = merge_overlapping_puls(integrated_table, keep_original=False)
    print(f"After merging overlapping PULs: {integrated_table.shape[0]} PULs\n")

    return integrated_table


def get_pulpy_annotations(pulpy_output_path):
    pulpy_annotations = polars.DataFrame()
    for pulpy_annotation_file in Path(pulpy_output_path).iterdir():
        # get only summary files, which contain the PUL annotations
        if "sum" in pulpy_annotation_file.stem and os.path.getsize(pulpy_annotation_file) > 0:
            pul_annotations = polars.read_csv(pulpy_annotation_file, separator='\t').with_columns(polars.col("pulid").map_elements(lambda x: f"PULpy_{x}").alias("pulid"))
            pulpy_annotations = pulpy_annotations.vstack(pul_annotations)

    return pulpy_annotations


def main():
    clusters_table = polars.read_csv("src/data/results/cblaster_results.tsv", separator='\t', infer_schema_length=600)
    pulpy_annotations = get_pulpy_annotations("src/PULpy-master/puls/")
    # integrate annotations
    integrated_table = integrate_PULpy_annotations(clusters_table, pulpy_annotations)
    print(f"Adding PULpy resulted in {integrated_table.shape[0]} puls")
    integrated_table.write_csv("src/data/results/clusters_with_pulpy.tsv", separator='\t')


if __name__ == "__main__":
    main()