"""
File for running miscellaneous functions/scripts
"""

import polars

def merge_overlapping_puls(df, group_col='sequence_id', start_col='start', end_col='end'):
    merged_puls = polars.DataFrame(schema=df.schema)
    merged_ids = []

    for sequence_id, group in df.group_by(group_col):
        if group.shape[0] == 1 or sequence_id[0] is None:
            merged_puls = merged_puls.vstack(polars.DataFrame(group))
            continue

        # sort by start position
        current_pul = None
        for row in group.sort(start_col).iter_rows(named=True):
            if current_pul is None:
                current_pul = row
            else:
                # check if there is an overlap with the current PUL
                if row[start_col] <= current_pul[end_col]:
                    # merge the PULs by updating the end position to the maximum end position
                    current_pul[end_col] = max(current_pul[end_col], row[end_col])
                    # merge cluster_id by concatenating with an underscore
                    current_pul['cluster_id'] = f"{current_pul['cluster_id']}_{row['cluster_id']}"
                    merged_ids.append({'cluster_id': current_pul['cluster_id'], 'merged': "merged"})                        
                else:
                    merged_puls = merged_puls.vstack(polars.DataFrame([current_pul]))
                    current_pul = row

        # add the last PUL after processing all rows for this sequence_id
        if current_pul is not None:
            merged_puls = merged_puls.vstack(polars.DataFrame([current_pul]))

    # add column for which puls are merged
    merged_puls = (
        merged_puls
        .join(polars.DataFrame(merged_ids), on="cluster_id", how="left", suffix="_new")
    )
    return merged_puls.sort('cluster_id').sort('merged')


def count_potential_bacteroidetes_puls(predicted_clusters, clusters_table, model_name):
    predicted_clusters_filtered = (
        predicted_clusters
        .join(
            clusters_table.select("sequence_id", "phylum").unique(), # get phylum
            on="sequence_id",
            how="left"
        )
        .filter(
            polars.col("phylum").eq("Bacteroidota"), # filter out non-bacteroidota
            polars.col("gene_count").ge(2)
        )
        .with_columns(
            polars.concat_str(polars.col("sequence_id"), polars.col("start")).alias("cluster_id") # add cluster id col
        )
        .select("sequence_id", "cluster_id", "start", "end")
    )
    pulpy_clusters = (
        polars.read_csv("src/data/data_collection/pulpy_annotations.tsv", separator="\t")
        .select("genome", "pulid", "start", "end")
        .rename({"genome": "sequence_id", "pulid": "cluster_id"})
    )
    # combine with pulpy anntoations and merge to find which ones overlap (merged puls=overlapping puls)
    predicted_and_pulpy = predicted_clusters_filtered.vstack(pulpy_clusters)
    predicted_and_pulpy = merge_overlapping_puls(predicted_and_pulpy)

    # filter to find which non-merged PULs are from our model    
    predicted_without_pulpy = (predicted_and_pulpy.filter(
        polars.col("merged").is_null(),
        ~polars.col("cluster_id").str.contains("PULpy")
    ))
    print(f"{model_name} found a total of {predicted_without_pulpy.height} PULs that were not predicted by PULpy")


if __name__ == "__main__":
    # model_name = "genecat_finetuned_cazy_masked"
    model_name = "gecco_pfam"
    clusters_table = polars.read_csv("src/data/data_collection/clusters_deduplicated_cblaster.tsv", separator="\t", infer_schema_length=700)
    predicted_clusters_path = f"src/data/results/{model_name}/predicted_clusters.parquet"
    predicted_clusters = polars.read_parquet(predicted_clusters_path)
    count_potential_bacteroidetes_puls(predicted_clusters, clusters_table, model_name)

