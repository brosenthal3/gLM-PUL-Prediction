import polars
from pathlib import Path
import os

def get_pulpy_annotations(pulpy_output_path):
    pulpy_annotations = polars.DataFrame()
    for pulpy_annotation_file in Path(pulpy_output_path).iterdir():
        # get only summary files, which contain the PUL annotations
        if "sum" in pulpy_annotation_file.stem and os.path.getsize(pulpy_annotation_file) > 0:
            pul_annotations = polars.read_csv(pulpy_annotation_file, separator='\t').with_columns(polars.col("pulid").map_elements(lambda x: f"PULpy_{x}").alias("pulid"))
            pulpy_annotations = pulpy_annotations.vstack(pul_annotations)

    return pulpy_annotations

if __name__ == "__main__":
    pulpy_annotations = get_pulpy_annotations("src/PULpy-master/puls/")
    pulpy_annotations.write_csv("src/data/data_collection/pulpy_annotations.tsv", separator='\t')