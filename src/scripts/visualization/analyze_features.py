from visualization_utilities import PredictionEvaluator
import matplotlib.pyplot as plt
import polars
import numpy as np
import os
import gb_io
import argparse



def main(args):
    model_name = args.model
    model_evaluator = PredictionEvaluator(
        labeled_results_path = f"src/data/results/{model_name}/labeled_results_test",
        model_name=model_name,
        k=5,
        output_path=f"results/plots/{model_name}"
    )
    features = polars.read_parquet("src/data/genecat_output/pfam.features.parquet")
    features_with_predictions = (
        model_evaluator.labeled_results[0]
        .select("protein_id", "average_p", "is_PUL_pulpy", "is_PUL")
        .join(
            features.select("protein_id", "domain"),
            how="inner",
            on="protein_id"
        )
       .group_by("domain")
        .agg(
            polars.col("is_PUL_pulpy").mean().alias("count_pulpy"),
            polars.col("average_p").mean().alias("sum_p"),
        )
        .sort("sum_p")
    )
    print(features_with_predictions)

    # TODO: proper enrichment analysis, and volcano plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize Pfam domain enrichment in model predictions"
    )
    parser.add_argument("--model", type=str, help="Name of model to evaluate", required=True)
    args = parser.parse_args()
    main(args)
