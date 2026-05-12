from visualization_utilities import PredictionEvaluator
import matplotlib.pyplot as plt
import polars
import numpy as np
import os
import gb_io
import argparse
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import plotly.express as px
import requests
from tqdm import tqdm
from time import sleep

def analyze_domain_enrichment(features_with_predictions, label_column="is_PUL_pred"):
    total_positive = (features_with_predictions.select(polars.col(label_column).sum()).item())
    total_negative = len(features_with_predictions) - total_positive

    results = []
    for domain in features_with_predictions["domain"].unique():
        subset = features_with_predictions.filter(polars.col("domain") == domain)

        # Domain present
        a = subset.select(polars.col(label_column).sum()).item()
        b = len(subset) - a

        # Domain absent
        c = total_positive - a
        d = total_negative - b

        # Fisher exact test
        contingency = [[a, b], [c, d]]
        oddsratio, pvalue = fisher_exact(contingency)

        # log2 enrichment
        log2_enrichment = np.log2(
            ((a + 0.5) / (a + b + 1)) /
            ((c + 0.5) / (c + d + 1))
        )

        results.append({
            "domain": domain,
            "positive_with_domain": a,
            "negative_with_domain": b,
            "oddsratio": oddsratio,
            "log2_enrichment": log2_enrichment,
            "pvalue": pvalue,
        })
    stats_df = polars.DataFrame(results)

    # multiple testing correction
    fdr = multipletests(stats_df["pvalue"], method="fdr_bh")[1]
    stats_df = stats_df.with_columns([
        polars.Series("fdr", fdr),
        (-np.log10(polars.Series(fdr))).alias("neglog10_fdr"),
    ])

    # add significance column 
    stats_df = stats_df.with_columns([((polars.col("fdr") < 0.05) & (polars.col("log2_enrichment").abs() > 1)).alias("significant")])
    return stats_df



def main(args):
    # get all predictions for the model
    model_name = args.model
    model_evaluator = PredictionEvaluator(
        labeled_results_path = f"src/data/results/{model_name}/labeled_results_test",
        model_name=model_name,
        k=5,
        output_path=f"results/plots/{model_name}"
    )
    model_evaluator.aggregate_all_folds()
    model_evaluator.recompute_predictions(0, threshold=0.2) # recompute labels at threshold

    features = polars.read_parquet("src/data/genecat_output/pfam.features.parquet")
    features_with_predictions = (
        model_evaluator.labeled_results[0]
        .select("protein_id", "is_PUL_pred", "is_PUL_pulpy", "is_PUL")
        .join(
            features.select("protein_id", "domain"),
            how="inner",
            on="protein_id"
        )
    )

    label_columns = ["is_PUL_pred", "is_PUL_pulpy", "is_PUL"]
    all_results = {}

    for label in label_columns:
        # get stats
        stats_df = analyze_domain_enrichment(features_with_predictions, label)
        all_results[label] = stats_df

        # convert to pandas for plotting
        pdf = stats_df.to_pandas()

        # Label only strong hits
        pdf["label"] = np.where(((pdf["fdr"] < 0.01) & (np.abs(pdf["log2_enrichment"]) > 2)), pdf["domain"], "")

        fig = px.scatter(
            pdf,
            x="log2_enrichment",
            y="neglog10_fdr",
            color="significant",
            size="positive_with_domain",
            hover_data=[
                "domain",
                "positive_with_domain",
                "oddsratio",
                "pvalue",
                "fdr",
            ],
            text="label",
            title=f"Pfam enrichment volcano plot: {label}",
        )

        # threshold lines
        fig.add_vline(x=-1, line_dash="dash")
        fig.add_vline(x=1, line_dash="dash")

        fig.add_hline(y=-np.log10(0.05), line_dash="dash")

        fig.update_traces(textposition="top center")

        fig.update_layout(
            height=900,
            width=1200,
            xaxis_title="log2 enrichment",
            yaxis_title="-log10(FDR)",
        )

        fig.write_html(os.path.join(model_evaluator.output_path, f"pfam_enrichment_{label}.html"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize Pfam domain enrichment in model predictions"
    )
    parser.add_argument("--model", type=str, help="Name of model to evaluate", required=True)
    args = parser.parse_args()
    main(args)
