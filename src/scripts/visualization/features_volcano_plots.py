from visualization_utilities import PredictionEvaluator
import matplotlib.pyplot as plt
import polars
import numpy as np
import os
import argparse
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

def analyze_domain_enrichment(features_with_predictions, label_column="is_PUL_pred"):
    features_with_predictions = features_with_predictions.select("domain", label_column)
    total_positive = (features_with_predictions.select(polars.col(label_column).sum()).item())
    total_negative = len(features_with_predictions) - total_positive

    results = []
    for domain in tqdm(features_with_predictions["domain"].unique(), desc="Computing enrichments"):
        subset = features_with_predictions.filter(polars.col("domain") == domain)

        # create contingency table
        present_positives = subset.select(polars.col(label_column).sum()).item()
        present_negatives = len(subset) - present_positives
        absent_positives = total_positive - present_positives
        absent_negatives = total_negative - present_negatives
        contingency = [[present_positives, present_negatives], [absent_positives, absent_negatives]]

        # perform fishers exact test on the contingency table
        oddsratio, pvalue = fisher_exact(contingency)

        # calculate log2 enrichment
        log2_enrichment = np.log2(
            ((present_positives + 0.5) / (present_positives + present_negatives + 1)) /
            ((absent_positives + 0.5) / (absent_positives + absent_negatives + 1))
        )

        results.append({
            "domain": domain,
            "positive_with_domain": present_positives,
            "negative_with_domain": present_negatives,
            "oddsratio": oddsratio,
            "log2_enrichment": log2_enrichment,
            "pvalue": pvalue,
        })
    stats_df = polars.DataFrame(results)

    # multiple testing correction
    fdr = multipletests(stats_df["pvalue"], method="fdr_bh")[1]
    stats_df = stats_df.with_columns([
        polars.Series("fdr", fdr),
        (-np.log10(polars.Series(fdr))).clip(upper_bound=400).alias("neglog10_fdr"),
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

    feature_descriptions = polars.read_parquet("src/data/analysis/pfam_metadata.parquet")
    if args.features == "pfam":
        features = polars.read_parquet("src/data/genecat_output/pfam.features.parquet")
    elif args.features == "cazy":
        features = polars.read_parquet("src/data/genecat_output/dbcan.pfam.features.parquet")

    features_with_predictions = (
        model_evaluator.labeled_results[0]
        .select("protein_id", "is_PUL_pred", "is_PUL_pulpy", "is_PUL")
        .join(
            features.select("protein_id", "domain"),
            how="inner",
            on="protein_id"
        )
        .join(
            feature_descriptions,
            on="domain",
            how="left"
        )
    )

    label_columns = ["is_PUL_pred", "is_PUL_pulpy", "is_PUL"]
    titles = [
        f"Predicted PULs ({model_evaluator.model_name})",
        "PULpy PULs",
        "Experimental PULs"
    ]
    fig, axs = plt.subplots(2, 3, figsize=(16, 10), height_ratios=[3, 2])

    for i, label in enumerate(label_columns):
        # get stats
        stats_df = (
            analyze_domain_enrichment(features_with_predictions, label)
            .join(
                features_with_predictions.select(["domain", "name"]).unique(),
                on="domain",
                how="left"
            )
        )
        # convert to pandas for plotting
        pdf = stats_df.to_pandas()

        # scatter plot
        ax = axs[0, i]
        scatter = ax.scatter(
            pdf["log2_enrichment"],
            pdf["neglog10_fdr"],
            s=np.clip(pdf["positive_with_domain"], 5, 100), # use size to indicate how many PUL proteins have this domain
            alpha=0.65,
        )

        ax.axvline(x=-1, linestyle="--")
        ax.axvline(x=1, linestyle="--")
        ax.axhline(y=-np.log10(0.05), linestyle="--")
        significant = pdf[(pdf["neglog10_fdr"] > 100) & (np.abs(pdf["log2_enrichment"]) > 2)]

        for _, row in significant.iterrows():
            ax.text(
                row["log2_enrichment"],
                row["neglog10_fdr"],
                row["domain"],
                fontsize=8,
            )

        if i == 0:
            ax.set_ylabel("-log10(FDR)")
            ax.scatter([], [], s=10, label="Found in 10 PUL proteins", color="gray")
            ax.scatter([], [], s=50, label="Found in 50 PUL proteins", color="gray")
            ax.scatter([], [], s=100, label="Found in >100 PUL proteins", color="gray")
            ax.legend(loc="upper left")
    
        ax.set_xlabel("log2 enrichment")
        ax.set_title(titles[i])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # add bar plot of top domains
        bar_ax = axs[1, i]
        top_domains = (pdf.sort_values("neglog10_fdr", ascending=False).head(10).copy())
        # add label for name
        top_domains["label"] = np.where(
            top_domains["name"].notna(),
            top_domains["name"],
            top_domains["domain"]
        )
        top_domains = top_domains.iloc[::-1]
        bars = bar_ax.barh(
            top_domains["domain"],
            top_domains["neglog10_fdr"]
        )
        for bar, (_, row) in zip(bars, top_domains.iterrows()):
            width = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            x = 10
            ha = "left"
            color = "black"

            bar_ax.text(
                x,
                y,
                row["label"],
                va="center",
                ha=ha,
                fontsize=9,
                color=color
            )

        if i == 0:
            bar_ax.set_ylabel("Feature")

        bar_ax.set_xlabel("-log10(FDR)")
        if i == 1:
            bar_ax.set_title(f"Top enriched domains")

        bar_ax.spines["top"].set_visible(False)
        bar_ax.spines["right"].set_visible(False)

        # Smaller labels
        bar_ax.tick_params(axis='y', labelsize=7)

    fig.suptitle("Volcano plots of feature enrichment")
    fig.tight_layout()
    fig.savefig(os.path.join(model_evaluator.output_path, f"feature_enrichment_analysis.png"))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize feature (pfam/cazy) enrichment in model predictions"
    )
    parser.add_argument("--model", type=str, help="Name of model to evaluate", required=True)
    parser.add_argument("--features", type=str, help="Type of feature set to analyze. Options: pfam, cazy", default="pfam")
    args = parser.parse_args()
    main(args)
