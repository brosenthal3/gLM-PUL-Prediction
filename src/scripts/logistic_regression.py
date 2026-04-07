import multiprocessing
import os
from pathlib import Path

# import anndata  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import polars
import rich
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import (  # type: ignore
    average_precision_score,
    # f1_score,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV  # type: ignore
from tap import Tap
from tqdm import tqdm  # type: ignore

"""
export PYTHONPATH=$PYTHONPATH:'/g/scb/zeller/rhackett/GeneCat/src/'

cd /g/scb/zeller/rhackett/bacbench/genecat_evaluations/bacformer_benchmarks

#######################################
TASKDIR="/g/scb/zeller/rhackett/bacbench/genecat_evaluations/data/bgc_membership/data/bgc_data.embeddings"
TASKDIR="/home/ray/Documents/BacBench/genecat_evaluations/data/bacbench-essential-genes-protein-sequences/data/essential_genes.embeddings"
MODEL=gene_multilabel_untied_o1fknuqz_v0_context_embedding
DF=$TASKDIR/test.parquet
python ../../bacbench/tasks/binary_task/run_linear_cls.py --input-df-file-path $DF --output-dir $TASKDIR/results/ --model-name ${MODEL}_l2 --contig-col sequence_id --label-col is_BGC --norm-type l2 --gridsearch --normalize

"""

def prepare_labeled_genes_df(
        df: pd.DataFrame,
        embeddings_col: str,
        label_col: str
    ) -> pd.DataFrame:
    """Prepare the labeled genes DataFrame."""
    # check if the embeddings column is already in the correct format
    if isinstance(df[embeddings_col].iloc[0], np.ndarray):
        return df
    # if embeddings is List[List[np.ndarray]], we need to make it List[np.ndarray]
    if isinstance(df[embeddings_col].iloc[0], list):
        df[embeddings_col] = df[embeddings_col].apply(lambda x: x[0])
    # explode the DF
    df = df.explode([embeddings_col, label_col, "protein_id", "product", "start", "end"])
    return df


def get_linear_model(gridsearch: bool, n_jobs: int, random_state: int):
    """Returns an sklearn model you can fit"""
    if gridsearch:
        clf = LogisticRegression(
            solver="liblinear",
            max_iter=10000,
            fit_intercept=True,
            random_state=random_state,
        )
        param_grid = {
            "penalty": ["l1", "l2"],
            "C": [0.01, 0.1, 1, 10, 100, 1000],
        }
        # score = make_scorer(
        #     f1_score,
        #     average="micro",
        #     response_method="predict",
        # )
        score = make_scorer(
            average_precision_score,
            average="macro",
            response_method="predict_proba",
        )

        # GridSearch with 5-fold CV
        model = GridSearchCV(
            clf,
            param_grid,
            scoring=score,
            cv=5,
            n_jobs=n_jobs,
            refit=True,
            error_score="raise",
        )

    else:
        # NOTE
        # gridsearch of parameters showed that the optimal ones for BGC prediction were:
        # l2-lorn, C=1, with l2 penalty.
        # Possibly worth considering is the z-feature norm.
        # Cuction: See Note on z-feature
        # z-feature-norm, C=0.01, with l2 penalty.
        model = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            fit_intercept=True,
            max_iter=10000,
            C=1,
            random_state=random_state,
        )

    return model


def calculate_global_metrics(df: pd.DataFrame):
    """Calculate metrics for all"""
    probas = df["probas"].to_numpy()
    labels = df["label"].to_numpy()
    rich.print("AP:", average_precision_score(labels, probas))
    rich.print("ROC AUC:", roc_auc_score(labels, probas))


def calculate_metrics_per_genome(df: pd.DataFrame, contig_col: str) -> pd.DataFrame:
    """Calculate AUROC and AUPRC per genome."""
    gdf = df.groupby(contig_col)[["label", "probas"]].agg(list).reset_index()

    gdf["auroc"] = gdf.apply(
        lambda x: roc_auc_score(x["label"], x["probas"]), axis=1
    )
    gdf["auprc"] = gdf.apply(
        lambda x: average_precision_score(x["label"], x["probas"]), axis=1
    )
    print("Per genome metrics:")
    print("Mean AUROC:", gdf["auroc"].mean(), "Median AUROC:", gdf["auroc"].median())
    print("Mean AUPRC:", gdf["auprc"].mean(), "Median AUPRC:", gdf["auprc"].median())
    return gdf


def normalize_embeddings(
        df: pd.DataFrame,
        embedding_col: str,
        norm_type: str = 'l2'
    ) -> pd.DataFrame:
    """
    Normalize embeddings in a DataFrame column.

    Args:
        df: pandas DataFrame containing the embeddings.
        column: name of the column with embeddings (expects numpy arrays or lists).
        norm_type: type of normalization. Options:
            'l2'      - L2 norm per sample (unit vector)
            'z_feature' - z-score per feature (mean/std per column)
            'z_sample'  - z-score per sample (mean/std per row)

    Note:
        The z-feature norm is a bit problematic since it requires a mean
        and std over the samples. Since these are split across train and test,
        calculating these values over both sets could constitute a form of leakage.
        I need in effect prior knowlege of test features. Having seperate means
        and std for train and test would probably really cause a mess.
        If l2 works, its best to stick with that.

    Note:
        A sample norm will probably mess up and significantly degrade performance.
        If feature magnitude ever matters which is very likely in embeddings!

    Returns
    -------
        DataFrame with the normalized embeddings in the same column.
    """
    embeddings = np.stack(df[embedding_col].values) # shape: (num_samples, embedding_dim)

    if norm_type == 'l2':
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings_normed = embeddings / norms
    elif norm_type == 'z_feature':
        mean = embeddings.mean(axis=0, keepdims=True)
        std = embeddings.std(axis=0, keepdims=True) + 1e-8
        embeddings_normed = (embeddings - mean) / std
    elif norm_type == 'z_sample':
        mean = embeddings.mean(axis=1, keepdims=True)
        std = embeddings.std(axis=1, keepdims=True) + 1e-8
        embeddings_normed = (embeddings - mean) / std
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")

    df = df.copy()
    df[embedding_col] = list(embeddings_normed)
    return df


def main(
    input_df_file_path: str,
    output_dir: str,
    n_jobs: int = 1,
    normalize: bool = False,
    norm_type: str = "l2",
    gridsearch: bool = False,
    random_state: int = 1,
    embeddings_col: str = "embeddings",
    label_col: str = "label",
    contig_col: str = "genome_name",
):
    """Run the training of the Linear model."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # read input file
    df = pd.read_parquet(input_df_file_path)
    # explode the embeddings column as after embedding it is a list of lists
    df = prepare_labeled_genes_df(df, embeddings_col=embeddings_col, label_col=label_col)

    if normalize:
        df = normalize_embeddings(df, embedding_col=embeddings_col, norm_type=norm_type)

    genome2idx = {g: i for i, g in enumerate(df[contig_col].unique())}
    df["genome_idx"] = df[contig_col].map(genome2idx)

    # TODO whatever it is! make it binary
    if "true" in df[label_col].unique():
        df["label"] = df[label_col].map({"true": 1, "false": 0})
    elif "Yes" in df[label_col].unique():
        df["label"] = df[label_col].map({"Yes": 1, "No": 0})
    else:
        raise ValueError(f"unknown binary labels {df[label_col].unique()}")

    dim = df[embeddings_col].iloc[0].shape[0]
    rich.print(f"Embedding has {dim} dimensions")

    ############################## Create datasets #####################################

    # check the split column.
    if {"train", "test", "validation"} == set(df["split"].unique()):
        # join train and val because LogisticRegression is convex optimization
        # no validation needed.
        train_df = df[(df["split"] == "train") | (df["split"] == "validation")].copy()
    elif {"train", "test"} == (set(df["split"].unique())):
        train_df = df[df["split"] == "train"].copy()
    else:
        raise ValueError(f"Unexpected splits {df['split'].unique()}")
    test_df = df[df["split"] == "test"].copy()

    ############################ train model ###########################################

    model = get_linear_model(
        gridsearch=gridsearch,
        n_jobs=n_jobs,
        random_state=random_state
    )
    model.fit(X=np.stack(train_df[embeddings_col].tolist()), y=train_df["label"].to_numpy())

    if gridsearch:
        rich.print(
            f"Best scores are {model.best_params_} with macro-ap score {model.best_score_}"
        )
        # write gridsearch results to tsv
        df = (
            polars.from_dict(
                model.cv_results_,
                strict=False,
            )
            .select(polars.exclude("^split.*$"))
            .drop(["mean_score_time", "std_score_time", "params"])
        )
        df.write_csv(
            file=Path(output_dir, "gridsearch_results.tsv"),
            include_header=True,
            separator="\t",
        )
        model = model.best_estimator_

    # # optionally dump the model
    # joblib.dump(model, Path(output, "single_gene_model.pkl"))

    ########################## predict on test set #####################################

    test_probas = model.predict_proba(X=np.stack(test_df[embeddings_col].tolist()))[:, 1]

    test_df["probas"] = test_probas

    calculate_global_metrics(df=test_df)

    genome_df = calculate_metrics_per_genome(test_df, contig_col=contig_col)

    return test_df, genome_df


class ArgumentParser(Tap):
    """Argument parser for sklearn logistic regeression model."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    input_df_file_path: str
    output_dir: str
    n_jobs: int = max(0, len(os.sched_getaffinity(0)) - 1)
    normalize: bool = False
    embeddings_col: str = "embeddings"
    model_name: str | None = None
    label_col: str = "essential"
    contig_col: str = "genome_name"
    norm_type: str = "l2"
    gridsearch: bool = False


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    args = ArgumentParser().parse_args()
    output = []
    output_genome = []
    for random_state in tqdm([1]):
        print(f"Running state {random_state}")
        test_df, genome_df = main(
            input_df_file_path=args.input_df_file_path,
            output_dir=args.output_dir,
            n_jobs=args.n_jobs,
            normalize=args.normalize,
            embeddings_col=args.embeddings_col,
            label_col=args.label_col,
            contig_col=args.contig_col,
            norm_type=args.norm_type,
            gridsearch=args.gridsearch,
        )
        test_df = test_df[[args.contig_col, "genome_idx", "protein_id", "probas", "label", args.label_col]]
        test_df["random_state"] = random_state
        genome_df["random_state"] = random_state
        output.append(test_df)
        output_genome.append(genome_df)

    output_df = pd.concat(output)
    output_df.to_parquet(os.path.join(args.output_dir, f"linmodel_results_{args.model_name}.parquet"))

    output_genome_df = pd.concat(output_genome)
    output_genome_df.to_parquet(os.path.join(args.output_dir, f"linmodel_genome_metric_results_{args.model_name}.parquet"))


    # ########################## save test predictions ###################################

    # adata_test = anndata.AnnData(
    #     X=test_probas,
    #     obs=test_df,
    #     uns={
    #         "label_type": label_col,
    #         "linear_model_info": model.get_params(),
    #         "data": input_df_file_path,
    #     },
    # )
    # filename = Path(output_dir, f"results.{args.kegg_type}.test.h5ad")
    # adata_test.write(
    #     filename=filename,
    #     compression="gzip",
    # )
    #

    rich.print(f"[bold blue]{'Saving test evaluation to':>12}[/] {args.output_dir}")