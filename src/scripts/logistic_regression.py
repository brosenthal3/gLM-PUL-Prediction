"""
Run logistic regression on gene embeddings.
"""

import multiprocessing
import os
from pathlib import Path
import numpy as np
import pandas as pd  # type: ignore
import polars
import rich
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import (  # type: ignore
    average_precision_score,
    make_scorer,
    roc_auc_score,
    matthews_corrcoef
)
from sklearn.model_selection import GridSearchCV, train_test_split  # type: ignore
from tap import Tap
from tqdm import tqdm  # type: ignore
from utility_scripts import join_gene_and_PUL_table
import pickle


def prepare_labeled_genes_df(df: pd.DataFrame, embeddings_col: str, label_col: str) -> pd.DataFrame:
    """Prepare the labeled genes DataFrame."""
    df = df.dropna(subset=embeddings_col)    
    # check if the embeddings column is already in the correct format
    if isinstance(df[embeddings_col].iloc[0], np.ndarray):
        df[embeddings_col] = df[embeddings_col].apply(
            lambda x: np.nan_to_num(x, nan=0.0)
        )
        return df
    # if embeddings is List[List[np.ndarray]], we need to make it List[np.ndarray]
    if isinstance(df[embeddings_col].iloc[0], list):
        df[embeddings_col] = df[embeddings_col].apply(lambda x: x[0])
    # explode the DF
    df = df.explode([embeddings_col, label_col, "protein_id"])
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
            "l1_ratio": [0, 1],
            "C": [1, 10, 500],
        }
        score = make_scorer(
            average_precision_score,
            average="macro",
            response_method="predict_proba",
        )

        # GridSearch with 3-fold CV, reduce complexity
        model = GridSearchCV(
            clf,
            param_grid,
            scoring=score,
            cv=3,
            n_jobs=n_jobs,
            refit=True,
            error_score="raise",
        )

    else:
        model = LogisticRegression(
            l1_ratio=0,
            solver="liblinear",
            fit_intercept=True,
            max_iter=10000,
            C=10, # found to be optimal by gridsearch (for most models)
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

    gdf["auroc"] = gdf.apply(lambda x: roc_auc_score(x["label"], x["probas"]), axis=1)
    gdf["auprc"] = gdf.apply(lambda x: average_precision_score(x["label"], x["probas"]), axis=1)

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

def calculate_optimal_threshold(train_df, embeddings_col, random_state, model):
    rich.print("Computing optimal threshold using MCC...")
    # split into validation (10%)
    train_df_split, validation_df = train_test_split(train_df, stratify=train_df["label"].to_numpy(), test_size=0.2, random_state=random_state)
    model.fit(X=np.stack(train_df_split[embeddings_col].tolist()), y=train_df_split["label"].to_numpy())
    
    # predict on validation df
    validaton_probas = model.predict_proba(X=np.stack(validation_df[embeddings_col].tolist()))[:, 1]
    true_labels = validation_df["label"].to_numpy()

    # calculate best threshold based on MCC
    thresholds = mcc_thresholds = np.linspace(0, 0.75, num=20) # assume threshold is below 0.75
    mmc_scores = []
    for threshold in tqdm(thresholds, desc="Calculating MCC for thresholds"):
        binary_pred = [1 if p >= threshold else 0 for p in validaton_probas]
        mmc = matthews_corrcoef(true_labels, binary_pred)
        mmc_scores.append(mmc)
    
    return model, thresholds[np.argmax(mmc_scores)]
    

def main(
    input_df_file_path: str,
    output_dir: str,
    n_jobs: int = 1,
    normalize: bool = False,
    norm_type: str = "l2",
    gridsearch: bool = False,
    random_state: int = 1,
    embeddings_col: str = "embedding",
    label_col: str = "label",
    contig_col: str = "genome_name",
    mask_cryptic_puls: bool = False
):
    """Run the training of the Linear model."""
    os.makedirs(output_dir, exist_ok=True)

    # read input file
    df_polars = polars.read_parquet(input_df_file_path)
    df = df_polars.to_pandas()
    # explode the embeddings column as after embedding it is a list of lists
    df = prepare_labeled_genes_df(df, embeddings_col=embeddings_col, label_col=label_col)
    # read cryptic PULs
    cryptic_puls = pd.read_csv("src/data/data_collection/cryptic_puls_genes.tsv", sep='\t')

    if normalize:
        df = normalize_embeddings(df, embedding_col=embeddings_col, norm_type=norm_type)

    genome2idx = {g: i for i, g in enumerate(df[contig_col].unique())}
    df["genome_idx"] = df[contig_col].map(genome2idx)

    # Binarize labels
    if "True" in df[label_col].unique():
        df["label"] = df[label_col].map({"True": 1, "False": 0})
    elif "Yes" in df[label_col].unique():
        df["label"] = df[label_col].map({"Yes": 1, "No": 0})
    elif True in df[label_col].unique():
        df['label'] = df[label_col].map({True: 1, False: 0})
    else:
        raise ValueError(f"unknown binary labels {df[label_col].unique()}")

    dim = df[embeddings_col].iloc[0].shape[0]
    rich.print(f"Embedding has {dim} dimensions")

    ############################## Create datasets #####################################

    if {"train", "test"} == (set(df["split"].unique())):
        train_df = df[df["split"] == "train"].copy()
    else:
        raise ValueError(f"Unexpected splits {df['split'].unique()}")
    test_df = df[df["split"] == "test"].copy()

    ############################ train model ###########################################

    model = get_linear_model(gridsearch=gridsearch, n_jobs=n_jobs, random_state=random_state)
    if mask_cryptic_puls:
        train_df = train_df[~train_df["protein_id"].isin(cryptic_puls["protein_id"])]
    
    # if only training once, split 10% off for validation, to calculate the optimal threshold
    rich.print("Training model...")
    if not gridsearch:
        model, threshold = calculate_optimal_threshold(train_df, embeddings_col, random_state, model)
        rich.print(f"Found optimal threshold at {threshold}")
    else:
        # train on full dataset
        model.fit(X=np.stack(train_df[embeddings_col].tolist()), y=train_df ["label"].to_numpy())
        threshold = 0.25


    if gridsearch:
        rich.print(f"Best scores are {model.best_params_} with macro-ap score {model.best_score_}")
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

    ########################## predict on test set #####################################

    test_probas = model.predict_proba(X=np.stack(test_df[embeddings_col].tolist()))[:, 1]
    train_probas = model.predict_proba(X=np.stack(train_df[embeddings_col].tolist()))[:, 1]

    test_df["probas"] = test_probas
    train_df["probas"] = train_probas
    rich.print("Test set evaluation:")
    calculate_global_metrics(df=test_df)

    rich.print("Train set evaluation:")
    calculate_global_metrics(df=train_df)

    return test_df, train_df, model, threshold


def save_results(clusters, genecat_results, genes, fold, output_dir, split="test", threshold=0.25):
    # get only genes in test set
    test_genes = (genes.join(clusters, on="sequence_id", how="semi"))

    # join genes with test clusters and predicted clusters
    cols = ["protein_id", "sequence_id", "cluster_id", "is_PUL", "start", "end"]
    labeled_test_genes = join_gene_and_PUL_table(test_genes, clusters).select(cols)

    # join gene tables of predicted clusters with test clusters
    labeled_table = (
        labeled_test_genes
        .join(genecat_results.select("protein_id", "probas").rename({"probas": "average_p"}), on="protein_id", how="left")
        .with_columns(
            polars.when(polars.col("is_PUL").is_null()).then(False).otherwise(polars.col("is_PUL")).alias("is_PUL"),
            polars.when(polars.col("average_p").ge(threshold)).then(True).otherwise(False).alias("is_PUL_pred"),
        )
        .sort("protein_id")
        .sort("sequence_id")
    )

    labeled_table.write_csv(output_dir + f"/labeled_results_{split}_{fold}.tsv", separator='\t')


def save_model(model, fold, output_dir):
    output_dir = output_dir + "/models"
    os.makedirs(output_dir, exist_ok=True)

    # if gridsearch, save best estimator
    if hasattr(model, "best_estimator_"):
        model_to_save = model.best_estimator_
    else:
        model_to_save = model

    # pickle model
    model_path = f"{output_dir}/fold_{fold}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_to_save, f)

    # save coeffs and intercept
    np.save(f"{output_dir}/coef_fold_{fold}.npy", model_to_save.coef_)
    np.save(f"{output_dir}/intercept_fold{fold}.npy", model_to_save.intercept_)


class ArgumentParser(Tap):
    """Argument parser for sklearn logistic regeression model."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    input_df_file_path: str
    output_dir: str
    n_jobs: int = -1
    normalize: bool = False
    embeddings_col: str = "embedding"
    model_name: str | None = None
    label_col: str = "label"
    contig_col: str = "sequence_id"
    norm_type: str = "l2"
    gridsearch: bool = False
    k: int = 7
    mask_cryptic_puls: bool = False
    save_results: str = "True"


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    args = ArgumentParser().parse_args()
    genes = polars.read_parquet("src/data/genecat_output/genome.genes.parquet")
    output_dir = args.output_dir
    thresholds = {"fold": [], "threshold": []}

    random_state = 1
    print(f"Running state {random_state}")
    
    # run all folds
    for fold in tqdm(range(args.k)):
        rich.print(f"[bold blue]Running fold {fold}...[/]")
        input_df_file_path = args.input_df_file_path + f"/fold_{fold}_data.parquet"
        output = []
        test_df, train_df, model, threshold = main(
            input_df_file_path=input_df_file_path,
            output_dir=output_dir,
            n_jobs=args.n_jobs,
            normalize=args.normalize,
            embeddings_col=args.embeddings_col,
            label_col=args.label_col,
            contig_col=args.contig_col,
            norm_type=args.norm_type,
            gridsearch=args.gridsearch,
            mask_cryptic_puls=args.mask_cryptic_puls
        )
        # make train and test dfs
        test_df = test_df[[args.contig_col, "genome_idx", "protein_id", "probas", args.label_col]]
        test_df["random_state"] = random_state
        train_df["random_state"] = random_state
        output.append(test_df)
        # record optimal threshold for this fold
        thresholds["fold"].append(fold)
        thresholds["threshold"].append(threshold)

        if args.save_results == "True":
            genecat_results = pd.concat(output)
            genecat_results = polars.from_pandas(genecat_results)
            rich.print(f"[bold blue]{'Saving test evaluation to':>12}[/] {args.output_dir}")
            
            # get all genes in test set
            test_clusters = polars.read_csv(f"src/data/splits/test_fold_{fold}.tsv", separator='\t')
            save_results(test_clusters, genecat_results, genes, fold, output_dir, threshold=threshold)
            # get all genes in train set
            train_clusters = polars.read_csv(f"src/data/splits/train_fold_{fold}.tsv", separator='\t')
            save_results(train_clusters, genecat_results, genes, fold, output_dir, split="train", threshold=threshold)

            try:
                save_model(model, fold, output_dir)
            except Exception as e:
                print("model could not be saved for some reason:")
                print(e)

    # save thresholds for all folds
    rich.print(f"[bold blue]{'Saving threshold values to':>12}[/] {args.output_dir}")
    thresholds = polars.DataFrame(thresholds).write_csv(f"{args.output_dir}/thresholds.tsv", separator="\t")

"""
### ALL PULs ###

# for genecat pfam:
python src/scripts/logistic_regression.py --input-df-file-path src/data/results/genecat_zeroshot_pfam/fold_data --output-dir src/data/results/genecat_zeroshot_pfam --model-name pfam --norm-type l2 --normalize

# for genecat cazy:
python src/scripts/logistic_regression.py --input-df-file-path src/data/results/genecat_zeroshot_cazy/fold_data --output-dir src/data/results/genecat_zeroshot_cazy --model-name cazy --norm-type l2 --normalize

# for ESM-C:
python src/scripts/logistic_regression.py --input-df-file-path src/data/results/esmc/fold_data --output-dir src/data/results/esmc --model-name esmc --norm-type l2 --normalize --embeddings-col embedding

# for Bacformer:
python src/scripts/logistic_regression.py --input-df-file-path src/data/results/bacformer/fold_data --output-dir src/data/results/bacformer --model-name bacformer --norm-type l2 --normalize --embeddings-col embedding

### MASKED CRYPTIC PULs ###

# for genecat pfam
python src/scripts/logistic_regression.py --input-df-file-path src/data/results/genecat_zeroshot_pfam/fold_data --output-dir src/data/results/genecat_zeroshot_pfam_masked --model-name pfam_masked --norm-type l2 --normalize --mask-cryptic-puls

# for genecat cazy
python src/scripts/logistic_regression.py --input-df-file-path src/data/results/genecat_zeroshot_cazy/fold_data --output-dir src/data/results/genecat_zeroshot_cazy_masked --model-name cazy_masked --norm-type l2 --normalize --mask-cryptic-puls

# for ESM-C:
python src/scripts/logistic_regression.py --input-df-file-path src/data/results/esmc/fold_data --output-dir src/data/results/esmc_masked --model-name esmc_masked --norm-type l2 --normalize --embeddings-col embedding

# for Bacformer:
python src/scripts/logistic_regression.py --input-df-file-path src/data/results/bacformer/fold_data --output-dir src/data/results/bacformer_masked --model-name bacformer_masked --norm-type l2 --normalize --embeddings-col embedding

"""
