from __future__ import annotations

import argparse
import os
from pathlib import Path

from genecat import __package__ as _module
from genecat import __version__
from genecat.model.defaults import (FinetuneOptimizerConfig,
                                    PreTrainMultilabelReconstructionConfig,
                                    PreTrainReconstructionConfig)


def configure_general_arguments(
    parser: argparse.ArgumentParser,
) -> argparse._ArgumentGroup:

    group = parser.add_argument_group()

    group.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"{_module} {__version__}",
        help="Show the program version number and exit.",
    )
    group.add_argument(
        "-j",
        "--jobs",
        default=max(0, len(os.sched_getaffinity(0))),  # works with slurms cpus-per-task
        type=int,
        help="The number of jobs to run in parallel sections.",
    )
    group.add_argument(
        "--seed",
        default=42,
        type=int,
        help="The seed to use for initializing pseudo-random number generators.",
    )

    return group


def configure_preprocessing(
    parser: argparse.ArgumentParser,
) -> argparse._ArgumentGroup:

    group = parser.add_argument_group(
        title="Preprocessing Arguments",
        description="Prepare a GeneCAT sql database from fna or genbank files.",
    )
    group.add_argument(
        "-i",
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="Input fna fasta or genbank files.",
    )
    group.add_argument(
        "-v",
        "--vocab",
        type=Path,
        required=False,
        help="Vocab file for Pfams. This is required to build a database.",
    )
    group.add_argument(
        "--hmms",
        type=Path,
        required=True,
        help="Path to hmms to use for domain annotation.",
    )
    group.add_argument(
        "--hmm-type",
        choices=("pfam", "cazy", "other"),
        required=False,
        default="pfam",
        help="The type of hmms passed via --hmms flag.",
    )
    group.add_argument(
        "-o", "--output-dir", type=Path, required=True, help="Output Directory"
    )
    group.add_argument(
        "--output-name",
        type=str,
        required=False,
        default="genome",
        help="The stem to give to the output file(s)",
    )
    group.add_argument(
        "-f",
        "--format",
        choices=("fasta", "genbank", "faa+gff"),
        required=False,
        default=None,
        help="The file format of the input. Optional. Will detect fasta, genbank and embl files.",
    )
    group.add_argument(
        "--min-record-length",
        type=int,
        required=False,
        default=0,
        help="Minimum record length to pass loading.",
    )
    group.add_argument(
        "--min-gene-number",
        type=int,
        required=False,
        default=0,
        help="Minimum number of genes a sequence must have during gene calling. Does not apply to CDS features!",
    )
    group.add_argument(
        "--cds-feature",
        type=str,
        required=False,
        help="Extract genes from annotated records using a feature rather than running de-novo gene-calling.",
    )
    group.add_argument(
        "--call-genes",
        action="store_true",
        help="Wether to force re-calling the genes with pyrodigal. Mutually exclusive with cds-feature",
    )
    group.add_argument(
        "--locus-tag",
        type=str,
        required=False,
        default="locus_tag",
        help=" The name of the feature qualifier to use for naming extracted genes when using the ``--cds-feature`` flag.",
    )
    group.add_argument(
        "-p",
        "--p-filter",
        type=float,
        required=False,
        default=1e-09,
        help="The p-value cutoff for protein domains to be included.",
    )
    group.add_argument(
        "--index-db",
        action="store_true",
        help="Wether to index the returned sql database. False by default to index after merge.",
    )
    group.add_argument(
        "--write-tables",
        action="store_true",
        help="Wether to write gene and feature tables as parquet files too",
    )
    group.add_argument(
        "--domain-number",
        required=False,
        type=int,
        help="Number of feature domains per gene to generate Vocab from. Default all",
    )
    group.add_argument(
        "--unique-domains",
        action="store_true",
        help="Consider only unique domains per Gene for building the Vocab. (Affects db creation too!)",
    )

    return group


def configure_merging_databases(
    parser: argparse.ArgumentParser,
) -> argparse._ArgumentGroup:

    group = parser.add_argument_group(
        title="Merge Database Arguments",
        description="Merge a list of genome sql databases into a new larger one.",
    )

    group.add_argument(
        "-i",
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="Input sql databases with genome schema",
    )

    group.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to the output database to write",
    )

    return group


def configure_pretraining_input(
    parser: argparse.ArgumentParser,
) -> argparse._ArgumentGroup:

    group = parser.add_argument_group(
        "Input", "Mandatory input files required by the command."
    )
    group.add_argument(
        "-i", "--input", required=True, type=Path, help="The database to train on."
    )
    group.add_argument(
        "-t",
        "--test",
        required=False,
        default=None,
        type=Path,
        help="The database to test on.",
    )
    group.add_argument(
        "--resume",
        required=False,
        type=Path,
        help="Path to model checkpoint to resume training from.",
    )
    group.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="The path where to write the trained model",
    )

    return group


def configure_PreTrainModule(
    parser: argparse.ArgumentParser,
) -> argparse._ArgumentGroup:

    group = parser.add_argument_group(
        "Gene Reconstruction Module", "Optional arguments to configure the Transformer"
    )
    group.add_argument(
        "--gene-count",
        type=int,
        default=PreTrainReconstructionConfig.gene_count,
        help="Set the number of genes per window",
    )
    group.add_argument(
        "--domain-count",
        type=int,
        default=PreTrainReconstructionConfig.domain_count,
        help="Set the number of domains per window",
    )
    group.add_argument(
        "--domain-order",
        type=str,
        choices=["domain_start", "pvalue_rank"],
        default=PreTrainReconstructionConfig.domain_order,
        help="Determine the order of domain tokens in genes.",
    )
    group.add_argument(
        "--d-model",
        type=int,
        default=PreTrainReconstructionConfig.d_model,
        help="Set the dimensionality of the model",
    )
    group.add_argument(
        "--num-layers",
        type=int,
        default=PreTrainReconstructionConfig.num_layers,
        help="Set the number of transformer layers",
    )
    group.add_argument(
        "--encoding-dropout",
        type=float,
        default=PreTrainReconstructionConfig.encoding_dropout,
        help="Set the dropout rate for encoding layers",
    )
    group.add_argument(
        "--transformer-dropout",
        type=float,
        default=PreTrainReconstructionConfig.transformer_dropout,
        help="Set the dropout rate for the transformer layers",
    )
    group.add_argument(
        "--mask-probability",
        type=float,
        default=PreTrainReconstructionConfig.mask_probability,
        help="Set the probability of masking during training",
    )
    group.add_argument(
        "--untie-weights",
        action="store_false",
        default=PreTrainReconstructionConfig.tie_weights,
        help="Tie weights of embedding and output layers",
    )
    group.add_argument(
        "--position-unit",
        type=str,
        choices=["gene", "domain", "both"],
        default=PreTrainReconstructionConfig.position_unit,
        help="Set the unit for position encoding: 'gene' or 'domain'",
    )
    group.add_argument(
        "--max-lr",
        type=float,
        default=PreTrainReconstructionConfig.max_lr,
        help="Set the maximum learning rate",
    )
    group.add_argument(
        "--base-lr",
        type=float,
        default=PreTrainReconstructionConfig.base_lr,
        help="Set the base learning rate",
    )
    group.add_argument(
        "--pca-start",
        type=float,
        default=PreTrainReconstructionConfig.pca_start,
        help="Set the PCA starting threshold",
    )
    group.add_argument(
        "--multilabel",
        action="store_true",
        help="Use a multilabel BCEWithLogitsLoss to train the model",
    )
    # Add additonal arguments which configure the Trainer:
    group.add_argument(
        "--matmul-precision", choices=["highest", "high", "medium"], default="medium"
    )
    group.add_argument(
        "--batch-size", type=int, default=512, help="The batch size for training."
    )
    group.add_argument(
        "--tune-batch-size",
        action="store_true",
        help="Tune batch size and overwrite passed batch size. Does not support DDP. Single GPU only",
    )
    group.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to train for"
    )

    return group


def configure_PreTrainMultilabelModule(
    parser: argparse.ArgumentParser,
) -> argparse._ArgumentGroup:

    group = parser.add_argument_group(
        "Gene Reconstruction Module", "Optional arguments to configure the Transformer"
    )
    group.add_argument(
        "--gene-count",
        type=int,
        default=PreTrainMultilabelReconstructionConfig.gene_count,
        help="Set the number of genes per window",
    )
    group.add_argument(
        "--d-model",
        type=int,
        default=PreTrainMultilabelReconstructionConfig.d_model,
        help="Set the dimensionality of the model",
    )
    group.add_argument(
        "--num-layers",
        type=int,
        default=PreTrainMultilabelReconstructionConfig.num_layers,
        help="Set the number of transformer layers",
    )
    group.add_argument(
        "--zero-inflate",
        action="store_true",
        help="Wether to zero-inflate the output logits to encourage learning number of labels per gene",
    )
    group.add_argument(
        "--encoding-dropout",
        type=float,
        default=PreTrainMultilabelReconstructionConfig.encoding_dropout,
        help="Set the dropout rate for encoding layers",
    )
    group.add_argument(
        "--transformer-dropout",
        type=float,
        default=PreTrainMultilabelReconstructionConfig.transformer_dropout,
        help="Set the dropout rate for the transformer layers",
    )
    group.add_argument(
        "--full-mask-proba",
        type=float,
        default=PreTrainMultilabelReconstructionConfig.full_mask_proba,
        help="Set the probability of masking genes fully during training",
    )
    group.add_argument(
        "--partial-mask-proba",
        type=float,
        default=PreTrainMultilabelReconstructionConfig.partial_mask_proba,
        help="Set the probability of masking genes partially during training",
    )
    group.add_argument(
        "--untie-weights",
        action="store_false",
        default=PreTrainMultilabelReconstructionConfig.tie_weights,
        help="Untie weights of embedding and output layers",
    )
    group.add_argument(
        "--max-lr",
        type=float,
        default=PreTrainMultilabelReconstructionConfig.max_lr,
        help="Set the maximum learning rate",
    )
    group.add_argument(
        "--base-lr",
        type=float,
        default=PreTrainMultilabelReconstructionConfig.base_lr,
        help="Set the base learning rate",
    )
    group.add_argument(
        "--pca-start",
        type=float,
        default=PreTrainMultilabelReconstructionConfig.pca_start,
        help="Set the PCA starting threshold",
    )
    # Add additonal arguments which configure the Trainer:
    group.add_argument(
        "--matmul-precision", choices=["highest", "high", "medium"], default="medium"
    )
    group.add_argument(
        "--batch-size", type=int, default=512, help="The batch size for training."
    )
    group.add_argument(
        "--tune-batch-size",
        action="store_true",
        help="Tune batch size and overwrite passed batch size. Does not support DDP. Single GPU only",
    )
    group.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to train for"
    )

    return group


def configure_WeightsAndBiases(
    parser: argparse.ArgumentParser,
) -> argparse._ArgumentGroup:

    group = parser.add_argument_group(
        "Weights and Biases",
        "Optional arguments to configure logging to Weights and Biases",
    )
    group.add_argument(
        "--offline",
        action="store_true",
        help="Do not upload logging results to Weights and Biases. Instead store only locally.",
    )
    group.add_argument(
        "--name",
        required=True,
        type=str,
        help="Descriptive name for the run on Weights and Biases.",
    )
    group.add_argument(
        "--tags",
        nargs="+",
        default=[],
        help="Tags to add to the run on Weights and Biases.",
    )

    return group


def configure_split_genetable_input(
    parser: argparse.ArgumentParser,
) -> argparse._ArgumentGroup:

    group = parser.add_argument_group(
        "Train-Test Split gene-table",
        "Arguments to perform a 5-fold CV train-test split by taxonomy",
    )
    group.add_argument(
        "-g",
        "--gene-table",
        required=True,
        type=Path,
        help="Path to Gene Table containing PG3 representatives",
    )
    group.add_argument(
        "-o",
        "--outdir",
        required=True,
        type=Path,
        help="Output Directory to which split Gene and Feature Tables will be written",
    )
    # group.add_argument(
    #     "--min-gene-number",
    #     required=False,
    #     default=16,
    #     type=int,
    #     help="The minimum number of genes per contig. Contigs with fewer genes are dropped.",
    # )
    group.add_argument(
        "--tax-level",
        choices={
            "domain",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "species",
        },
        required=True,
        type=str,
        help="Taxonomic level on which to split the data",
    )
    group.add_argument(
        "--create-subfolds",
        action="store_true",
        help="Wether to create 5-fold CV subfolds for each pretraining fold.",
    )
    return group


def configure_build_vocab_input(
    parser: argparse.ArgumentParser,
) -> argparse._ArgumentGroup:

    group = parser.add_argument_group(
        "Build Vocab",
        "Arguments to build a Vocab of PFAMs from a gene and a feature table",
    )
    group.add_argument(
        "-g",
        "--genes",
        required=False,
        help="If provided, only a Vocab for the features of the supplied genes will be generated",
        type=Path,
    )
    group.add_argument(
        "-f",
        "--features",
        required=True,
        help="Path to the PFAM features table as .tsv or .parquet file",
        type=Path,
    )
    group.add_argument(
        "--domain-number",
        required=False,
        type=int,
        help="Number of feature domains per gene to generate Vocab from. Default all",
    )
    group.add_argument(
        "--unique-domains",
        action="store_true",
        help="Consider only unique domains per Gene for building the Vocab. (Affects db creation too!)",
    )
    group.add_argument(
        "--min-domain-occurence",
        default=0,
        type=int,
        help="Minimum occurence of a pfam domain to be included in Vocab. Default no minimum.",
    )
    group.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to the output vocab to write",
        type=Path,
    )
    group.add_argument(
        "--genome-database",
        required=False,
        help="Optional. Also requires genes. If passed will write a GenomeDatabase too!",
        type=Path,
    )

    return group


def configure_build_database_input(
    parser: argparse.ArgumentParser,
) -> argparse._ArgumentGroup:

    group = parser.add_argument_group(
        "Build Database",
        "Arguments to build a genome SQL database on which the GeneCAT Transformer operates",
    )
    group.add_argument(
        "-f",
        "--features",
        required=True,
        help="Path to the PFAM features table as .tsv or .parquet file",
        type=Path,
    )
    group.add_argument(
        "-g",
        "--genes",
        required=True,
        help="Path to the genes table as .tsv or .parquet file",
        type=Path,
    )
    group.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to the output database to write",
        type=Path,
    )
    group.add_argument(
        "--vocab",
        required=True,
        help="Path to the input vocab to encode domains with",
        type=Path,
    )
    group.add_argument(
        "--domain-number",
        required=False,
        type=int,
        help="Number of feature domains per gene to generate Vocab from. Default all",
    )
    group.add_argument(
        "--unique-domains",
        action="store_true",
        help="Consider only unique domains per Gene for building the Vocab. (Affects db creation too!)",
    )

    return group


def configure_build_labled_database_input(
    parser: argparse.ArgumentParser,
) -> argparse._ArgumentGroup:

    group = parser.add_argument_group(
        "Build Database",
        "Arguments to build a genome SQL database on which the GeneCAT Transformer operates",
    )
    group.add_argument(
        "-g",
        "--genes",
        required=True,
        help="Path to the genes table as .tsv or .parquet file",
        type=Path,
    )
    group.add_argument(
        "--pfams",
        required=True,
        help="Path to the PFAM features table as .tsv or .parquet file",
        type=Path,
    )
    group.add_argument(
        "--kofams",
        required=True,
        help="Path to the KOfam features table as .tsv or .parquet file",
        type=Path,
    )
    group.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to the output database to write",
        type=Path,
    )
    group.add_argument(
        "--pfam-vocab",
        required=True,
        help="Path to the input vocab to encode domains with",
        type=Path,
    )
    group.add_argument(
        "--kofam-vocab",
        required=True,
        help="Path to the kofam vocab to encode kofams with",
        type=Path,
    )
    group.add_argument(
        "--n-samples",
        required=False,
        help="How many kofam annotated genes to sample.",
        type=int,
    )

    return group


def configure_bgc_finetuning_input(
    parser: argparse.ArgumentParser,
) -> argparse._ArgumentGroup:

    group = parser.add_argument_group(
        "FineTune BGC model",
        "Arguments to finetune a binary BGC membership prediction model",
    )
    group.add_argument(
        "-g",
        "--gene-table",
        required=True,
        type=Path,
        help="The gene table finetune on.",
    )
    group.add_argument(
        "-d",
        "--domain-table",
        required=True,
        type=Path,
        help="The Pfam domain feature table finetune on.",
    )
    group.add_argument(
        "-c",
        "--cluster-table",
        required=True,
        type=Path,
        help="The cluster table finetune on.",
    )
    group.add_argument(
        "-m",
        "--model",
        required=True,
        type=Path,
        help="Path to foundation PreTrained Model to finetune",
    )
    group.add_argument(
        "--task-type",
        choices=["binary", "multiclass"],
        default="binary",
        help="'binary' for finetuning a BGC prediction model, 'multiclass' for finetuning on BGC type",
    )
    group.add_argument(
        "--resume",
        required=False,
        type=Path,
        help="Path to model checkpoint to resume training from.",
    )
    group.add_argument(
        "--vocab",
        required=True,
        type=Path,
        help="The Vocab of the Database on which the model was trained on.",
    )
    group.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="The path where to write the tested model",
    )
    return group


def configure_bgc_finetuneing_test(
    parser: argparse.ArgumentParser,
) -> argparse._ArgumentGroup:

    group = parser.add_argument_group(
        "Configure BGC Finetuning",
        "Optional Arguments to perform testing of finetuned model",
    )
    group.add_argument(
        "--test-gene-table",
        required=False,
        type=Path,
        help="The gene table to test on.",
    )
    group.add_argument(
        "--test-domain-table",
        required=False,
        type=Path,
        help="The domain table to test on.",
    )
    group.add_argument(
        "--test-cluster-table",
        required=False,
        type=Path,
        help="The cluster table to test on.",
    )

    return group


def configure_FineTuneModel(
    parser: argparse.ArgumentParser,
) -> argparse._ArgumentGroup:

    group = parser.add_argument_group(
        "FineTuneModel", "Optional Arguments to configure a FineTuneModel"
    )
    parser.add_argument(
        "--untrained",
        action="store_true",
        help="Do not load model weights but use default parameters instead.",
    )
    group.add_argument(
        "--frozen",
        action="store_true",
        help="If set, the weights of the pretrained model will be frozen during finetuning",
    )
    group.add_argument(
        "--max-lr",
        type=float,
        default=FinetuneOptimizerConfig.max_lr,
        help="Set the maximum learning rate",
    )
    group.add_argument(
        "--base-lr",
        type=float,
        default=FinetuneOptimizerConfig.base_lr,
        help="Set the base learning rate",
    )
    group.add_argument(
        "--pca-start",
        type=float,
        default=FinetuneOptimizerConfig.pca_start,
        help="Set the PCA starting threshold",
    )
    group.add_argument(
        "--dropout",
        type=float,
        default=FinetuneOptimizerConfig.dropout,
        help="Set the dropout",
    )
    group.add_argument(
        "--middle-focus",
        action="store_true",
        help="If set, the model will only learn on the middle gene in each window",
    )

    # Add additonal arguments which configure the Trainer:
    parser.add_argument(
        "--matmul-precision", choices=["highest", "high", "medium"], default="medium"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="The batch size.")
    group.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to train for"
    )
    return group


def configure_extract_embeddings_IO(
    parser: argparse.ArgumentParser,
) -> argparse._ArgumentGroup:

    group = parser.add_argument_group(
        "IO Extract Embeddings", "Required Arguments to Extract Embeddings"
    )
    group.add_argument(
        "-i",
        "--input",
        required=False,
        type=Path,
        help="Path to SQL DB with PFAM annotated Genes",
    )
    group.add_argument(
        "-g",
        "--genes",
        required=False,
        help="If provided, only a Vocab for the features of the supplied genes will be generated",
        type=Path,
    )
    group.add_argument(
        "-f",
        "--features",
        required=False,
        help="Path to the PFAM features table as .tsv or .parquet file",
        type=Path,
    )
    group.add_argument(
        "--vocab",
        required=False,
        help="Path to the input vocab to encode domains with",
        type=Path,
    )
    group.add_argument(
        "-l",
        "--layer",
        required=False,
        help="Which type of embeddings to generate from the model.",
        default="context_embedding",
        choices=["initial_embedding", "context_embedding"],
    )
    group.add_argument(
        "-m",
        "--model",
        required=True,
        type=Path,
        help="Path to PreTrained Model for generating embeddings",
    )
    group.add_argument(
        "--untrained",
        action="store_true",
        help="If the model should be loaded without weights.",
    )
    group.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the embeddings.",
    )
    group.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output file to which SQL Embedding DB is be written",
    )
    group.add_argument(
        "--outtypes",
        required=True,
        nargs="+",
        choices=["df", "db", "bin"],
        help="What kind of output to produce",
    )
    group.add_argument(
        "--batch-size",
        type=int,
        default=32,
        required=False,
        help="The batch size for generating embeddings.",
    )
    group.add_argument(
        "--matmul-precision",
        choices=["highest", "high", "medium"],
        default="medium",
        required=False,
    )

    return group


def configure_extract_embeddings_options(
    parser: argparse.ArgumentParser,
) -> argparse._ArgumentGroup:

    group = parser.add_argument_group(
        "Configure Extract Embeddings",
        "Optional Arguments to configure Extract Embeddings",
    )
    # create mutually exclusive argument group
    # either random sampling, or window sparcity sampling
    sampling = group.add_mutually_exclusive_group(required=False)
    sampling.add_argument(
        "--random-sample-size",
        type=int,
        required=False,
        help="Number of randomly selected genes to generate embeddings for.",
    )
    sampling.add_argument(
        "--sparcity-threshold",
        type=float,
        required=False,
        help="Theshold fraction of annotated genes in a window to generate embeddings for",
    )
    sampling.add_argument(
        "--kofam-classes",
        nargs=3,
        type=int,
        required=False,
        help="3 ints: Top N Kofam classes, min number of genes per class, max number of genes per class",
    )
    group.add_argument(
        "--mask-middle-gene",
        action="store_true",
        help="Mask the middle Gene to create context-only embeddings",
    )
    group.add_argument(
        "--shuffle-gene-order",
        action="store_true",
        help="Shuffle the genes in the window around the middle to test Gene order versus proximity",
    )
    group.add_argument(
        "--shuffle-strand-order",
        action="store_true",
        help="Shuffle the srand information in the window to test if strand information is predictive",
    )

    return group


def configure_model_eval_input(
    parser: argparse.ArgumentParser,
) -> argparse._ArgumentGroup:

    group = parser.add_argument_group(
        "Configure Inference",
        "Optional Arguments to configure Inference",
    )
    group.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="The database to test on",
    )
    group.add_argument(
        "-m",
        "--model",
        required=True,
        type=Path,
        help="Path to the Model to be tested. Expecting .pt file",
    )
    group.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="The path where to write the results of testing",
    )
    group.add_argument(
        "--matmul-precision", choices=["highest", "high", "medium"], default="medium"
    )
    group.add_argument(
        "--batch-size", type=int, default=32, help="The batch size for testing."
    )

    return group
