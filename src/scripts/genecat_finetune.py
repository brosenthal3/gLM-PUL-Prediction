from __future__ import annotations

import multiprocessing
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from pathlib import Path

import anndata
import polars
import pytorch_lightning as lightning
import torch
import torch.cuda
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint, RichProgressBar)
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from rich.console import Console

from genecat.database import (FeatureTable, GeneTable, LabeledGeneTable,
                              LabeledGenomeDatabase)
from genecat.database.vocab import Vocab
from genecat.model import TrainValTestDataModule
from genecat.model.defaults import FinetuneOptimizerConfig
from genecat.model.finetuning import (FineTuneBinaryModel,
                                      FineTuneMultiClassModel,
                                      _BaseFineTuneModel)
from genecat.model.model_registry import load_model

from ._parser import (configure_bgc_finetuneing_test,
                      configure_bgc_finetuning_input, configure_FineTuneModel,
                      configure_general_arguments, configure_WeightsAndBiases)
from ._utils import create_versioned_path, show_device_summary

"""
export PYTHONPATH='/home/ray/Documents/GeneCAT/src/'
MODEL='/home/ray/Documents/GeneCAT/models_model_test/model_test_uyj4bjmg_v0.pt'
VOCAB='/home/ray/Documents/GeneCAT/data/test_data/genes_1000.train.vocab.txt'
GENES='/home/ray/Documents/GeneCAT/data/bgc_cluster_data/mibig-1.3.proG2.fna.top_genes.tsv'
DOMAINS='/home/ray/Documents/GeneCAT/data/bgc_cluster_data/mibig-1.3.proG2.fna.top_features.tsv'
CLUSTERS='/home/ray/Documents/GeneCAT/data/bgc_cluster_data/mibig-1.3.proG2.clusters.tsv'

python -m genecat.cli bgc-finetune -g $GENES -d $DOMAINS -c $CLUSTERS --vocab $VOCAB -m $MODEL -o bgc_model --batch-size 10 -j 1 --offline --name somebs --middle-focus --test-gene-table $GENES --test-domain-table $DOMAINS --test-cluster-table $CLUSTERS

python -m genecat.cli bgc-finetune -g $GENES -d $DOMAINS -c $CLUSTERS --vocab $VOCAB -m $MODEL -o bgc_model --batch-size 10 -j 1 --offline --name test --test-gene-table $GENES --test-domain-table $DOMAINS --test-cluster-table $CLUSTERS
"""


def configure_parser(parser: ArgumentParser):
    """Configure the main parser with argument groups for different namespaces"""

    configure_general_arguments(parser)

    configure_bgc_finetuning_input(parser)

    configure_bgc_finetuneing_test(parser)

    configure_WeightsAndBiases(parser)

    configure_FineTuneModel(parser)

    parser.set_defaults(run=run)


def join_gene_and_cluster_table(
    gene_table: GeneTable,
    cluster_table: polars.LazyFrame,
    label_col_name: str,
) -> LabeledGeneTable:
    """
    Join a gene table with a BGC cluster table.

    Note:
        For the BGC benchmark we get a regular gene table and a regular feature table.
        Then we get a special cluster table which is organized by contig.
        In the Cluster table Start and End coordinates of the contig indicate
        which genes blong to a BGC. Additionally there is a single BGC type label.
    """
    labled_gene_table = (
        cluster_table
        # to avoid conflict with start and end in gene table!
        .rename({"start": "cluster_start", "end": "cluster_end"})
        .join(
            gene_table.table,
            on="sequence_id",
            how="inner",
            validate="1:m",
        )
        .with_columns(
            # TODO there might be a minor error here if the cluster table is 1-based
            polars.when(
                polars.col("start") >= polars.col("cluster_start"),
                polars.col("end") <= polars.col("cluster_end"),
            )
            .then(True)
            .otherwise(False)
            .cast(polars.String)
            .alias("is_BGC")
        )
        .sort(by=["sequence_id", "start", "end"])
        .with_row_index(name="gene_id", offset=0)  # important
        .select(["sequence_id", "protein_id", "start", "end", "strand", label_col_name])
    )

    return LabeledGeneTable(labled_gene_table, label_column=label_col_name)


def run(args: Namespace, console: Console) -> int:
    """Run function for the CLI."""

    multiprocessing.set_start_method("spawn")

    lightning.seed_everything(42)

    torch.set_float32_matmul_precision(args.matmul_precision)

    show_device_summary()

    if args.untrained:
        console.print("Loading untrained default Model")
        pretrained_model, hyper_params = load_model(
            path=args.model,
            with_weights=False,  # do not load the model weights
            return_hyperparams=True,
        )
    else:
        console.print("Loading pretrained Model")
        pretrained_model, hyper_params = load_model(
            path=args.model,
            with_weights=True,  # load the pretrained model
            return_hyperparams=True,
        )

    config = FinetuneOptimizerConfig.from_namespace(args)

    console.print("Loading training/validation data")
    train_gene_table = GeneTable.load(args.gene_table)
    train_domain_table = FeatureTable.load(args.domain_table)
    train_cluster_table = polars.scan_csv(args.cluster_table, separator="\t").select(
        ["sequence_id", "start", "end", "type"]
    )

    finetune_model: _BaseFineTuneModel
    if args.task_type == "binary":
        console.print("Initializing binary Finetuning Model")
        finetune_model = FineTuneBinaryModel(
            pretrained_model=pretrained_model,
            middle_focus=args.middle_focus,
            config=config,
        )
        early_stopping_metric = "val_auroc"
        label_col_name = "is_BGC"

    elif args.task_type == "multiclass":

        raise NotImplementedError(
            "Before doing multiclass, figure out if you want to do binary prediction of BGCs first. Or if you want to filter out non-BGC genes!"
        )
        console.print("Initializing multiclass Finetuning Model")
        train_cluster_table = (
            train_cluster_table.with_columns(
                polars.col("type")
                .str.split(";")
                .list.first()  # NOTE the crude handling to force multiclass!
            )
            .filter(polars.col("type") != "Unknown")
            .unique("type")
        )
        early_stopping_metric = "val_macro_f1"
        label_col_name = "type"

    train_labeled_genes = join_gene_and_cluster_table(
        train_gene_table,
        train_cluster_table,
        label_col_name=label_col_name,
    )
    label_vocab = train_labeled_genes.build_label_vocab()

    if args.task_type == "multiclass":
        finetune_model = FineTuneMultiClassModel(
            pretrained_model=pretrained_model,
            middle_focus=args.middle_focus,
            n_classes=len(label_vocab),
            config=config,
        )

    with args.vocab.open() as f:
        vocab = Vocab.load(f)

    console.print("Building LabeledGenomeDatabase for training!")
    labeled_genome_db = LabeledGenomeDatabase.build_labeled_database(
        labeled_genes=train_labeled_genes,
        pfam_features=train_domain_table,
        pfam_vocab=vocab,
        label_vocab=label_vocab,
        output=":memory:",
        console=console,
    )

    finetune_model.check_model_params()

    console.print("Building SQLFinetuneDataset for testing!")
    labeled_dataset = finetune_model.build_dataset(
        pretrained_model=pretrained_model,
        label_type=args.task_type,
        database=labeled_genome_db,
    )

    if (
        args.test_gene_table is not None
        and args.test_domain_table is not None
        and args.test_cluster_table is not None
    ):
        console.print("Loading testing data")
        test_gene_table = GeneTable.load(args.test_gene_table)
        test_domain_table = FeatureTable.load(args.test_domain_table)
        test_cluster_table = (
            polars.scan_csv(args.cluster_table, separator="\t")
            .select(["sequence_id", "start", "end", "type"])
            .with_columns(
                # NOTE the crude handling to force multiclass!
                polars.col("type")
                .str.split(";")
                .list.first()
            )
        )
        test_labeled_genes = join_gene_and_cluster_table(
            gene_table=test_gene_table,
            cluster_table=test_cluster_table,
            label_col_name=label_col_name,
        )

        console.print("Building LabeledGenomeDatabase for testing!")
        test_labeled_genome_db = LabeledGenomeDatabase.build_labeled_database(
            labeled_genes=test_labeled_genes,
            pfam_features=test_domain_table,
            pfam_vocab=vocab,
            label_vocab=label_vocab,
            output=":memory:",
            console=console,
        )

        console.print("Building SQLFinetuneDataset for testing!")
        test_labeled_dataset = finetune_model.build_dataset(
            pretrained_model=pretrained_model,
            label_type=args.task_type,
            database=test_labeled_genome_db,
        )

    else:
        test_labeled_dataset = None

    datamodule = TrainValTestDataModule(
        dataset_train_val=labeled_dataset,
        dataset_test=test_labeled_dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle=True,
        pin_memory=True,
        num_workers=0,  # jobs_to_workers(args.jobs),
        collate_fn=finetune_model.collate_fn,
        persistent_workers=False,  # jobs_to_workers(args.jobs) > 0,
    )

    args.output.parent.mkdir(exist_ok=True, parents=True)
    log_save_dir = Path(args.output.parent, f"logs_{args.output.stem}")
    log_save_dir.mkdir(parents=True, exist_ok=True)

    wandb_logger = WandbLogger(
        name=args.name,
        project="GeneCAT_BGC",
        log_model=(False if args.offline else "all"),
        prefix="finetune",
        save_dir=log_save_dir,
        offline=args.offline,
        tags=[
            "FineTuning",
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        ]
        + args.tags,
    )
    wandb_logger.experiment.config.update(
        {
            "name": wandb_logger._name,
            "project": wandb_logger._project,
            "run_id": wandb_logger.experiment.id,
            "task_type": args.task_type,
            "untrained": args.untrained,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "matmul_precision": args.matmul_precision,
            "gene_table": str(args.gene_table.absolute()),
            "domain_table": str(args.domain_table.absolute()),
            "cluster_table": str(args.cluster_table.absolute()),
            "output": str(args.output.absolute()),
            "resume": None if args.resume is None else str(args.resume.absolute()),
            "seed": int(args.seed),
            "middle_focus": args.middle_focus,
            **asdict(config),
        }
    )
    ckpt_fn = wandb_logger.experiment.name + "-" + wandb_logger.experiment.id

    # ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir,
        filename=ckpt_fn + "-{epoch:02d}-{step}-{val_loss:.2f}",
    )

    trainer = lightning.Trainer(
        accelerator="auto",
        precision="32-true",
        logger=[
            CSVLogger(
                wandb_logger.experiment.dir,
                name=f"{wandb_logger._name}_finetune_log_{wandb_logger.experiment.id}",
                flush_logs_every_n_steps=100,
            ),
            wandb_logger,
        ],
        min_epochs=1,
        max_epochs=args.epochs,
        val_check_interval=0.25,
        callbacks=[
            RichProgressBar(),
            LearningRateMonitor(logging_interval="step"),
            EarlyStopping(monitor=early_stopping_metric, mode="max", verbose=True),
            checkpoint_callback,
        ],
    )

    # train and save model
    console.print("[bold blue]Finetuning Model")
    trainer.fit(model=finetune_model, datamodule=datamodule, ckpt_path=args.resume)

    model_save_dir = Path(args.output.parent, f"models_{args.output.stem}")
    mdl_filepath = Path(
        model_save_dir, f"model_{args.name}_{wandb_logger.experiment.id}.pt"
    )
    saved_mdl_path = create_versioned_path(mdl_filepath)

    best_model: _BaseFineTuneModel
    if args.task_type == "binary":
        best_model = FineTuneBinaryModel.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
    elif args.task_type == "multiclass":
        best_model = FineTuneMultiClassModel.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
    model_info = dict(wandb_logger.experiment.config)

    with open(saved_mdl_path, "wb") as f:
        best_model.save_model(file=f, info=model_info)
    console.print(f"[bold blue]{'Saving model':>12}[/] to {saved_mdl_path.resolve()}")

    if test_labeled_dataset is not None:
        console.print("[bold blue]Testing Finetuned Model")
        trainer.test(model=best_model, datamodule=datamodule)

        adata = anndata.AnnData(
            # TODO if you do multiclass you need to fix this too
            # because X needs to be 2dimensional
            X=best_model.test_probas.reshape(-1, 1),  # type: ignore
            obs=test_labeled_genes.table.collect().to_pandas(),
            uns={
                "pretrained_model_path": str(args.model.resolve()),
                "pretrained_model_info": hyper_params["model_info"],
                "finetune_model_path": str(saved_mdl_path.resolve()),
                "finetune_model_info": model_info,
                "test_gene_table": str(args.test_gene_table.resolve()),
                "test_domain_table": str(args.test_domain_table.resolve()),
                "test_cluster_table": str(args.test_cluster_table.resolve()),
            },
        )
        filename = Path(wandb_logger.experiment.dir, "bgc_predictions.h5ad")
        adata.write(
            filename=filename,
            compression="gzip",
        )
        console.print(
            f"[bold blue]{'Saving middle gene BGC prediction to':>12}[/] to {filename.resolve()}"
        )

    console.print(f"[bold green]{'Finished':>12}[/] finetuning model.")

    return 0
