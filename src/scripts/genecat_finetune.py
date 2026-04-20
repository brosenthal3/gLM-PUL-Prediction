from __future__ import annotations

import multiprocessing
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from pathlib import Path
from rich.console import Console

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

from scripts._genecat_parser import (configure_bgc_finetuneing_test,
                      configure_bgc_finetuning_input, configure_FineTuneModel,
                      configure_general_arguments, configure_WeightsAndBiases)
from scripts._genecat_utils import create_versioned_path, show_device_summary
from utility_scripts import join_gene_and_PUL_table

"""
export PYTHONPATH='/home/ray/Documents/GeneCAT/src/'
MODEL='/home/ray/Documents/GeneCAT/models_model_test/model_test_uyj4bjmg_v0.pt'
VOCAB='/home/ray/Documents/GeneCAT/data/test_data/genes_1000.train.vocab.txt'
GENES='/home/ray/Documents/GeneCAT/data/bgc_cluster_data/mibig-1.3.proG2.fna.top_genes.tsv'
DOMAINS='/home/ray/Documents/GeneCAT/data/bgc_cluster_data/mibig-1.3.proG2.fna.top_features.tsv'
CLUSTERS='/home/ray/Documents/GeneCAT/data/bgc_cluster_data/mibig-1.3.proG2.clusters.tsv'

python -m genecat.cli bgc-finetune -g $GENES -d $DOMAINS -c $CLUSTERS --vocab $VOCAB\
     -m $MODEL -o bgc_model --batch-size 10 -j 1 --offline --name test --test-gene-table\
        $GENES --test-domain-table $DOMAINS --test-cluster-table $CLUSTERS

## PULS ##
export PYTHONPATH='/home/benr/thesis/genecat/src/:/home/benr/thesis/gLM-PUL-Prediction/src/'
python src/scripts/genecat_finetune.py -g src/data/genecat_output/genome.genes.parquet -d src/data/genecat_output/genome.features.parquet 
-c src/data/splits/train_fold_0.tsv --vocab src/data/genecat_output/test_vocab.txt -m src/data/models/model_test.pt -o src/data/results/genecat/fine_tuned --batch-size 10 -j 1 --offline --name test
"""


def configure_parser(parser: ArgumentParser):
    """Configure the main parser with argument groups for different namespaces"""

    configure_general_arguments(parser)
    configure_bgc_finetuning_input(parser)
    configure_bgc_finetuneing_test(parser)
    configure_WeightsAndBiases(parser)
    configure_FineTuneModel(parser)
    parser.set_defaults(run=run)

def reset_start_end(table: polars.DataFrame) -> polars.DataFrame:
    return table.with_columns(
        polars.when(polars.col("start") < polars.col("end")).then(polars.col("start")).otherwise(polars.col("end")).alias("start"),
        polars.when(polars.col("start") < polars.col("end")).then(polars.col("end")).otherwise(polars.col("start")).alias("end"),
    )

def join_gene_and_cluster_table(
    gene_table: GeneTable,
    cluster_table: polars.LazyFrame,
    label_col_name: str = "is_PUL",
    buffer: int = 100 # allow for buffer around PUL boundaries
) -> LabeledGeneTable:
    gene_table = reset_start_end(gene_table.table)
    cluster_table = reset_start_end(cluster_table)

    labled_gene_table = (
        cluster_table
        .rename({"start": "pul_start", "end": "pul_end"}) # avoid column name conflicts
        .join(
            gene_table,
            on="sequence_id",
            how="inner",
            validate="m:m",
        )
        .with_columns(
            polars.when(
                polars.col("start") >= polars.col("pul_start") - buffer,
                polars.col("end") <= polars.col("pul_end") + buffer,
            )
            .then(True)
            .otherwise(False)
            .cast(polars.Boolean)
            .alias(label_col_name)
        )
        # aggregate by protein_id to determine if protein is in any PUL
        .group_by("protein_id")
        .agg(
            polars.col(label_col_name).any().alias(label_col_name),
            polars.col("sequence_id").first().alias("sequence_id"),
            polars.col("start").first().alias("start"),
            polars.col("end").first().alias("end"),
            polars.col("strand").first().alias("strand")
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
        ["sequence_id", "start", "end", "cluster_id"]
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
        label_col_name = "is_PUL"
    else: 
        raise NotImplementedError("Only binary classification available")

    train_labeled_genes = join_gene_and_cluster_table(
        train_gene_table,
        train_cluster_table,
        label_col_name=label_col_name,
    )
    label_vocab = train_labeled_genes.build_label_vocab()

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
            .select(["sequence_id", "start", "end", "cluster_id"])
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
        project="GeneCAT_PUL",
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
            checkpoint_callback.best_model_path,
            weights_only=False,
        )
    model_info = dict(wandb_logger.experiment.config)

    with open(saved_mdl_path, "wb") as f:
        best_model.save_model(file=f, info=model_info)
    console.print(f"[bold blue]{'Saving model':>12}[/] to {saved_mdl_path.resolve()}")

    if test_labeled_dataset is not None:
        console.print("[bold blue]Testing Finetuned Model")
        trainer.test(model=best_model, datamodule=datamodule)

        adata = anndata.AnnData(
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
        filename = Path(wandb_logger.experiment.dir, "pul_predictions.h5ad")
        adata.write(
            filename=filename,
            compression="gzip",
        )
        console.print(
            f"[bold blue]{'Saving middle gene PUL prediction to':>12}[/] to {filename.resolve()}"
        )

    console.print(f"[bold green]{'Finished':>12}[/] finetuning model.")

    return 0


if __name__ == "__main__":
    console = Console()
    parser = ArgumentParser(description="Fine-tune a pretrained GeneCAT model on BGC data")
    configure_parser(parser)
    args = parser.parse_args()

    run(args, console)

