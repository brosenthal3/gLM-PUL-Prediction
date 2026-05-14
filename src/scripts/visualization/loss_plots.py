import polars
import matplotlib.pyplot as plt
import os
from palettable.cartocolors.qualitative import Bold_10

def plot_loss(pfam, cazy, out_path, model_name):
    df_pfam = polars.read_csv(pfam)
    df_cazy = polars.read_csv(cazy)

    # Figure size in inches
    cm = 1 / 2.54  # centimeters in inches
    textwidth = 19 * cm

    figsize = (textwidth, cm*16)
    dpi = 300
    xlim = None  # can set manually if you want
    ylim_loss = (0, 0.5)
    ylim_pred = (0, 6.5)
    left, right, bottom, top = 0.1, 0.9, 0.15, 0.9  # fixed axes layout
    fig, axis = plt.subplots(2, 1, figsize=figsize, sharex=True)
    bold10 = Bold_10.mpl_colors

    for i, df in enumerate([df_pfam, df_cazy]):
        axs = axis[i]
        train_df = (
            df.filter(polars.col("train_loss").is_not_null())
        )

        if "val_loss" in df.columns:
            val_df = df.filter(polars.col("val_loss").is_not_null())
            axs.scatter(val_df["step"].to_numpy(), val_df["val_loss"].cast(polars.Float32).to_numpy(),label="Validation Loss", color="tab:orange", s=25, zorder=20)

        if "test_loss" in df.columns:
            test_df = df.filter(polars.col("test_loss").is_not_null())
            axs.scatter(test_df["step"].to_numpy(), test_df["test_loss"].cast(polars.Float32).to_numpy(), label="Test Loss", color="tab:green", s=25, zorder=20)

        axs.plot(train_df["step"].to_numpy(), train_df["train_loss"].cast(polars.Float32).to_numpy(), label="Train Loss", linestyle='-', linewidth=1, zorder=10)

        axs.set_ylabel("Loss")
        axs.set_ylim(ylim_loss)
        axs.legend(loc="upper right")
        axs.set_title("Loss in training " + model_name + (" (Pfam features)" if i==0 else " (CAZy+Pfam features)"))

    axis[1].set_xlabel("Step")
    print(f"Saving plot to {out_path}...")
    plt.tight_layout()
    plt.savefig(out_path)


out_path_masked = "results/plots/loss_plots/train_loss_masked.png"
pfam_masked = "src/data/results/genecat_finetuned_pfam_masked/logs_fold_0/wandb/latest-run/files/pfam_fold_0_finetune_log_al0c8pxe/version_0/metrics.csv"
cazy_masked = "src/data/results/genecat_finetuned_cazy_masked/logs_fold_0/wandb/latest-run/files/cazy_fold_0_finetune_log_bqrf1ino/version_0/metrics.csv"
plot_loss(pfam_masked, cazy_masked, out_path_masked, "genecat_finetuned_masked")

out_path = "results/plots/loss_plots/train_loss.png"
pfam = "src/data/results/genecat_finetuned_pfam/logs_fold_0/wandb/latest-run/files/pfam_fold_0_finetune_log_txdd0ge4/version_0/metrics.csv"
cazy = "src/data/results/genecat_finetuned_cazy/logs_fold_0/wandb/latest-run/files/cazy_fold_0_finetune_log_q8uzmlkl/version_0/metrics.csv"
plot_loss(pfam, cazy, out_path, "genecat_finetuned")